#!/usr/bin/env python3
"""Train an STC noise shaper and/or flow-guided motion adapter on frozen SD."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.models.brushnet import BrushNetModel
from diffusers.models.brushnet_motion_adapter import BrushNetFlowMotionAdapter
from diffusers.models.stc_noise_shaper import STCConditionedNoiseShaper
from diffusers.models.motion_adapter_training import (
    FrozenBrushNetMotionModel,
    build_flow_confidence,
    build_stable_bg_confidence,
    temporal_warp_loss,
)
from motion_adapter_dataset import BrushNetMotionClipDataset
from ip_adapter import FusionIPAdapter
from transformers import CLIPTextModel, CLIPTokenizer

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Checkpoint directory. Overrides model.resume in the JSON config.",
    )
    parser.add_argument(
        "--device",
        help="For example cuda, cuda:0, or cpu. Overrides trainer.device.",
    )
    return parser.parse_args()


def setup_logger(log_path: Path, enabled=True):
    logger = logging.getLogger("motion_adapter")
    logger.handlers.clear()
    logger.propagate = False
    if not enabled:
        logger.setLevel(logging.CRITICAL)
        logger.addHandler(logging.NullHandler())
        return logger
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    for handler in (
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def distributed_context(requested_device):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(requested_device)
    return distributed, rank, local_rank, world_size, device


def global_average(totals, count, device):
    keys = sorted(totals)
    values = [float(totals[key]) for key in keys] + [float(count)]
    packed = torch.tensor(values, device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    denominator = packed[-1].clamp_min(1.0)
    return {
        key: float(packed[index] / denominator)
        for index, key in enumerate(keys)
    }


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataset(config, split, random_horizontal_flip):
    data = config["data"]
    split_stride = data.get(f"{split}_stride", data.get("stride", 1))
    flow_root = data.get("flow_root", data.get("refined_flow_root"))
    if not flow_root:
        raise ValueError("data.flow_root or data.refined_flow_root is required")
    return BrushNetMotionClipDataset(
        dataset_root=data["dataset_root"],
        refined_flow_root=flow_root,
        manifest=data["manifest"],
        split=split,
        clip_length=data["clip_length"],
        height=data.get("height", 512),
        width=data.get("width", 512),
        stride=split_stride,
        random_horizontal_flip=random_horizontal_flip,
        caption=data.get("caption", ""),
        flow_source=data.get("flow_source", "refined"),
    )


def frozen(module, device, dtype):
    module.requires_grad_(False).eval()
    return module.to(device=device, dtype=dtype)


def encode_images(vae, images, stochastic=True):
    shape = images.shape
    flat = images.reshape(-1, *shape[2:])
    distribution = vae.encode(flat).latent_dist
    latents = distribution.sample() if stochastic else distribution.mode()
    latents = latents * vae.config.scaling_factor
    return latents.reshape(*shape[:2], *latents.shape[1:])


def make_conditioning(decoded_latents, roi_masks, conditioning_channels):
    if conditioning_channels != 5:
        raise ValueError(
            "This V1 trainer expects the deployed 5-channel BrushNet condition "
            "(4 decoded-image latents + 1 background mask), but the loaded "
            f"BrushNet expects {conditioning_channels} channels."
        )
    bg = 1.0 - roi_masks
    bg = F.interpolate(
        bg.flatten(0, 1),
        size=decoded_latents.shape[-2:],
        mode="nearest",
    ).reshape(*bg.shape[:2], 1, *decoded_latents.shape[-2:])
    return torch.cat((decoded_latents, bg), dim=2), bg


def make_temporal_noise(latents, bg_mask, shared_bg_strength):
    independent = torch.randn_like(latents)
    if shared_bg_strength <= 0:
        return independent
    rho = min(float(shared_bg_strength), 1.0)
    shared = torch.randn_like(latents[:, :1]).expand_as(latents)
    mixed_bg = math.sqrt(1.0 - rho) * independent + math.sqrt(rho) * shared
    return independent * (1.0 - bg_mask) + mixed_bg * bg_mask


def prediction_target(noise_scheduler, latents, noise, timesteps):
    flat_latents = latents.flatten(0, 1)
    flat_noise = noise.flatten(0, 1)
    flat_timesteps = timesteps.repeat_interleave(latents.shape[1])
    prediction_type = noise_scheduler.config.prediction_type
    if prediction_type == "epsilon":
        target = flat_noise
    elif prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(
            flat_latents, flat_noise, flat_timesteps
        )
    else:
        raise ValueError(f"Unsupported prediction type: {prediction_type}")
    return target.reshape_as(noise)


def predicted_clean_latents(
    noise_scheduler, noisy_latents, model_prediction, timesteps
):
    alpha = noise_scheduler.alphas_cumprod.to(
        device=noisy_latents.device, dtype=noisy_latents.dtype
    )[timesteps]
    alpha = alpha[:, None, None, None, None]
    sigma = (1.0 - alpha).sqrt()
    sqrt_alpha = alpha.sqrt()
    prediction_type = noise_scheduler.config.prediction_type
    if prediction_type == "epsilon":
        return (noisy_latents - sigma * model_prediction) / sqrt_alpha.clamp_min(
            1e-6
        )
    if prediction_type == "v_prediction":
        return sqrt_alpha * noisy_latents - sigma * model_prediction
    raise ValueError(f"Unsupported prediction type: {prediction_type}")


class _UNetCarrier:
    """Minimal object required by the repository's FusionIPAdapter loader."""

    def __init__(self, unet):
        self.unet = unet

    def to(self, *args, **kwargs):
        self.unet.to(*args, **kwargs)
        return self


def build_ip_conditioner(unet, config, device):
    ip_config = config.get("ip_adapter", {})
    if not ip_config.get("enabled", False):
        return None
    conditioner = FusionIPAdapter(
        _UNetCarrier(unet),
        ip_config["image_encoder"],
        ip_config["weights"],
        ip_config["fusion_weights"],
        device,
        num_tokens=int(ip_config.get("num_tokens", 4)),
    )
    for module in (
        conditioner.image_encoder,
        conditioner.image_proj_model,
        conditioner.fusion_module,
    ):
        module.requires_grad_(False).eval()
    conditioner.set_scale(float(ip_config.get("scale", 1.0)))
    return conditioner


def _masked_pil_frames(decoded_frames, roi_masks):
    images = ((decoded_frames.float() + 1.0) * 127.5).clamp(0, 255)
    roi = roi_masks.float()
    foreground = images * roi
    background = images * (1.0 - roi)

    def convert(tensor):
        arrays = (
            tensor.flatten(0, 1)
            .permute(0, 2, 3, 1)
            .byte()
            .cpu()
            .numpy()
        )
        return [Image.fromarray(array, mode="RGB") for array in arrays]

    return convert(foreground), convert(background)


def text_embeddings(
    tokenizer,
    text_encoder,
    captions,
    device,
    decoded_frames,
    roi_masks,
    ip_conditioner=None,
):
    tokens = tokenizer(
        list(captions),
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    text = text_encoder(tokens.input_ids.to(device), return_dict=False)[0]
    if ip_conditioner is None:
        return text
    foreground, background = _masked_pil_frames(decoded_frames, roi_masks)
    image_tokens, _ = ip_conditioner.get_fgbg_image_embeds(
        fg_pil_image=foreground,
        bg_pil_image=background,
    )
    batch, frames = decoded_frames.shape[:2]
    text = text[:, None].expand(batch, frames, *text.shape[1:])
    image_tokens = image_tokens.reshape(
        batch, frames, *image_tokens.shape[1:]
    ).to(device=device, dtype=text.dtype)
    return torch.cat((text, image_tokens), dim=2)


def cosine_schedule(warmup_steps, total_steps, min_ratio):
    def schedule(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step + 1) / warmup_steps, 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return schedule


def compute_batch(
    batch,
    model,
    vae,
    tokenizer,
    text_encoder,
    noise_scheduler,
    device,
    weight_dtype,
    loss_config,
    stochastic_latents,
    ip_conditioner=None,
    noise_shaper=None,
    noise_shaper_config=None,
):
    decoded_cpu = batch["decoded_frames"]
    roi_cpu = batch["roi_masks"]
    gt = batch["gt_frames"].to(device=device, dtype=weight_dtype)
    decoded = decoded_cpu.to(device=device, dtype=weight_dtype)
    roi = roi_cpu.to(device=device, dtype=weight_dtype)
    flow_forward = batch["flow_forward"].to(
        device=device, dtype=weight_dtype
    )
    flow_backward = batch["flow_backward"].to(device=device, dtype=weight_dtype)
    if "motion_confidence" in batch and "temporal_confidence" in batch:
        motion_confidence = batch["motion_confidence"].to(
            device=device, dtype=weight_dtype
        )
        temporal_confidence = batch["temporal_confidence"].to(
            device=device, dtype=weight_dtype
        )
    else:
        flow_forward_float = flow_forward.float()
        flow_backward_float = flow_backward.float()
        batch_size, frames = roi.shape[:2]
        flow_height, flow_width = flow_backward.shape[-2:]
        bg_flow = F.interpolate(
            (1.0 - roi.float()).flatten(0, 1),
            size=(flow_height, flow_width),
            mode="nearest",
        ).reshape(batch_size, frames, 1, flow_height, flow_width)
        motion_confidence = build_flow_confidence(
            flow_forward_float,
            flow_backward_float,
        )
        temporal_confidence = build_stable_bg_confidence(
            flow_forward_float,
            flow_backward_float,
            bg_flow,
            flow_confidence=motion_confidence,
        )
        motion_confidence = motion_confidence.to(weight_dtype)
        temporal_confidence = temporal_confidence.to(weight_dtype)
    with torch.no_grad():
        latents = encode_images(vae, gt, stochastic=stochastic_latents)
        decoded_latents = encode_images(
            vae, decoded, stochastic=stochastic_latents
        )
        conditioning, bg_latent = make_conditioning(
            decoded_latents,
            roi,
            int(model.brushnet.config.conditioning_channels),
        )
        text = text_embeddings(
            tokenizer,
            text_encoder,
            batch["caption"],
            device,
            decoded_cpu,
            roi_cpu,
            ip_conditioner,
        )

    batch_size, frames = latents.shape[:2]
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (batch_size,),
        device=device,
        dtype=torch.long,
    )
    noise_shaper_config = noise_shaper_config or {}
    shaped = None
    if noise_shaper is not None:
        shaped = noise_shaper(
            independent_noise=torch.randn_like(latents),
            decoded_latents=decoded_latents.detach(),
            bg_mask=bg_latent.detach(),
            backward_flow=flow_backward,
            motion_confidence=motion_confidence,
            stable_bg=temporal_confidence,
            strength=float(noise_shaper_config.get("strength", 1.0)),
        )
        noise = shaped["noise"]
    else:
        noise = make_temporal_noise(
            latents,
            bg_latent,
            loss_config.get("shared_bg_noise_strength", 1.0),
        )
    noisy = noise_scheduler.add_noise(
        latents.flatten(0, 1),
        noise.flatten(0, 1),
        timesteps.repeat_interleave(frames),
    ).reshape_as(latents)
    target_noise = (
        noise.detach()
        if noise_shaper is not None
        and noise_shaper_config.get("detach_target", True)
        else noise
    )
    target = prediction_target(
        noise_scheduler,
        latents,
        target_noise,
        timesteps,
    )

    prediction, residual_reg = model(
        noisy_latents=noisy,
        timesteps=timesteps,
        encoder_hidden_states=text,
        conditioning_latents=conditioning,
        flow_backward=flow_backward,
        motion_confidence=motion_confidence,
        motion_scale=loss_config.get("motion_scale", 1.0),
    )
    noise_loss = F.mse_loss(prediction.float(), target.float())
    clean_prediction = predicted_clean_latents(
        noise_scheduler, noisy, prediction, timesteps
    )
    temporal_loss = temporal_warp_loss(
        clean_prediction.float(),
        flow_backward.float(),
        temporal_confidence.float(),
        charbonnier_eps=loss_config.get("charbonnier_eps", 1e-3),
    )
    temporal_timestep_weight = (
        noise_scheduler.alphas_cumprod.to(device=device)[timesteps]
        .sqrt()
        .mean()
        if loss_config.get("temporal_timestep_weighting", True)
        else temporal_loss.new_ones(())
    )
    weighted_temporal_loss = temporal_timestep_weight * temporal_loss
    flow_prior_loss = (
        shaped["flow_residual_energy"]
        if shaped is not None
        else noise_loss.new_zeros(())
    )
    noise_stats_loss = (
        shaped["noise_mean"].square()
        + (shaped["noise_std"] - 1.0).square()
        if shaped is not None
        else noise_loss.new_zeros(())
    )
    if shaped is not None:
        gate_target = float(noise_shaper_config.get("gate_target_beta", 0.8))
        gate_prior_loss = (shaped["beta_mean"] - gate_target).square()
    else:
        gate_prior_loss = noise_loss.new_zeros(())
    total = (
        float(loss_config.get("noise_weight", 1.0)) * noise_loss
        + float(loss_config.get("temporal_weight", 0.05))
        * weighted_temporal_loss
        + float(loss_config.get("residual_weight", 1e-4)) * residual_reg
        + float(loss_config.get("flow_prior_weight", 0.01))
        * flow_prior_loss
        + float(loss_config.get("noise_stats_weight", 0.01))
        * noise_stats_loss
        + float(loss_config.get("gate_prior_weight", 0.01))
        * gate_prior_loss
    )
    return {
        "total": total,
        "noise": noise_loss,
        "temporal": temporal_loss,
        "temporal_weighted": weighted_temporal_loss,
        "residual": residual_reg,
        "confidence_ratio": motion_confidence.float().mean(),
        "stable_bg_ratio": temporal_confidence.float().mean(),
        "flow_prior": flow_prior_loss,
        "noise_stats": noise_stats_loss,
        "gate_prior": gate_prior_loss,
        "noise_beta": (
            shaped["beta_mean"] if shaped is not None else noise_loss.new_zeros(())
        ),
        "noise_effective_beta": (
            shaped["effective_beta_mean"]
            if shaped is not None
            else noise_loss.new_zeros(())
        ),
        "noise_warp": (
            shaped["noise_warp_error"]
            if shaped is not None
            else noise_loss.new_zeros(())
        ),
        "shaped_noise_mean_abs": (
            shaped["noise_mean"].abs()
            if shaped is not None
            else noise.mean().abs()
        ),
        "shaped_noise_std": (
            shaped["noise_std"]
            if shaped is not None
            else noise.std(unbiased=False)
        ),
    }


@torch.no_grad()
def validate(
    loader,
    batches,
    model,
    vae,
    tokenizer,
    text_encoder,
    noise_scheduler,
    device,
    weight_dtype,
    loss_config,
    amp,
    ip_conditioner,
    validation_seed,
    noise_shaper=None,
    noise_shaper_config=None,
):
    if model.motion_adapter is not None:
        model.motion_adapter.eval()
    if noise_shaper is not None:
        noise_shaper.eval()
    totals = {}
    count = 0
    cpu_rng = torch.get_rng_state()
    cuda_rng = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )
    torch.manual_seed(validation_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(validation_seed)
    try:
        for batch in loader:
            with (
                torch.autocast("cuda", dtype=weight_dtype)
                if amp
                else nullcontext()
            ):
                losses = compute_batch(
                    batch,
                    model,
                    vae,
                    tokenizer,
                    text_encoder,
                    noise_scheduler,
                    device,
                    weight_dtype,
                    loss_config,
                    stochastic_latents=False,
                    ip_conditioner=ip_conditioner,
                    noise_shaper=noise_shaper,
                    noise_shaper_config=noise_shaper_config,
                )
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach())
            count += 1
            if batches > 0 and count >= batches:
                break
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
    if model.motion_adapter is not None:
        model.motion_adapter.train()
    if noise_shaper is not None:
        noise_shaper.train()
    return totals, count


def save_checkpoint(
    checkpoint_root,
    step,
    adapter,
    optimizer,
    scheduler,
    scaler,
    best_valid,
    config,
    noise_shaper=None,
    is_best=False,
):
    destination = checkpoint_root / f"checkpoint-{step:07d}"
    temporary = checkpoint_root / f".checkpoint-{step:07d}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    if adapter is not None:
        adapter.save_pretrained(temporary / "motion_adapter")
    if noise_shaper is not None:
        noise_shaper.save_pretrained(temporary / "noise_shaper")
    torch.save(
        {
            "format_version": 3,
            "training_mode": (
                "joint_motion_and_noise"
                if adapter is not None and noise_shaper is not None
                else "stc_noise_only"
                if noise_shaper is not None
                else "motion_adapter_only"
            ),
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_valid": best_valid,
            "config": config,
        },
        temporary / "trainer_state.pt",
    )
    temporary.replace(destination)
    latest = checkpoint_root / "latest.json"
    latest.write_text(
        json.dumps({"checkpoint": str(destination), "step": step}, indent=2),
        encoding="utf-8",
    )
    if is_best:
        (checkpoint_root / "best.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(destination),
                    "step": step,
                    "valid_total": best_valid,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return destination


def resolve_resume(config, cli_resume):
    resume = cli_resume or config.get("model", {}).get("resume")
    if not resume:
        return None
    resume = Path(resume).resolve()
    if resume.name in {"latest.json", "best.json"}:
        resume = Path(json.loads(resume.read_text())["checkpoint"])
    if not (resume / "trainer_state.pt").is_file():
        raise FileNotFoundError(f"Invalid resume checkpoint: {resume}")
    return resume


def component_checkpoint(path, component):
    if not path:
        return None
    path = Path(path).resolve()
    if path.name in {"latest.json", "best.json"}:
        path = Path(json.loads(path.read_text(encoding="utf-8"))["checkpoint"])
    nested = path / component
    return nested if nested.is_dir() else path


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    trainer = config["trainer"]
    distributed, rank, local_rank, world_size, device = distributed_context(
        args.device or trainer.get("device", "cuda")
    )
    is_main = rank == 0
    seed_everything(int(config.get("seed", 2026)) + rank)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    amp = bool(trainer.get("amp", True) and device.type == "cuda")
    weight_dtype = torch.float16 if amp else torch.float32
    output_dir = Path(config["output_dir"]).resolve()
    checkpoint_root = Path(config["checkpoint_dir"]).resolve()
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    logger = setup_logger(
        output_dir / "logs" / "train.log",
        enabled=is_main,
    )
    writer = (
        SummaryWriter(output_dir / "tensorboard")
        if is_main and SummaryWriter
        else None
    )
    if is_main and writer is None:
        logger.warning("TensorBoard is unavailable; install tensorboard to enable it")
    if is_main:
        (output_dir / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

    base = config["model"]["base_model"]
    brushnet_path = config["model"]["brushnet"]
    logger.info("Loading base diffusion model %s", base)
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = frozen(
        CLIPTextModel.from_pretrained(base, subfolder="text_encoder"),
        device,
        weight_dtype,
    )
    vae = frozen(
        AutoencoderKL.from_pretrained(base, subfolder="vae"),
        device,
        weight_dtype,
    )
    unet = frozen(
        UNet2DConditionModel.from_pretrained(base, subfolder="unet"),
        device,
        weight_dtype,
    )
    ip_conditioner = build_ip_conditioner(unet, config, device)
    brushnet = frozen(
        BrushNetModel.from_pretrained(brushnet_path),
        device,
        weight_dtype,
    )
    if hasattr(unet, "enable_gradient_checkpointing") and trainer.get(
        "gradient_checkpointing", True
    ):
        unet.enable_gradient_checkpointing()

    resume = resolve_resume(config, args.resume)
    adapter_config = config.get("adapter", {})
    adapter_enabled = bool(adapter_config.get("enabled", True))
    adapter_init = config.get("model", {}).get("motion_adapter_init")
    adapter = None
    if adapter_enabled and resume:
        resume_adapter_path = resume / "motion_adapter"
        if not resume_adapter_path.is_dir():
            raise ValueError(
                f"Resume checkpoint has no motion_adapter: {resume}"
            )
        adapter = BrushNetFlowMotionAdapter.from_pretrained(resume_adapter_path)
    elif adapter_enabled and adapter_init:
        adapter = BrushNetFlowMotionAdapter.from_pretrained(
            component_checkpoint(adapter_init, "motion_adapter")
        )
    elif adapter_enabled:
        adapter = BrushNetFlowMotionAdapter.from_brushnet(
            brushnet,
            bottleneck_channels=adapter_config.get("bottleneck_channels", 64),
            flow_channels=adapter_config.get("flow_channels", 16),
            use_down=adapter_config.get("use_down", True),
            use_mid=adapter_config.get("use_mid", True),
            use_up=adapter_config.get("use_up", False),
        )
    else:
        if adapter_init:
            logger.warning(
                "Ignoring model.motion_adapter_init because adapter.enabled=false"
            )
        logger.info("Motion adapter disabled: running STC noise-only mode")
    if adapter is not None:
        adapter = adapter.to(device=device, dtype=torch.float32).train()
    noise_shaper_config = config.get("noise_shaper", {})
    noise_shaper_enabled = bool(noise_shaper_config.get("enabled", False))
    noise_shaper = None
    noise_shaper_has_resume_state = False
    if noise_shaper_enabled:
        resume_noise_path = resume / "noise_shaper" if resume else None
        noise_shaper_init = noise_shaper_config.get("checkpoint")
        if resume_noise_path is not None and resume_noise_path.is_dir():
            noise_shaper = STCConditionedNoiseShaper.from_pretrained(
                resume_noise_path
            )
            noise_shaper_has_resume_state = True
        elif noise_shaper_init:
            noise_shaper = STCConditionedNoiseShaper.from_pretrained(
                component_checkpoint(noise_shaper_init, "noise_shaper")
            )
        else:
            noise_shaper = STCConditionedNoiseShaper(
                latent_channels=int(noise_shaper_config.get("latent_channels", 4)),
                condition_channels=int(noise_shaper_config.get("condition_channels", 9)),
                hidden_channels=int(noise_shaper_config.get("hidden_channels", 64)),
                num_attention_heads=int(noise_shaper_config.get("num_attention_heads", 4)),
                num_transformer_layers=int(noise_shaper_config.get("num_transformer_layers", 1)),
                mlp_ratio=float(noise_shaper_config.get("mlp_ratio", 4.0)),
                dropout=float(noise_shaper_config.get("dropout", 0.0)),
                encoder_architecture=str(
                    noise_shaper_config.get("encoder_architecture", "native")
                ),
                condition_group_channels=tuple(
                    noise_shaper_config.get(
                        "condition_group_channels",
                        (5, 2, 2),
                    )
                ),
                videocomposer_pool_size=int(
                    noise_shaper_config.get("videocomposer_pool_size", 128)
                ),
                flow_residual_scale=float(noise_shaper_config.get("flow_residual_scale", 2.0)),
                use_refined_flow_prior=bool(noise_shaper_config.get("use_refined_flow_prior", True)),
                use_input_flow_prior=noise_shaper_config.get("use_input_flow_prior"),
                predict_flow_residual=bool(noise_shaper_config.get("predict_flow_residual", True)),
                beta_min=float(noise_shaper_config.get("beta_min", 0.0)),
                beta_max=float(noise_shaper_config.get("beta_max", 0.95)),
                initial_beta=float(noise_shaper_config.get("initial_beta", 0.10)),
                warp_region=str(
                    noise_shaper_config.get("warp_region", "stable_bg")
                ),
                channel_normalize=bool(noise_shaper_config.get("channel_normalize", False)),
                global_normalize=bool(noise_shaper_config.get("global_normalize", False)),
                norm_eps=float(noise_shaper_config.get("norm_eps", 1e-5)),
            )
        noise_shaper = noise_shaper.to(
            device=device, dtype=torch.float32
        ).train()
    if adapter is None and noise_shaper is None:
        raise ValueError(
            "No trainable module: enable adapter and/or noise_shaper"
        )
    model = FrozenBrushNetMotionModel(brushnet, unet, adapter)
    noise_shaper_runner = noise_shaper
    if distributed:
        if adapter is not None:
            model.motion_adapter = DistributedDataParallel(
                adapter,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        if noise_shaper is not None:
            noise_shaper_runner = DistributedDataParallel(
                noise_shaper,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
    if trainer.get("gradient_checkpointing", True):
        # Diffusers activates checkpointing only while the frozen UNet is in
        # train mode. Its parameters remain frozen; SD 1.5 uses zero dropout.
        model.unet.train()
        if adapter is None and hasattr(model.brushnet, "enable_gradient_checkpointing"):
            model.brushnet.enable_gradient_checkpointing()
            model.brushnet.train()

    train_dataset = make_dataset(config, "train", True)
    valid_dataset = make_dataset(config, "valid", False)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(config.get("seed", 2026)),
            drop_last=True,
        )
        if distributed
        else None
    )
    valid_sampler = (
        DistributedSampler(
            valid_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        if distributed
        else None
    )
    loader_kwargs = {
        "num_workers": int(trainer.get("num_workers", 4)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(trainer.get("batch_size", 1)),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(trainer.get("valid_batch_size", 1)),
        shuffle=False,
        sampler=valid_sampler,
        drop_last=False,
        **loader_kwargs,
    )

    lr = float(trainer.get("lr", 1e-4))
    adapter_lr = float(trainer.get("motion_adapter_lr", lr))
    noise_shaper_lr = float(trainer.get("noise_shaper_lr", lr))
    min_lr = float(trainer.get("min_lr", lr * 0.01))
    trainable_parameters = []
    parameter_groups = []
    group_names = []
    group_min_lrs = []
    if adapter is not None:
        adapter_parameters = [
            parameter for parameter in adapter.parameters() if parameter.requires_grad
        ]
        adapter_min_lr = float(
            trainer.get("motion_adapter_min_lr", min(min_lr, adapter_lr))
        )
        if not 0.0 <= adapter_min_lr <= adapter_lr:
            raise ValueError(
                "motion_adapter_min_lr must be in [0,motion_adapter_lr]"
            )
        trainable_parameters.extend(adapter_parameters)
        parameter_groups.append({"params": adapter_parameters, "lr": adapter_lr})
        group_names.append("motion_adapter")
        group_min_lrs.append(adapter_min_lr)
    if noise_shaper is not None:
        noise_shaper_parameters = [
            parameter
            for parameter in noise_shaper.parameters()
            if parameter.requires_grad
        ]
        noise_shaper_min_lr = float(
            trainer.get("noise_shaper_min_lr", min(min_lr, noise_shaper_lr))
        )
        if not 0.0 <= noise_shaper_min_lr <= noise_shaper_lr:
            raise ValueError(
                "noise_shaper_min_lr must be in [0,noise_shaper_lr]"
            )
        trainable_parameters.extend(noise_shaper_parameters)
        parameter_groups.append(
            {"params": noise_shaper_parameters, "lr": noise_shaper_lr}
        )
        group_names.append("noise_shaper")
        group_min_lrs.append(noise_shaper_min_lr)
    if not trainable_parameters:
        raise ValueError("No parameters require gradients")
    optimizer = AdamW(
        parameter_groups,
        lr=lr,
        betas=tuple(trainer.get("betas", [0.9, 0.999])),
        weight_decay=float(trainer.get("weight_decay", 0.01)),
    )
    max_steps = int(trainer["max_steps"])
    warmup_steps = int(trainer.get("warmup_steps", 500))
    lr_lambdas = [
        cosine_schedule(
            warmup_steps,
            max_steps,
            minimum / float(group["lr"]),
        )
        for group, minimum in zip(parameter_groups, group_min_lrs)
    ]
    scheduler = LambdaLR(
        optimizer,
        lr_lambdas,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    step = 0
    best_valid = math.inf
    resume_has_noise_shaper = bool(
        resume is not None and (resume / "noise_shaper").is_dir()
    )
    resume_has_adapter = bool(
        resume is not None and (resume / "motion_adapter").is_dir()
    )
    can_resume_training_state = bool(
        resume
        and resume_has_adapter == adapter_enabled
        and resume_has_noise_shaper == noise_shaper_enabled
        and (not noise_shaper_enabled or noise_shaper_has_resume_state)
    )
    if resume and not can_resume_training_state:
        logger.warning(
            "Checkpoint %s components do not match adapter.enabled=%s and "
            "noise_shaper.enabled=%s; using available weights as a warm "
            "start and resetting optimizer/step to zero",
            resume,
            adapter_enabled,
            noise_shaper_enabled,
        )
    if can_resume_training_state:
        state = torch.load(resume / "trainer_state.pt", map_location="cpu")
        saved_groups = len(state["optimizer"].get("param_groups", []))
        current_groups = len(optimizer.param_groups)
        if saved_groups != current_groups:
            raise ValueError(
                "Resume optimizer-group mismatch: checkpoint has "
                f"{saved_groups}, current config has {current_groups}. Use "
                "model.motion_adapter_init/noise_shaper.checkpoint for a "
                "weights-only warm start."
            )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step = int(state["step"])
        best_valid = float(state.get("best_valid", math.inf))
        logger.info("Resumed %s at step %d", resume, step)

    trainable = sum(parameter.numel() for parameter in trainable_parameters)
    logger.info(
        "train_clips=%d valid_clips=%d T=%d B_per_gpu=%d world=%d "
        "global_batch=%d trainable=%d adapter=%s noise_shaper=%s "
        "flow_source=%s device=%s amp=%s",
        len(train_dataset),
        len(valid_dataset),
        config["data"]["clip_length"],
        trainer.get("batch_size", 1),
        world_size,
        int(trainer.get("batch_size", 1))
        * world_size
        * int(trainer.get("gradient_accumulation_steps", 1)),
        trainable,
        adapter_enabled,
        noise_shaper_enabled,
        config["data"].get("flow_source", "refined"),
        device,
        amp,
    )
    if noise_shaper is not None:
        logger.info(
            "STC noise shaper enabled encoder=%s warp_region=%s "
            "parameters=%d beta_max=%.3f "
            "initial_beta=%.3f input_flow_prior=%s predict_flow_residual=%s "
            "channel_norm=%s global_norm=%s",
            str(getattr(noise_shaper.config, "encoder_architecture", "native")),
            str(getattr(noise_shaper.config, "warp_region", "stable_bg")),
            sum(
                parameter.numel()
                for parameter in noise_shaper.parameters()
                if parameter.requires_grad
            ),
            float(noise_shaper.config.beta_max),
            float(noise_shaper.config.initial_beta),
            bool(
                noise_shaper.config.use_input_flow_prior
                if noise_shaper.config.use_input_flow_prior is not None
                else noise_shaper.config.use_refined_flow_prior
            ),
            bool(noise_shaper.config.predict_flow_residual),
            bool(noise_shaper.config.channel_normalize),
            bool(noise_shaper.config.global_normalize),
        )
    accumulation = int(trainer.get("gradient_accumulation_steps", 1))
    log_freq = int(trainer.get("log_freq", 20))
    valid_freq = int(trainer.get("valid_freq", 500))
    save_freq = int(trainer.get("save_freq", 500))
    optimizer.zero_grad(set_to_none=True)
    sampler_epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(sampler_epoch)
    train_iterator = iter(train_loader)
    micro_step = 0
    running = {}
    running_count = 0
    started = time.time()

    while step < max_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            sampler_epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(sampler_epoch)
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        will_step = (micro_step + 1) % accumulation == 0
        with ExitStack() as sync_context:
            if distributed and not will_step:
                if model.motion_adapter is not None:
                    sync_context.enter_context(model.motion_adapter.no_sync())
                if noise_shaper_runner is not None:
                    sync_context.enter_context(noise_shaper_runner.no_sync())
            autocast = (
                torch.autocast("cuda", dtype=weight_dtype)
                if amp
                else nullcontext()
            )
            with autocast:
                losses = compute_batch(
                    batch,
                    model,
                    vae,
                    tokenizer,
                    text_encoder,
                    noise_scheduler,
                    device,
                    weight_dtype,
                    config["loss"],
                    stochastic_latents=True,
                    ip_conditioner=ip_conditioner,
                    noise_shaper=noise_shaper_runner,
                    noise_shaper_config=noise_shaper_config,
                )
                scaled_loss = losses["total"] / accumulation
            scaler.scale(scaled_loss).backward()
        micro_step += 1
        for key, value in losses.items():
            running[key] = running.get(key, 0.0) + float(value.detach())
        running_count += 1
        if micro_step % accumulation:
            continue

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, float(trainer.get("grad_clip", 1.0))
        )
        scale_before_update = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step_skipped = bool(
            amp and scaler.get_scale() < scale_before_update
        )
        if optimizer_step_skipped:
            if is_main:
                logger.warning(
                    "AMP overflow: optimizer update skipped; keeping the same "
                    "global step and learning rate"
                )
            continue
        scheduler.step()
        step += 1

        if step % log_freq == 0 or step == 1:
            averaged = global_average(running, running_count, device)
            current_lrs = {
                name: optimizer.param_groups[index]["lr"]
                for index, name in enumerate(group_names)
            }
            current_noise_lr = current_lrs.get(
                "noise_shaper", current_lrs.get("motion_adapter", 0.0)
            )
            current_adapter_lr = current_lrs.get(
                "motion_adapter", current_noise_lr
            )
            logger.info(
                "step=%d/%d total=%.5f noise=%.5f temporal=%.5f "
                "residual=%.6f beta=%.3f eff_beta=%.3f nstd=%.3f nwarp=%.5f "
                "conf=%.3f grad=%.3f lr=%.3e noise_lr=%.3e %.2fit/s",
                step,
                max_steps,
                averaged["total"],
                averaged["noise"],
                averaged["temporal"],
                averaged["residual"],
                averaged["noise_beta"],
                averaged["noise_effective_beta"],
                averaged["shaped_noise_std"],
                averaged["noise_warp"],
                averaged["confidence_ratio"],
                float(grad_norm),
                current_adapter_lr,
                current_noise_lr,
                step / max(time.time() - started, 1e-6),
            )
            if writer:
                for key, value in averaged.items():
                    writer.add_scalar(f"train/{key}", value, step)
                writer.add_scalar("train/lr", current_adapter_lr, step)
                writer.add_scalar("train/noise_shaper_lr", current_noise_lr, step)
                writer.add_scalar("train/grad_norm", float(grad_norm), step)
            running.clear()
            running_count = 0

        valid_metrics = None
        if valid_freq > 0 and step % valid_freq == 0:
            valid_totals, valid_count = validate(
                valid_loader,
                int(trainer.get("valid_batches", 0)),
                model,
                vae,
                tokenizer,
                text_encoder,
                noise_scheduler,
                device,
                weight_dtype,
                config["loss"],
                amp,
                ip_conditioner,
                int(config.get("seed", 2026)) + 100000 + rank,
                noise_shaper=noise_shaper_runner,
                noise_shaper_config=noise_shaper_config,
            )
            valid_metrics = global_average(
                valid_totals, valid_count, device
            )
            logger.info(
                "VALID step=%d total=%.5f noise=%.5f temporal=%.5f "
                "residual=%.6f beta=%.3f eff_beta=%.3f "
                "nstd=%.3f nwarp=%.5f conf=%.3f",
                step,
                valid_metrics["total"],
                valid_metrics["noise"],
                valid_metrics["temporal"],
                valid_metrics["residual"],
                valid_metrics["noise_beta"],
                valid_metrics["noise_effective_beta"],
                valid_metrics["shaped_noise_std"],
                valid_metrics["noise_warp"],
                valid_metrics["confidence_ratio"],
            )
            if writer:
                for key, value in valid_metrics.items():
                    writer.add_scalar(f"valid/{key}", value, step)

        should_save = save_freq > 0 and step % save_freq == 0
        improved = (
            valid_metrics is not None
            and valid_metrics["total"] < best_valid
        )
        if improved:
            best_valid = valid_metrics["total"]
        if should_save or improved or step == max_steps:
            if is_main:
                destination = save_checkpoint(
                    checkpoint_root,
                    step,
                    adapter,
                    optimizer,
                    scheduler,
                    scaler,
                    best_valid,
                    config,
                    noise_shaper=noise_shaper,
                    is_best=improved,
                )
                logger.info(
                    "Saved %s%s", destination, " (best)" if improved else ""
                )
            if distributed:
                dist.barrier()
        if writer:
            writer.flush()

    logger.info("Training finished best_valid=%.6f", best_valid)
    if writer:
        writer.close()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
