#!/usr/bin/env python3
"""Stage-2 training for STC noise fusion and a temporal U-Net adapter.

This trainer is independent of the legacy flow-guided motion adapter. It uses
the hierarchical ``SFU_STC_flow`` dataset, loads the best Stage-1 full-flow
predictor, freezes its STC encoder and flow decoder, adds a learned beta head,
and can add zero-initialized temporal residual blocks to the frozen 2D U-Net.
Configuration flags allow beta-only, temporal-only warm-up, or joint training
without modifying Stage 1 or the pretrained BrushNet/U-Net parameters.
"""

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
from typing import Dict, Mapping, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.models.brushnet import BrushNetModel
from diffusers.models.stc_noise_shaper import STCConditionedNoiseShaper
from diffusers.models.unet_temporal_adapter import DiffusionUNetTemporalAdapter
from ip_adapter import FusionIPAdapter
from stc_flow_dataset import STCFlowDataset
from stc_noise_fusion_training import (
    beta_parameters,
    beta_spatial_smoothness,
    build_stage2_noise_shaper,
    set_beta_only_trainable,
    temporal_latent_loss,
)
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
        help="Stage-2 checkpoint directory or latest.json/best.json pointer",
    )
    parser.add_argument(
        "--device",
        help="For example cuda, cuda:0, or cpu; overrides trainer.device",
    )
    return parser.parse_args()


def setup_logger(path: Path, enabled: bool) -> logging.Logger:
    logger = logging.getLogger("stc_noise_fusion_stage2")
    logger.handlers.clear()
    logger.propagate = False
    if not enabled:
        logger.setLevel(logging.CRITICAL)
        logger.addHandler(logging.NullHandler())
        return logger
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    for handler in (logging.FileHandler(path), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def distributed_context(requested_device: str):
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


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_schedule(warmup_steps: int, total_steps: int, min_ratio: float):
    def schedule(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return max(float(step + 1) / warmup_steps, 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return schedule


def global_average(totals: Mapping[str, float], count: int, device: torch.device):
    keys = sorted(totals)
    packed = torch.tensor(
        [float(totals[key]) for key in keys] + [float(count)],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    denominator = packed[-1].clamp_min(1.0)
    return {key: float(packed[index] / denominator) for index, key in enumerate(keys)}


def resolve_resume(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path).expanduser().resolve()
    if path.name in {"latest.json", "best.json"}:
        path = Path(json.loads(path.read_text(encoding="utf-8"))["checkpoint"])
    if not (path / "trainer_state.pt").is_file():
        raise FileNotFoundError(f"Invalid Stage-2 checkpoint: {path}")
    if not (path / "noise_shaper").is_dir():
        raise FileNotFoundError(f"Stage-2 checkpoint has no noise_shaper: {path}")
    return path


def make_dataset(config: Mapping, split: str) -> STCFlowDataset:
    data = dict(config["data"])
    data["load_gt"] = True
    # Stage 2 consumes the frozen predictor and no longer needs teacher caches.
    data.pop("teacher_flow_root", None)
    data.setdefault("seed", int(config.get("seed", 0)))
    return STCFlowDataset(data, split=split)


def make_loader(dataset, trainer, split, distributed, world_size, rank):
    training = split == "train"
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=training,
            drop_last=training,
            seed=int(getattr(dataset, "seed", 0)),
        )
        if distributed
        else None
    )
    batch_size = int(
        trainer.get("batch_size", 1)
        if training
        else trainer.get("valid_batch_size", 1)
    )
    workers = int(trainer.get("num_workers", 2))
    persistent = bool(trainer.get("persistent_workers", False) and workers > 0)
    if training and float(getattr(dataset, "horizontal_flip_probability", 0.0)) > 0:
        persistent = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        drop_last=training,
        num_workers=workers,
        pin_memory=bool(trainer.get("pin_memory", True)),
        persistent_workers=persistent,
    ), sampler


def frozen(module, device, dtype):
    module.requires_grad_(False).eval()
    return module.to(device=device, dtype=dtype)


def resolve_pretrained_component(path, component):
    """Resolve a component directory, checkpoint directory, or JSON pointer."""
    path = Path(path).expanduser().resolve()
    if path.name in {"latest.json", "best.json"}:
        path = Path(json.loads(path.read_text(encoding="utf-8"))["checkpoint"])
    nested = path / component
    if nested.is_dir():
        path = nested
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"Cannot find {component!r} component at {path}")
    return path


class FrozenBrushNetTemporalDenoiser(torch.nn.Module):
    """Frozen BrushNet/U-Net path with an optional trainable temporal adapter."""

    def __init__(self, brushnet, unet, temporal_adapter=None, adapter_scale=1.0):
        super().__init__()
        self.brushnet = brushnet
        self.unet = unet
        self.temporal_adapter = temporal_adapter
        self.adapter_scale = float(adapter_scale)
        self.brushnet.requires_grad_(False)
        self.unet.requires_grad_(False)

    def forward(
        self,
        noisy_latents,
        timesteps,
        encoder_hidden_states,
        conditioning_latents,
    ):
        return frozen_brushnet_unet_forward(
            self.brushnet,
            self.unet,
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            conditioning_latents,
            temporal_adapter=self.temporal_adapter,
            temporal_adapter_scale=self.adapter_scale,
        )


def encode_video(vae, images, device, dtype, stochastic: bool):
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError("images must have shape [B,T,3,H,W]")
    batch, frames = images.shape[:2]
    flat = images.flatten(0, 1).to(device=device, dtype=dtype)
    posterior = vae.encode(flat).latent_dist
    latents = posterior.sample() if stochastic else posterior.mode()
    latents = latents * vae.config.scaling_factor
    return latents.reshape(batch, frames, *latents.shape[1:])


def make_brushnet_condition(decoded_latents, roi_masks, conditioning_channels, device):
    if int(conditioning_channels) != 5:
        raise ValueError("Stage 2 requires a 5-channel BrushNet condition")
    batch, frames = decoded_latents.shape[:2]
    background = 1.0 - roi_masks.float().to(device)
    background = F.interpolate(
        background.flatten(0, 1),
        size=decoded_latents.shape[-2:],
        mode="nearest",
    ).reshape(batch, frames, 1, *decoded_latents.shape[-2:])
    conditioning = torch.cat((decoded_latents, background.to(decoded_latents)), dim=2)
    return conditioning, background


class _UNetCarrier:
    def __init__(self, unet):
        self.unet = unet

    def to(self, *args, **kwargs):
        self.unet.to(*args, **kwargs)
        return self


def build_ip_conditioner(unet, config, device):
    ip_config = config.get("ip_adapter", {})
    if not ip_config.get("enabled", False):
        return None
    target_dtype = next(unet.parameters()).dtype
    conditioner = FusionIPAdapter(
        _UNetCarrier(unet),
        ip_config["image_encoder"],
        ip_config["weights"],
        ip_config["fusion_weights"],
        device,
        num_tokens=int(ip_config.get("num_tokens", 4)),
    )
    # The legacy FusionIPAdapter constructor creates the image modules and new
    # U-Net attention processors in FP16 unconditionally. Align them with the
    # deployed U-Net so both FP32 diagnostic runs and FP16 AMP runs receive
    # matching Linear/attention operands.
    unet.to(device=device, dtype=target_dtype)
    for module in (
        conditioner.image_encoder,
        conditioner.image_proj_model,
        conditioner.fusion_module,
    ):
        module.to(device=device, dtype=target_dtype).requires_grad_(False).eval()
    conditioner.set_scale(float(ip_config.get("scale", 1.0)))
    return conditioner


def _masked_pil_frames(decoded_frames, roi_masks):
    images = ((decoded_frames.float() + 1.0) * 127.5).clamp(0, 255)
    roi = roi_masks.float()

    def convert(tensor):
        arrays = tensor.flatten(0, 1).permute(0, 2, 3, 1).byte().cpu().numpy()
        return [Image.fromarray(array, mode="RGB") for array in arrays]

    return convert(images * roi), convert(images * (1.0 - roi))


def _decoded_pil_frames(decoded_frames):
    images = ((decoded_frames.float() + 1.0) * 127.5).clamp(0, 255)
    arrays = images.flatten(0, 1).permute(0, 2, 3, 1).byte().cpu().numpy()
    return [Image.fromarray(array, mode="RGB") for array in arrays]


@torch.no_grad()
def conditioning_embeddings(
    tokenizer,
    text_encoder,
    captions,
    device,
    decoded_frames,
    roi_masks,
    ip_conditioner,
    include_base_image=False,
    fusion_scale=1.0,
    v8_mask_order=False,
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
    if v8_mask_order:
        # custom_dataset_v04.py names the inverted-mask (background) crop
        # ``fg_clip_images`` and the ROI crop ``bg_clip_images``. Preserve that
        # historical ordering when consuming the released V8 fusion weights.
        foreground, background = background, foreground
    if include_base_image:
        # Match train_brushnet_VCM_ipadapter_v8_coco_nulltext.py exactly:
        # final image condition = base + fusion_scale * fused(FG, BG).
        full_frames = _decoded_pil_frames(decoded_frames)
        processor = ip_conditioner.clip_image_processor
        encoder_dtype = next(ip_conditioner.image_encoder.parameters()).dtype

        def encode(images):
            pixels = processor(images=images, return_tensors="pt").pixel_values
            return ip_conditioner.image_encoder(
                pixels.to(device=device, dtype=encoder_dtype)
            ).image_embeds

        base_embeds = encode(full_frames)
        foreground_embeds = encode(foreground)
        background_embeds = encode(background)
        fused_embeds = ip_conditioner.fusion_module(
            foreground_embeds, background_embeds
        )
        combined_embeds = base_embeds + float(fusion_scale) * fused_embeds
        image_tokens = ip_conditioner.image_proj_model(combined_embeds)
    else:
        image_tokens, _ = ip_conditioner.get_fgbg_image_embeds(
            fg_pil_image=foreground,
            bg_pil_image=background,
        )
    batch, frames = decoded_frames.shape[:2]
    text = text[:, None].expand(batch, frames, *text.shape[1:])
    image_tokens = image_tokens.reshape(batch, frames, *image_tokens.shape[1:]).to(
        device=device, dtype=text.dtype
    )
    return torch.cat((text, image_tokens), dim=2)


def prediction_target(noise_scheduler, latents, noise, timesteps):
    flat_latents = latents.flatten(0, 1)
    flat_noise = noise.flatten(0, 1)
    flat_timesteps = timesteps.repeat_interleave(latents.shape[1])
    prediction_type = noise_scheduler.config.prediction_type
    if prediction_type == "epsilon":
        target = flat_noise
    elif prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(flat_latents, flat_noise, flat_timesteps)
    else:
        raise ValueError(f"Unsupported prediction type: {prediction_type}")
    return target.reshape_as(noise)


def predicted_clean_latents(noise_scheduler, noisy, prediction, timesteps):
    alpha = noise_scheduler.alphas_cumprod.to(
        device=noisy.device, dtype=noisy.dtype
    )[timesteps][:, None, None, None, None]
    sigma = (1.0 - alpha).sqrt()
    sqrt_alpha = alpha.sqrt()
    if noise_scheduler.config.prediction_type == "epsilon":
        return (noisy - sigma * prediction) / sqrt_alpha.clamp_min(1e-6)
    if noise_scheduler.config.prediction_type == "v_prediction":
        return sqrt_alpha * noisy - sigma * prediction
    raise ValueError(
        f"Unsupported prediction type: {noise_scheduler.config.prediction_type}"
    )


def frozen_brushnet_unet_forward(
    brushnet,
    unet,
    noisy_latents,
    timesteps,
    encoder_hidden_states,
    conditioning_latents,
    temporal_adapter=None,
    temporal_adapter_scale=1.0,
):
    """Frozen denoising path that retains gradients to the shaped input noise."""

    batch, frames = noisy_latents.shape[:2]
    noisy_flat = noisy_latents.flatten(0, 1)
    condition_flat = conditioning_latents.flatten(0, 1)
    timestep_flat = timesteps.repeat_interleave(frames)
    if encoder_hidden_states.ndim == 3:
        text_flat = encoder_hidden_states.repeat_interleave(frames, dim=0)
    elif encoder_hidden_states.ndim == 4:
        text_flat = encoder_hidden_states.flatten(0, 1)
    else:
        raise ValueError("encoder_hidden_states must be [B,L,D] or [B,T,L,D]")
    down, mid, up = brushnet(
        noisy_flat,
        timestep_flat,
        encoder_hidden_states=text_flat,
        brushnet_cond=condition_flat,
        return_dict=False,
    )
    prediction = unet(
        noisy_flat,
        timestep_flat,
        encoder_hidden_states=text_flat,
        down_block_add_samples=list(down),
        mid_block_add_sample=mid,
        up_block_add_samples=list(up),
        temporal_adapter=temporal_adapter,
        temporal_num_frames=frames if temporal_adapter is not None else None,
        temporal_adapter_scale=float(temporal_adapter_scale),
        return_dict=False,
    )[0]
    return prediction.reshape(batch, frames, *prediction.shape[1:])


def compute_batch(
    batch,
    noise_shaper,
    denoiser,
    vae,
    tokenizer,
    text_encoder,
    noise_scheduler,
    device,
    weight_dtype,
    config,
    stochastic_gt: bool,
    ip_conditioner=None,
):
    bare_denoiser = denoiser.module if hasattr(denoiser, "module") else denoiser
    gt_cpu = batch["gt_frames"]
    decoded_cpu = batch["decoded_frames"]
    roi_cpu = batch["roi_masks"]
    with torch.no_grad():
        gt_latents = encode_video(
            vae, gt_cpu, device, weight_dtype, stochastic=stochastic_gt
        )
        # Match Stage 1: the degraded condition always uses posterior mode.
        decoded_latents = encode_video(
            vae, decoded_cpu, device, weight_dtype, stochastic=False
        )
        conditioning, bg_mask = make_brushnet_condition(
            decoded_latents,
            roi_cpu,
            bare_denoiser.brushnet.config.conditioning_channels,
            device,
        )
        batch_size = gt_latents.shape[0]
        caption = str(config["data"].get("caption", ""))
        text = conditioning_embeddings(
            tokenizer,
            text_encoder,
            [caption] * batch_size,
            device,
            decoded_cpu,
            roi_cpu,
            ip_conditioner,
            include_base_image=bool(
                config.get("ip_adapter", {}).get("include_base_image", False)
            ),
            fusion_scale=float(
                config.get("ip_adapter", {}).get("fusion_scale", 1.0)
            ),
            v8_mask_order=bool(
                config.get("ip_adapter", {}).get("v8_mask_order", False)
            ),
        )

    batch_size, frames = gt_latents.shape[:2]
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (batch_size,),
        device=device,
        dtype=torch.long,
    )
    independent_noise = torch.randn_like(gt_latents)
    shaped = noise_shaper(
        independent_noise=independent_noise,
        decoded_latents=decoded_latents.detach(),
        bg_mask=bg_mask.detach(),
        strength=float(config.get("noise_fusion", {}).get("strength", 1.0)),
    )
    noise = shaped["noise"]
    noisy = noise_scheduler.add_noise(
        gt_latents.flatten(0, 1),
        noise.flatten(0, 1),
        timesteps.repeat_interleave(frames),
    ).reshape_as(gt_latents)
    # The target is a label. The beta head is optimized through the noisy-input
    # path of the frozen denoiser, not by moving its own target.
    target = prediction_target(
        noise_scheduler,
        gt_latents,
        noise.detach(),
        timesteps,
    )
    prediction = denoiser(
        noisy,
        timesteps,
        text,
        conditioning,
    )
    noise_loss = F.mse_loss(prediction.float(), target.float())
    predicted_clean = predicted_clean_latents(
        noise_scheduler, noisy, prediction, timesteps
    )
    temporal = temporal_latent_loss(
        predicted_clean,
        shaped["predicted_flow_backward"].detach(),
        eps=float(config.get("loss", {}).get("charbonnier_eps", 1e-3)),
    )
    timestep_weight = (
        noise_scheduler.alphas_cumprod.to(device=device)[timesteps].sqrt().mean()
        if config.get("loss", {}).get("temporal_timestep_weighting", True)
        else temporal.new_ones(())
    )
    weighted_temporal = timestep_weight * temporal
    noise_stats = shaped["noise_mean"].square() + (
        shaped["noise_std"] - 1.0
    ).square()
    beta_target = float(config.get("loss", {}).get("beta_target", 0.5))
    beta_prior = (shaped["beta_mean"] - beta_target).square()
    beta_smooth = beta_spatial_smoothness(shaped["beta"])
    loss_config = config.get("loss", {})
    total = (
        float(loss_config.get("noise_weight", 1.0)) * noise_loss
        + float(loss_config.get("temporal_weight", 0.05)) * weighted_temporal
        + float(loss_config.get("noise_stats_weight", 0.01)) * noise_stats
        + float(loss_config.get("beta_prior_weight", 0.001)) * beta_prior
        + float(loss_config.get("beta_smoothness_weight", 0.001)) * beta_smooth
    )
    return {
        "total": total,
        "noise": noise_loss,
        "temporal": temporal,
        "temporal_weighted": weighted_temporal,
        "noise_stats": noise_stats,
        "beta_prior": beta_prior,
        "beta_smoothness": beta_smooth,
        "beta": shaped["beta_mean"],
        "effective_beta": shaped["effective_beta_mean"],
        "noise_mean_abs": shaped["noise_mean"].abs(),
        "noise_std": shaped["noise_std"],
        "noise_warp": shaped["noise_warp_error"],
        "flow_energy": shaped["flow_prediction_energy"],
    }


def set_stage2_train_mode(model):
    bare = model.module if hasattr(model, "module") else model
    bare.train()
    # Stage-1 dropout must remain disabled while its weights are frozen.
    bare.stc_encoder.eval()
    bare.full_flow_head.eval()
    bare.beta_head.train()


@torch.no_grad()
def validate(
    loader,
    max_batches,
    noise_shaper,
    denoiser,
    vae,
    tokenizer,
    text_encoder,
    noise_scheduler,
    device,
    weight_dtype,
    config,
    ip_conditioner,
    seed,
):
    bare = noise_shaper.module if hasattr(noise_shaper, "module") else noise_shaper
    bare_denoiser = denoiser.module if hasattr(denoiser, "module") else denoiser
    bare.eval()
    bare_denoiser.eval()
    totals: Dict[str, float] = {}
    count = 0
    devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=devices):
        seed_everything(seed)
        for batch_index, batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            losses = compute_batch(
                batch,
                noise_shaper,
                denoiser,
                vae,
                tokenizer,
                text_encoder,
                noise_scheduler,
                device,
                weight_dtype,
                config,
                stochastic_gt=False,
                ip_conditioner=ip_conditioner,
            )
            batch_size = int(batch["decoded_frames"].shape[0])
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch_size
            count += batch_size
    set_stage2_train_mode(noise_shaper)
    bare_denoiser.brushnet.train()
    bare_denoiser.unet.train()
    if bare_denoiser.temporal_adapter is not None:
        if any(
            parameter.requires_grad
            for parameter in bare_denoiser.temporal_adapter.parameters()
        ):
            bare_denoiser.temporal_adapter.train()
        else:
            bare_denoiser.temporal_adapter.eval()
    return global_average(totals, count, device)


def save_checkpoint(
    checkpoint_root,
    model,
    temporal_adapter,
    optimizer,
    scheduler,
    scaler,
    step,
    best_valid,
    config,
    is_best,
    extra_state=None,
):
    destination = checkpoint_root / f"checkpoint-{int(step):07d}"
    temporary = checkpoint_root / f".checkpoint-{int(step):07d}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    bare = model.module if hasattr(model, "module") else model
    bare.save_pretrained(temporary / "noise_shaper")
    if temporal_adapter is not None:
        temporal_adapter.save_pretrained(temporary / "temporal_adapter")
    state = {
        "step": int(step),
        "best_valid": float(best_valid),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }
    state.update(dict(extra_state or {}))
    torch.save(state, temporary / "trainer_state.pt")
    (temporary / "config.json").write_text(
        json.dumps(dict(config), indent=2), encoding="utf-8"
    )
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)
    pointer = {"checkpoint": str(destination), "step": int(step)}
    (checkpoint_root / "latest.json").write_text(
        json.dumps(pointer, indent=2), encoding="utf-8"
    )
    if is_best:
        pointer["valid_total"] = float(best_valid)
        (checkpoint_root / "best.json").write_text(
            json.dumps(pointer, indent=2), encoding="utf-8"
        )
    return destination


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    trainer = config["trainer"]
    distributed, rank, local_rank, world_size, device = distributed_context(
        args.device or trainer.get("device", "cuda")
    )
    is_main = rank == 0
    seed_everything(int(config.get("seed", 2026)) + rank)
    amp = bool(trainer.get("amp", True) and device.type == "cuda")
    weight_dtype = torch.float16 if amp else torch.float32
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    checkpoint_root = Path(config["checkpoint_dir"]).expanduser().resolve()
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    logger = setup_logger(output_dir / "logs" / "train.log", is_main)
    writer = (
        SummaryWriter(output_dir / "tensorboard")
        if is_main and SummaryWriter is not None
        else None
    )
    if is_main:
        (output_dir / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

    resume = resolve_resume(
        args.resume
        or (
            Path(config["model"]["resume"])
            if config.get("model", {}).get("resume")
            else None
        )
    )
    if resume is not None:
        noise_shaper = STCConditionedNoiseShaper.from_pretrained(
            resume / "noise_shaper"
        )
        set_beta_only_trainable(noise_shaper)
        stage1_component = Path(config["model"]["stage1_checkpoint"])
    else:
        noise_shaper, stage1_component = build_stage2_noise_shaper(
            Path(config["model"]["stage1_checkpoint"]),
            config.get("noise_fusion", {}),
        )
    noise_shaper = noise_shaper.to(device=device, dtype=torch.float32)
    train_beta = bool(config.get("noise_fusion", {}).get("trainable", True))
    beta_trainable_parameters = list(beta_parameters(noise_shaper))
    if not train_beta:
        noise_shaper.requires_grad_(False)
        beta_trainable_parameters = []

    base = config["model"]["base_model"]
    logger.info("Loading frozen diffusion/BrushNet models")
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
    # FusionIPAdapter installs attention processors after the initial freeze.
    # Freeze the complete deployed U-Net again so Stage 2 cannot accumulate or
    # optimize IP-Adapter/UNet parameters accidentally.
    unet.requires_grad_(False)
    brushnet = frozen(
        BrushNetModel.from_pretrained(config["model"]["brushnet"]),
        device,
        weight_dtype,
    )

    temporal_config = dict(config.get("temporal_adapter", {}))
    temporal_enabled = bool(temporal_config.get("enabled", False))
    train_temporal = bool(temporal_config.get("trainable", True))
    temporal_adapter = None
    if temporal_enabled:
        resume_temporal = resume / "temporal_adapter" if resume is not None else None
        temporal_checkpoint = temporal_config.get("checkpoint")
        if resume_temporal is not None and resume_temporal.is_dir():
            temporal_adapter = DiffusionUNetTemporalAdapter.from_pretrained(
                resume_temporal
            )
        elif resume is not None:
            raise FileNotFoundError(
                "Resume config enables temporal_adapter but the checkpoint "
                f"has no temporal_adapter component: {resume}"
            )
        elif temporal_checkpoint:
            temporal_adapter = DiffusionUNetTemporalAdapter.from_pretrained(
                resolve_pretrained_component(
                    temporal_checkpoint, "temporal_adapter"
                )
            )
        else:
            temporal_adapter = DiffusionUNetTemporalAdapter.from_unet(
                unet,
                down_block_indices=tuple(
                    temporal_config.get("down_block_indices", (0, 1, 2))
                ),
                use_mid=bool(temporal_config.get("use_mid", True)),
                up_block_indices=tuple(
                    temporal_config.get("up_block_indices", ())
                ),
                bottleneck_channels=int(
                    temporal_config.get("bottleneck_channels", 64)
                ),
                temporal_kernel_size=int(
                    temporal_config.get("temporal_kernel_size", 3)
                ),
                dropout=float(temporal_config.get("dropout", 0.0)),
            )
        temporal_adapter = temporal_adapter.to(
            device=device, dtype=torch.float32
        )
        temporal_adapter.requires_grad_(train_temporal)
        temporal_adapter.train() if train_temporal else temporal_adapter.eval()
    elif temporal_config.get("checkpoint"):
        raise ValueError(
            "temporal_adapter.checkpoint was provided while "
            "temporal_adapter.enabled=false"
        )

    temporal_trainable_parameters = (
        [
            parameter
            for parameter in temporal_adapter.parameters()
            if parameter.requires_grad
        ]
        if temporal_adapter is not None
        else []
    )
    if not beta_trainable_parameters and not temporal_trainable_parameters:
        raise ValueError(
            "No trainable Stage-2 component. Enable noise_fusion.trainable "
            "and/or temporal_adapter.trainable."
        )
    if trainer.get("gradient_checkpointing", True):
        if hasattr(unet, "enable_gradient_checkpointing"):
            unet.enable_gradient_checkpointing()
        if hasattr(brushnet, "enable_gradient_checkpointing"):
            brushnet.enable_gradient_checkpointing()
    # Frozen weights, but train mode is required by Diffusers checkpointing to
    # retain the input-gradient path back to beta.
    unet.train()
    brushnet.train()

    denoiser = FrozenBrushNetTemporalDenoiser(
        brushnet,
        unet,
        temporal_adapter=temporal_adapter,
        adapter_scale=float(temporal_config.get("scale", 1.0)),
    )
    noise_runner = noise_shaper
    denoiser_runner = denoiser
    if distributed and train_beta:
        noise_runner = DistributedDataParallel(
            noise_shaper,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    if distributed and train_temporal:
        denoiser_runner = DistributedDataParallel(
            denoiser,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    set_stage2_train_mode(noise_runner)

    train_dataset = make_dataset(config, "train")
    valid_dataset = make_dataset(config, "valid")
    train_loader, train_sampler = make_loader(
        train_dataset, trainer, "train", distributed, world_size, rank
    )
    valid_loader, _ = make_loader(
        valid_dataset, trainer, "valid", distributed, world_size, rank
    )
    if len(train_loader) == 0:
        raise ValueError("Training DataLoader is empty")

    lr = float(trainer.get("lr", 1e-4))
    min_lr = float(trainer.get("min_lr", lr * 0.01))
    parameter_groups = []
    group_names = []
    group_min_lrs = []
    parameters = []
    if beta_trainable_parameters:
        beta_lr = float(trainer.get("noise_fusion_lr", lr))
        beta_min_lr = float(
            trainer.get("noise_fusion_min_lr", min(min_lr, beta_lr))
        )
        if not 0.0 <= beta_min_lr <= beta_lr:
            raise ValueError("noise_fusion_min_lr must be in [0,noise_fusion_lr]")
        parameter_groups.append({"params": beta_trainable_parameters, "lr": beta_lr})
        group_names.append("noise_fusion")
        group_min_lrs.append(beta_min_lr)
        parameters.extend(beta_trainable_parameters)
    if temporal_trainable_parameters:
        temporal_lr = float(trainer.get("temporal_adapter_lr", lr))
        temporal_min_lr = float(
            trainer.get("temporal_adapter_min_lr", min(min_lr, temporal_lr))
        )
        if not 0.0 <= temporal_min_lr <= temporal_lr:
            raise ValueError(
                "temporal_adapter_min_lr must be in [0,temporal_adapter_lr]"
            )
        parameter_groups.append(
            {"params": temporal_trainable_parameters, "lr": temporal_lr}
        )
        group_names.append("temporal_adapter")
        group_min_lrs.append(temporal_min_lr)
        parameters.extend(temporal_trainable_parameters)
    optimizer = AdamW(
        parameter_groups,
        lr=lr,
        betas=tuple(trainer.get("betas", [0.9, 0.999])),
        weight_decay=float(trainer.get("weight_decay", 0.01)),
    )
    max_steps = int(trainer["max_steps"])
    scheduler = LambdaLR(
        optimizer,
        [
            cosine_schedule(
                int(trainer.get("warmup_steps", 500)),
                max_steps,
                minimum / float(group["lr"]),
            )
            for group, minimum in zip(parameter_groups, group_min_lrs)
        ],
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp,
        init_scale=float(trainer.get("amp_init_scale", 2048.0)),
    )
    step = 0
    best_valid = math.inf
    resume_state = None
    if resume is not None:
        resume_state = torch.load(resume / "trainer_state.pt", map_location="cpu")
        saved_groups = len(resume_state["optimizer"].get("param_groups", []))
        if saved_groups != len(optimizer.param_groups):
            raise ValueError(
                "Resume optimizer-group mismatch: checkpoint has "
                f"{saved_groups}, current config has {len(optimizer.param_groups)}"
            )
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        scaler.load_state_dict(resume_state["scaler"])
        step = int(resume_state["step"])
        best_valid = float(resume_state.get("best_valid", math.inf))
        logger.info("Resumed %s at step %d", resume, step)

    accumulation = int(trainer.get("gradient_accumulation_steps", 1))
    log_freq = int(trainer.get("log_freq", 20))
    valid_freq = int(trainer.get("valid_freq", 500))
    save_freq = int(trainer.get("save_freq", 500))
    valid_batches = int(trainer.get("valid_batches", 0))
    logger.info(
        "train_clips=%d valid_clips=%d T=%d B_per_gpu=%d world=%d "
        "effective_batch=%d beta_trainable=%d temporal_trainable=%d "
        "temporal_stages=down%s/mid%s/up%s stage1=%s device=%s amp=%s",
        len(train_dataset),
        len(valid_dataset),
        int(config["data"]["clip_length"]),
        int(trainer.get("batch_size", 1)),
        world_size,
        int(trainer.get("batch_size", 1)) * world_size * accumulation,
        sum(parameter.numel() for parameter in beta_trainable_parameters),
        sum(parameter.numel() for parameter in temporal_trainable_parameters),
        list(temporal_config.get("down_block_indices", ())) if temporal_enabled else [],
        bool(temporal_config.get("use_mid", False)) if temporal_enabled else False,
        list(temporal_config.get("up_block_indices", ())) if temporal_enabled else [],
        stage1_component,
        device,
        amp,
    )

    optimizer.zero_grad(set_to_none=True)
    epoch = int(resume_state.get("epoch", 0)) if resume_state else 0
    micro_step = int(resume_state.get("micro_step", step * accumulation)) if resume_state else 0
    if hasattr(train_dataset, "set_epoch"):
        train_dataset.set_epoch(epoch)
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    iterator = iter(train_loader)
    running: Dict[str, float] = {}
    running_count = 0
    started = time.time()

    try:
        while step < max_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                if hasattr(train_dataset, "set_epoch"):
                    train_dataset.set_epoch(epoch)
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                iterator = iter(train_loader)
                batch = next(iterator)
            micro_step += 1
            update_now = micro_step % accumulation == 0
            with ExitStack() as sync_context:
                if not update_now:
                    if hasattr(noise_runner, "no_sync"):
                        sync_context.enter_context(noise_runner.no_sync())
                    if hasattr(denoiser_runner, "no_sync"):
                        sync_context.enter_context(denoiser_runner.no_sync())
                autocast = (
                    torch.autocast("cuda", dtype=torch.float16)
                    if amp
                    else nullcontext()
                )
                with autocast:
                    losses = compute_batch(
                        batch,
                        noise_runner,
                        denoiser_runner,
                        vae,
                        tokenizer,
                        text_encoder,
                        noise_scheduler,
                        device,
                        weight_dtype,
                        config,
                        stochastic_gt=True,
                        ip_conditioner=ip_conditioner,
                    )
                    scaled_loss = losses["total"] / accumulation
                scaler.scale(scaled_loss).backward()
            batch_size = int(batch["decoded_frames"].shape[0])
            for key, value in losses.items():
                running[key] = running.get(key, 0.0) + float(value.detach()) * batch_size
            running_count += batch_size
            if not update_now:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(trainer.get("grad_clip", 1.0))
            )
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            update_succeeded = scaler.get_scale() >= scale_before
            optimizer.zero_grad(set_to_none=True)
            if not update_succeeded:
                if is_main:
                    logger.warning("AMP overflow: skipped Stage-2 optimizer update")
                running.clear()
                running_count = 0
                continue
            scheduler.step()
            step += 1

            if step % log_freq == 0 or step == 1:
                averaged = global_average(running, running_count, device)
                current_lrs = {
                    name: optimizer.param_groups[index]["lr"]
                    for index, name in enumerate(group_names)
                }
                if is_main:
                    logger.info(
                        "step=%d/%d total=%.5f noise=%.5f temporal=%.5f "
                        "beta=%.3f eff_beta=%.3f beta_smooth=%.5f nstd=%.3f "
                        "nwarp=%.5f grad=%.3f beta_lr=%.3e temporal_lr=%.3e "
                        "%.2fit/s",
                        step,
                        max_steps,
                        averaged["total"],
                        averaged["noise"],
                        averaged["temporal"],
                        averaged["beta"],
                        averaged["effective_beta"],
                        averaged["beta_smoothness"],
                        averaged["noise_std"],
                        averaged["noise_warp"],
                        float(grad_norm),
                        current_lrs.get("noise_fusion", 0.0),
                        current_lrs.get("temporal_adapter", 0.0),
                        step / max(time.time() - started, 1e-6),
                    )
                    if writer is not None:
                        for key, value in averaged.items():
                            writer.add_scalar(f"train/{key}", value, step)
                        for name, current_lr in current_lrs.items():
                            writer.add_scalar(f"train/{name}_lr", current_lr, step)
                        writer.add_scalar("train/grad_norm", float(grad_norm), step)
                running.clear()
                running_count = 0

            valid_metrics = None
            is_best = False
            if step % valid_freq == 0 or step == max_steps:
                valid_metrics = validate(
                    valid_loader,
                    valid_batches,
                    noise_runner,
                    denoiser_runner,
                    vae,
                    tokenizer,
                    text_encoder,
                    noise_scheduler,
                    device,
                    weight_dtype,
                    config,
                    ip_conditioner,
                    int(config.get("seed", 2026)) + 100000 + rank,
                )
                is_best = valid_metrics["total"] < best_valid
                if is_best:
                    best_valid = valid_metrics["total"]
                if is_main:
                    logger.info(
                        "VALID step=%d total=%.5f noise=%.5f temporal=%.5f "
                        "beta=%.3f eff_beta=%.3f nstd=%.3f nwarp=%.5f%s",
                        step,
                        valid_metrics["total"],
                        valid_metrics["noise"],
                        valid_metrics["temporal"],
                        valid_metrics["beta"],
                        valid_metrics["effective_beta"],
                        valid_metrics["noise_std"],
                        valid_metrics["noise_warp"],
                        " (best)" if is_best else "",
                    )
                    if writer is not None:
                        for key, value in valid_metrics.items():
                            writer.add_scalar(f"valid/{key}", value, step)

            if step % save_freq == 0 or step == max_steps or is_best:
                if is_main:
                    destination = save_checkpoint(
                        checkpoint_root,
                        noise_runner,
                        temporal_adapter,
                        optimizer,
                        scheduler,
                        scaler,
                        step,
                        best_valid,
                        config,
                        is_best,
                        extra_state={"epoch": epoch, "micro_step": micro_step},
                    )
                    logger.info("Saved %s%s", destination, " (best)" if is_best else "")
            if distributed and (valid_metrics is not None or step % save_freq == 0):
                dist.barrier()
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()
    if is_main:
        logger.info("Stage-2 training finished best_valid=%.6f", best_valid)


if __name__ == "__main__":
    main()
