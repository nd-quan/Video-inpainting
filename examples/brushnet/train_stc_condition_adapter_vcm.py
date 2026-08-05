#!/usr/bin/env python3
"""Stage-3 STC feature injection for the VCM BrushNet/IP-Adapter pipeline.

The trainer accepts either a learned-beta Stage-2 noise shaper or a Stage-1
flow predictor rebuilt with fixed beta. It reuses the STC feature tensor and
trains a zero-initialized multi-scale adapter at the frozen BrushNet-to-U-Net
residual ports. It has no dependency on the legacy flow motion adapter.

This entry point is intentionally condition-only. The experimental U-Net
temporal path is isolated in ``train_stc_condition_temporal_adapter_vcm.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.models.brushnet import BrushNetModel
from diffusers.models.stc_condition_adapter import STCBrushNetConditionAdapter
from diffusers.models.stc_noise_shaper import STCConditionedNoiseShaper
from stc_condition_adapter_training import (
    FrozenBrushNetSTCConditionModel,
    build_stage3_adapter,
    configure_stage3_trainable,
    load_stage1_fixed_noise_shaper,
    load_stage2_noise_shaper,
    resolve_stage1_flow_predictor,
    set_stage3_train_mode,
    stage3_parameter_groups,
    stage3_trainable_parameters,
    validate_stage3_adapter,
    validate_stage3_noise_shaper,
)
from stc_noise_fusion_training import temporal_latent_loss
from train_stc_noise_fusion_vcm import (
    build_ip_conditioner,
    conditioning_embeddings,
    cosine_schedule,
    distributed_context,
    encode_video,
    frozen,
    global_average,
    make_brushnet_condition,
    make_dataset,
    make_loader,
    predicted_clean_latents,
    prediction_target,
    seed_everything,
    setup_logger,
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
        help="Stage-3 checkpoint directory or latest.json/best.json pointer",
    )
    parser.add_argument(
        "--device",
        help="For example cuda, cuda:0, or cpu; overrides trainer.device",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate fixed/learned source and local deployment paths, then exit",
    )
    return parser.parse_args()


def resolve_resume(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path).expanduser().resolve()
    if path.name in {"latest.json", "best.json"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        path = Path(payload["checkpoint"]).expanduser().resolve()
    required = (
        path / "trainer_state.pt",
        path / "noise_shaper" / "config.json",
        path / "condition_adapter" / "config.json",
    )
    missing = [str(candidate) for candidate in required if not candidate.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Invalid Stage-3 checkpoint {path}; missing {missing}"
        )
    return path


def prepare_stage3_config(config: Mapping) -> Dict:
    """Validate Stage-3 source semantics and materialize V8 component paths.

    The function mutates the config dictionary only when ``model.v8_checkpoint``
    is supplied, replacing it with the exact separated deployment artifacts
    expected by the existing frozen BrushNet/IP conditioning loaders.
    """

    model = config.get("model", {})
    data = config.get("data", {})
    adapter = config.get("adapter", {})
    ip = config.get("ip_adapter", {})
    noise_fusion = config.get("noise_fusion", {})
    mode = str(model.get("noise_source", "learned_stage2")).lower()
    if mode not in {"learned_stage2", "fixed_stage1"}:
        raise ValueError(
            "model.noise_source must be 'learned_stage2' or 'fixed_stage1'"
        )

    base = Path(str(model["base_model"])).expanduser()
    if base.is_absolute() or base.exists():
        base = base.resolve()
        for component in ("scheduler", "tokenizer", "text_encoder", "vae", "unet"):
            if not (base / component).is_dir():
                raise FileNotFoundError(
                    f"Base-model component {component!r} not found under {base}"
                )
        model["base_model"] = str(base)

    v8_root = model.get("v8_checkpoint")
    v8_components = None
    if v8_root:
        root = Path(v8_root).expanduser().resolve()
        v8_components = {
            "checkpoint": root,
            "brushnet": root / "brushnet",
            "ip_adapter": root / "ipadapter" / "model.safetensors",
            "fusion": root / "ipadapter" / "fusion_module.safetensors",
        }
        if not (v8_components["brushnet"] / "config.json").is_file():
            raise FileNotFoundError(
                f"V8 BrushNet component not found: {v8_components['brushnet']}"
            )
        for key in ("ip_adapter", "fusion"):
            if not v8_components[key].is_file():
                raise FileNotFoundError(
                    f"V8 {key} weights not found: {v8_components[key]}"
                )
        model["v8_checkpoint"] = str(root)
        model["brushnet"] = str(v8_components["brushnet"])
        ip["weights"] = str(v8_components["ip_adapter"])
        ip["fusion_weights"] = str(v8_components["fusion"])
        if not bool(ip.get("enabled", False)):
            raise ValueError("V8 Stage 3 requires ip_adapter.enabled=true")
        if not bool(ip.get("include_base_image", False)):
            raise ValueError("V8 Stage 3 requires include_base_image=true")
        if not bool(ip.get("v8_mask_order", False)):
            raise ValueError("V8 Stage 3 requires v8_mask_order=true")
        if int(ip.get("num_tokens", 4)) != 4:
            raise ValueError("V8 Stage 3 requires exactly four IP tokens")

    dataset_root = Path(data["dataset_root"]).expanduser().resolve()
    manifest = Path(data.get("manifest", dataset_root / "manifest.json"))
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
    data["dataset_root"] = str(dataset_root)
    data["manifest"] = str(manifest.resolve())

    strength = float(noise_fusion.get("strength", 1.0))
    if not 0.0 <= strength <= 1.0:
        raise ValueError("noise_fusion.strength must be in [0,1]")
    source_component = None
    source_config = None
    fixed_beta = None
    if mode == "fixed_stage1":
        if not model.get("stage1_checkpoint"):
            raise ValueError("model.stage1_checkpoint is required for fixed_stage1")
        if "fixed_beta" not in model:
            raise ValueError("model.fixed_beta is required for fixed_stage1")
        fixed_beta = float(model["fixed_beta"])
        if not 0.0 <= fixed_beta <= 1.0:
            raise ValueError("model.fixed_beta must be in [0,1]")
        if str(noise_fusion.get("warp_region", "all")).lower() != "all":
            raise ValueError("Fixed-beta Stage 3 requires warp_region='all'")
        source_component = resolve_stage1_flow_predictor(
            Path(model["stage1_checkpoint"])
        )
        source_config = json.loads(
            (source_component / "config.json").read_text(encoding="utf-8")
        )
        if str(source_config.get("flow_prediction_mode", "")).lower() != "full":
            raise ValueError("Fixed-beta Stage 3 requires Stage-1 full-flow mode")
        if str(source_config.get("beta_mode", "")).lower() != "fixed":
            raise ValueError("Fixed-beta Stage 3 requires fixed-beta Stage 1")
        hidden_channels = int(source_config["hidden_channels"])
        if int(adapter.get("input_channels", hidden_channels)) != hidden_channels:
            raise ValueError(
                "adapter.input_channels must match Stage-1 hidden_channels: "
                f"{adapter.get('input_channels')} vs {hidden_channels}"
            )
    else:
        if not model.get("stage2_checkpoint") and not (
            model.get("resume") or model.get("init_checkpoint")
        ):
            raise ValueError(
                "model.stage2_checkpoint is required for learned_stage2"
            )

    return {
        "noise_source": mode,
        "source_component": str(source_component) if source_component else None,
        "fixed_beta": fixed_beta,
        "v8_components": (
            {key: str(value) for key, value in v8_components.items()}
            if v8_components
            else None
        ),
        "dataset_root": str(dataset_root),
        "manifest": str(manifest.resolve()),
    }


def compute_batch(
    batch,
    runner,
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
    gt_cpu = batch["gt_frames"]
    decoded_cpu = batch["decoded_frames"]
    roi_cpu = batch["roi_masks"]
    bare = runner.module if hasattr(runner, "module") else runner
    with torch.no_grad():
        gt_latents = encode_video(
            vae, gt_cpu, device, weight_dtype, stochastic=stochastic_gt
        )
        decoded_latents = encode_video(
            vae, decoded_cpu, device, weight_dtype, stochastic=False
        )
        conditioning, bg_mask = make_brushnet_condition(
            decoded_latents,
            roi_cpu,
            bare.brushnet.config.conditioning_channels,
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
                config.get("ip_adapter", {}).get("include_base_image", True)
            ),
            fusion_scale=float(
                config.get("ip_adapter", {}).get("fusion_scale", 1.0)
            ),
            v8_mask_order=bool(
                config.get("ip_adapter", {}).get("v8_mask_order", True)
            ),
        )

    batch_size = gt_latents.shape[0]
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (batch_size,),
        device=device,
        dtype=torch.long,
    )
    independent_noise = torch.randn_like(gt_latents)
    output = runner(
        clean_latents=gt_latents,
        independent_noise=independent_noise,
        decoded_latents=decoded_latents.detach(),
        bg_mask=bg_mask.detach(),
        timesteps=timesteps,
        encoder_hidden_states=text,
        brushnet_condition=conditioning,
    )
    target = prediction_target(
        noise_scheduler,
        gt_latents,
        output["noise"],
        timesteps,
    )
    prediction = output["prediction"]
    noise_loss = F.mse_loss(prediction.float(), target.float())
    predicted_clean = predicted_clean_latents(
        noise_scheduler,
        output["noisy_latents"],
        prediction,
        timesteps,
    )
    temporal = temporal_latent_loss(
        predicted_clean,
        output["predicted_flow_backward"],
        eps=float(config.get("loss", {}).get("charbonnier_eps", 1e-3)),
    )
    timestep_weight = (
        noise_scheduler.alphas_cumprod.to(device=device)[timesteps].sqrt().mean()
        if config.get("loss", {}).get("temporal_timestep_weighting", True)
        else temporal.new_ones(())
    )
    weighted_temporal = timestep_weight * temporal
    residual = output["correction_energy"]
    loss_config = config.get("loss", {})
    total = (
        float(loss_config.get("noise_weight", 1.0)) * noise_loss
        + float(loss_config.get("temporal_weight", 0.05)) * weighted_temporal
        + float(loss_config.get("residual_weight", 1e-4)) * residual
    )
    return {
        "total": total,
        "noise": noise_loss,
        "temporal": temporal,
        "temporal_weighted": weighted_temporal,
        "residual": residual,
        "correction_rms": output["correction_rms"],
        "correction_abs": output["correction_abs"],
        "beta": output["beta_mean"],
        "effective_beta": output["effective_beta_mean"],
    }


@torch.no_grad()
def validate(
    loader,
    max_batches,
    runner,
    vae,
    tokenizer,
    text_encoder,
    noise_scheduler,
    device,
    weight_dtype,
    config,
    ip_conditioner,
    seed,
    train_stc_encoder,
    gradient_checkpointing,
):
    bare = runner.module if hasattr(runner, "module") else runner
    bare.eval()
    totals: Dict[str, float] = {}
    count = 0
    devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=devices):
        seed_everything(seed)
        for batch_index, batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            validation_autocast = (
                torch.autocast("cuda", dtype=weight_dtype)
                if device.type == "cuda"
                and weight_dtype in {torch.float16, torch.bfloat16}
                else nullcontext()
            )
            with validation_autocast:
                losses = compute_batch(
                    batch,
                    runner,
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
    set_stage3_train_mode(
        runner,
        train_stc_encoder=train_stc_encoder,
        gradient_checkpointing=gradient_checkpointing,
    )
    return global_average(totals, count, device)


def save_checkpoint(
    checkpoint_root,
    model,
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
    bare.noise_shaper.save_pretrained(temporary / "noise_shaper")
    bare.condition_adapter.save_pretrained(temporary / "condition_adapter")
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


def _load_components(config, resume, brushnet, device):
    model_config = config.get("model", {})
    noise_source = str(
        model_config.get("noise_source", "learned_stage2")
    ).lower()
    if noise_source not in {"learned_stage2", "fixed_stage1"}:
        raise ValueError(
            "model.noise_source must be 'learned_stage2' or 'fixed_stage1'"
        )
    fixed_beta = (
        float(model_config["fixed_beta"])
        if noise_source == "fixed_stage1" and "fixed_beta" in model_config
        else None
    )
    if noise_source == "fixed_stage1" and fixed_beta is None:
        raise ValueError("model.fixed_beta is required for fixed_stage1")

    if resume is not None:
        noise_shaper = STCConditionedNoiseShaper.from_pretrained(
            resume / "noise_shaper"
        )
        adapter = STCBrushNetConditionAdapter.from_pretrained(
            resume / "condition_adapter"
        )
        source = resume / "noise_shaper"
    elif model_config.get("init_checkpoint"):
        initialization = resolve_resume(
            Path(model_config["init_checkpoint"])
        )
        noise_shaper = STCConditionedNoiseShaper.from_pretrained(
            initialization / "noise_shaper"
        )
        adapter = STCBrushNetConditionAdapter.from_pretrained(
            initialization / "condition_adapter"
        )
        source = initialization / "noise_shaper"
    else:
        if noise_source == "fixed_stage1":
            if not model_config.get("stage1_checkpoint"):
                raise ValueError(
                    "model.stage1_checkpoint is required for fixed_stage1"
                )
            noise_shaper, source = load_stage1_fixed_noise_shaper(
                Path(model_config["stage1_checkpoint"]),
                fixed_beta=fixed_beta,
            )
        else:
            if not model_config.get("stage2_checkpoint"):
                raise ValueError(
                    "model.stage2_checkpoint is required for learned_stage2"
                )
            noise_shaper, source = load_stage2_noise_shaper(
                Path(model_config["stage2_checkpoint"])
            )
        adapter = build_stage3_adapter(
            brushnet, noise_shaper, config.get("adapter", {})
        )
    validate_stage3_noise_shaper(
        noise_shaper,
        expected_mode=noise_source,
        fixed_beta=fixed_beta,
    )
    validate_stage3_adapter(adapter, brushnet, noise_shaper)
    return noise_shaper.to(device=device, dtype=torch.float32), adapter.to(
        device=device, dtype=torch.float32
    ), source, noise_source


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    preflight = prepare_stage3_config(config)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return
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
        (output_dir / "preflight.json").write_text(
            json.dumps(preflight, indent=2, sort_keys=True), encoding="utf-8"
        )

    resume = resolve_resume(
        args.resume
        or (
            Path(config["model"]["resume"])
            if config.get("model", {}).get("resume")
            else None
        )
    )
    if resume is not None and config.get("model", {}).get("init_checkpoint"):
        raise ValueError("Use only one of model.resume or model.init_checkpoint")
    base = config["model"]["base_model"]
    logger.info("Loading frozen VAE, BrushNet, IP-Adapter, and diffusion U-Net")
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
    unet.requires_grad_(False)
    brushnet = frozen(
        BrushNetModel.from_pretrained(config["model"]["brushnet"]),
        device,
        weight_dtype,
    )
    noise_shaper, condition_adapter, noise_source_path, noise_source_mode = _load_components(
        config, resume, brushnet, device
    )

    adapter_config = config.get("adapter", {})
    train_stc_encoder = bool(adapter_config.get("train_stc_encoder", False))
    gradient_checkpointing = bool(trainer.get("gradient_checkpointing", True))
    if gradient_checkpointing:
        if hasattr(unet, "enable_gradient_checkpointing"):
            unet.enable_gradient_checkpointing()
        if hasattr(brushnet, "enable_gradient_checkpointing"):
            brushnet.enable_gradient_checkpointing()

    model = FrozenBrushNetSTCConditionModel(
        noise_shaper=noise_shaper,
        condition_adapter=condition_adapter,
        brushnet=brushnet,
        unet=unet,
        alphas_cumprod=noise_scheduler.alphas_cumprod,
        noise_strength=float(config.get("noise_fusion", {}).get("strength", 1.0)),
        injection_scale=float(adapter_config.get("injection_scale", 1.0)),
    ).to(device=device)
    # FrozenBrushNetSTCConditionModel creates alphas_cumprod as a new buffer.
    # Moving the wrapper is required even though its child modules were already
    # moved above; otherwise CUDA DDP attempts to broadcast this CPU buffer.
    adapter_lr = float(trainer.get("lr", 1e-4))
    parameter_groups = stage3_parameter_groups(
        model,
        adapter_lr=adapter_lr,
        train_stc_encoder=train_stc_encoder,
        stc_lr=float(adapter_config.get("stc_lr", adapter_lr * 0.1)),
    )
    parameters = list(stage3_trainable_parameters(model))
    if not parameters:
        raise ValueError("Stage 3 has no trainable parameters")
    runner = model
    if distributed:
        runner = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    set_stage3_train_mode(
        runner,
        train_stc_encoder=train_stc_encoder,
        gradient_checkpointing=gradient_checkpointing,
    )

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

    optimizer = AdamW(
        parameter_groups,
        betas=tuple(trainer.get("betas", [0.9, 0.999])),
        weight_decay=float(trainer.get("weight_decay", 0.01)),
    )
    max_steps = int(trainer["max_steps"])
    min_lr = float(trainer.get("min_lr", adapter_lr * 0.01))
    scheduler = LambdaLR(
        optimizer,
        cosine_schedule(
            int(trainer.get("warmup_steps", 500)),
            max_steps,
            min_lr / adapter_lr,
        ),
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
    max_consecutive_amp_overflows = int(
        trainer.get("max_consecutive_amp_overflows", 16)
    )
    if max_consecutive_amp_overflows < 1:
        raise ValueError("max_consecutive_amp_overflows must be positive")
    adapter_count = sum(
        parameter.numel() for parameter in model.condition_adapter.parameters()
    )
    stc_count = sum(
        parameter.numel()
        for parameter in model.noise_shaper.stc_encoder.parameters()
        if parameter.requires_grad
    )
    logger.info(
        "train_clips=%d valid_clips=%d T=%d B_per_gpu=%d world=%d "
        "effective_batch=%d adapter_trainable=%d stc_trainable=%d "
        "noise_source=%s source_path=%s beta_mode=%s fixed_beta=%s "
        "device=%s amp=%s",
        len(train_dataset),
        len(valid_dataset),
        int(config["data"]["clip_length"]),
        int(trainer.get("batch_size", 1)),
        world_size,
        int(trainer.get("batch_size", 1)) * world_size * accumulation,
        adapter_count,
        stc_count,
        noise_source_mode,
        noise_source_path,
        str(noise_shaper.config.beta_mode),
        (
            f"{float(noise_shaper.config.fixed_beta):.3f}"
            if str(noise_shaper.config.beta_mode).lower() == "fixed"
            else "n/a"
        ),
        device,
        amp,
    )

    optimizer.zero_grad(set_to_none=True)
    epoch = int(resume_state.get("epoch", 0)) if resume_state else 0
    micro_step = (
        int(resume_state.get("micro_step", step * accumulation))
        if resume_state
        else 0
    )
    if hasattr(train_dataset, "set_epoch"):
        train_dataset.set_epoch(epoch)
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    iterator = iter(train_loader)
    running: Dict[str, float] = {}
    running_count = 0
    consecutive_amp_overflows = 0
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
            sync_context = (
                nullcontext()
                if update_now or not hasattr(runner, "no_sync")
                else runner.no_sync()
            )
            with sync_context:
                autocast = (
                    torch.autocast("cuda", dtype=torch.float16)
                    if amp
                    else nullcontext()
                )
                with autocast:
                    losses = compute_batch(
                        batch,
                        runner,
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
                nonfinite = [
                    key
                    for key, value in losses.items()
                    if not bool(torch.isfinite(value.detach()).all())
                ]
                if nonfinite:
                    raise FloatingPointError(
                        "Non-finite Stage-3 forward losses before backward: "
                        + ", ".join(nonfinite)
                    )
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
            if not amp and not bool(torch.isfinite(grad_norm)):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "Non-finite Stage-3 gradient in FP32 smoke run; "
                    f"grad_norm={float(grad_norm)}"
                )
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            update_succeeded = scale_after >= scale_before
            optimizer.zero_grad(set_to_none=True)
            if not update_succeeded:
                consecutive_amp_overflows += 1
                if is_main:
                    logger.warning(
                        "AMP overflow %d/%d: skipped Stage-3 optimizer update "
                        "scale=%.3e->%.3e grad=%s total=%.6g noise=%.6g "
                        "temporal=%.6g residual=%.6g",
                        consecutive_amp_overflows,
                        max_consecutive_amp_overflows,
                        float(scale_before),
                        float(scale_after),
                        str(float(grad_norm)),
                        float(losses["total"].detach()),
                        float(losses["noise"].detach()),
                        float(losses["temporal"].detach()),
                        float(losses["residual"].detach()),
                    )
                running.clear()
                running_count = 0
                if consecutive_amp_overflows >= max_consecutive_amp_overflows:
                    raise FloatingPointError(
                        "Stage-3 AMP overflow did not recover after "
                        f"{consecutive_amp_overflows} consecutive attempts. "
                        "Use the FP32 smoke config to separate numerical "
                        "instability from model/data errors."
                    )
                continue
            consecutive_amp_overflows = 0
            scheduler.step()
            step += 1

            if step % log_freq == 0 or step == 1:
                averaged = global_average(running, running_count, device)
                if is_main:
                    lr_values = [group["lr"] for group in optimizer.param_groups]
                    logger.info(
                        "step=%d/%d total=%.5f noise=%.5f temporal=%.5f "
                        "corr_rms=%.5f residual=%.6f beta=%.3f grad=%.3f "
                        "lr=%s %.2fit/s",
                        step,
                        max_steps,
                        averaged["total"],
                        averaged["noise"],
                        averaged["temporal"],
                        averaged["correction_rms"],
                        averaged["residual"],
                        averaged["beta"],
                        float(grad_norm),
                        ",".join(f"{value:.3e}" for value in lr_values),
                        step / max(time.time() - started, 1e-6),
                    )
                    if writer is not None:
                        for key, value in averaged.items():
                            writer.add_scalar(f"train/{key}", value, step)
                        for index, value in enumerate(lr_values):
                            writer.add_scalar(f"train/lr_group_{index}", value, step)
                        writer.add_scalar("train/grad_norm", float(grad_norm), step)
                running.clear()
                running_count = 0

            valid_metrics = None
            is_best = False
            if step % valid_freq == 0 or step == max_steps:
                valid_metrics = validate(
                    valid_loader,
                    valid_batches,
                    runner,
                    vae,
                    tokenizer,
                    text_encoder,
                    noise_scheduler,
                    device,
                    weight_dtype,
                    config,
                    ip_conditioner,
                    int(config.get("seed", 2026)) + 200000 + rank,
                    train_stc_encoder,
                    gradient_checkpointing,
                )
                is_best = valid_metrics["total"] < best_valid
                if is_best:
                    best_valid = valid_metrics["total"]
                if is_main:
                    logger.info(
                        "VALID step=%d total=%.5f noise=%.5f temporal=%.5f "
                        "corr_rms=%.5f residual=%.6f beta=%.3f%s",
                        step,
                        valid_metrics["total"],
                        valid_metrics["noise"],
                        valid_metrics["temporal"],
                        valid_metrics["correction_rms"],
                        valid_metrics["residual"],
                        valid_metrics["beta"],
                        " (best)" if is_best else "",
                    )
                    if writer is not None:
                        for key, value in valid_metrics.items():
                            writer.add_scalar(f"valid/{key}", value, step)

            if step % save_freq == 0 or step == max_steps or is_best:
                if is_main:
                    destination = save_checkpoint(
                        checkpoint_root,
                        runner,
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
        logger.info("Stage-3 training finished best_valid=%.6f", best_valid)


if __name__ == "__main__":
    main()
