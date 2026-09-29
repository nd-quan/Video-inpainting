#!/usr/bin/env python3
"""Joint Stage-3 STC-condition and diffusion U-Net temporal-adapter trainer.

This is deliberately a separate entry point.  Running
``train_stc_condition_adapter_vcm.py`` continues to train the original
condition-only Stage 3 and never creates or updates a temporal adapter.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Mapping

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import train_stc_condition_adapter_vcm as stage3
from stc_condition_temporal_adapter_training import (
    FrozenBrushNetSTCConditionTemporalModel,
    joint_stage3_parameter_groups,
    joint_stage3_trainable_parameters,
    load_or_build_temporal_adapter,
    resolve_temporal_adapter_component,
    set_joint_stage3_train_mode,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Joint condition+temporal checkpoint or latest/best pointer",
    )
    parser.add_argument("--device", help="cuda, cuda:0, or cpu")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def prepare_joint_config(config: Mapping) -> Dict:
    preflight = stage3.prepare_stage3_config(config)
    temporal = config.get("temporal_adapter")
    if not isinstance(temporal, Mapping) or not bool(temporal.get("enabled", False)):
        raise ValueError("temporal_adapter.enabled=true is required by this trainer")
    down = tuple(int(value) for value in temporal.get("down_block_indices", (0, 1, 2)))
    up = tuple(int(value) for value in temporal.get("up_block_indices", ()))
    if not down and not up and not bool(temporal.get("use_mid", True)):
        raise ValueError("At least one temporal injection location is required")
    if len(set(down)) != len(down) or len(set(up)) != len(up):
        raise ValueError("Temporal block indices must be unique")
    if int(temporal.get("bottleneck_channels", 64)) <= 0:
        raise ValueError("temporal_adapter.bottleneck_channels must be positive")
    kernel = int(temporal.get("temporal_kernel_size", 3))
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError("temporal_kernel_size must be a positive odd number")
    dropout = float(temporal.get("dropout", 0.0))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("temporal_adapter.dropout must be in [0,1)")
    scale = float(temporal.get("scale", 1.0))
    if scale < 0.0:
        raise ValueError("temporal_adapter.scale must be non-negative")

    resume_value = config.get("model", {}).get("resume")
    resume = stage3.resolve_resume(Path(resume_value)) if resume_value else None
    if resume is not None:
        resolve_temporal_adapter_component(resume)
    result = dict(preflight)
    result["temporal_adapter"] = {
        "down_block_indices": list(down),
        "use_mid": bool(temporal.get("use_mid", True)),
        "up_block_indices": list(up),
        "bottleneck_channels": int(temporal.get("bottleneck_channels", 64)),
        "temporal_kernel_size": kernel,
        "dropout": dropout,
        "scale": scale,
        "resume_component": (
            str(resolve_temporal_adapter_component(resume)) if resume else None
        ),
    }
    return result


@torch.no_grad()
def validate_joint(
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
    metrics = stage3.validate(
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
    )
    set_joint_stage3_train_mode(
        runner,
        train_stc_encoder=train_stc_encoder,
        gradient_checkpointing=gradient_checkpointing,
    )
    return metrics


def save_joint_checkpoint(
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
    bare.temporal_adapter.save_pretrained(temporary / "temporal_adapter")
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
    preflight = prepare_joint_config(config)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    trainer = config["trainer"]
    distributed, rank, local_rank, world_size, device = stage3.distributed_context(
        args.device or trainer.get("device", "cuda")
    )
    is_main = rank == 0
    stage3.seed_everything(int(config.get("seed", 2026)) + rank)
    amp = bool(trainer.get("amp", True) and device.type == "cuda")
    weight_dtype = torch.float16 if amp else torch.float32
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    checkpoint_root = Path(config["checkpoint_dir"]).expanduser().resolve()
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    logger = stage3.setup_logger(output_dir / "logs" / "train.log", is_main)
    writer = (
        stage3.SummaryWriter(output_dir / "tensorboard")
        if is_main and stage3.SummaryWriter is not None
        else None
    )
    if is_main:
        (output_dir / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        (output_dir / "preflight.json").write_text(
            json.dumps(preflight, indent=2, sort_keys=True), encoding="utf-8"
        )

    resume = stage3.resolve_resume(
        args.resume
        or (
            Path(config["model"]["resume"])
            if config.get("model", {}).get("resume")
            else None
        )
    )
    initialization = (
        stage3.resolve_resume(Path(config["model"]["init_checkpoint"]))
        if config.get("model", {}).get("init_checkpoint")
        else None
    )
    if resume is not None and initialization is not None:
        raise ValueError("Use only one of model.resume or model.init_checkpoint")
    if resume is not None:
        resolve_temporal_adapter_component(resume)

    base = config["model"]["base_model"]
    logger.info(
        "Loading frozen VAE, BrushNet, IP-Adapter, diffusion U-Net, and temporal adapter"
    )
    noise_scheduler = stage3.DDPMScheduler.from_pretrained(
        base, subfolder="scheduler"
    )
    tokenizer = stage3.CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = stage3.frozen(
        stage3.CLIPTextModel.from_pretrained(base, subfolder="text_encoder"),
        device,
        weight_dtype,
    )
    vae = stage3.frozen(
        stage3.AutoencoderKL.from_pretrained(base, subfolder="vae"),
        device,
        weight_dtype,
    )
    unet = stage3.frozen(
        stage3.UNet2DConditionModel.from_pretrained(base, subfolder="unet"),
        device,
        weight_dtype,
    )
    ip_conditioner = stage3.build_ip_conditioner(unet, config, device)
    unet.requires_grad_(False)
    brushnet = stage3.frozen(
        stage3.BrushNetModel.from_pretrained(config["model"]["brushnet"]),
        device,
        weight_dtype,
    )
    noise_shaper, condition_adapter, noise_source_path, noise_source_mode = (
        stage3._load_components(config, resume, brushnet, device)
    )
    temporal_config = config["temporal_adapter"]
    temporal_adapter = load_or_build_temporal_adapter(
        unet,
        temporal_config,
        resume=resume,
        initialization=initialization,
    ).to(device=device, dtype=torch.float32)

    adapter_config = config.get("adapter", {})
    train_stc_encoder = bool(adapter_config.get("train_stc_encoder", False))
    gradient_checkpointing = bool(trainer.get("gradient_checkpointing", True))
    if gradient_checkpointing:
        if hasattr(unet, "enable_gradient_checkpointing"):
            unet.enable_gradient_checkpointing()
        if hasattr(brushnet, "enable_gradient_checkpointing"):
            brushnet.enable_gradient_checkpointing()

    model = FrozenBrushNetSTCConditionTemporalModel(
        noise_shaper=noise_shaper,
        condition_adapter=condition_adapter,
        temporal_adapter=temporal_adapter,
        brushnet=brushnet,
        unet=unet,
        alphas_cumprod=noise_scheduler.alphas_cumprod,
        noise_strength=float(config.get("noise_fusion", {}).get("strength", 1.0)),
        injection_scale=float(adapter_config.get("injection_scale", 1.0)),
        temporal_adapter_scale=float(temporal_config.get("scale", 1.0)),
    ).to(device=device)

    condition_lr = float(trainer.get("lr", 1e-4))
    temporal_lr = float(temporal_config.get("lr", condition_lr))
    parameter_groups = joint_stage3_parameter_groups(
        model,
        condition_lr=condition_lr,
        temporal_lr=temporal_lr,
        train_stc_encoder=train_stc_encoder,
        stc_lr=float(adapter_config.get("stc_lr", condition_lr * 0.1)),
    )
    parameters = list(joint_stage3_trainable_parameters(model))
    if not parameters:
        raise ValueError("Joint Stage 3 has no trainable parameters")
    runner = model
    if distributed:
        runner = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    set_joint_stage3_train_mode(
        runner,
        train_stc_encoder=train_stc_encoder,
        gradient_checkpointing=gradient_checkpointing,
    )

    train_dataset = stage3.make_dataset(config, "train")
    valid_dataset = stage3.make_dataset(config, "valid")
    train_loader, train_sampler = stage3.make_loader(
        train_dataset, trainer, "train", distributed, world_size, rank
    )
    valid_loader, _ = stage3.make_loader(
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
    min_lr = float(trainer.get("min_lr", condition_lr * 0.01))
    scheduler = LambdaLR(
        optimizer,
        stage3.cosine_schedule(
            int(trainer.get("warmup_steps", 500)),
            max_steps,
            min_lr / condition_lr,
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
        logger.info("Resumed joint checkpoint %s at step %d", resume, step)

    accumulation = int(trainer.get("gradient_accumulation_steps", 1))
    log_freq = int(trainer.get("log_freq", 20))
    valid_freq = int(trainer.get("valid_freq", 500))
    save_freq = int(trainer.get("save_freq", 500))
    valid_batches = int(trainer.get("valid_batches", 0))
    max_overflows = int(trainer.get("max_consecutive_amp_overflows", 16))
    if max_overflows < 1:
        raise ValueError("max_consecutive_amp_overflows must be positive")
    condition_count = sum(p.numel() for p in model.condition_adapter.parameters())
    temporal_count = sum(p.numel() for p in model.temporal_adapter.parameters())
    stc_count = sum(
        p.numel()
        for p in model.noise_shaper.stc_encoder.parameters()
        if p.requires_grad
    )
    logger.info(
        "train_clips=%d valid_clips=%d T=%d B_per_gpu=%d world=%d "
        "effective_batch=%d condition_trainable=%d temporal_trainable=%d "
        "stc_trainable=%d temporal_down=%s temporal_mid=%s noise_source=%s "
        "source_path=%s beta_mode=%s fixed_beta=%s device=%s amp=%s",
        len(train_dataset),
        len(valid_dataset),
        int(config["data"]["clip_length"]),
        int(trainer.get("batch_size", 1)),
        world_size,
        int(trainer.get("batch_size", 1)) * world_size * accumulation,
        condition_count,
        temporal_count,
        stc_count,
        list(temporal_adapter.config.down_block_indices),
        bool(temporal_adapter.config.use_mid),
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
    consecutive_overflows = 0
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
                    losses = stage3.compute_batch(
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
                        "Non-finite joint Stage-3 losses: " + ", ".join(nonfinite)
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
                raise FloatingPointError(f"Non-finite FP32 gradient: {float(grad_norm)}")
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            optimizer.zero_grad(set_to_none=True)
            if scale_after < scale_before:
                consecutive_overflows += 1
                if is_main:
                    logger.warning(
                        "AMP overflow %d/%d; skipped update scale=%.3e->%.3e grad=%s",
                        consecutive_overflows,
                        max_overflows,
                        float(scale_before),
                        float(scale_after),
                        str(float(grad_norm)),
                    )
                running.clear()
                running_count = 0
                if consecutive_overflows >= max_overflows:
                    raise FloatingPointError("Joint Stage-3 AMP overflow did not recover")
                continue
            consecutive_overflows = 0
            scheduler.step()
            step += 1

            if step % log_freq == 0 or step == 1:
                averaged = stage3.global_average(running, running_count, device)
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
                valid_metrics = validate_joint(
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
                    destination = save_joint_checkpoint(
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
        logger.info("Joint Stage-3 training finished best_valid=%.6f", best_valid)


if __name__ == "__main__":
    main()
