#!/usr/bin/env python3
"""Stage-1 training for full STC flow prediction on paired VCM clips.

This stage is intentionally independent of the diffusion-noise loss.  The STC
encoder sees degraded input clips and learns bidirectional adjacent-frame flow
from a clean-video teacher cache.  Latents use ``latent_dist.mode()`` so flow
supervision is deterministic across iterations and validation runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

from diffusers import AutoencoderKL
from diffusers.models.stc_flow_training import (
    compute_flow_training_losses,
    save_stage1_checkpoint,
)
from diffusers.models.stc_noise_shaper import STCConditionedNoiseShaper
from stc_flow_dataset import STCFlowDataset

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
        help="Stage-1 checkpoint directory or latest.json/best.json pointer",
    )
    parser.add_argument(
        "--device",
        help="For example cuda, cuda:0, or cpu; overrides trainer.device",
    )
    return parser.parse_args()


def setup_logger(path: Path, enabled: bool) -> logging.Logger:
    logger = logging.getLogger("stc_flow_stage1")
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


def resolve_checkpoint(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path).resolve()
    if path.name in {"latest.json", "best.json"}:
        path = Path(json.loads(path.read_text(encoding="utf-8"))["checkpoint"])
    if not (path / "trainer_state.pt").is_file():
        raise FileNotFoundError(f"Invalid Stage-1 checkpoint: {path}")
    if not (path / "flow_predictor").is_dir():
        raise FileNotFoundError(f"Checkpoint has no flow_predictor component: {path}")
    return path


def resolve_model_checkpoint(path: Path) -> Path:
    """Resolve a weights-only component path, including JSON pointers."""
    path = Path(path).resolve()
    if path.name in {"latest.json", "best.json"}:
        path = Path(json.loads(path.read_text(encoding="utf-8"))["checkpoint"])
    nested = path / "flow_predictor"
    path = nested if nested.is_dir() else path
    if not path.is_dir():
        raise FileNotFoundError(f"Flow-predictor checkpoint not found: {path}")
    return path


def make_dataset(config: Mapping, split: str) -> STCFlowDataset:
    """Build the shared dataset loader from one config dictionary.

    Dataset contract: items contain degraded ``decoded_frames``, aligned
    ``gt_frames``/``roi_masks``, cached ``teacher_flow_forward`` and
    ``teacher_flow_backward``, plus optional ``teacher_valid_*`` masks.
    """
    data = dict(config["data"])
    data.setdefault("seed", int(config.get("seed", 0)))
    if not data.get("teacher_flow_root"):
        raise ValueError("Stage-1 training requires data.teacher_flow_root")
    return STCFlowDataset(data, split=split)


def preflight_data_paths(config: Mapping):
    data = config["data"]
    dataset_root = Path(data["dataset_root"]).expanduser().resolve()
    manifest = Path(
        data.get("manifest", dataset_root / "manifest.json")
    ).expanduser().resolve()
    teacher_root = Path(data["teacher_flow_root"]).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
    if not teacher_root.is_dir():
        raise FileNotFoundError(
            f"Teacher-flow cache not found: {teacher_root}. Run "
            "precompute_sfu_stc_teacher_flows.py before Stage-1 training."
        )


def make_loader(
    dataset,
    trainer: Mapping,
    split: str,
    distributed: bool,
    world_size: int,
    rank: int,
):
    training = split == "train"
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size if distributed else 1,
            rank=rank if distributed else 0,
            shuffle=training,
            drop_last=training,
            seed=int(getattr(dataset, "seed", 0)),
        )
        if distributed or training
        else None
    )
    batch_size = int(
        trainer.get("batch_size", 1)
        if training
        else trainer.get("valid_batch_size", 1)
    )
    workers = int(trainer.get("num_workers", 2))
    persistent_workers = bool(
        trainer.get("persistent_workers", True) and workers > 0
    )
    # STCFlowDataset.set_epoch controls stateless horizontal augmentation.
    # Persistent worker copies would not observe later parent-process updates.
    if training and float(getattr(dataset, "horizontal_flip_probability", 0.0)) > 0:
        persistent_workers = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        drop_last=training,
        num_workers=workers,
        pin_memory=bool(trainer.get("pin_memory", True)),
        persistent_workers=persistent_workers,
    ), sampler


def _batch_value(batch: Mapping, primary: str, aliases: Iterable[str] = ()):
    for key in (primary, *aliases):
        if key in batch:
            return batch[key]
    raise KeyError(f"Batch is missing {primary!r}; accepted aliases: {tuple(aliases)}")


def _optional_batch_value(batch: Mapping, primary: str, aliases: Iterable[str] = ()):
    for key in (primary, *aliases):
        if key in batch:
            return batch[key]
    return None


@torch.no_grad()
def encode_deterministic_latents(
    vae: AutoencoderKL,
    images: torch.Tensor,
    device: torch.device,
    vae_dtype: torch.dtype,
) -> torch.Tensor:
    """Encode `[B,T,3,H,W]` with posterior mode, never posterior sampling."""
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError("decoded_frames must have shape [B,T,3,H,W]")
    batch, frames = images.shape[:2]
    flat = images.flatten(0, 1).to(device=device, dtype=vae_dtype)
    latents = vae.encode(flat).latent_dist.mode()
    latents = latents * vae.config.scaling_factor
    return latents.reshape(batch, frames, *latents.shape[1:]).float()


def latent_background_mask(roi_masks: torch.Tensor, size, device: torch.device):
    if roi_masks.ndim != 5 or roi_masks.shape[2] != 1:
        raise ValueError("roi_masks must have shape [B,T,1,H,W]")
    batch, frames = roi_masks.shape[:2]
    background = 1.0 - roi_masks.float().to(device)
    return F.interpolate(
        background.flatten(0, 1), size=size, mode="nearest"
    ).reshape(batch, frames, 1, *size)


def build_model(config: Mapping, checkpoint: Optional[Path] = None):
    if checkpoint is not None:
        return STCConditionedNoiseShaper.from_pretrained(
            checkpoint / "flow_predictor"
        )
    model_config = dict(config.get("flow_predictor", {}))
    model_config.pop("checkpoint", None)
    return STCConditionedNoiseShaper(**model_config)


def predict_batch_flow(model, decoded_latents, bg_mask):
    bare_model = model.module if hasattr(model, "module") else model
    # Calling through DDP is required for reducer hooks.  The model's forward
    # supports this explicit stage selector while the bare model exposes the
    # convenient predict_flow API for non-DDP use.
    if hasattr(model, "module"):
        return model(
            operation="predict_flow",
            decoded_latents=decoded_latents,
            bg_mask=bg_mask,
        )
    return bare_model.predict_flow(decoded_latents, bg_mask, return_dict=True)


def compute_batch(
    batch: Mapping,
    model,
    vae,
    device: torch.device,
    vae_dtype: torch.dtype,
    amp: bool,
    loss_config: Mapping,
) -> Dict[str, torch.Tensor]:
    decoded_cpu = _batch_value(batch, "decoded_frames", ("input_frames",))
    roi_cpu = _batch_value(batch, "roi_masks", ("masks",))
    decoded = decoded_cpu.to(device=device, dtype=torch.float32)
    decoded_latents = encode_deterministic_latents(
        vae, decoded_cpu, device, vae_dtype
    )
    bg_mask = latent_background_mask(
        roi_cpu, decoded_latents.shape[-2:], device
    )
    teacher_forward = _batch_value(
        batch, "teacher_flow_forward", ("teacher_forward", "flow_forward")
    ).to(device=device, dtype=torch.float32)
    teacher_backward = _batch_value(
        batch, "teacher_flow_backward", ("teacher_backward", "flow_backward")
    ).to(device=device, dtype=torch.float32)
    valid_forward = _optional_batch_value(
        batch, "teacher_valid_forward", ("valid_forward", "valid_f")
    )
    valid_backward = _optional_batch_value(
        batch, "teacher_valid_backward", ("valid_backward", "valid_b")
    )
    if valid_forward is not None:
        valid_forward = valid_forward.to(device=device, dtype=torch.float32)
    if valid_backward is not None:
        valid_backward = valid_backward.to(device=device, dtype=torch.float32)

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if amp
        else nullcontext()
    )
    with autocast:
        output = predict_batch_flow(model, decoded_latents, bg_mask)
    predicted_forward = output["predicted_flow_forward"].float()
    predicted_backward = output["predicted_flow_backward"].float()
    return compute_flow_training_losses(
        predicted_forward,
        predicted_backward,
        teacher_forward,
        teacher_backward,
        decoded,
        valid_forward=valid_forward,
        valid_backward=valid_backward,
        loss_config=loss_config,
    )


@torch.no_grad()
def validate(
    loader,
    model,
    vae,
    device,
    vae_dtype,
    amp,
    loss_config,
    max_batches: int,
):
    was_training = model.training
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        losses = compute_batch(
            batch, model, vae, device, vae_dtype, amp, loss_config
        )
        batch_size = int(_batch_value(batch, "decoded_frames", ("input_frames",)).shape[0])
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * batch_size
        count += batch_size
    metrics = global_average(totals, count, device)
    if was_training:
        model.train()
    return metrics


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
    vae_dtype = torch.float16 if amp else torch.float32
    output_dir = Path(config["output_dir"]).resolve()
    checkpoint_root = Path(config["checkpoint_dir"]).resolve()
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
        if writer is None:
            logger.warning("TensorBoard unavailable; install tensorboard to enable it")

    preflight_data_paths(config)

    resume = resolve_checkpoint(
        args.resume
        or (
            Path(config["model"]["resume"])
            if config.get("model", {}).get("resume")
            else None
        )
    )
    init_checkpoint = config.get("flow_predictor", {}).get("checkpoint")
    if resume is not None and init_checkpoint:
        logger.warning("Ignoring flow_predictor.checkpoint because resume is active")
    model_checkpoint = resume
    if model_checkpoint is None and init_checkpoint:
        candidate = resolve_model_checkpoint(Path(init_checkpoint))
        model = STCConditionedNoiseShaper.from_pretrained(candidate)
    else:
        model = build_model(config, model_checkpoint)
    if str(model.config.flow_prediction_mode).lower() != "full":
        raise ValueError("Stage-1 trainer requires flow_prediction_mode='full'")
    flow_parameter_ids = {id(parameter) for parameter in model.flow_parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in flow_parameter_ids)
    model = model.to(device=device, dtype=torch.float32).train()

    vae = AutoencoderKL.from_pretrained(
        config["model"]["base_model"], subfolder="vae"
    )
    vae.requires_grad_(False).eval().to(device=device, dtype=vae_dtype)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
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
        raise ValueError(
            "Training DataLoader is empty; reduce batch_size/world_size or add clips"
        )
    bare_model = model.module if hasattr(model, "module") else model
    trainable_parameters = [
        parameter
        for parameter in bare_model.flow_parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("STC flow predictor has no trainable parameters")
    lr = float(trainer.get("lr", 1e-4))
    min_lr = float(trainer.get("min_lr", lr * 0.01))
    if not 0.0 <= min_lr <= lr:
        raise ValueError("Require 0 <= min_lr <= lr")
    optimizer = AdamW(
        trainable_parameters,
        lr=lr,
        betas=tuple(trainer.get("betas", [0.9, 0.999])),
        weight_decay=float(trainer.get("weight_decay", 0.01)),
    )
    max_steps = int(trainer["max_steps"])
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    scheduler = LambdaLR(
        optimizer,
        cosine_schedule(
            int(trainer.get("warmup_steps", 1000)),
            max_steps,
            min_lr / lr,
        ),
    )
    amp_init_scale = float(trainer.get("amp_init_scale", 2048.0))
    if amp_init_scale <= 0:
        raise ValueError("amp_init_scale must be positive")
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp,
        init_scale=amp_init_scale,
    )
    step = 0
    best_valid_epe = math.inf
    resume_state = None
    if resume is not None:
        resume_state = torch.load(resume / "trainer_state.pt", map_location="cpu")
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        if resume_state.get("scaler") is not None:
            scaler.load_state_dict(resume_state["scaler"])
        step = int(resume_state["step"])
        best_valid_epe = float(resume_state.get("best_valid_epe", math.inf))
        logger.info("Resumed %s at step %d", resume, step)

    accumulation = int(trainer.get("gradient_accumulation_steps", 1))
    if accumulation <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    logger.info(
        "train_clips=%d valid_clips=%d T=%d B_per_gpu=%d world=%d "
        "effective_batch=%d trainable=%d deterministic_vae=True "
        "epe_units=latent_pixels device=%s amp=%s",
        len(train_dataset),
        len(valid_dataset),
        int(config["data"]["clip_length"]),
        int(trainer.get("batch_size", 1)),
        world_size,
        int(trainer.get("batch_size", 1)) * world_size * accumulation,
        sum(parameter.numel() for parameter in trainable_parameters),
        device,
        amp,
    )
    log_freq = int(trainer.get("log_freq", 20))
    valid_freq = int(trainer.get("valid_freq", 500))
    save_freq = int(trainer.get("save_freq", 500))
    if min(log_freq, valid_freq, save_freq) <= 0:
        raise ValueError("log_freq, valid_freq, and save_freq must be positive")
    valid_batches = int(trainer.get("valid_batches", 0))
    grad_clip = float(trainer.get("grad_clip", 1.0))
    optimizer.zero_grad(set_to_none=True)
    epoch = int(resume_state.get("epoch", 0)) if resume_state else 0
    micro_step = (
        int(resume_state.get("micro_step", step * accumulation))
        if resume_state
        else 0
    )
    batches_in_epoch = (
        int(resume_state.get("batches_in_epoch", 0)) if resume_state else 0
    )
    epoch += batches_in_epoch // len(train_loader)
    batches_in_epoch %= len(train_loader)
    if hasattr(train_dataset, "set_epoch"):
        train_dataset.set_epoch(epoch)
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    train_iterator = iter(train_loader)
    for _ in range(batches_in_epoch):
        next(train_iterator)
    running: Dict[str, float] = {}
    running_count = 0
    started = time.time()

    try:
        while step < max_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                epoch += 1
                batches_in_epoch = 0
                if hasattr(train_dataset, "set_epoch"):
                    train_dataset.set_epoch(epoch)
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            batches_in_epoch += 1
            micro_step += 1
            update_now = micro_step % accumulation == 0
            sync_context = (
                nullcontext()
                if update_now or not hasattr(model, "no_sync")
                else model.no_sync()
            )
            with sync_context:
                losses = compute_batch(
                    batch,
                    model,
                    vae,
                    device,
                    vae_dtype,
                    amp,
                    config.get("loss", {}),
                )
                scaler.scale(losses["total"] / accumulation).backward()
            batch_size = int(
                _batch_value(batch, "decoded_frames", ("input_frames",)).shape[0]
            )
            for key, value in losses.items():
                running[key] = running.get(key, 0.0) + float(value.detach()) * batch_size
            running_count += batch_size
            if not update_now:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            update_succeeded = scale_after >= scale_before
            optimizer.zero_grad(set_to_none=True)

            if not update_succeeded:
                if is_main:
                    logger.warning(
                        "Skipped optimizer update because AMP detected non-finite "
                        "gradients (scale %.0f -> %.0f)",
                        scale_before,
                        scale_after,
                    )
                running.clear()
                running_count = 0
                continue

            scheduler.step()
            step += 1

            if step % log_freq == 0 or step == 1:
                averaged = global_average(running, running_count, device)
                elapsed = max(time.time() - started, 1e-6)
                speed = step / elapsed
                if is_main:
                    logger.info(
                        "step=%d/%d total=%.5f teacher=%.5f epe=%.4f "
                        "fb=%.5f smooth=%.5f valid_f=%.3f valid_b=%.3f "
                        "grad=%.3f lr=%.3e %.2fit/s",
                        step,
                        max_steps,
                        averaged["total"],
                        averaged["teacher"],
                        averaged["epe"],
                        averaged["fb"],
                        averaged["smoothness"],
                        averaged["valid_forward"],
                        averaged["valid_backward"],
                        float(grad_norm),
                        optimizer.param_groups[0]["lr"],
                        speed,
                    )
                    if writer is not None:
                        for key, value in averaged.items():
                            writer.add_scalar(f"train/{key}", value, step)
                        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                        writer.add_scalar("train/grad_norm", float(grad_norm), step)
                running.clear()
                running_count = 0

            validate_now = step % valid_freq == 0 or step == max_steps
            valid_metrics = None
            is_best = False
            if validate_now:
                valid_metrics = validate(
                    valid_loader,
                    model,
                    vae,
                    device,
                    vae_dtype,
                    amp,
                    config.get("loss", {}),
                    valid_batches,
                )
                is_best = valid_metrics["epe"] < best_valid_epe
                if is_best:
                    best_valid_epe = valid_metrics["epe"]
                if is_main:
                    logger.info(
                        "VALID step=%d total=%.5f teacher=%.5f epe=%.4f "
                        "epe_f=%.4f epe_b=%.4f fb=%.5f smooth=%.5f%s",
                        step,
                        valid_metrics["total"],
                        valid_metrics["teacher"],
                        valid_metrics["epe"],
                        valid_metrics["epe_forward"],
                        valid_metrics["epe_backward"],
                        valid_metrics["fb"],
                        valid_metrics["smoothness"],
                        " (best)" if is_best else "",
                    )
                    if writer is not None:
                        for key, value in valid_metrics.items():
                            writer.add_scalar(f"valid/{key}", value, step)

            save_now = step % save_freq == 0 or step == max_steps or is_best
            if is_main and save_now:
                destination = save_stage1_checkpoint(
                    checkpoint_root,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    step,
                    best_valid_epe,
                    config,
                    is_best=is_best,
                    extra_state={
                        "epoch": epoch,
                        "micro_step": micro_step,
                        "batches_in_epoch": batches_in_epoch,
                    },
                )
                logger.info(
                    "Saved %s%s", destination, " (best)" if is_best else ""
                )
            if distributed and (validate_now or save_now):
                dist.barrier()
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()
    if is_main:
        logger.info("Training finished best_valid_epe=%.4f", best_valid_epe)


if __name__ == "__main__":
    main()
