#!/usr/bin/env python
"""Stage-1 diagnostic: train only the V4++ flow head against teacher flow.

No diffusion, BrushNet, temporal attention, delta head, or alignment fusion is
executed.  The frozen RGB spatial encoder supplies pair features and the shared
bidirectional flow head is the only trainable component.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPImageProcessor


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from diffusers.models.stc_flow_training import (  # noqa: E402
    endpoint_error,
    prepare_teacher_flow,
)
from diffusers.optimization import get_scheduler  # noqa: E402
from STC_encoder_v3_rgb_flow.flow_supervision import (  # noqa: E402
    compute_teacher_flow_loss,
)
from STC_encoder_v3_rgb_flow.teacher_flow_data import (  # noqa: E402
    TeacherFlowV8ClipDataset,
    collate_teacher_flow_clips,
)
from STC_encoder_v4pp_bg_feature.bg_focused_flow_aligned_stc_adapter import (  # noqa: E402
    BGFocusedFlowAlignedRGBSTCAdapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init_checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--teacher_flow_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--valid_split", default="valid")
    parser.add_argument("--clip_length", type=int, default=16)
    parser.add_argument("--clip_stride", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--checkpointing_steps", type=int, default=250)
    parser.add_argument("--validation_steps", type=int, default=250)
    parser.add_argument("--valid_clips_per_sequence", type=int, default=1)
    parser.add_argument("--checkpoints_total_limit", type=int, default=10)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--flow_charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--log_steps", type=int, default=20)
    return parser.parse_args()


def resolve_component(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name in {"latest.json", "best.json"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate = Path(payload["checkpoint"])
        path = candidate if candidate.is_absolute() else path.parent / candidate
        path = path.resolve()
    nested = path / "stc_flow_model"
    component = nested if (nested / "config.json").is_file() else path
    if not (component / "config.json").is_file():
        raise FileNotFoundError(f"No V4++ stc_flow_model below {path}")
    return component


class FlowHeadOnlyModel(nn.Module):
    def __init__(self, backbone: BGFocusedFlowAlignedRGBSTCAdapter):
        super().__init__()
        self.backbone = backbone

    def train(self, mode: bool = True):
        # Keep the frozen representation deterministic even if a checkpoint
        # was configured with dropout. Only the flow head changes mode.
        super().train(mode)
        self.backbone.eval()
        self.backbone.flow_head.train(mode)
        return self

    def forward(self, rgb_sequence: torch.Tensor, bg_mask_sequence: torch.Tensor):
        batch, frames, height, width = self.backbone.stc_adapter._validate_inputs(
            rgb_sequence, bg_mask_sequence
        )
        with torch.no_grad():
            condition = self.backbone.build_pixel_condition(
                rgb_sequence, bg_mask_sequence
            ).flatten(0, 1)
            spatial_flat = self.backbone.stc_adapter.spatial_encoder(condition)
            channels, encoded_height, encoded_width = (
                spatial_flat.shape[1],
                spatial_flat.shape[2],
                spatial_flat.shape[3],
            )
            spatial = spatial_flat.reshape(
                batch, frames, channels, encoded_height, encoded_width
            )
        return self.backbone._decode_bidirectional_flow(spatial)


def make_dataset(args: argparse.Namespace, split: str, tokenizer):
    return TeacherFlowV8ClipDataset(
        dataset_root=args.dataset_root,
        split=split,
        teacher_flow_root=args.teacher_flow_root,
        tokenizer=tokenizer,
        clip_image_processor=CLIPImageProcessor(),
        clip_length=args.clip_length,
        stride=args.clip_stride,
        resolution=args.resolution,
    )


def select_balanced_validation(dataset, maximum: int) -> None:
    if maximum <= 0:
        return
    grouped: Dict[str, List] = defaultdict(list)
    for clip in dataset.clips:
        grouped[str(clip[0])].append(clip)
    selected = []
    for sequence in sorted(grouped):
        clips = grouped[sequence]
        count = min(maximum, len(clips))
        positions = np.linspace(0, len(clips) - 1, count).round().astype(int)
        selected.extend(clips[int(position)] for position in positions)
    dataset.clips = selected


def batch_tensors(batch, args, device):
    clips = int(batch["clip_batch_size"])
    frames = int(batch["num_frames"])
    if frames != args.clip_length:
        raise ValueError(f"Batch T={frames}, expected {args.clip_length}")
    rgb = batch["conditioning_pixel_values"].to(
        device=device, dtype=torch.float32, non_blocking=True
    ).reshape(clips, frames, 3, args.resolution, args.resolution)
    bg = batch["masks"].to(
        device=device, dtype=torch.float32, non_blocking=True
    ).reshape(clips, frames, 1, args.resolution, args.resolution)
    teacher = {
        key: batch[key].to(device=device, dtype=torch.float32, non_blocking=True)
        for key in (
            "teacher_flow_forward",
            "teacher_flow_backward",
            "teacher_valid_forward",
            "teacher_valid_backward",
        )
    }
    return clips, rgb, bg, teacher


def compute_outputs(model, rgb, bg, teacher, args):
    predicted_forward, predicted_backward = model(rgb, bg)
    report = compute_teacher_flow_loss(
        predicted_forward=predicted_forward,
        predicted_backward=predicted_backward,
        teacher_forward=teacher["teacher_flow_forward"],
        teacher_backward=teacher["teacher_flow_backward"],
        bg_mask_sequence=bg,
        valid_forward=teacher["teacher_valid_forward"],
        valid_backward=teacher["teacher_valid_backward"],
        region="all",
        charbonnier_eps=args.flow_charbonnier_eps,
    )
    size = tuple(predicted_forward.shape[-2:])
    teacher_forward, valid_forward = prepare_teacher_flow(
        teacher["teacher_flow_forward"], size, teacher["teacher_valid_forward"]
    )
    teacher_backward, valid_backward = prepare_teacher_flow(
        teacher["teacher_flow_backward"], size, teacher["teacher_valid_backward"]
    )
    zero_epe = 0.5 * (
        endpoint_error(torch.zeros_like(predicted_forward), teacher_forward, valid_forward)
        + endpoint_error(torch.zeros_like(predicted_backward), teacher_backward, valid_backward)
    )
    return report, zero_epe


def reduce_sums(values: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


@torch.no_grad()
def validate(model, loader, args, accelerator) -> Dict[str, float]:
    model.eval()
    totals = torch.zeros(6, device=accelerator.device, dtype=torch.float64)
    for batch in loader:
        clips, rgb, bg, teacher = batch_tensors(batch, args, accelerator.device)
        with accelerator.autocast():
            report, zero_epe = compute_outputs(model, rgb, bg, teacher, args)
        values = (
            report.loss,
            report.epe,
            zero_epe,
            report.predicted_magnitude,
            report.valid_forward_ratio,
        )
        totals[:5] += torch.stack([value.detach().double() for value in values]) * clips
        totals[5] += clips
    totals = reduce_sums(totals)
    count = totals[5].clamp_min(1.0)
    metrics = {
        "flow_loss": float(totals[0] / count),
        "epe_pred": float(totals[1] / count),
        "epe_zero": float(totals[2] / count),
        "predicted_magnitude": float(totals[3] / count),
        "valid_ratio": float(totals[4] / count),
    }
    metrics["epe_gain_over_zero"] = 1.0 - metrics["epe_pred"] / max(
        metrics["epe_zero"], 1e-8
    )
    model.train()
    return metrics


def json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    temporary.replace(path)


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def prune_checkpoints(output_dir: Path, limit: int) -> None:
    if limit <= 0:
        return
    paths = sorted(
        (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
        key=checkpoint_step,
    )
    for path in paths[: max(0, len(paths) - limit)]:
        shutil.rmtree(path)


def save_checkpoint(
    accelerator, model, optimizer, scheduler, args, step, metrics, is_best
) -> None:
    checkpoint = args.output_dir / f"checkpoint-{step}"
    accelerator.save_state(str(checkpoint / "accelerator_state"))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.backbone.save_pretrained(
            checkpoint / "stc_flow_model", safe_serialization=True
        )
        metadata = {
            "experiment": "v4pp_flow_head_only_stage1",
            "global_step": step,
            "trainable_components": ["flow_head"],
            "frozen_components": [
                "spatial_encoder",
                "temporal_blocks",
                "alignment_fusion",
                "zero_conv",
            ],
            "flow_region": "all",
            "loss": "bidirectional clean-teacher Charbonnier only",
            "validation": metrics,
            "init_checkpoint": str(args.init_checkpoint.resolve()),
        }
        json_dump(checkpoint / "metadata.json", metadata)
        json_dump(
            args.output_dir / "latest.json",
            {"checkpoint": checkpoint.name, "global_step": step},
        )
        if is_best:
            json_dump(
                args.output_dir / "best.json",
                {"checkpoint": checkpoint.name, "global_step": step, **metrics},
            )
        prune_checkpoints(args.output_dir, args.checkpoints_total_limit)
    accelerator.wait_for_everyone()


def resolve_resume(args) -> Optional[Path]:
    value = args.resume_from_checkpoint
    if not value:
        return None
    if value == "latest":
        pointer = args.output_dir / "latest.json"
        if not pointer.is_file():
            raise FileNotFoundError(pointer)
        value = json.loads(pointer.read_text())["checkpoint"]
    path = Path(value)
    if not path.is_absolute():
        path = args.output_dir / path
    if not (path / "accelerator_state").is_dir():
        raise FileNotFoundError(path / "accelerator_state")
    return path.resolve()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.teacher_flow_root = args.teacher_flow_root.expanduser().resolve()
    for value in (
        args.clip_length,
        args.clip_stride,
        args.train_batch_size,
        args.gradient_accumulation_steps,
        args.max_train_steps,
    ):
        if value <= 0:
            raise ValueError("Clip, batch and step settings must be positive")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed)
    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_dump(args.output_dir / "run_config.json", vars(args))

    resume = resolve_resume(args)
    init_component = resolve_component(
        resume if resume is not None else args.init_checkpoint
    )
    backbone = BGFocusedFlowAlignedRGBSTCAdapter.from_pretrained(str(init_component))
    backbone.requires_grad_(False)
    backbone.flow_head.requires_grad_(True)
    model = FlowHeadOnlyModel(backbone)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or not all(name.startswith("backbone.flow_head.") for name in trainable_names):
        raise RuntimeError(f"Unexpected trainable parameters: {trainable_names[:10]}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.pretrained_model_name_or_path.expanduser().resolve()),
        subfolder="tokenizer",
        use_fast=False,
    )
    train_dataset = make_dataset(args, args.train_split, tokenizer)
    valid_dataset = make_dataset(args, args.valid_split, tokenizer)
    select_balanced_validation(valid_dataset, args.valid_clips_per_sequence)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_teacher_flow_clips,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        collate_fn=collate_teacher_flow_clips,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        # Accelerate's prepared scheduler steps once per process when batches
        # are sharded. Match the audited V3 trainer so the user-facing warmup
        # and total-step settings remain optimizer-update counts under DDP.
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )
    global_step = 0
    if resume is not None:
        accelerator.load_state(str(resume / "accelerator_state"))
        global_step = checkpoint_step(resume)

    initial = validate(model, valid_loader, args, accelerator)
    if accelerator.is_main_process:
        json_dump(args.output_dir / "initial_validation.json", initial)
        print("INITIAL " + json.dumps(initial, sort_keys=True), flush=True)
        print(
            f"trainable={sum(p.numel() for p in trainable):,} names={trainable_names[:3]}...",
            flush=True,
        )

    best_epe = initial["epe_pred"]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    while global_step < args.max_train_steps:
        for batch in train_loader:
            with accelerator.accumulate(model):
                _, rgb, bg, teacher = batch_tensors(batch, args, accelerator.device)
                with accelerator.autocast():
                    report, zero_epe = compute_outputs(model, rgb, bg, teacher, args)
                    loss = report.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            if accelerator.is_main_process and (
                global_step == 1 or global_step % args.log_steps == 0
            ):
                print(
                    json.dumps(
                        {
                            "step": global_step,
                            "loss_flow": float(loss.detach()),
                            "epe_pred": float(report.epe.detach()),
                            "epe_zero": float(zero_epe.detach()),
                            "predicted_magnitude": float(report.predicted_magnitude.detach()),
                            "lr": float(scheduler.get_last_lr()[0]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            validate_now = (
                global_step % args.validation_steps == 0
                or global_step == args.max_train_steps
            )
            save_now = (
                global_step % args.checkpointing_steps == 0
                or global_step == args.max_train_steps
            )
            metrics = None
            is_best = False
            if validate_now:
                metrics = validate(model, valid_loader, args, accelerator)
                is_best = metrics["epe_pred"] <= best_epe
                best_epe = min(best_epe, metrics["epe_pred"])
                if accelerator.is_main_process:
                    print(
                        f"VALID step={global_step} " + json.dumps(metrics, sort_keys=True),
                        flush=True,
                    )
            if save_now or is_best:
                if metrics is None:
                    metrics = validate(model, valid_loader, args, accelerator)
                    is_best = metrics["epe_pred"] <= best_epe
                    best_epe = min(best_epe, metrics["epe_pred"])
                save_checkpoint(
                    accelerator,
                    model,
                    optimizer,
                    scheduler,
                    args,
                    global_step,
                    metrics,
                    is_best,
                )
            model.train()
            if global_step >= args.max_train_steps:
                break
        epoch += 1
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"Finished at step={global_step}, best_valid_epe={best_epe:.6f}")


if __name__ == "__main__":
    main()
