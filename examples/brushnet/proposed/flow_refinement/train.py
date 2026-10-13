#!/usr/bin/env python
"""Train/evaluate a frozen-V7, clean-teacher-supervised flow refiner."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
PROPOSED = HERE.parent
V7 = PROPOSED / "STC_encoder_v7_raft_flow_distillation"
for directory in (str(PROPOSED), str(V7)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from flow_refinement.model import FlowRefinementBlock, RefinerConfig, resize_flow
from flow_refinement.losses import refinement_losses
from raft_teacher_pair_data import RAFTTeacherFlowPairDataset, evenly_limit_pairs_per_sequence
from STC_encoder_v8_raft_deformable.raft_flow_provider import (
    FrozenV7RAFTFlowProvider, resolve_raft_student_component,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--teacher_flow_root", type=Path, required=True)
    parser.add_argument("--raft_student_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--valid_split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--valid_pairs_per_sequence", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pair_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--edge_weight", type=float, default=.05)
    parser.add_argument("--edge_threshold", type=float, default=1.0)
    parser.add_argument("--bg_weight", type=float, default=1.0)
    parser.add_argument("--quarter_max_displacement", type=float, default=8.0)
    parser.add_argument("--half_max_displacement", type=float, default=4.0)
    parser.add_argument("--search_radius", type=int, default=4)
    parser.add_argument("--quarter_only", action="store_true")
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--validation_every", type=int, default=250)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--evaluate_only", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None,
                        help="A checkpoint-N directory with accelerator_state")
    args = parser.parse_args()
    if args.resolution < 8 or args.resolution % 8:
        parser.error("resolution must be divisible by eight")
    if args.batch_size < 1 or args.pair_batch_size < 1 or args.max_steps < 1:
        parser.error("batch_size, pair_batch_size, and max_steps must be positive")
    if args.gradient_accumulation_steps < 1 or args.validation_every < 1 or args.checkpoint_every < 1:
        parser.error("accumulation, validation_every, and checkpoint_every must be positive")
    if args.resume and args.evaluate_only:
        parser.error("--resume and --evaluate_only are mutually exclusive")
    return args


def _loader(args, split: str, train: bool):
    dataset = RAFTTeacherFlowPairDataset(args.dataset_root, args.teacher_flow_root,
                                         split, resolution=args.resolution)
    if not train:
        evenly_limit_pairs_per_sequence(dataset, args.valid_pairs_per_sequence)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=train,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=False)
    return dataset, loader


def _batch_to_device(batch, device):
    keys = ("degraded0", "degraded1", "teacher_forward", "teacher_backward",
            "valid_forward", "valid_backward", "bg0", "bg1")
    result = {key: batch[key].to(device=device, dtype=torch.float32, non_blocking=True)
              for key in keys}
    size = result["degraded0"].shape[-2:]
    for key in ("teacher_forward", "teacher_backward"):
        if result[key].shape[-2:] != size:
            result[key] = resize_flow(result[key], size)
    for key in ("valid_forward", "valid_backward"):
        if result[key].shape[-2:] != size:
            result[key] = F.interpolate(result[key], size=size, mode="nearest")
    return result


def _coarse(provider, batch):
    rgb = torch.stack((batch["degraded0"], batch["degraded1"]), dim=1)
    flows = provider.predict_sequence(rgb)
    # inference_mode tensors must become ordinary detached tensors before
    # they enter the trainable refiner's convolution/warping autograd graph.
    return flows.forward[:, 0].clone(), flows.backward[:, 0].clone()


def _losses(model, batch, coarse_f, coarse_b, args):
    refined_f, refined_b = model(batch["degraded0"], batch["degraded1"],
                                  coarse_f, coarse_b)
    losses = refinement_losses(
        refined_f, refined_b, batch["teacher_forward"], batch["teacher_backward"],
        batch["valid_forward"], batch["valid_backward"], batch["bg0"], batch["bg1"],
        edge_weight=args.edge_weight, edge_threshold=args.edge_threshold,
        bg_weight=args.bg_weight,
    )
    with torch.no_grad():
        raw = refinement_losses(coarse_f, coarse_b, batch["teacher_forward"],
                                batch["teacher_backward"], batch["valid_forward"],
                                batch["valid_backward"], batch["bg0"], batch["bg1"],
                                edge_weight=0, edge_threshold=args.edge_threshold,
                                bg_weight=args.bg_weight)
    for metric in ("epe", "bg_epe", "boundary_epe"):
        losses["raw_"+metric] = raw[metric]
    return losses


@torch.no_grad()
def validate(accelerator, model, provider, loader, args):
    model.eval()
    keys = ("loss_total", "loss_flow", "loss_edge", "epe", "bg_epe",
            "boundary_epe", "raw_epe", "raw_bg_epe", "raw_boundary_epe")
    totals = torch.zeros(len(keys)+1, device=accelerator.device, dtype=torch.float64)
    for raw_batch in loader:
        batch = _batch_to_device(raw_batch, accelerator.device)
        coarse_f, coarse_b = _coarse(provider, batch)
        with accelerator.autocast():
            losses = _losses(model, batch, coarse_f, coarse_b, args)
        count = batch["degraded0"].shape[0]
        totals[:-1] += torch.stack([losses[k].detach().double() for k in keys])*count
        totals[-1] += count
    totals = accelerator.reduce(totals, reduction="sum")
    output = {key: (totals[index]/totals[-1].clamp_min(1)).item()
              for index, key in enumerate(keys)}
    model.train()
    return output


def _write_checkpoint(accelerator, model, args, step, metrics, base_component, best=False):
    if not accelerator.is_main_process:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_state": {k: v.detach().cpu() for k, v in accelerator.unwrap_model(model).state_dict().items()},
        "model_config": accelerator.unwrap_model(model).config_dict(),
        "base_raft_component": str(base_component),
        "step": int(step),
        "metrics": metrics,
        "loss_config": {"edge_weight": args.edge_weight,
                        "edge_threshold": args.edge_threshold, "bg_weight": args.bg_weight},
    }
    name = "best.pt" if best else f"checkpoint-{step:07d}.pt"
    target = args.output_dir / name
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    print(json.dumps({"saved": str(target), "step": step, **metrics}, sort_keys=True), flush=True)


def _load_payload(path: Path, expected_component: Path):
    payload = torch.load(path, map_location="cpu")
    if str(Path(payload["base_raft_component"]).resolve()) != str(expected_component):
        raise ValueError("Refiner checkpoint belongs to a different RAFT student component")
    return payload


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    base_component = resolve_raft_student_component(args.raft_student_path)
    config = RefinerConfig(search_radius=args.search_radius,
                           quarter_max_displacement=args.quarter_max_displacement,
                           half_max_displacement=args.half_max_displacement,
                           quarter_only=args.quarter_only)
    if args.preflight_only:
        train_set, _ = _loader(args, args.train_split, True)
        valid_set, _ = _loader(args, args.valid_split, False)
        config.validate()
        print(json.dumps({"preflight": "ok", "train_pairs": len(train_set),
                          "valid_pairs": len(valid_set), "raft_student": str(base_component),
                          "model_config": args.__dict__}, default=str, indent=2))
        return
    accelerator = Accelerator(mixed_precision=args.mixed_precision,
                              gradient_accumulation_steps=args.gradient_accumulation_steps,
                              kwargs_handlers=[DistributedDataParallelKwargs(
                                  find_unused_parameters=bool(args.quarter_only))])
    if accelerator.device.type != "cuda":
        raise ValueError("The frozen V7 RAFT provider requires CUDA")
    _, valid_loader = _loader(args, args.valid_split, False)
    if args.evaluate_only:
        payload = _load_payload(args.evaluate_only, base_component)
        args.edge_weight = float(payload["loss_config"]["edge_weight"])
        args.edge_threshold = float(payload["loss_config"]["edge_threshold"])
        args.bg_weight = float(payload["loss_config"]["bg_weight"])
        model = FlowRefinementBlock(RefinerConfig(**payload["model_config"]))
        model.load_state_dict(payload["model_state"], strict=True)
        model, valid_loader = accelerator.prepare(model, valid_loader)
        provider = FrozenV7RAFTFlowProvider(base_component, device=accelerator.device,
                                            pair_batch_size=args.pair_batch_size,
                                            mixed_precision=args.mixed_precision != "no")
        metrics = validate(accelerator, model, provider, valid_loader, args)
        if accelerator.is_main_process:
            print(json.dumps({"checkpoint": str(args.evaluate_only), "validation": metrics}, indent=2))
        return
    _, train_loader = _loader(args, args.train_split, True)
    model = FlowRefinementBlock(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader)
    provider = FrozenV7RAFTFlowProvider(base_component, device=accelerator.device,
                                        pair_batch_size=args.pair_batch_size,
                                        mixed_precision=args.mixed_precision != "no")
    step = 0
    best_bg_epe = float("inf")
    if args.resume:
        metadata = json.loads((args.resume / "metadata.json").read_text())
        if metadata["base_raft_component"] != str(base_component) or metadata["model_config"] != config.__dict__:
            raise ValueError("Resume checkpoint is incompatible with model/base flow")
        expected_loss = {"edge_weight": args.edge_weight,
                         "edge_threshold": args.edge_threshold,
                         "bg_weight": args.bg_weight}
        expected_data = {"dataset_root": str(args.dataset_root.resolve()),
                         "teacher_flow_root": str(args.teacher_flow_root.resolve()),
                         "resolution": args.resolution}
        if metadata.get("loss_config") != expected_loss or metadata.get("data_config") != expected_data:
            raise ValueError("Resume checkpoint is incompatible with loss/data configuration")
        accelerator.load_state(str(args.resume / "accelerator_state"))
        step = int(metadata["step"])
        best_bg_epe = (float(metadata["best_bg_epe"])
                       if metadata["best_bg_epe"] is not None else float("inf"))
    model.train()
    skipped_updates = 0
    while step < args.max_steps:
        for raw_batch in train_loader:
            batch = _batch_to_device(raw_batch, accelerator.device)
            coarse_f, coarse_b = _coarse(provider, batch)
            scale_before = accelerator.scaler.get_scale() if accelerator.scaler is not None else None
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    losses = _losses(model, batch, coarse_f, coarse_b, args)
                accelerator.backward(losses["loss_total"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            if not accelerator.sync_gradients:
                continue
            scale_was_reduced = (scale_before is not None and
                                 accelerator.scaler.get_scale() < scale_before)
            if accelerator.optimizer_step_was_skipped or scale_was_reduced:
                skipped_updates += 1
                if accelerator.is_main_process and skipped_updates <= 5:
                    print(json.dumps({"skipped_fp16_update": skipped_updates,
                                      "optimizer_step": step}), flush=True)
                if skipped_updates >= 100:
                    raise FloatingPointError("100 consecutive optimizer updates were skipped; try --mixed_precision no")
                continue
            skipped_updates = 0
            step += 1
            if accelerator.is_main_process and (step == 1 or step % 20 == 0):
                print(json.dumps({"step": step, "loss": float(losses["loss_total"].detach()),
                                  "raw_bg_epe": float(losses["raw_bg_epe"].detach()),
                                  "refined_bg_epe": float(losses["bg_epe"].detach())}), flush=True)
            if step % args.validation_every == 0 or step == args.max_steps:
                metrics = validate(accelerator, model, provider, valid_loader, args)
                if metrics["bg_epe"] < min(best_bg_epe, metrics["raw_bg_epe"]):
                    best_bg_epe = metrics["bg_epe"]
                    _write_checkpoint(accelerator, model, args, step, metrics, base_component, best=True)
            if step % args.checkpoint_every == 0 or step == args.max_steps:
                _write_checkpoint(accelerator, model, args, step,
                                  metrics if step % args.validation_every == 0 or step == args.max_steps else {},
                                  base_component)
                accelerator.wait_for_everyone()
                state_dir = args.output_dir / f"checkpoint-{step:07d}"
                state_dir.mkdir(parents=True, exist_ok=True)
                accelerator.save_state(str(state_dir / "accelerator_state"))
                if accelerator.is_main_process:
                    (state_dir / "metadata.json").write_text(json.dumps({
                        "step": step, "best_bg_epe": (best_bg_epe if best_bg_epe < float("inf") else None),
                        "base_raft_component": str(base_component),
                        "model_config": config.__dict__,
                        "loss_config": {"edge_weight": args.edge_weight,
                                        "edge_threshold": args.edge_threshold,
                                        "bg_weight": args.bg_weight},
                        "data_config": {"dataset_root": str(args.dataset_root.resolve()),
                                        "teacher_flow_root": str(args.teacher_flow_root.resolve()),
                                        "resolution": args.resolution},
                    }, indent=2)+"\n")
                accelerator.wait_for_everyone()
            if step >= args.max_steps:
                break


if __name__ == "__main__":
    main()
