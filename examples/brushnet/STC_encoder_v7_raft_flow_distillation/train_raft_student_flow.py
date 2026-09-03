#!/usr/bin/env python
"""Standalone clean-to-degraded RAFT flow distillation.

This is intentionally a *flow-only* stage.  It trains a pretrained RAFT on
degraded RGB pairs to reproduce frozen clean-RAFT flow cached offline.  No
BrushNet, STC temporal block, V5 memory, V6 DCN, or diffusion objective is
constructed here.  The selected RAFT student is integrated later as V6's base
feature-flow prior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from diffusers.models.stc_flow_training import (  # noqa: E402
    charbonnier_flow_loss,
    edge_aware_smoothness_loss,
    endpoint_error,
    forward_backward_consistency_loss,
    prepare_teacher_flow,
    resize_flow_sequence,
)
from diffusers.optimization import get_scheduler  # noqa: E402
from raft_student import RAFTStudentFlowPredictor  # noqa: E402
from raft_teacher_pair_data import (  # noqa: E402
    RAFTTeacherFlowPairDataset,
    evenly_limit_pairs_per_sequence,
)


DEFAULT_PROPAINTER = Path("/home/cilab/ndquan/videoInpainting/pretrained/ProPainter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--teacher_flow_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--propainter_root", type=Path, default=DEFAULT_PROPAINTER)
    parser.add_argument("--raft_checkpoint", type=Path, required=True)
    parser.add_argument("--train_split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--valid_split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--include_branches", nargs="*", default=None)
    parser.add_argument("--valid_pairs_per_sequence", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--raft_pair_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=5000)
    parser.add_argument("--checkpointing_steps", type=int, default=250)
    parser.add_argument("--validation_steps", type=int, default=250)
    parser.add_argument("--checkpoints_total_limit", type=int, default=20)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--raft_iterations", type=int, default=20)
    parser.add_argument("--final_flow_only", action="store_true")
    parser.add_argument("--raft_sequence_gamma", type=float, default=0.8)
    parser.add_argument("--teacher_weight", type=float, default=1.0)
    parser.add_argument("--fb_weight", type=float, default=0.1)
    parser.add_argument("--smoothness_weight", type=float, default=0.01)
    parser.add_argument("--smoothness_edge_weight", type=float, default=10.0)
    parser.add_argument("--flow_charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--flow_loss_region", choices=("all", "bg"), default="all")
    parser.add_argument("--deform_feature_size", type=int, default=64)
    parser.add_argument("--deform_residual_range", type=float, default=2.0)
    parser.add_argument("--best_metric", choices=("bg_epe", "epe", "large_motion_epe"), default="bg_epe")
    parser.add_argument("--metric_histogram_max", type=float, default=64.0)
    parser.add_argument("--metric_histogram_bins", type=int, default=512)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="fp16")
    # Python 3.8 in the guided_diff environment predates
    # argparse.BooleanOptionalAction, so retain an explicit inverse flag.
    parser.add_argument("--freeze_batchnorm", dest="freeze_batchnorm", action="store_true", default=True)
    parser.add_argument("--train_batchnorm", dest="freeze_batchnorm", action="store_false")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--evaluate_only", default=None)
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--allow_teacher_checkpoint_mismatch", action="store_true")
    parser.add_argument("--log_steps", type=int, default=20)
    return parser.parse_args()


def json_dump(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Checkpoint directory must be checkpoint-N: {path}")
    return int(match.group(1))


def _resolve_pointer(root: Path, name: str) -> Path:
    pointer = root / f"{name}.json"
    if not pointer.is_file():
        raise FileNotFoundError(pointer)
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    path = Path(payload["checkpoint"])
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def resolve_checkpoint(value: Optional[str], output_dir: Path) -> Optional[Path]:
    if not value:
        return None
    if value in {"latest", "best"}:
        path = _resolve_pointer(output_dir, value)
    else:
        path = Path(value)
        if not path.is_absolute():
            path = output_dir / path
        path = path.resolve()
    if not (path / "raft_student" / "pytorch_model.bin").is_file():
        raise FileNotFoundError(f"No RAFT student weights under {path}")
    return path


def _five(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(value.shape)}")
    return value.unsqueeze(1)


def _pair_batch(batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    tensor_keys = (
        "degraded0",
        "degraded1",
        "bg0",
        "bg1",
        "teacher_forward",
        "teacher_backward",
        "valid_forward",
        "valid_backward",
    )
    return {
        key: batch[key].to(device=device, non_blocking=True, dtype=torch.float32)
        for key in tensor_keys
    }


def _targets_and_region_weights(
    batch: Dict[str, torch.Tensor],
    output_size: Tuple[int, int],
    region: str,
):
    target_forward, valid_forward = prepare_teacher_flow(
        _five(batch["teacher_forward"]), output_size, _five(batch["valid_forward"])
    )
    target_backward, valid_backward = prepare_teacher_flow(
        _five(batch["teacher_backward"]), output_size, _five(batch["valid_backward"])
    )
    bg = F.interpolate(
        torch.cat((batch["bg0"], batch["bg1"]), dim=0),
        size=output_size,
        mode="nearest",
    )
    batch_size = batch["bg0"].shape[0]
    bg_forward = bg[:batch_size].unsqueeze(1)
    bg_backward = bg[batch_size:].unsqueeze(1)
    if region == "bg":
        valid_forward = valid_forward * bg_forward
        valid_backward = valid_backward * bg_backward
    elif region != "all":
        raise ValueError(f"Unknown flow supervision region {region!r}")
    return target_forward, target_backward, valid_forward, valid_backward, bg_forward, bg_backward


def _teacher_loss(
    predicted_forward: torch.Tensor,
    predicted_backward: torch.Tensor,
    target_forward: torch.Tensor,
    target_backward: torch.Tensor,
    valid_forward: torch.Tensor,
    valid_backward: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return 0.5 * (
        charbonnier_flow_loss(_five(predicted_forward), target_forward, valid_forward, eps)
        + charbonnier_flow_loss(_five(predicted_backward), target_backward, valid_backward, eps)
    )


def compute_training_losses(
    predicted_forward: torch.Tensor,
    predicted_backward: torch.Tensor,
    all_forward: Sequence[torch.Tensor],
    all_backward: Sequence[torch.Tensor],
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    output_size = tuple(predicted_forward.shape[-2:])
    (
        target_forward,
        target_backward,
        valid_forward,
        valid_backward,
        _,
        _,
    ) = _targets_and_region_weights(batch, output_size, args.flow_loss_region)
    eps = float(args.flow_charbonnier_eps)
    final_teacher = _teacher_loss(
        predicted_forward,
        predicted_backward,
        target_forward,
        target_backward,
        valid_forward,
        valid_backward,
        eps,
    )
    if args.final_flow_only:
        iterative_teacher = final_teacher
        iterative_count = predicted_forward.new_tensor(1.0)
    else:
        if len(all_forward) != len(all_backward) or not all_forward:
            raise ValueError("RAFT iterative forward/backward outputs are inconsistent")
        weights = predicted_forward.new_tensor(
            [float(args.raft_sequence_gamma) ** (len(all_forward) - 1 - index) for index in range(len(all_forward))]
        )
        # Normalization keeps the configured teacher weight comparable to a
        # final-only run, instead of multiplying it by roughly 1/(1-gamma).
        weights = weights / weights.sum().clamp_min(1e-8)
        iterative_teacher = sum(
            weight
            * _teacher_loss(
                forward,
                backward,
                target_forward,
                target_backward,
                valid_forward,
                valid_backward,
                eps,
            )
            for weight, forward, backward in zip(weights, all_forward, all_backward)
        )
        iterative_count = predicted_forward.new_tensor(float(len(all_forward)))

    fb = forward_backward_consistency_loss(
        _five(predicted_forward), _five(predicted_backward), valid_forward, valid_backward, eps
    )
    smooth = 0.5 * (
        edge_aware_smoothness_loss(
            _five(predicted_forward), _five(batch["degraded0"]), valid_forward, args.smoothness_edge_weight
        )
        + edge_aware_smoothness_loss(
            _five(predicted_backward), _five(batch["degraded1"]), valid_backward, args.smoothness_edge_weight
        )
    )
    total = (
        float(args.teacher_weight) * iterative_teacher
        + float(args.fb_weight) * fb
        + float(args.smoothness_weight) * smooth
    )
    final_epe = 0.5 * (
        endpoint_error(_five(predicted_forward), target_forward, valid_forward)
        + endpoint_error(_five(predicted_backward), target_backward, valid_backward)
    )
    return {
        "loss_total": total,
        "loss_teacher_iterative": iterative_teacher,
        "loss_teacher_final": final_teacher,
        "loss_fb": fb,
        "loss_smooth": smooth,
        "flow_epe": final_epe,
        "flow_valid_ratio": 0.5 * (valid_forward.mean() + valid_backward.mean()),
        "raft_prediction_count": iterative_count,
    }


class FlowMetricAccumulator:
    """Distributed, valid-mask-aware EPE and V6-range diagnostics."""

    MOTION_EDGES = (0.0, 1.0, 2.0, 4.0, 8.0, math.inf)

    def __init__(self, device: torch.device, histogram_max: float, histogram_bins: int):
        if histogram_max <= 0.0 or histogram_bins < 8:
            raise ValueError("Invalid histogram configuration")
        self.device = device
        self.histogram_max = float(histogram_max)
        self.histogram_bins = int(histogram_bins)
        self.scalar = torch.zeros(14, device=device, dtype=torch.float64)
        self.residual_hist = torch.zeros(histogram_bins, device=device, dtype=torch.float64)
        self.magnitude_hist = torch.zeros(histogram_bins, device=device, dtype=torch.float64)
        self.motion_epe_sum = torch.zeros(len(self.MOTION_EDGES) - 1, device=device, dtype=torch.float64)
        self.motion_count = torch.zeros(len(self.MOTION_EDGES) - 1, device=device, dtype=torch.float64)
        self.maximum = torch.zeros(2, device=device, dtype=torch.float64)

    def _histogram(self, values: torch.Tensor, mask: torch.Tensor, destination: torch.Tensor) -> None:
        values = values[mask]
        if values.numel() == 0:
            return
        scale = self.histogram_bins / self.histogram_max
        indices = (values.detach().double().clamp(0.0, self.histogram_max - 1e-12) * scale).long()
        destination.scatter_add_(0, indices, torch.ones_like(indices, dtype=destination.dtype))

    def _update_direction(self, prediction, target, valid, bg) -> None:
        error = (prediction - target).float().square().sum(dim=2, keepdim=True).sqrt()
        magnitude = target.float().square().sum(dim=2, keepdim=True).sqrt()
        valid_mask = valid > 0.5
        bg_mask = valid_mask & (bg > 0.5)
        valid_weight = valid.float()
        bg_weight = valid.float() * bg.float()
        self.scalar[0] += (error.double() * valid_weight.double()).sum()
        self.scalar[1] += valid_weight.double().sum()
        self.scalar[2] += (error.double() * bg_weight.double()).sum()
        self.scalar[3] += bg_weight.double().sum()
        self.scalar[4] += (magnitude.double() * valid_weight.double()).sum()
        self.scalar[5] += (magnitude.double() * bg_weight.double()).sum()
        self.scalar[6] += (error.gt(1.0) & valid_mask).double().sum()
        self.scalar[7] += (error.gt(2.0) & valid_mask).double().sum()
        self.scalar[8] += valid_mask.double().sum()
        self.maximum[0] = torch.maximum(self.maximum[0], error[valid_mask].double().max() if valid_mask.any() else self.maximum[0])
        self._histogram(error, valid_mask, self.residual_hist)
        self._histogram(magnitude, valid_mask, self.magnitude_hist)
        for index, (lower, upper) in enumerate(zip(self.MOTION_EDGES[:-1], self.MOTION_EDGES[1:])):
            selected = valid_mask & (magnitude >= lower) & (magnitude < upper)
            self.motion_epe_sum[index] += error[selected].double().sum()
            self.motion_count[index] += selected.double().sum()

    def update(
        self,
        predicted_forward: torch.Tensor,
        predicted_backward: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        deform_feature_size: int,
        deform_residual_range: float,
    ) -> None:
        size = tuple(predicted_forward.shape[-2:])
        target_f, target_b, valid_f, valid_b, bg_f, bg_b = _targets_and_region_weights(batch, size, "all")
        pred_f, pred_b = _five(predicted_forward), _five(predicted_backward)
        self._update_direction(pred_f, target_f, valid_f, bg_f)
        self._update_direction(pred_b, target_b, valid_b, bg_b)
        pred_f64 = resize_flow_sequence(pred_f, (deform_feature_size, deform_feature_size))
        pred_b64 = resize_flow_sequence(pred_b, (deform_feature_size, deform_feature_size))
        target_f64 = resize_flow_sequence(target_f, (deform_feature_size, deform_feature_size))
        target_b64 = resize_flow_sequence(target_b, (deform_feature_size, deform_feature_size))
        valid_f64 = F.interpolate(valid_f.flatten(0, 1), (deform_feature_size, deform_feature_size), mode="nearest").reshape_as(pred_f64[:, :, :1])
        valid_b64 = F.interpolate(valid_b.flatten(0, 1), (deform_feature_size, deform_feature_size), mode="nearest").reshape_as(pred_b64[:, :, :1])
        error_f64 = (pred_f64 - target_f64).float().square().sum(dim=2, keepdim=True).sqrt()
        error_b64 = (pred_b64 - target_b64).float().square().sum(dim=2, keepdim=True).sqrt()
        valid64 = torch.cat((valid_f64, valid_b64), dim=0) > 0.5
        error64 = torch.cat((error_f64, error_b64), dim=0)
        self.scalar[9] += (error64.gt(float(deform_residual_range)) & valid64).double().sum()
        self.scalar[10] += valid64.double().sum()
        self.maximum[1] = torch.maximum(self.maximum[1], error64[valid64].double().max() if valid64.any() else self.maximum[1])

    @staticmethod
    def _quantile(histogram: torch.Tensor, maximum: float, quantile: float) -> float:
        total = histogram.sum()
        if total <= 0:
            return 0.0
        index = int(torch.searchsorted(histogram.cumsum(0), total * quantile, right=False).item())
        return float((index + 0.5) * maximum / histogram.numel())

    def finalize(self) -> Dict[str, float]:
        if dist.is_available() and dist.is_initialized():
            for value in (self.scalar, self.residual_hist, self.magnitude_hist, self.motion_epe_sum, self.motion_count):
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.maximum, op=dist.ReduceOp.MAX)
        count = self.scalar[1].clamp_min(1e-12)
        bg_count = self.scalar[3].clamp_min(1e-12)
        feature_count = self.scalar[10].clamp_min(1e-12)
        epe = self.scalar[0] / count
        bg_epe = self.scalar[2] / bg_count
        zero_epe = self.scalar[4] / count
        bg_zero_epe = self.scalar[5] / bg_count
        result = {
            "flow_epe": float(epe),
            "bg_flow_epe": float(bg_epe),
            "zero_flow_epe": float(zero_epe),
            "bg_zero_flow_epe": float(bg_zero_epe),
            "zero_flow_gain": float(1.0 - epe / zero_epe.clamp_min(1e-12)),
            "bg_zero_flow_gain": float(1.0 - bg_epe / bg_zero_epe.clamp_min(1e-12)),
            "teacher_flow_magnitude_mean": float(self.scalar[4] / count),
            "teacher_bg_flow_magnitude_mean": float(self.scalar[5] / bg_count),
            "flow_residual_p50": self._quantile(self.residual_hist, self.histogram_max, 0.50),
            "flow_residual_p90": self._quantile(self.residual_hist, self.histogram_max, 0.90),
            "flow_residual_p95": self._quantile(self.residual_hist, self.histogram_max, 0.95),
            "flow_residual_max": float(self.maximum[0]),
            "teacher_flow_magnitude_p50": self._quantile(self.magnitude_hist, self.histogram_max, 0.50),
            "teacher_flow_magnitude_p90": self._quantile(self.magnitude_hist, self.histogram_max, 0.90),
            "teacher_flow_magnitude_p95": self._quantile(self.magnitude_hist, self.histogram_max, 0.95),
            "flow_residual_gt_1px_ratio": float(self.scalar[6] / count),
            "flow_residual_gt_2px_ratio": float(self.scalar[7] / count),
            "flow_error_gt_deform_range_ratio": float(self.scalar[9] / feature_count),
            "flow_error_within_deform_range_ratio": float(1.0 - self.scalar[9] / feature_count),
            "flow_residual_feature_max": float(self.maximum[1]),
            "valid_pixel_count": float(count),
            "bg_valid_pixel_count": float(bg_count),
        }
        for index, (lower, upper) in enumerate(zip(self.MOTION_EDGES[:-1], self.MOTION_EDGES[1:])):
            label = f"motion_{lower:g}_{'inf' if math.isinf(upper) else f'{upper:g}'}"
            bin_count = self.motion_count[index]
            result[f"{label}_count"] = float(bin_count)
            result[f"{label}_epe"] = float(self.motion_epe_sum[index] / bin_count.clamp_min(1e-12))
        return result


def validate(
    model: RAFTStudentFlowPredictor,
    loader: DataLoader,
    args: argparse.Namespace,
    accelerator: Accelerator,
) -> Dict[str, float]:
    model.eval()
    metrics = FlowMetricAccumulator(
        accelerator.device, args.metric_histogram_max, args.metric_histogram_bins
    )
    with torch.no_grad():
        for raw_batch in loader:
            batch = _pair_batch(raw_batch, accelerator.device)
            predicted_forward, predicted_backward = model.predict_bidirectional(
                batch["degraded0"],
                batch["degraded1"],
                return_all=False,
                pair_batch_size=args.raft_pair_batch_size,
            )
            metrics.update(
                predicted_forward,
                predicted_backward,
                batch,
                args.deform_feature_size,
                args.deform_residual_range,
            )
    result = metrics.finalize()
    model.train()
    return result


def _metric_score(metrics: Dict[str, float], best_metric: str) -> float:
    if best_metric == "bg_epe":
        return float(metrics["bg_flow_epe"])
    if best_metric == "epe":
        return float(metrics["flow_epe"])
    if best_metric == "large_motion_epe":
        return float(metrics["motion_8_inf_epe"])
    raise ValueError(best_metric)


def _config_contract(args: argparse.Namespace) -> Dict:
    keys = (
        "dataset_root",
        "teacher_flow_root",
        "propainter_root",
        "raft_checkpoint",
        "train_split",
        "valid_split",
        "resolution",
        "raft_iterations",
        "final_flow_only",
        "raft_sequence_gamma",
        "flow_loss_region",
        "freeze_batchnorm",
        "mixed_precision",
    )
    return {key: str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key) for key in keys}


def save_checkpoint(
    accelerator: Accelerator,
    model: RAFTStudentFlowPredictor,
    output_dir: Path,
    step: int,
    best_score: float,
    args: argparse.Namespace,
    validation: Dict[str, float],
    is_best: bool,
) -> None:
    checkpoint = output_dir / f"checkpoint-{step:07d}"
    state_dir = checkpoint / "accelerator_state"
    accelerator.save_state(str(state_dir))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        bare_model = accelerator.unwrap_model(model)
        bare_model.save_pretrained(checkpoint / "raft_student")
        metadata = {
            "experiment": "v7_clean_to_degraded_raft_flow_distillation",
            "global_step": int(step),
            "trainable_components": ["raft_student"],
            "frozen_components": ["clean_raft_teacher_cache", "BrushNet", "STC", "V5", "V6_DCN"],
            "objective": "teacher-flow distillation + FB + edge-aware smoothness",
            "best_metric": args.best_metric,
            "best_score": float(best_score),
            "validation": validation,
            "contract": _config_contract(args),
        }
        json_dump(checkpoint / "metadata.json", metadata)
        json_dump(output_dir / "latest.json", {"checkpoint": checkpoint.name, "global_step": int(step)})
        if is_best:
            json_dump(output_dir / "best.json", {"checkpoint": checkpoint.name, "global_step": int(step), **validation})
        checkpoints = sorted(
            (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
            key=checkpoint_step,
        )
        limit = int(args.checkpoints_total_limit)
        if limit > 0:
            for stale in checkpoints[: max(0, len(checkpoints) - limit)]:
                shutil.rmtree(stale)
    accelerator.wait_for_everyone()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "resolution",
        "train_batch_size",
        "raft_pair_batch_size",
        "gradient_accumulation_steps",
        "max_train_steps",
        "checkpointing_steps",
        "validation_steps",
        "raft_iterations",
        "metric_histogram_bins",
        "deform_feature_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(args.resolution) % 8:
        raise ValueError("resolution must be divisible by 8")
    for name in (
        "learning_rate",
        "teacher_weight",
        "fb_weight",
        "smoothness_weight",
        "flow_charbonnier_eps",
        "smoothness_edge_weight",
        "deform_residual_range",
        "metric_histogram_max",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if not 0.0 < float(args.raft_sequence_gamma) <= 1.0:
        raise ValueError("raft_sequence_gamma must lie in (0,1]")
    if not (args.propainter_root / "RAFT" / "raft.py").is_file():
        raise FileNotFoundError(args.propainter_root / "RAFT" / "raft.py")
    if not args.raft_checkpoint.is_file():
        raise FileNotFoundError(args.raft_checkpoint)


def preflight(args: argparse.Namespace) -> Dict:
    metadata_path = args.teacher_flow_root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cached_hash = metadata.get("raft_sha256")
    student_hash = sha256(args.raft_checkpoint)
    if cached_hash and cached_hash != student_hash and not args.allow_teacher_checkpoint_mismatch:
        raise ValueError(
            "RAFT student checkpoint differs from the frozen teacher cache. "
            "Pass --allow_teacher_checkpoint_mismatch only for an explicit ablation."
        )
    train_set = RAFTTeacherFlowPairDataset(
        args.dataset_root, args.teacher_flow_root, args.train_split, args.resolution, args.include_branches
    )
    valid_set = RAFTTeacherFlowPairDataset(
        args.dataset_root, args.teacher_flow_root, args.valid_split, args.resolution, args.include_branches
    )
    original_valid_pairs = len(valid_set)
    evenly_limit_pairs_per_sequence(valid_set, args.valid_pairs_per_sequence)
    return {
        "status": "ok",
        "model": "pretrained ProPainter RAFT student on degraded RGB",
        "teacher": "frozen cached clean ProPainter RAFT flow",
        "dataset_root": str(args.dataset_root),
        "teacher_flow_root": str(args.teacher_flow_root),
        "teacher_cache_resolution": [metadata.get("height"), metadata.get("width")],
        "teacher_checkpoint": metadata.get("raft_checkpoint"),
        "teacher_checkpoint_sha256": cached_hash,
        "student_checkpoint": str(args.raft_checkpoint),
        "student_checkpoint_sha256": student_hash,
        "train_pair_count": len(train_set),
        "valid_pair_count_before_limit": original_valid_pairs,
        "valid_pair_count": len(valid_set),
        "resolution": int(args.resolution),
        "raft_iterations": int(args.raft_iterations),
        "flow_loss_region": args.flow_loss_region,
        "iterative_supervision": not bool(args.final_flow_only),
        "deform_feature_size": int(args.deform_feature_size),
        "deform_residual_range_feature_px": float(args.deform_residual_range),
        "normalization": "degraded RGB [0,1] -> [-1,1]",
        "flow_convention": "forward=t->t+1 on t; backward=t+1->t on t+1; [dx,dy]",
    }


def main(args: argparse.Namespace) -> None:
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.teacher_flow_root = args.teacher_flow_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.propainter_root = args.propainter_root.expanduser().resolve()
    args.raft_checkpoint = args.raft_checkpoint.expanduser().resolve()
    validate_args(args)
    preflight_report = preflight(args)
    print(json.dumps(preflight_report, indent=2, sort_keys=True), flush=True)
    if args.preflight_only:
        return

    resume = resolve_checkpoint(args.resume_from_checkpoint, args.output_dir)
    evaluate_checkpoint = resolve_checkpoint(args.evaluate_only, args.output_dir)
    if resume is not None and evaluate_checkpoint is not None:
        raise ValueError("resume_from_checkpoint and evaluate_only are mutually exclusive")
    existing = [path for path in args.output_dir.glob("checkpoint-*") if path.is_dir()]
    if existing and resume is None and evaluate_checkpoint is None:
        raise ValueError(
            f"Output already contains RAFT-student results: {args.output_dir}; use --resume_from_checkpoint latest or a new output directory"
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed, device_specific=True)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_dump(args.output_dir / "run_config.json", vars(args))

    train_set = RAFTTeacherFlowPairDataset(
        args.dataset_root, args.teacher_flow_root, args.train_split, args.resolution, args.include_branches
    )
    valid_set = RAFTTeacherFlowPairDataset(
        args.dataset_root, args.teacher_flow_root, args.valid_split, args.resolution, args.include_branches
    )
    evenly_limit_pairs_per_sequence(valid_set, args.valid_pairs_per_sequence)
    loader_args = {
        "num_workers": int(args.dataloader_num_workers),
        "pin_memory": accelerator.device.type == "cuda",
        "persistent_workers": int(args.dataloader_num_workers) > 0,
    }
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, drop_last=True, **loader_args)
    valid_loader = DataLoader(valid_set, batch_size=args.train_batch_size, shuffle=False, drop_last=False, **loader_args)

    if evaluate_checkpoint is not None:
        model = RAFTStudentFlowPredictor.from_pretrained(
            evaluate_checkpoint / "raft_student",
            propainter_root=args.propainter_root,
            raft_checkpoint=args.raft_checkpoint,
            mixed_precision=args.mixed_precision != "no",
        )
        model, valid_loader = accelerator.prepare(model, valid_loader)
        metrics = validate(accelerator.unwrap_model(model), valid_loader, args, accelerator)
        if accelerator.is_main_process:
            summary_path = args.output_dir / f"evaluation_{evaluate_checkpoint.name}.json"
            json_dump(summary_path, {"checkpoint": str(evaluate_checkpoint), **metrics})
            print("EVALUATION " + json.dumps(metrics, sort_keys=True), flush=True)
        accelerator.wait_for_everyone()
        return

    model = RAFTStudentFlowPredictor(
        args.propainter_root,
        args.raft_checkpoint,
        iterations=args.raft_iterations,
        mixed_precision=args.mixed_precision != "no",
        freeze_batchnorm=args.freeze_batchnorm,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
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
        metadata_path = resume / "metadata.json"
        if metadata_path.is_file():
            contract = json.loads(metadata_path.read_text(encoding="utf-8")).get("contract", {})
            for key, saved in contract.items():
                current = _config_contract(args).get(key)
                if saved != current:
                    raise ValueError(f"Resume contract mismatch for {key}: {saved!r} != {current!r}")

    writer = SummaryWriter(str(args.output_dir / "tensorboard")) if accelerator.is_main_process else None
    initial = validate(accelerator.unwrap_model(model), valid_loader, args, accelerator)
    best_score = _metric_score(initial, args.best_metric)
    has_best_checkpoint = (args.output_dir / "best.json").is_file()
    if accelerator.is_main_process:
        json_dump(args.output_dir / "initial_validation.json", initial)
        print("INITIAL " + json.dumps(initial, sort_keys=True), flush=True)
        for key, value in initial.items():
            writer.add_scalar(f"valid/{key}", value, global_step)
        writer.flush()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    while global_step < args.max_train_steps:
        for raw_batch in train_loader:
            with accelerator.accumulate(model):
                batch = _pair_batch(raw_batch, accelerator.device)
                forward_all, backward_all = model(
                    batch["degraded0"],
                    batch["degraded1"],
                    return_all=not args.final_flow_only,
                    pair_batch_size=args.raft_pair_batch_size,
                )
                if args.final_flow_only:
                    predicted_forward, predicted_backward = forward_all, backward_all
                    all_forward, all_backward = [forward_all], [backward_all]
                else:
                    predicted_forward, predicted_backward = forward_all[-1], backward_all[-1]
                    all_forward, all_backward = forward_all, backward_all
                losses = compute_training_losses(
                    predicted_forward,
                    predicted_backward,
                    all_forward,
                    all_backward,
                    batch,
                    args,
                )
                accelerator.backward(losses["loss_total"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            if accelerator.is_main_process and (global_step == 1 or global_step % args.log_steps == 0):
                logged = {key: float(value.detach()) for key, value in losses.items()}
                logged.update({"step": global_step, "lr": float(scheduler.get_last_lr()[0])})
                print("TRAIN " + json.dumps(logged, sort_keys=True), flush=True)
                for key, value in logged.items():
                    if key not in {"step"}:
                        writer.add_scalar(f"train/{key}", value, global_step)
            should_validate = global_step % args.validation_steps == 0 or global_step == args.max_train_steps
            should_save = global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps
            metrics = None
            is_best = False
            if should_validate:
                metrics = validate(accelerator.unwrap_model(model), valid_loader, args, accelerator)
                score = _metric_score(metrics, args.best_metric)
                is_best = (not has_best_checkpoint) or score < best_score
                best_score = min(best_score, score)
                has_best_checkpoint = has_best_checkpoint or is_best
                if accelerator.is_main_process:
                    print("VALID " + json.dumps({"step": global_step, **metrics}, sort_keys=True), flush=True)
                    for key, value in metrics.items():
                        writer.add_scalar(f"valid/{key}", value, global_step)
                    writer.add_scalar("valid/checkpoint_selection_score", score, global_step)
                    writer.flush()
            if should_save:
                if metrics is None:
                    metrics = validate(accelerator.unwrap_model(model), valid_loader, args, accelerator)
                    score = _metric_score(metrics, args.best_metric)
                    is_best = (not has_best_checkpoint) or score < best_score
                    best_score = min(best_score, score)
                    has_best_checkpoint = has_best_checkpoint or is_best
                save_checkpoint(
                    accelerator,
                    model,
                    args.output_dir,
                    global_step,
                    best_score,
                    args,
                    metrics,
                    is_best,
                )
            if global_step >= args.max_train_steps:
                break
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main(parse_args())
