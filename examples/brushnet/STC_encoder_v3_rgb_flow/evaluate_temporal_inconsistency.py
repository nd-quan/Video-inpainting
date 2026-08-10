#!/usr/bin/env python
"""Diagnose local versus coarse/global temporal inconsistency.

The evaluator targets the clip folders produced by
``STC_encoder_v2_rgb/evaluate_rgb_stc_shared_noise.py``.  It uses clean-GT
backward teacher flow, rather than optical flow estimated from a generated
frame, and measures the temporal evolution of *restoration error*::

    error_t    = prediction_t - clean_gt_t
    residual_t = error_t - warp(error_(t-1), teacher_backward_t)

This removes genuine clean-video motion from the temporal score.  Raw masks
are white in the high-quality ROI and black in the degraded BG; only the BG is
evaluated.  The residual is split with a mask-normalized Gaussian low pass:

    coarse = G(mask * residual) / (G(mask) + eps)
    local  = residual - coarse

The local/coarse label is explicitly a scale-dependent diagnostic, not a
universal perceptual judgment.  Always interpret it together with absolute
RMSE and the spatial-localization metrics written by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


DEFAULT_DATASET = Path("/home/cilab/ndquan/videoInpainting/SFU_STC_flow")
DEFAULT_TEACHER = DEFAULT_DATASET / "teacher_flows_512x512"


@dataclass(frozen=True)
class FrameRef:
    video: str
    frame_id: int
    prediction: Path
    clip_dir: Path
    clip_id: str
    clip_order: int


@dataclass(frozen=True)
class EvaluationUnit:
    unit_id: str
    video: str
    frames: Tuple[FrameRef, ...]


def parse_scales(value: str) -> Tuple[float, ...]:
    try:
        scales = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("scales must be comma-separated numbers") from error
    if not scales or any(scale <= 0 for scale in scales):
        raise argparse.ArgumentTypeError("all Gaussian scales must be positive")
    return tuple(sorted(set(scales)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Motion-compensated local/global temporal inconsistency diagnosis."
    )
    parser.add_argument(
        "--eval_root",
        type=Path,
        required=True,
        help="Root containing <clip>/clip_metrics.json and <clip>/{final,raw}/*.png.",
    )
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--teacher_flow_root", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--frame_kind", choices=("final", "raw"), default="final")
    parser.add_argument(
        "--overlap_mode",
        choices=("per_clip", "first", "last"),
        default="per_clip",
        help=(
            "per_clip avoids artificial transitions between diffusion windows; "
            "first/last reproduce a de-overlapped stitched video and mark window boundaries."
        ),
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--video_filter", default=None, help="Optional case-insensitive substring.")
    parser.add_argument("--max_units", type=int, default=None)
    parser.add_argument("--scales", type=parse_scales, default=parse_scales("8,16,32"))
    parser.add_argument("--primary_scale", type=float, default=16.0)
    parser.add_argument("--mask_erode_radius", type=int, default=2)
    parser.add_argument("--min_valid_pixels", type=float, default=256.0)
    parser.add_argument("--min_blurred_support", type=float, default=0.25)
    parser.add_argument("--consistent_rmse_threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--local_share_threshold", type=float, default=0.60)
    parser.add_argument("--static_motion_threshold", type=float, default=0.50)
    parser.add_argument("--moving_motion_threshold", type=float, default=2.0)
    parser.add_argument("--visualizations_per_unit", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow_missing", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_units is not None and args.max_units < 1:
        parser.error("--max_units must be positive")
    if args.mask_erode_radius < 0 or args.min_valid_pixels < 0:
        parser.error("erosion radius and minimum valid pixels must be non-negative")
    if not 0.0 <= args.min_blurred_support <= 1.0:
        parser.error("--min_blurred_support must be in [0,1]")
    if not 0.5 < args.local_share_threshold < 1.0:
        parser.error("--local_share_threshold must be in (0.5,1)")
    if args.consistent_rmse_threshold < 0:
        parser.error("--consistent_rmse_threshold must be non-negative")
    if args.static_motion_threshold < 0 or args.moving_motion_threshold <= args.static_motion_threshold:
        parser.error("motion thresholds must satisfy 0 <= static < moving")
    if args.visualizations_per_unit < 0:
        parser.error("--visualizations_per_unit must be non-negative")
    if not any(math.isclose(args.primary_scale, scale) for scale in args.scales):
        parser.error("--primary_scale must be one of --scales")
    return args


def _safe_video(video: str) -> str:
    parts = Path(video).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Unsafe/invalid video identifier: {video!r}")
    return "__".join(parts)


def discover_units(
    eval_root: Path,
    frame_kind: str,
    overlap_mode: str,
    video_filter: Optional[str] = None,
) -> List[EvaluationUnit]:
    """Read authoritative video/frame IDs from every clip_metrics.json."""
    clips: List[Tuple[int, str, Path, List[FrameRef]]] = []
    for order, metadata_path in enumerate(sorted(eval_root.glob("*/clip_metrics.json"))):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        video = str(metadata.get("video", ""))
        frame_ids = [int(value) for value in metadata.get("frame_ids", [])]
        if not video or len(frame_ids) < 2:
            raise ValueError(f"Invalid clip metadata: {metadata_path}")
        if video_filter and video_filter.lower() not in video.lower():
            continue
        clip_dir = metadata_path.parent
        refs = [
            FrameRef(
                video=video,
                frame_id=frame_id,
                prediction=clip_dir / frame_kind / f"{frame_id:06d}.png",
                clip_dir=clip_dir,
                clip_id=clip_dir.name,
                clip_order=order,
            )
            for frame_id in frame_ids
        ]
        clips.append((frame_ids[0], video, clip_dir, refs))
    if not clips:
        raise FileNotFoundError(
            f"No matching */clip_metrics.json records found below {eval_root}"
        )

    if overlap_mode == "per_clip":
        return [
            EvaluationUnit(clip_dir.name, video, tuple(refs))
            for _, video, clip_dir, refs in clips
        ]

    candidates: Dict[str, Dict[int, List[FrameRef]]] = defaultdict(lambda: defaultdict(list))
    for _, video, _, refs in clips:
        for ref in refs:
            candidates[video][ref.frame_id].append(ref)
    units = []
    for video in sorted(candidates):
        selected = []
        for frame_id in sorted(candidates[video]):
            refs = sorted(candidates[video][frame_id], key=lambda item: item.clip_order)
            selected.append(refs[0] if overlap_mode == "first" else refs[-1])
        units.append(EvaluationUnit(_safe_video(video), video, tuple(selected)))
    return units


def _find_numbered_image(directory: Path, frame_id: int) -> Optional[Path]:
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = directory / f"{frame_id:06d}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def reference_paths(ref: FrameRef, dataset_root: Path, split: str) -> Tuple[Optional[Path], Optional[Path]]:
    # Prefer the exact transformed references saved by the diffusion evaluator.
    # Falling back to canonical source files can introduce resize-kernel differences
    # that are large enough to contaminate a temporal residual near fine edges.
    gt = _find_numbered_image(ref.clip_dir / "gt", ref.frame_id)
    roi_mask = _find_numbered_image(ref.clip_dir / "mask_roi", ref.frame_id)
    canonical = dataset_root / split
    if gt is None:
        gt = _find_numbered_image(canonical / "GT" / ref.video, ref.frame_id)
    if roi_mask is None:
        roi_mask = _find_numbered_image(canonical / "mask" / ref.video, ref.frame_id)
    return gt, roi_mask


def teacher_pair_path(root: Path, split: str, video: str, previous: int, current: int) -> Path:
    split_root = root / split if (root / split).is_dir() else root
    return split_root / video / f"{previous:06d}_{current:06d}.npz"


def preflight(
    units: Sequence[EvaluationUnit], dataset_root: Path, teacher_root: Path, split: str
) -> Dict[str, object]:
    missing_predictions: List[str] = []
    missing_references: List[str] = []
    missing_flows: List[str] = []
    nonconsecutive = 0
    pair_count = 0
    boundary_pairs = 0
    physical_pairs: Counter = Counter()
    for unit in units:
        for ref in unit.frames:
            if not ref.prediction.is_file():
                missing_predictions.append(str(ref.prediction))
            gt, mask = reference_paths(ref, dataset_root, split)
            if gt is None or mask is None:
                missing_references.append(f"{unit.video}:{ref.frame_id}")
        for previous, current in zip(unit.frames[:-1], unit.frames[1:]):
            if current.frame_id != previous.frame_id + 1:
                nonconsecutive += 1
                continue
            pair_count += 1
            physical_pairs[(unit.video, previous.frame_id, current.frame_id)] += 1
            boundary_pairs += int(previous.clip_id != current.clip_id)
            flow = teacher_pair_path(
                teacher_root, split, unit.video, previous.frame_id, current.frame_id
            )
            if not flow.is_file():
                missing_flows.append(str(flow))
    return {
        "units": len(units),
        "frames_with_context": sum(len(unit.frames) for unit in units),
        "candidate_consecutive_pairs": pair_count,
        "unique_physical_pairs": len(physical_pairs),
        "duplicate_pair_contexts": pair_count - len(physical_pairs),
        "window_boundary_pairs": boundary_pairs,
        "nonconsecutive_transitions_skipped": nonconsecutive,
        "missing_prediction_count": len(missing_predictions),
        "missing_reference_count": len(missing_references),
        "missing_teacher_flow_count": len(missing_flows),
        "missing_examples": {
            "prediction": missing_predictions[:5],
            "reference": missing_references[:5],
            "teacher_flow": missing_flows[:5],
        },
    }


def load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).to(device=device)


def load_bg_mask(path: Path, size: Tuple[int, int], device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0
    roi = torch.from_numpy(array)[None, None].to(device=device)
    if roi.shape[-2:] != size:
        roi = F.interpolate(roi, size=size, mode="nearest")
    # Raw/saved mask: black=degraded BG, white=high-quality ROI.
    return (roi[0] < 0.5).float()


def resize_chw(tensor: torch.Tensor, size: Tuple[int, int], mode: str) -> torch.Tensor:
    if tensor.shape[-2:] == size:
        return tensor
    kwargs = {"align_corners": True} if mode in ("bilinear", "bicubic") else {}
    return F.interpolate(tensor[None], size=size, mode=mode, **kwargs)[0]


def load_teacher_backward(
    path: Path, size: Tuple[int, int], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    with np.load(path) as cache:
        if "teacher_b" not in cache or "valid_b" not in cache:
            raise KeyError(f"{path} must contain teacher_b and valid_b")
        flow = torch.from_numpy(np.asarray(cache["teacher_b"], dtype=np.float32).copy())
        valid = torch.from_numpy(np.asarray(cache["valid_b"], dtype=np.float32).copy())
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError(f"teacher_b must be [2,H,W], got {tuple(flow.shape)} in {path}")
    if valid.ndim == 2:
        valid = valid.unsqueeze(0)
    if valid.ndim != 3 or valid.shape[0] != 1:
        raise ValueError(f"valid_b must be [1,H,W], got {tuple(valid.shape)} in {path}")
    flow = flow.to(device=device)
    valid = valid.to(device=device)
    source_h, source_w = flow.shape[-2:]
    finite = torch.isfinite(flow).all(dim=0, keepdim=True).float()
    flow = torch.nan_to_num(flow)
    valid = valid.clamp(0, 1) * finite
    if (source_h, source_w) != size:
        flow = resize_chw(flow, size, "bilinear").clone()
        flow[0].mul_(size[1] / source_w)
        flow[1].mul_(size[0] / source_h)
        valid = resize_chw(valid, size, "nearest")
    valid = valid * flow_in_bounds(flow)
    return flow, valid


def flow_in_bounds(flow: torch.Tensor) -> torch.Tensor:
    _, height, width = flow.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    sample_x = xx + flow[0]
    sample_y = yy + flow[1]
    return (
        torch.isfinite(flow).all(dim=0, keepdim=True)
        & (sample_x[None] >= 0)
        & (sample_x[None] <= max(width - 1, 0))
        & (sample_y[None] >= 0)
        & (sample_y[None] <= max(height - 1, 0))
    ).float()


def backward_warp(source: torch.Tensor, backward_flow: torch.Tensor) -> torch.Tensor:
    """Warp CHW source into current coordinates with current->source flow."""
    if source.ndim != 3 or backward_flow.ndim != 3 or backward_flow.shape[0] != 2:
        raise ValueError("source must be CHW and backward_flow must be [2,H,W]")
    if source.shape[-2:] != backward_flow.shape[-2:]:
        raise ValueError("source and flow must have the same spatial resolution")
    _, height, width = source.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=source.device, dtype=source.dtype),
        torch.arange(width, device=source.device, dtype=source.dtype),
        indexing="ij",
    )
    sample_x = xx + backward_flow[0].to(source.dtype)
    sample_y = yy + backward_flow[1].to(source.dtype)
    grid = torch.stack(
        (
            2.0 * sample_x / max(width - 1, 1) - 1.0,
            2.0 * sample_y / max(height - 1, 1) - 1.0,
        ),
        dim=-1,
    )
    return F.grid_sample(
        source[None], grid[None], mode="bilinear", padding_mode="zeros", align_corners=True
    )[0]


def erode_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius == 0:
        return mask
    padded = F.pad(mask[None], (radius, radius, radius, radius), value=0.0)
    return -F.max_pool2d(-padded, kernel_size=2 * radius + 1, stride=1)[0]


_GAUSSIAN_KERNELS: Dict[Tuple[float, str, torch.dtype], torch.Tensor] = {}


def gaussian_blur(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur for CHW tensors."""
    key = (float(sigma), str(tensor.device), tensor.dtype)
    kernel = _GAUSSIAN_KERNELS.get(key)
    if kernel is None:
        radius = max(1, int(math.ceil(3.0 * sigma)))
        x = torch.arange(-radius, radius + 1, device=tensor.device, dtype=tensor.dtype)
        kernel = torch.exp(-0.5 * (x / sigma).square())
        kernel = kernel / kernel.sum()
        _GAUSSIAN_KERNELS[key] = kernel
    radius = kernel.numel() // 2
    channels = tensor.shape[0]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    padding_mode = "reflect" if min(tensor.shape[-2:]) > radius else "replicate"
    value = F.pad(tensor[None], (radius, radius, 0, 0), mode=padding_mode)
    value = F.conv2d(value, horizontal, groups=channels)
    value = F.pad(value, (0, 0, radius, radius), mode=padding_mode)
    return F.conv2d(value, vertical, groups=channels)[0]


def weighted_l1(value: torch.Tensor, weight: torch.Tensor) -> float:
    channels = value.shape[0]
    return float((value.abs() * weight).sum() / (weight.sum() * channels).clamp_min(1e-12))


def weighted_mse(value: torch.Tensor, weight: torch.Tensor) -> float:
    channels = value.shape[0]
    return float((value.square() * weight).sum() / (weight.sum() * channels).clamp_min(1e-12))


def weighted_scalar_mean(value: torch.Tensor, weight: torch.Tensor) -> float:
    return float((value * weight[0]).sum() / weight.sum().clamp_min(1e-12))


def masked_percentile(value: torch.Tensor, weight: torch.Tensor, q: float) -> float:
    selected = value[weight[0] > 0.5]
    if selected.numel() == 0:
        selected = value[weight[0] > 0]
    return float(torch.quantile(selected.float(), q)) if selected.numel() else 0.0


def frequency_key(scale: float) -> str:
    text = f"{scale:g}".replace(".", "p")
    return f"sigma_{text}"


def classify_frequency(
    raw_rmse: float,
    local_rmse: float,
    coarse_rmse: float,
    consistent_threshold: float,
    local_share_threshold: float,
) -> Tuple[str, float]:
    denominator = local_rmse + coarse_rmse
    local_share = local_rmse / denominator if denominator > 1e-12 else 0.5
    if raw_rmse <= consistent_threshold:
        return "consistent", local_share
    if local_share >= local_share_threshold:
        return "local_high_frequency_dominant", local_share
    if local_share <= 1.0 - local_share_threshold:
        return "coarse_global_low_frequency_dominant", local_share
    return "mixed_frequency", local_share


def classify_spatial_scope(
    raw_rmse: float,
    support90: float,
    coherence: float,
    consistent_threshold: float,
) -> str:
    if raw_rmse <= consistent_threshold:
        return "consistent"
    if support90 <= 0.25:
        return "spatially_localized"
    if support90 >= 0.50 and coherence >= 0.70:
        return "globally_coherent"
    return "distributed_or_mixed"


def localization_metrics(residual: torch.Tensor, weight: torch.Tensor) -> Dict[str, float]:
    energy = residual.square().mean(dim=0)
    valid = weight[0] > 0.5
    if not torch.any(valid):
        valid = weight[0] > 0
    values = energy[valid]
    if values.numel() == 0 or float(values.sum()) <= 1e-20:
        top10 = 0.0
        support90 = 0.0
    else:
        ordered = torch.sort(values, descending=True).values
        total = ordered.sum()
        top_count = max(1, int(math.ceil(0.10 * ordered.numel())))
        top10 = float(ordered[:top_count].sum() / total)
        cumulative = torch.cumsum(ordered, dim=0)
        needed = int(torch.searchsorted(cumulative, 0.90 * total).item()) + 1
        support90 = needed / ordered.numel()
    signed_sum = (residual * weight).sum(dim=(1, 2))
    numerator = torch.linalg.vector_norm(signed_sum)
    denominator = (
        torch.linalg.vector_norm(residual, dim=0) * weight[0]
    ).sum().clamp_min(1e-12)
    coherence = float((numerator / denominator).clamp(0, 1))
    return {
        "energy_top10_fraction": top10,
        "energy_support90_fraction": support90,
        "directional_coherence": coherence,
    }


@torch.no_grad()
def evaluate_pair(
    prediction_previous: torch.Tensor,
    prediction_current: torch.Tensor,
    gt_previous: torch.Tensor,
    gt_current: torch.Tensor,
    bg_previous: torch.Tensor,
    bg_current: torch.Tensor,
    backward_flow: torch.Tensor,
    flow_valid: torch.Tensor,
    *,
    scales: Sequence[float] = (8.0, 16.0, 32.0),
    primary_scale: float = 16.0,
    erode_radius: int = 2,
    min_valid_pixels: float = 256.0,
    min_blurred_support: float = 0.25,
    consistent_rmse_threshold: float = 1.0 / 255.0,
    local_share_threshold: float = 0.60,
    static_motion_threshold: float = 0.50,
    moving_motion_threshold: float = 2.0,
    return_maps: bool = False,
) -> Tuple[Dict[str, object], Optional[Dict[str, torch.Tensor]]]:
    """Evaluate one adjacent pair; all image tensors are CHW in [0,1]."""
    size = prediction_current.shape[-2:]
    if prediction_current.ndim != 3 or prediction_current.shape[0] != 3:
        raise ValueError("prediction tensors must be [3,H,W]")
    prediction_previous = resize_chw(prediction_previous, size, "bilinear")
    gt_previous = resize_chw(gt_previous, size, "bilinear")
    gt_current = resize_chw(gt_current, size, "bilinear")
    bg_previous = resize_chw(bg_previous, size, "nearest")
    bg_current = resize_chw(bg_current, size, "nearest")
    if backward_flow.shape[-2:] != size:
        source_h, source_w = backward_flow.shape[-2:]
        backward_flow = resize_chw(backward_flow, size, "bilinear").clone()
        backward_flow[0].mul_(size[1] / source_w)
        backward_flow[1].mul_(size[0] / source_h)
        flow_valid = resize_chw(flow_valid, size, "nearest")
    flow_valid = flow_valid.clamp(0, 1) * flow_in_bounds(backward_flow)

    previous_bg = erode_mask(bg_previous.clamp(0, 1), erode_radius)
    current_bg = erode_mask(bg_current.clamp(0, 1), erode_radius)
    warped_previous_bg = backward_warp(previous_bg, backward_flow).clamp(0, 1)
    weight = flow_valid * current_bg * warped_previous_bg
    valid_pixels = float(weight.sum())
    base: Dict[str, object] = {
        "status": "ok" if valid_pixels >= min_valid_pixels else "insufficient_valid_area",
        "raw_weight": valid_pixels,
        "valid_coverage": valid_pixels / (size[0] * size[1]),
        "bg_current_fraction": float(current_bg.mean()),
        "flow_valid_fraction": float(flow_valid.mean()),
    }
    if valid_pixels < min_valid_pixels:
        return base, None

    error_previous = prediction_previous - gt_previous
    error_current = prediction_current - gt_current
    warped_error_previous = backward_warp(error_previous, backward_flow)
    residual = error_current - warped_error_previous
    raw_mse = weighted_mse(residual, weight)
    raw_rmse = math.sqrt(max(raw_mse, 0.0))
    base.update(
        {
            "raw_temporal_l1": weighted_l1(residual, weight),
            "raw_temporal_mse": raw_mse,
            "raw_temporal_rmse": raw_rmse,
        }
    )
    spatial_mean = (residual * weight).sum(dim=(1, 2)) / weight.sum().clamp_min(1e-12)
    base["global_dc_l1"] = float(spatial_mean.abs().mean())
    base["global_dc_rmse"] = float(spatial_mean.square().mean().sqrt())
    base.update(localization_metrics(residual, weight))

    flow_magnitude = backward_flow.square().sum(dim=0).sqrt()
    motion_mean = weighted_scalar_mean(flow_magnitude, weight)
    base["motion_mean_px"] = motion_mean
    base["motion_p90_px"] = masked_percentile(flow_magnitude, weight, 0.90)
    if motion_mean < static_motion_threshold:
        base["motion_band"] = "static"
    elif motion_mean < moving_motion_threshold:
        base["motion_band"] = "moderate"
    else:
        base["motion_band"] = "moving"

    primary_maps = None
    primary_prefix = frequency_key(primary_scale)
    for scale in scales:
        prefix = frequency_key(scale)
        blurred_weight = gaussian_blur(weight, float(scale))
        coarse = gaussian_blur(weight * residual, float(scale)) / blurred_weight.clamp_min(1e-6)
        scale_weight = weight * (blurred_weight >= min_blurred_support).float()
        scale_pixels = float(scale_weight.sum())
        base[f"{prefix}_weight"] = scale_pixels
        if scale_pixels < min_valid_pixels:
            for suffix in ("coarse_l1", "coarse_mse", "coarse_rmse", "local_l1", "local_mse", "local_rmse"):
                base[f"{prefix}_{suffix}"] = None
            continue
        local = residual - coarse
        coarse_mse = weighted_mse(coarse, scale_weight)
        local_mse = weighted_mse(local, scale_weight)
        base.update(
            {
                f"{prefix}_coarse_l1": weighted_l1(coarse, scale_weight),
                f"{prefix}_coarse_mse": coarse_mse,
                f"{prefix}_coarse_rmse": math.sqrt(max(coarse_mse, 0.0)),
                f"{prefix}_local_l1": weighted_l1(local, scale_weight),
                f"{prefix}_local_mse": local_mse,
                f"{prefix}_local_rmse": math.sqrt(max(local_mse, 0.0)),
            }
        )
        if prefix == primary_prefix:
            primary_maps = (coarse, local, scale_weight)

    local_rmse = base.get(f"{primary_prefix}_local_rmse")
    coarse_rmse = base.get(f"{primary_prefix}_coarse_rmse")
    if local_rmse is None or coarse_rmse is None:
        base["status"] = "insufficient_primary_scale_support"
        return base, None
    diagnosis, local_share = classify_frequency(
        raw_rmse,
        float(local_rmse),
        float(coarse_rmse),
        consistent_rmse_threshold,
        local_share_threshold,
    )
    base["local_rmse_share"] = local_share
    # Backward-compatible descriptive alias. This is an RMSE ratio, not an
    # additive residual-energy fraction.
    base["local_frequency_share"] = local_share
    base["frequency_diagnosis"] = diagnosis
    support90 = float(base["energy_support90_fraction"])
    coherence = float(base["directional_coherence"])
    base["spatial_scope_diagnosis"] = classify_spatial_scope(
        raw_rmse, support90, coherence, consistent_rmse_threshold
    )

    if return_maps and primary_maps is not None:
        coarse, local, scale_weight = primary_maps
        maps = {
            "warped_prediction_previous": backward_warp(prediction_previous, backward_flow),
            "warped_gt_previous": backward_warp(gt_previous, backward_flow),
            "effective_mask": weight,
            "analysis_weight": scale_weight,
            "residual": residual,
            "coarse": coarse,
            "local": local,
        }
        return base, maps
    return base, None


def _uint8_rgb(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)


def _heatmap(value: torch.Tensor, mask: torch.Tensor, scale: float) -> np.ndarray:
    x = (value.detach().float().cpu() / max(scale, 1e-8)).clamp(0, 1).numpy()
    # Compact jet-like map; invalid pixels remain black.
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0, 1)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0, 1)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0, 1)
    rgb = np.stack((red, green, blue), axis=-1)
    rgb *= mask.detach().float().cpu()[0].clamp(0, 1).numpy()[..., None]
    return (rgb * 255).round().astype(np.uint8)


def save_montage(
    path: Path,
    prediction_current: torch.Tensor,
    gt_current: torch.Tensor,
    maps: Mapping[str, torch.Tensor],
    row: Mapping[str, object],
) -> None:
    weight = maps["analysis_weight"]
    effective_mask = maps["effective_mask"]
    magnitude = maps["residual"].square().mean(dim=0).sqrt()
    valid_values = magnitude[weight[0] > 0]
    color_scale = float(torch.quantile(valid_values, 0.99)) if valid_values.numel() else 1.0
    panels = [
        ("prediction current", _uint8_rgb(prediction_current)),
        ("warped prediction previous", _uint8_rgb(maps["warped_prediction_previous"])),
        ("clean GT current", _uint8_rgb(gt_current)),
        ("warped clean GT previous", _uint8_rgb(maps["warped_gt_previous"])),
        ("effective BG validity Q", np.repeat((effective_mask[0].detach().cpu().numpy()[..., None] * 255).astype(np.uint8), 3, axis=2)),
        ("|motion-corrected residual|", _heatmap(magnitude, weight, color_scale)),
        ("|coarse / global-lowfreq|", _heatmap(maps["coarse"].square().mean(dim=0).sqrt(), weight, color_scale)),
        ("|local / highfreq|", _heatmap(maps["local"].square().mean(dim=0).sqrt(), weight, color_scale)),
    ]
    height, width = panels[0][1].shape[:2]
    label_height = 30
    footer_height = 24
    canvas = Image.new(
        "RGB", (4 * width, 2 * (height + label_height) + footer_height), "black"
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(panels):
        column, row_index = index % 4, index // 4
        x, y = column * width, row_index * (height + label_height)
        canvas.paste(Image.fromarray(array, mode="RGB"), (x, y + label_height))
        draw.text((x + 5, y + 7), label, fill="white")
    draw.text(
        (5, canvas.height - footer_height + 4),
        f"diagnosis={row.get('frequency_diagnosis')}  raw_RMSE={float(row.get('raw_temporal_rmse', 0)):.5f}",
        fill="white",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _weighted_average(rows: Sequence[Mapping[str, object]], key: str, weight_key: str) -> Optional[float]:
    pairs = [
        (float(row[key]), float(row[weight_key]))
        for row in rows
        if row.get(key) is not None and float(row.get(weight_key, 0)) > 0
    ]
    if not pairs:
        return None
    denominator = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / denominator


def aggregate_rows(
    rows: Sequence[Mapping[str, object]],
    primary_scale: float,
    consistent_threshold: float,
    local_share_threshold: float,
) -> Dict[str, object]:
    valid = [row for row in rows if row.get("status") == "ok"]
    prefix = frequency_key(primary_scale)
    result: Dict[str, object] = {
        "pair_count": len(rows),
        "valid_pair_count": len(valid),
        "skipped_pair_count": len(rows) - len(valid),
        "diagnosis_counts": dict(Counter(str(row.get("frequency_diagnosis")) for row in valid)),
        "spatial_scope_counts": dict(
            Counter(str(row.get("spatial_scope_diagnosis")) for row in valid)
        ),
        "motion_band_counts": dict(Counter(str(row.get("motion_band")) for row in valid)),
    }
    if not valid:
        result["frequency_diagnosis"] = "no_valid_pairs"
        return result
    for key in (
        "raw_temporal_l1",
        "global_dc_l1",
        "global_dc_rmse",
        "motion_mean_px",
        "energy_top10_fraction",
        "energy_support90_fraction",
        "directional_coherence",
    ):
        result[key] = _weighted_average(valid, key, "raw_weight")
    # Coverage is already normalized per pair. Weighting it by valid area once
    # more would introduce an area-squared bias.
    result["valid_coverage"] = float(
        np.mean([float(row["valid_coverage"]) for row in valid])
    )
    raw_mse = _weighted_average(valid, "raw_temporal_mse", "raw_weight")
    coarse_mse = _weighted_average(valid, f"{prefix}_coarse_mse", f"{prefix}_weight")
    local_mse = _weighted_average(valid, f"{prefix}_local_mse", f"{prefix}_weight")
    raw_rmse = math.sqrt(max(raw_mse or 0.0, 0.0))
    coarse_rmse = math.sqrt(max(coarse_mse or 0.0, 0.0))
    local_rmse = math.sqrt(max(local_mse or 0.0, 0.0))
    diagnosis, share = classify_frequency(
        raw_rmse, local_rmse, coarse_rmse, consistent_threshold, local_share_threshold
    )
    result.update(
        {
            "raw_temporal_rmse": raw_rmse,
            f"{prefix}_coarse_rmse": coarse_rmse,
            f"{prefix}_local_rmse": local_rmse,
            "local_rmse_share": share,
            "local_frequency_share": share,
            "frequency_diagnosis": diagnosis,
            "pair_raw_rmse_median": float(np.median([row["raw_temporal_rmse"] for row in valid])),
            "pair_raw_rmse_p90": float(np.quantile([row["raw_temporal_rmse"] for row in valid], 0.90)),
            "window_boundary_pair_count": sum(int(row.get("window_boundary", 0)) for row in valid),
        }
    )
    result["spatial_scope_diagnosis"] = classify_spatial_scope(
        raw_rmse,
        float(result.get("energy_support90_fraction") or 0.0),
        float(result.get("directional_coherence") or 0.0),
        consistent_threshold,
    )
    return result


def evaluate_unit(
    unit: EvaluationUnit,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], List[Tuple[float, Path, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, object]]]]:
    rows: List[Dict[str, object]] = []
    # Keep only top-K payloads while streaming. A stitched sequence can contain
    # hundreds of pairs, and each visualization payload contains multiple RGB
    # tensors at full resolution.
    visual_candidates = []
    for previous, current in zip(unit.frames[:-1], unit.frames[1:]):
        identity = {
            "unit_id": unit.unit_id,
            "video": unit.video,
            "frame_previous": previous.frame_id,
            "frame_current": current.frame_id,
            "clip_previous": previous.clip_id,
            "clip_current": current.clip_id,
            "window_boundary": int(previous.clip_id != current.clip_id),
        }
        if current.frame_id != previous.frame_id + 1:
            rows.append({**identity, "status": "nonconsecutive_transition"})
            continue
        gt_previous_path, mask_previous_path = reference_paths(previous, args.dataset_root, args.split)
        gt_current_path, mask_current_path = reference_paths(current, args.dataset_root, args.split)
        flow_path = teacher_pair_path(
            args.teacher_flow_root,
            args.split,
            unit.video,
            previous.frame_id,
            current.frame_id,
        )
        required = [
            previous.prediction,
            current.prediction,
            gt_previous_path,
            gt_current_path,
            mask_previous_path,
            mask_current_path,
            flow_path,
        ]
        missing = [str(path) for path in required if path is None or not Path(path).is_file()]
        if missing:
            if not args.allow_missing:
                raise FileNotFoundError("Missing pair inputs:\n" + "\n".join(missing))
            rows.append({**identity, "status": "missing_inputs", "missing": " | ".join(missing)})
            continue
        prediction_previous = load_rgb(previous.prediction, device)
        prediction_current = load_rgb(current.prediction, device)
        size = prediction_current.shape[-2:]
        gt_previous = resize_chw(load_rgb(Path(gt_previous_path), device), size, "bilinear")
        gt_current = resize_chw(load_rgb(Path(gt_current_path), device), size, "bilinear")
        bg_previous = load_bg_mask(Path(mask_previous_path), size, device)
        bg_current = load_bg_mask(Path(mask_current_path), size, device)
        flow, valid = load_teacher_backward(flow_path, size, device)
        metrics, maps = evaluate_pair(
            prediction_previous,
            prediction_current,
            gt_previous,
            gt_current,
            bg_previous,
            bg_current,
            flow,
            valid,
            scales=args.scales,
            primary_scale=args.primary_scale,
            erode_radius=args.mask_erode_radius,
            min_valid_pixels=args.min_valid_pixels,
            min_blurred_support=args.min_blurred_support,
            consistent_rmse_threshold=args.consistent_rmse_threshold,
            local_share_threshold=args.local_share_threshold,
            static_motion_threshold=args.static_motion_threshold,
            moving_motion_threshold=args.moving_motion_threshold,
            return_maps=args.visualizations_per_unit > 0,
        )
        row = {**identity, "teacher_flow": str(flow_path), **metrics}
        rows.append(row)
        if maps is not None:
            visual_path = (
                args.output_dir
                / "visualizations"
                / unit.unit_id
                / f"{previous.frame_id:06d}_{current.frame_id:06d}.png"
            )
            visual_candidates.append(
                (
                    float(metrics.get("raw_temporal_rmse", 0.0)),
                    visual_path,
                    prediction_current.detach().cpu(),
                    gt_current.detach().cpu(),
                    {key: value.detach().cpu() for key, value in maps.items()},
                    row,
                )
            )
            visual_candidates.sort(key=lambda item: item[0], reverse=True)
            del visual_candidates[args.visualizations_per_unit :]
    return rows, visual_candidates


def main() -> None:
    args = parse_args()
    args.eval_root = args.eval_root.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.teacher_flow_root = args.teacher_flow_root.expanduser().resolve()
    if not args.eval_root.is_dir():
        raise FileNotFoundError(f"Evaluation root not found: {args.eval_root}")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")
    if not args.teacher_flow_root.is_dir():
        raise FileNotFoundError(f"Teacher-flow root not found: {args.teacher_flow_root}")
    if args.output_dir is None:
        args.output_dir = args.eval_root / f"temporal_diagnosis_{args.frame_kind}_{args.overlap_mode}"
    else:
        args.output_dir = args.output_dir.expanduser().resolve()

    units = discover_units(
        args.eval_root, args.frame_kind, args.overlap_mode, args.video_filter
    )
    if args.max_units is not None:
        units = units[: args.max_units]
    report = preflight(units, args.dataset_root, args.teacher_flow_root, args.split)
    report.update(
        {
            "eval_root": str(args.eval_root),
            "frame_kind": args.frame_kind,
            "overlap_mode": args.overlap_mode,
            "mask_semantics": "raw 0=degraded BG, 255=HQ ROI; evaluator uses M_BG=1",
            "flow_convention": "teacher_b on current coordinates, current->previous",
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    missing_total = sum(
        int(report[key])
        for key in (
            "missing_prediction_count",
            "missing_reference_count",
            "missing_teacher_flow_count",
        )
    )
    if missing_total and not args.allow_missing:
        raise FileNotFoundError(
            "Preflight found missing inputs. Re-run with --allow_missing only if skipping "
            "those pairs is intentional."
        )
    if args.preflight_only:
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    if args.output_dir.exists() and (args.output_dir / "summary.json").is_file() and not args.overwrite:
        raise FileExistsError(
            f"Completed output exists: {args.output_dir}. Use --overwrite to replace reports."
        )
    visualization_root = args.output_dir / "visualizations"
    if args.overwrite and visualization_root.is_dir():
        # --overwrite explicitly requests replacement. Restrict cleanup to this
        # one known child so prediction/reference data can never be touched.
        shutil.rmtree(visualization_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: ([float(item) for item in value] if key == "scales" else str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    all_rows: List[Dict[str, object]] = []
    for index, unit in enumerate(units, start=1):
        rows, candidates = evaluate_unit(unit, args, device)
        all_rows.extend(rows)
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, path, prediction, gt, maps, row in candidates[: args.visualizations_per_unit]:
            save_montage(path, prediction, gt, maps, row)
        if index == 1 or index % 25 == 0 or index == len(units):
            print(f"[{index}/{len(units)}] evaluated {unit.unit_id}")

    if not all_rows:
        raise RuntimeError("No frame pairs were discovered")
    write_csv(args.output_dir / "per_pair_metrics.csv", all_rows)
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in all_rows:
        grouped[str(row["video"])].append(row)
    per_sequence = []
    for video in sorted(grouped):
        aggregate = aggregate_rows(
            grouped[video],
            args.primary_scale,
            args.consistent_rmse_threshold,
            args.local_share_threshold,
        )
        per_sequence.append({"video": video, **aggregate})
    write_csv(args.output_dir / "per_sequence_metrics.csv", per_sequence)
    overall = aggregate_rows(
        all_rows,
        args.primary_scale,
        args.consistent_rmse_threshold,
        args.local_share_threshold,
    )
    summary = {
        "contract": {
            "residual": "(prediction_t-GT_t) - warp(prediction_t-1-GT_t-1, teacher_b_t)",
            "effective_mask": "valid_b * eroded_M_BG_t * warp(eroded_M_BG_t-1, teacher_b_t)",
            "frequency_split": "masked normalized Gaussian low-pass; local=residual-low",
            "primary_sigma_pixels": args.primary_scale,
            "classification": (
                "heuristic and scale-dependent; local_rmse_share is "
                "local_RMSE/(local_RMSE+coarse_RMSE), while severity is "
                "raw_temporal_rmse"
            ),
        },
        "preflight": report,
        "overall_micro_weighted": overall,
        "per_sequence_count": len(per_sequence),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["overall_micro_weighted"], indent=2, sort_keys=True))
    print(f"Reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
