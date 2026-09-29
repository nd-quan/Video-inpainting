#!/usr/bin/env python
"""Visualize V4++ predicted flow against cached clean-video teacher flow.

The comparison is deliberately performed at the STC feature resolution.  For
display, both fields are resized with displacement scaling and rendered with
one shared magnitude scale; predicted and teacher flow are never normalized
independently.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from diffusers.models.stc_flow_training import (  # noqa: E402
    prepare_teacher_flow,
    resize_flow_sequence,
)
from STC_encoder_v3_rgb_flow.teacher_flow_data import (  # noqa: E402
    TeacherFlowV8ClipDataset,
)
from STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (  # noqa: E402
    backward_warp_feature,
)
from STC_encoder_v4pp_bg_feature.bg_focused_flow_aligned_stc_adapter import (  # noqa: E402
    BGFocusedFlowAlignedRGBSTCAdapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare V4++ flow alignment with cached teacher flow."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--teacher_flow_root", type=Path, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--clip_length", type=int, default=16)
    parser.add_argument("--clip_stride", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--clips_per_sequence", type=int, default=1)
    parser.add_argument("--pairs_per_clip", type=int, default=2)
    parser.add_argument(
        "--large_motion_threshold",
        type=float,
        default=0.25,
        help=(
            "Teacher-flow magnitude threshold in STC feature-grid pixels. "
            "With downsample_factor=8, 0.25 corresponds to about 2 RGB pixels."
        ),
    )
    parser.add_argument(
        "--direction_motion_threshold",
        type=float,
        default=0.125,
        help=(
            "Minimum teacher magnitude used for cosine direction metrics, in "
            "STC feature-grid pixels (0.125 is about 1 RGB pixel at x8)."
        ),
    )
    parser.add_argument(
        "--sequence_branches",
        nargs="*",
        default=None,
        help="Optional Class/Sequence branches.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_component(checkpoint: Path) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.name in {"latest.json", "best.json"}:
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        value = Path(payload["checkpoint"])
        checkpoint = value if value.is_absolute() else checkpoint.parent / value
        checkpoint = checkpoint.resolve()
    candidates = (checkpoint / "stc_flow_model", checkpoint)
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Cannot find stc_flow_model/config.json below " + str(checkpoint)
    )


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def select_clip_indices(dataset, clips_per_sequence: int) -> List[int]:
    if clips_per_sequence < 1:
        raise ValueError("--clips_per_sequence must be positive")
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, clip in enumerate(dataset.clips):
        grouped[str(clip[0])].append(index)
    selected: List[int] = []
    for sequence in sorted(grouped):
        indices = grouped[sequence]
        count = min(clips_per_sequence, len(indices))
        positions = np.linspace(0, len(indices) - 1, count).round().astype(int)
        selected.extend(indices[int(position)] for position in positions)
    return selected


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> float:
    value = value.float()
    weight = weight.float()
    return float((value * weight).sum() / weight.sum().clamp_min(1e-6))


def direction_metrics(
    predicted: torch.Tensor,
    teacher: torch.Tensor,
    weight: torch.Tensor,
    *,
    large_motion_threshold: float,
    direction_motion_threshold: float,
) -> Dict[str, float]:
    epe = (predicted.float() - teacher.float()).square().sum(0, keepdim=True).sqrt()
    pred_mag = predicted.float().square().sum(0, keepdim=True).sqrt()
    teacher_mag = teacher.float().square().sum(0, keepdim=True).sqrt()
    epe_pred = weighted_mean(epe, weight)
    epe_zero = weighted_mean(teacher_mag, weight)
    pred_magnitude = weighted_mean(pred_mag, weight)
    teacher_magnitude = weighted_mean(teacher_mag, weight)

    # Direction is undefined for a zero teacher vector. Restrict cosine to
    # pixels whose teacher motion is large enough to have a meaningful angle.
    direction_weight = weight * (teacher_mag >= direction_motion_threshold)
    cosine = (predicted.float() * teacher.float()).sum(0, keepdim=True) / (
        pred_mag * teacher_mag
    ).clamp_min(1e-8)

    # Report motion-region EPE against its own zero-flow baseline. This makes
    # it explicit whether the head helps where transport is actually needed,
    # rather than benefiting mainly from static pixels.
    large_weight = weight * (teacher_mag >= large_motion_threshold)
    valid_weight = weight.sum().clamp_min(1e-6)
    large_weight_sum = large_weight.sum()
    has_large_motion = bool(large_weight_sum.item() > 0)
    large_epe_pred = weighted_mean(epe, large_weight) if has_large_motion else float("nan")
    large_epe_zero = (
        weighted_mean(teacher_mag, large_weight) if has_large_motion else float("nan")
    )
    return {
        "epe_pred": epe_pred,
        "epe_zero": epe_zero,
        "epe_gain_over_zero": 1.0 - epe_pred / max(epe_zero, 1e-8),
        "pred_magnitude": pred_magnitude,
        "teacher_magnitude": teacher_magnitude,
        "magnitude_ratio": pred_magnitude / max(teacher_magnitude, 1e-8),
        "cosine_direction": weighted_mean(cosine, direction_weight),
        "direction_pixel_ratio": float(direction_weight.sum() / valid_weight),
        "large_motion_epe_pred": large_epe_pred,
        "large_motion_epe_zero": large_epe_zero,
        "large_motion_epe_gain_over_zero": (
            1.0 - large_epe_pred / max(large_epe_zero, 1e-8)
            if has_large_motion
            else float("nan")
        ),
        "large_motion_pixel_ratio": float(large_weight_sum / valid_weight),
        "valid_bg_ratio": float((weight > 0).float().mean()),
    }


def tensor_rgb_to_bgr(frame: torch.Tensor) -> np.ndarray:
    array = (
        ((frame.detach().float().cpu() + 1.0) * 127.5)
        .clamp(0, 255)
        .permute(1, 2, 0)
        .byte()
        .numpy()
    )
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def flow_to_bgr(flow: torch.Tensor, max_magnitude: float) -> np.ndarray:
    array = flow.detach().float().cpu().permute(1, 2, 0).numpy()
    magnitude, angle = cv2.cartToPolar(array[..., 0], array[..., 1])
    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((angle * 90.0 / np.pi) % 180.0).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(
        magnitude / max(float(max_magnitude), 1e-6) * 255.0, 0, 255
    ).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def scalar_heatmap(value: torch.Tensor, vmax: float | None = None) -> np.ndarray:
    array = value.detach().float().cpu().squeeze().numpy()
    finite = array[np.isfinite(array)]
    if vmax is None:
        vmax = float(np.percentile(finite, 99.0)) if finite.size else 1.0
    normalized = np.clip(array / max(float(vmax), 1e-6) * 255.0, 0, 255).astype(
        np.uint8
    )
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def mask_to_bgr(mask: torch.Tensor) -> np.ndarray:
    array = (
        mask.detach().float().cpu().squeeze().clamp(0, 1).mul(255).byte().numpy()
    )
    return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (7, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def montage(
    tiles: Sequence[Tuple[str, np.ndarray]],
    columns: int = 4,
    tile_size: Tuple[int, int] = (320, 320),
) -> np.ndarray:
    rendered = [
        add_label(cv2.resize(image, tile_size, interpolation=cv2.INTER_AREA), label)
        for label, image in tiles
    ]
    blank = np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
    while len(rendered) % columns:
        rendered.append(blank.copy())
    return np.concatenate(
        [
            np.concatenate(rendered[start : start + columns], axis=1)
            for start in range(0, len(rendered), columns)
        ],
        axis=0,
    )


def image_flow(flow: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    return resize_flow_sequence(flow[None, None], image_size)[0, 0]


def select_informative_pairs(
    teacher_forward: torch.Tensor,
    teacher_backward: torch.Tensor,
    weight_forward: torch.Tensor,
    weight_backward: torch.Tensor,
    count: int,
) -> List[int]:
    scores = []
    for pair in range(teacher_forward.shape[0]):
        forward_mag = teacher_forward[pair].square().sum(0, keepdim=True).sqrt()
        backward_mag = teacher_backward[pair].square().sum(0, keepdim=True).sqrt()
        score = 0.5 * (
            weighted_mean(forward_mag, weight_forward[pair])
            + weighted_mean(backward_mag, weight_backward[pair])
        )
        scores.append((score, pair))
    return [pair for _, pair in sorted(scores, reverse=True)[: max(1, count)]]


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.clip_length < 2 or args.clip_stride < 1 or args.pairs_per_clip < 1:
        raise ValueError("Invalid clip/pair arguments")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} is non-empty; pass --overwrite")
    visual_dir = output_dir / "montages"
    visual_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.pretrained_model_name_or_path.expanduser().resolve()),
        subfolder="tokenizer",
        use_fast=False,
    )
    dataset = TeacherFlowV8ClipDataset(
        dataset_root=args.dataset_root,
        split=args.split,
        teacher_flow_root=args.teacher_flow_root,
        tokenizer=tokenizer,
        clip_image_processor=CLIPImageProcessor(),
        clip_length=args.clip_length,
        stride=args.clip_stride,
        resolution=args.resolution,
        include_branches=args.sequence_branches,
    )
    selected = select_clip_indices(dataset, args.clips_per_sequence)

    component = resolve_component(args.checkpoint)
    model = BGFocusedFlowAlignedRGBSTCAdapter.from_pretrained(str(component))
    model.requires_grad_(False).eval().to(device=device, dtype=torch.float32)
    amp = device.type == "cuda" and not args.no_amp

    metric_rows: List[Dict[str, object]] = []
    visual_rows: List[Dict[str, object]] = []
    for dataset_index in selected:
        sample = dataset[dataset_index]
        rgb = sample["conditioning_pixel_values"].unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        bg = sample["masks"].unsqueeze(0).to(device=device, dtype=torch.float32)
        gt = sample["pixel_values"].to(device=device, dtype=torch.float32)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            output = model(rgb, bg, predict_flow=True, return_dict=True)
        pred_forward = output.predicted_flow_forward[0].float()
        pred_backward = output.predicted_flow_backward[0].float()
        confidence = output.alignment_confidence[0].float()
        prediction_size = tuple(pred_forward.shape[-2:])

        teacher_forward, valid_forward = prepare_teacher_flow(
            sample["teacher_flow_forward"].unsqueeze(0).to(device),
            prediction_size,
            sample["teacher_valid_forward"].unsqueeze(0).to(device),
        )
        teacher_backward, valid_backward = prepare_teacher_flow(
            sample["teacher_flow_backward"].unsqueeze(0).to(device),
            prediction_size,
            sample["teacher_valid_backward"].unsqueeze(0).to(device),
        )
        teacher_forward = teacher_forward[0]
        teacher_backward = teacher_backward[0]
        valid_forward = valid_forward[0]
        valid_backward = valid_backward[0]
        bg_feature = F.interpolate(
            bg.flatten(0, 1), size=prediction_size, mode="nearest"
        ).reshape(args.clip_length, 1, *prediction_size)
        weight_forward = valid_forward * bg_feature[:-1]
        weight_backward = valid_backward * bg_feature[1:]

        sequence = str(sample["video"])
        frame_ids = [int(value) for value in sample["frame_ids"].tolist()]
        # Metrics cover every adjacent pair. Visualization below remains
        # limited to the most informative pairs to keep output manageable.
        for pair in range(pred_forward.shape[0]):
            forward_all = direction_metrics(
                pred_forward[pair],
                teacher_forward[pair],
                weight_forward[pair],
                large_motion_threshold=args.large_motion_threshold,
                direction_motion_threshold=args.direction_motion_threshold,
            )
            backward_all = direction_metrics(
                pred_backward[pair],
                teacher_backward[pair],
                weight_backward[pair],
                large_motion_threshold=args.large_motion_threshold,
                direction_motion_threshold=args.direction_motion_threshold,
            )
            metric_row: Dict[str, object] = {
                "sequence": sequence,
                "dataset_index": dataset_index,
                "frame_t": frame_ids[pair],
                "frame_t1": frame_ids[pair + 1],
            }
            metric_row.update(
                {f"forward_{key}": value for key, value in forward_all.items()}
            )
            metric_row.update(
                {f"backward_{key}": value for key, value in backward_all.items()}
            )
            metric_rows.append(metric_row)

        pair_indices = select_informative_pairs(
            teacher_forward,
            teacher_backward,
            weight_forward,
            weight_backward,
            args.pairs_per_clip,
        )
        for pair in pair_indices:
            forward_stats = direction_metrics(
                pred_forward[pair],
                teacher_forward[pair],
                weight_forward[pair],
                large_motion_threshold=args.large_motion_threshold,
                direction_motion_threshold=args.direction_motion_threshold,
            )
            backward_stats = direction_metrics(
                pred_backward[pair],
                teacher_backward[pair],
                weight_backward[pair],
                large_motion_threshold=args.large_motion_threshold,
                direction_motion_threshold=args.direction_motion_threshold,
            )

            image_size = tuple(gt.shape[-2:])
            pf = image_flow(pred_forward[pair], image_size)
            pb = image_flow(pred_backward[pair], image_size)
            tf = image_flow(teacher_forward[pair], image_size)
            tb = image_flow(teacher_backward[pair], image_size)
            shared_flow_scale = max(
                float(field.square().sum(0).sqrt().quantile(0.99))
                for field in (pf, pb, tf, tb)
            )
            shared_flow_scale = max(shared_flow_scale, 1e-6)

            pred_warp, _ = backward_warp_feature(
                gt[pair : pair + 1], pb.unsqueeze(0)
            )
            teacher_warp, _ = backward_warp_feature(
                gt[pair : pair + 1], tb.unsqueeze(0)
            )
            current = gt[pair + 1]
            pred_warp_error = (pred_warp[0] - current).abs().mean(0, keepdim=True)
            teacher_warp_error = (
                teacher_warp[0] - current
            ).abs().mean(0, keepdim=True)
            zero_warp_error = (gt[pair] - current).abs().mean(0, keepdim=True)
            error_scale = max(
                float(value.quantile(0.99))
                for value in (pred_warp_error, teacher_warp_error, zero_warp_error)
            )

            epe_forward_map = (
                pred_forward[pair] - teacher_forward[pair]
            ).square().sum(0, keepdim=True).sqrt()
            epe_backward_map = (
                pred_backward[pair] - teacher_backward[pair]
            ).square().sum(0, keepdim=True).sqrt()
            shared_epe_scale = max(
                float(epe_forward_map.quantile(0.99)),
                float(epe_backward_map.quantile(0.99)),
                1e-6,
            )

            confidence_image = F.interpolate(
                confidence[pair : pair + 1], size=image_size, mode="bilinear", align_corners=False
            )[0]
            bg_image = F.interpolate(
                bg[:, pair + 1], size=image_size, mode="nearest"
            )[0]
            valid_backward_image = F.interpolate(
                valid_backward[pair : pair + 1], size=image_size, mode="nearest"
            )[0]

            tiles = [
                ("Degraded t", tensor_rgb_to_bgr(rgb[0, pair])),
                ("Degraded t+1", tensor_rgb_to_bgr(rgb[0, pair + 1])),
                ("BG mask t+1 (white=BG)", mask_to_bgr(bg_image)),
                ("Alignment confidence", scalar_heatmap(confidence_image, 1.0)),
                ("Teacher backward", flow_to_bgr(tb, shared_flow_scale)),
                ("Predicted backward", flow_to_bgr(pb, shared_flow_scale)),
                ("Backward EPE", scalar_heatmap(epe_backward_map, shared_epe_scale)),
                ("Teacher valid backward", mask_to_bgr(valid_backward_image)),
                ("Teacher forward", flow_to_bgr(tf, shared_flow_scale)),
                ("Predicted forward", flow_to_bgr(pf, shared_flow_scale)),
                ("Forward EPE", scalar_heatmap(epe_forward_map, shared_epe_scale)),
                (f"Shared flow max={shared_flow_scale:.2f}px", np.zeros_like(tensor_rgb_to_bgr(current))),
                ("GT t+1", tensor_rgb_to_bgr(current)),
                ("Pred-warped GT t", tensor_rgb_to_bgr(pred_warp[0])),
                ("Teacher-warped GT t", tensor_rgb_to_bgr(teacher_warp[0])),
                ("Zero-flow GT t", tensor_rgb_to_bgr(gt[pair])),
                ("Pred warp error", scalar_heatmap(pred_warp_error, error_scale)),
                ("Teacher warp error", scalar_heatmap(teacher_warp_error, error_scale)),
                ("Zero warp error", scalar_heatmap(zero_warp_error, error_scale)),
                (f"B gain={backward_stats['epe_gain_over_zero']:.3f}", np.zeros_like(tensor_rgb_to_bgr(current))),
            ]
            name = (
                f"{sanitize(sequence)}_clip{dataset_index:04d}_"
                f"f{frame_ids[pair]:06d}_{frame_ids[pair + 1]:06d}.png"
            )
            cv2.imwrite(str(visual_dir / name), montage(tiles))

            row: Dict[str, object] = {
                "sequence": sequence,
                "dataset_index": dataset_index,
                "frame_t": frame_ids[pair],
                "frame_t1": frame_ids[pair + 1],
                "shared_visual_flow_scale_px": shared_flow_scale,
                "confidence_mean": float(confidence[pair].mean()),
                "warp_l1_pred": float(pred_warp_error.mean()),
                "warp_l1_teacher": float(teacher_warp_error.mean()),
                "warp_l1_zero": float(zero_warp_error.mean()),
            }
            row.update({f"forward_{key}": value for key, value in forward_stats.items()})
            row.update({f"backward_{key}": value for key, value in backward_stats.items()})
            visual_rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    write_csv(output_dir / "per_pair_metrics.csv", metric_rows)
    write_csv(output_dir / "visualized_pair_metrics.csv", visual_rows)
    numeric_keys = [
        key
        for key, value in metric_rows[0].items()
        if isinstance(value, (int, float))
        and key not in {"dataset_index", "frame_t", "frame_t1"}
    ] if metric_rows else []

    def finite_average(key: str) -> float:
        values = np.asarray(
            [float(row[key]) for row in metric_rows], dtype=np.float64
        )
        return float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else float("nan")

    def aggregate_direction(prefix: str) -> Dict[str, float]:
        def values(key: str) -> np.ndarray:
            return np.asarray(
                [float(row[f"{prefix}_{key}"]) for row in metric_rows],
                dtype=np.float64,
            )

        valid = values("valid_bg_ratio")

        def weighted(key: str, weight: np.ndarray) -> float:
            value = values(key)
            keep = np.isfinite(value) & np.isfinite(weight) & (weight > 0)
            return (
                float(np.sum(value[keep] * weight[keep]) / np.sum(weight[keep]))
                if np.any(keep)
                else float("nan")
            )

        epe_pred = weighted("epe_pred", valid)
        epe_zero = weighted("epe_zero", valid)
        pred_magnitude = weighted("pred_magnitude", valid)
        teacher_magnitude = weighted("teacher_magnitude", valid)
        direction_weight = valid * values("direction_pixel_ratio")
        large_weight = valid * values("large_motion_pixel_ratio")
        large_epe_pred = weighted("large_motion_epe_pred", large_weight)
        large_epe_zero = weighted("large_motion_epe_zero", large_weight)
        return {
            "epe_pred": epe_pred,
            "epe_zero": epe_zero,
            "epe_gain_over_zero": 1.0 - epe_pred / max(epe_zero, 1e-8),
            "pred_magnitude": pred_magnitude,
            "teacher_magnitude": teacher_magnitude,
            "magnitude_ratio": pred_magnitude / max(teacher_magnitude, 1e-8),
            "cosine_direction": weighted("cosine_direction", direction_weight),
            "direction_valid_fraction": float(direction_weight.sum() / valid.sum()),
            "large_motion_epe_pred": large_epe_pred,
            "large_motion_epe_zero": large_epe_zero,
            "large_motion_epe_gain_over_zero": (
                1.0 - large_epe_pred / max(large_epe_zero, 1e-8)
            ),
            "large_motion_valid_fraction": float(large_weight.sum() / valid.sum()),
        }

    summary = {
        "checkpoint_component": str(component),
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "teacher_flow_root": str(args.teacher_flow_root.expanduser().resolve()),
        "split": args.split,
        "clip_length": args.clip_length,
        "clip_stride": args.clip_stride,
        "selected_clip_count": len(selected),
        "evaluated_pair_count": len(metric_rows),
        "visualized_pair_count": len(visual_rows),
        "large_motion_threshold_feature_px": args.large_motion_threshold,
        "large_motion_threshold_rgb_px_approx": (
            args.large_motion_threshold * float(model.config.downsample_factor)
        ),
        "direction_motion_threshold_feature_px": args.direction_motion_threshold,
        "direction_motion_threshold_rgb_px_approx": (
            args.direction_motion_threshold * float(model.config.downsample_factor)
        ),
        "flow_units": "STC feature-grid pixels for metrics; RGB pixels for visualization",
        "flow_color_policy": "predicted and teacher share one magnitude scale per montage",
        "averages": {
            key: finite_average(key)
            for key in numeric_keys
        },
        "weighted_bg_diagnostics": {
            direction: aggregate_direction(direction)
            for direction in ("forward", "backward")
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
