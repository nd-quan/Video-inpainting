#!/usr/bin/env python
"""Evaluate saved RAFT-student flow against one cached clean-RAFT teacher pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student_flows", type=Path, required=True,
                        help="flows.npz written by visualize_raft_output_input_pair.py")
    parser.add_argument("--teacher_flow", type=Path, required=True,
                        help="Cached NPZ containing teacher_f/b and valid_f/b")
    parser.add_argument("--student_prefix", choices=("input", "output"), default="input")
    parser.add_argument("--mask_t", type=Path, default=None,
                        help="ROI mask at t; white means ROI")
    parser.add_argument("--mask_t1", type=Path, default=None,
                        help="ROI mask at t+1; white means ROI")
    parser.add_argument("--region", choices=("all", "bg", "roi"), default="bg")
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--tile_size", type=int, default=384)
    return parser.parse_args()


def flow_hwc(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 3:
        raise ValueError(f"{name} must be a 3D array, got {value.shape}")
    if value.shape[-1] == 2:
        return value
    if value.shape[0] == 2:
        return value.transpose(1, 2, 0)
    raise ValueError(f"{name} must have two flow channels, got {value.shape}")


def valid_hw(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"{name} must be HxW or 1xHxW, got {value.shape}")
    return np.isfinite(value) & (value >= 0.5)


def read_roi(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    value = cv2.imread(str(path.expanduser().resolve()), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    if value.shape != shape:
        raise ValueError(f"Mask {path} has shape {value.shape}, expected {shape}")
    return value >= 128


def flow_color(flow: np.ndarray, scale: float) -> np.ndarray:
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((angle * 90 / np.pi) % 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude / max(scale, 1e-8) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def heatmap(value: np.ndarray, scale: float, support: np.ndarray) -> np.ndarray:
    normalized = np.clip(value / max(scale, 1e-8) * 255, 0, 255).astype(np.uint8)
    image = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    image[~support] = (70, 70, 70)
    return image


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(result, text, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA)
    return result


def montage(tiles: Sequence[Tuple[str, np.ndarray]], tile_size: int) -> np.ndarray:
    images = [label(cv2.resize(image, (tile_size, tile_size), cv2.INTER_AREA), title)
              for title, image in tiles]
    while len(images) % 4:
        images.append(np.zeros_like(images[0]))
    return np.concatenate([np.concatenate(images[i:i + 4], axis=1)
                           for i in range(0, len(images), 4)], axis=0)


def direction_metrics(student: np.ndarray, teacher: np.ndarray,
                      support: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    epe = np.linalg.norm(student - teacher, axis=2)
    teacher_magnitude = np.linalg.norm(teacher, axis=2)
    selected = epe[support]
    zero = teacher_magnitude[support]
    if not selected.size:
        raise ValueError("Evaluation support is empty")
    mean_epe, zero_epe = float(selected.mean()), float(zero.mean())
    metrics = {
        "epe_mean_px": mean_epe,
        "epe_median_px": float(np.median(selected)),
        "epe_p95_px": float(np.percentile(selected, 95)),
        "epe_p99_px": float(np.percentile(selected, 99)),
        "outlier_over_1px_ratio": float((selected > 1).mean()),
        "outlier_over_3px_ratio": float((selected > 3).mean()),
        "outlier_over_5px_ratio": float((selected > 5).mean()),
        "zero_flow_epe_mean_px": zero_epe,
        "gain_vs_zero_flow": float(1 - mean_epe / max(zero_epe, 1e-8)),
        "teacher_flow_magnitude_mean_px": zero_epe,
        "student_flow_magnitude_mean_px": float(
            np.linalg.norm(student, axis=2)[support].mean()
        ),
        "support_pixels": int(support.sum()),
        "support_ratio": float(support.mean()),
    }
    return epe, metrics


def main() -> None:
    args = parse_args()
    student_path = args.student_flows.expanduser().resolve()
    teacher_path = args.teacher_flow.expanduser().resolve()
    with np.load(student_path, allow_pickle=False) as data:
        student_f = flow_hwc(data[f"{args.student_prefix}_forward"], "student forward")
        student_b = flow_hwc(data[f"{args.student_prefix}_backward"], "student backward")
    with np.load(teacher_path, allow_pickle=False) as data:
        teacher_f = flow_hwc(data["teacher_f"], "teacher_f")
        teacher_b = flow_hwc(data["teacher_b"], "teacher_b")
        valid_f = valid_hw(data["valid_f"], "valid_f")
        valid_b = valid_hw(data["valid_b"], "valid_b")
    shape = teacher_f.shape[:2]
    if any(value.shape[:2] != shape for value in (student_f, student_b, teacher_b)):
        raise ValueError("Student and teacher flow must have the same spatial resolution")
    if args.region == "all":
        region_t = region_t1 = np.ones(shape, dtype=bool)
    else:
        if args.mask_t is None or args.mask_t1 is None:
            raise ValueError("--mask_t and --mask_t1 are required for BG/ROI EPE")
        roi_t, roi_t1 = read_roi(args.mask_t, shape), read_roi(args.mask_t1, shape)
        region_t, region_t1 = (roi_t, roi_t1) if args.region == "roi" else (~roi_t, ~roi_t1)
    support_f, support_b = valid_f & region_t, valid_b & region_t1
    epe_f, forward = direction_metrics(student_f, teacher_f, support_f)
    epe_b, backward = direction_metrics(student_b, teacher_b, support_b)
    flow_scale = max(float(np.percentile(np.concatenate((
        np.linalg.norm(student_f, axis=2).ravel(),
        np.linalg.norm(student_b, axis=2).ravel(),
        np.linalg.norm(teacher_f, axis=2).ravel(),
        np.linalg.norm(teacher_b, axis=2).ravel(),
    )), 99)), 1e-8)
    epe_scale = max(float(np.percentile(np.concatenate((
        epe_f[support_f], epe_b[support_b]
    )), 99)), 1e-8)
    support_f_image = cv2.cvtColor(support_f.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    support_b_image = cv2.cvtColor(support_b.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    tiles = [
        (f"Teacher forward, p99={flow_scale:.2f}px", flow_color(teacher_f, flow_scale)),
        ("Student forward", flow_color(student_f, flow_scale)),
        (f"Forward EPE, mean={forward['epe_mean_px']:.3f}px", heatmap(epe_f, epe_scale, support_f)),
        (f"Forward {args.region} support", support_f_image),
        ("Teacher backward", flow_color(teacher_b, flow_scale)),
        ("Student backward", flow_color(student_b, flow_scale)),
        (f"Backward EPE, mean={backward['epe_mean_px']:.3f}px", heatmap(epe_b, epe_scale, support_b)),
        (f"Backward {args.region} support", support_b_image),
    ]
    save_dir = args.save_dir.expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    assets = {
        "teacher_student_epe_montage.png": montage(tiles, args.tile_size),
        "forward_epe.png": heatmap(epe_f, epe_scale, support_f),
        "backward_epe.png": heatmap(epe_b, epe_scale, support_b),
        "forward_support.png": support_f_image,
        "backward_support.png": support_b_image,
    }
    for name, image in assets.items():
        if not cv2.imwrite(str(save_dir / name), image):
            raise OSError(save_dir / name)
    np.savez_compressed(save_dir / "teacher_student_epe_maps.npz",
                        forward_epe=epe_f, backward_epe=epe_b,
                        forward_support=support_f, backward_support=support_b)
    result = {
        "student_flows": str(student_path),
        "student_prefix": args.student_prefix,
        "teacher_flow": str(teacher_path),
        "region": args.region,
        "mask_semantics": "White=ROI, black=BG",
        "forward": forward,
        "backward": backward,
        "bidirectional_epe_mean_px": float(
            0.5 * (forward["epe_mean_px"] + backward["epe_mean_px"])
        ),
        "bidirectional_gain_vs_zero_flow": float(
            0.5 * (forward["gain_vs_zero_flow"] + backward["gain_vs_zero_flow"])
        ),
        "display_epe_p99_px": epe_scale,
    }
    (save_dir / "teacher_epe_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
