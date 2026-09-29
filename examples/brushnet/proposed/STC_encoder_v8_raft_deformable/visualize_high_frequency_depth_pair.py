#!/usr/bin/env python
"""Visualize high-frequency content and monocular depth for two input images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


DEFAULT_DEPTH_MODEL = "Intel/dpt-hybrid-midas"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_a", type=Path, required=True)
    parser.add_argument("--input_b", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--depth_model", default=DEFAULT_DEPTH_MODEL,
                        help="Local directory or Hugging Face model ID.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--tile_size", type=int, default=384)
    parser.add_argument("--highpass_sigma", type=float, default=2.0)
    parser.add_argument("--display_percentile", type=float, default=99.0)
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path.expanduser().resolve()), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def features(image: np.ndarray, sigma: float):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    smooth = cv2.GaussianBlur(gray, (0, 0), sigma)
    highpass = np.abs(gray - smooth)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    dx, dy = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(dx, dy)
    return gray, highpass, laplacian, gradient


def percentile(values: Sequence[np.ndarray], q: float) -> float:
    joined = np.concatenate([value[np.isfinite(value)].reshape(-1) for value in values])
    return max(float(np.percentile(joined, q)), 1e-8)


def heatmap(value: np.ndarray, maximum: float) -> np.ndarray:
    image = np.clip(value / max(maximum, 1e-8) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(image, cv2.COLORMAP_TURBO)


def signed_map(value: np.ndarray, maximum: float) -> np.ndarray:
    normalized = np.clip(value / max(maximum, 1e-8), -1, 1)
    # Blue means B is smaller; red means B is larger.
    image = np.ones((*normalized.shape, 3), dtype=np.float32)
    image[..., 0] -= np.maximum(normalized, 0)
    image[..., 1] -= np.abs(normalized)
    image[..., 2] -= np.maximum(-normalized, 0)
    return (image * 255).astype(np.uint8)


def predict_depth(images: Sequence[np.ndarray], model_name: str, device: torch.device,
                  local_files_only: bool) -> Tuple[np.ndarray, np.ndarray]:
    processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForDepthEstimation.from_pretrained(
        model_name, local_files_only=local_files_only
    ).to(device).eval()
    inputs = processor(images=[Image.fromarray(image) for image in images], return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        prediction = model(**inputs).predicted_depth[:, None]
    output = []
    for index, image in enumerate(images):
        resized = F.interpolate(prediction[index:index + 1], image.shape[:2], mode="bicubic",
                                align_corners=False)[0, 0]
        output.append(resized.float().cpu().numpy())
    return output[0], output[1]


def affine_align(source: np.ndarray, target: np.ndarray):
    matrix = np.stack((source.reshape(-1), np.ones(source.size)), axis=1)
    scale, shift = np.linalg.lstsq(matrix, target.reshape(-1), rcond=None)[0]
    return source * scale + shift, float(scale), float(shift)


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(result, text, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
                (255, 255, 255), 1, cv2.LINE_AA)
    return result


def montage(tiles: Sequence[Tuple[str, np.ndarray]], tile_size: int) -> np.ndarray:
    rendered = [label(cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA), title)
                for title, image in tiles]
    while len(rendered) % 4:
        rendered.append(np.zeros_like(rendered[0]))
    return np.concatenate([np.concatenate(rendered[i:i + 4], axis=1)
                           for i in range(0, len(rendered), 4)], axis=0)


def main() -> None:
    args = parse_args()
    if args.tile_size < 64 or args.highpass_sigma <= 0 or not 0 < args.display_percentile <= 100:
        raise ValueError("Invalid tile size, sigma, or percentile")
    image_a, image_b = read_rgb(args.input_a), read_rgb(args.input_b)
    if image_a.shape != image_b.shape:
        raise ValueError(f"Images must have identical H/W, got {image_a.shape} and {image_b.shape}")
    maps_a, maps_b = features(image_a, args.highpass_sigma), features(image_b, args.highpass_sigma)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu")
    depth_a, depth_b = predict_depth((image_a, image_b), args.depth_model, device,
                                     args.local_files_only)
    aligned_b, depth_scale_fit, depth_shift_fit = affine_align(depth_b, depth_a)

    names = ("High-pass", "Laplacian", "Gradient")
    hf_scales = [percentile((maps_a[i], maps_b[i]), args.display_percentile) for i in range(1, 4)]
    depth_low = float(np.percentile(np.concatenate((depth_a.ravel(), depth_b.ravel())), 1))
    depth_high = float(np.percentile(np.concatenate((depth_a.ravel(), depth_b.ravel())), 99))
    depth_range = max(depth_high - depth_low, 1e-8)
    depth_a_vis = heatmap(np.clip(depth_a - depth_low, 0, None), depth_range)
    depth_b_vis = heatmap(np.clip(depth_b - depth_low, 0, None), depth_range)
    depth_delta = aligned_b - depth_a
    delta_scale = percentile((np.abs(depth_delta),), args.display_percentile)
    tiles = [
        ("Input A", cv2.cvtColor(image_a, cv2.COLOR_RGB2BGR)),
        ("Input B", cv2.cvtColor(image_b, cv2.COLOR_RGB2BGR)),
        ("Absolute RGB difference", heatmap(np.abs(image_b.astype(np.float32) - image_a).mean(2), 80)),
        ("Grayscale", cv2.cvtColor((maps_a[0] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)),
    ]
    for offset, name, maximum in zip(range(1, 4), names, hf_scales):
        tiles.extend(((f"{name} A", heatmap(maps_a[offset], maximum)),
                      (f"{name} B", heatmap(maps_b[offset], maximum)),
                      (f"|B-A|, shared p{args.display_percentile:g}={maximum:.3g}",
                       heatmap(np.abs(maps_b[offset] - maps_a[offset]), maximum))))
    tiles.extend((
        ("Relative depth A", depth_a_vis),
        ("Relative depth B", depth_b_vis),
        ("Depth B-A after affine alignment", signed_map(depth_delta, delta_scale)),
        (f"Depth delta +/-{delta_scale:.3g}", signed_map(np.linspace(-delta_scale, delta_scale, image_a.shape[1])[None].repeat(image_a.shape[0], 0), delta_scale)),
    ))
    save_dir = args.save_dir.expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    output = montage(tiles, args.tile_size)
    visual_assets = {
        "montage.png": output,
        "highpass_a.png": heatmap(maps_a[1], hf_scales[0]),
        "highpass_b.png": heatmap(maps_b[1], hf_scales[0]),
        "laplacian_a.png": heatmap(maps_a[2], hf_scales[1]),
        "laplacian_b.png": heatmap(maps_b[2], hf_scales[1]),
        "gradient_a.png": heatmap(maps_a[3], hf_scales[2]),
        "gradient_b.png": heatmap(maps_b[3], hf_scales[2]),
        "depth_a.png": depth_a_vis,
        "depth_b.png": depth_b_vis,
        "depth_aligned_difference.png": signed_map(depth_delta, delta_scale),
    }
    for name, image in visual_assets.items():
        if not cv2.imwrite(str(save_dir / name), image):
            raise OSError(save_dir / name)
    np.save(save_dir / "depth_a.npy", depth_a)
    np.save(save_dir / "depth_b.npy", depth_b)
    np.savez_compressed(save_dir / "high_frequency_maps.npz",
                        highpass_a=maps_a[1], highpass_b=maps_b[1],
                        laplacian_a=maps_a[2], laplacian_b=maps_b[2],
                        gradient_a=maps_a[3], gradient_b=maps_b[3])
    metrics = {
        "input_a": str(args.input_a.expanduser().resolve()),
        "input_b": str(args.input_b.expanduser().resolve()),
        "depth_model": args.depth_model,
        "depth_semantics": "Model-relative depth/disparity; absolute metric distance is not implied.",
        "depth_b_to_a_affine_scale": depth_scale_fit,
        "depth_b_to_a_affine_shift": depth_shift_fit,
        "depth_aligned_mae": float(np.abs(depth_delta).mean()),
        "depth_aligned_rmse": float(np.sqrt(np.square(depth_delta).mean())),
        "highpass_mean_a": float(maps_a[1].mean()),
        "highpass_mean_b": float(maps_b[1].mean()),
        "laplacian_mean_a": float(maps_a[2].mean()),
        "laplacian_mean_b": float(maps_b[2].mean()),
        "gradient_mean_a": float(maps_a[3].mean()),
        "gradient_mean_b": float(maps_b[3].mean()),
        "display_scales": {name.lower(): scale for name, scale in zip(names, hf_scales)},
        "display_note": "Each A/B high-frequency pair shares a scale. Depth A/B share p1-p99. Depth difference is shown after global affine alignment; red means B larger and blue means B smaller.",
    }
    (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
