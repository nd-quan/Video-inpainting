#!/usr/bin/env python
"""Compare a fine-tuned V7 flow student on consecutive input/output pairs.

The saved student can be either the original ProPainter RAFT or SEA-RAFT.
It is evaluated separately on ``input_t -> input_t1`` and ``output_t ->
output_t1``.  The visualization compares temporal flow and frame similarity
before and after motion compensation for the two domains.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
V7_DIR = BRUSHNET_DIR / "STC_encoder_v7_raft_flow_distillation"
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))
if str(V7_DIR) not in sys.path:
    sys.path.insert(0, str(V7_DIR))

from STC_encoder_v8_raft_deformable.raft_flow_provider import (  # noqa: E402
    FrozenV7RAFTFlowProvider,
    resolve_raft_student_component,
)
from sea_raft_student import SEAStudentFlowPredictor  # noqa: E402


DEFAULT_ROOT = Path("/home/cilab/ndquan/videoInpainting/code/BrushNet")
DEFAULT_STUDENT = (
    DEFAULT_ROOT
    / "experiments/train_v7_raft_student_flow/checkpoint-0004750/raft_student"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_t", type=Path, required=True)
    parser.add_argument("--input_t1", type=Path, required=True)
    parser.add_argument("--output_t", type=Path, required=True)
    parser.add_argument("--output_t1", type=Path, required=True)
    parser.add_argument("--mask_t", type=Path, default=None,
                        help="ROI mask at t; white means inpainting ROI.")
    parser.add_argument("--mask_t1", type=Path, default=None,
                        help="ROI mask at t+1; white means inpainting ROI.")
    parser.add_argument("--region", choices=("full", "bg", "roi"), default="full",
                        help="Region used for warping diagnostics and metrics.")
    parser.add_argument(
        "--raft_student_path",
        type=Path,
        default=DEFAULT_STUDENT,
        help="V7 run pointer, checkpoint, or raft_student folder; supports ProPainter RAFT and SEA-RAFT.",
    )
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--tile_size", type=int, default=384)
    parser.add_argument("--flow_percentile", type=float, default=99.0)
    parser.add_argument("--map_percentile", type=float, default=99.0)
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path.expanduser().resolve()), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_roi(path: Path, image_shape: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path.expanduser().resolve()), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if mask.shape != image_shape:
        raise ValueError(f"Mask {path} has shape {mask.shape}, expected {image_shape}")
    return mask >= 128


def to_raft(image: np.ndarray, device: torch.device) -> torch.Tensor:
    value = torch.from_numpy(image).permute(2, 0, 1).float().div(127.5).sub(1.0)
    return value.unsqueeze(0).to(device)


def pad_pair(frame0: torch.Tensor, frame1: torch.Tensor):
    height, width = frame0.shape[-2:]
    pad_h, pad_w = (-height) % 8, (-width) % 8
    padding = (0, pad_w, 0, pad_h)
    return F.pad(frame0, padding, mode="replicate"), F.pad(frame1, padding, mode="replicate")


def flow_color(flow: np.ndarray, scale: float) -> np.ndarray:
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*magnitude.shape, 3), np.uint8)
    hsv[..., 0] = ((angle * 90.0 / np.pi) % 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude / max(scale, 1e-8) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def heatmap(value: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.clip(value / max(scale, 1e-8) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def robust_scale(values: Sequence[np.ndarray], percentile: float) -> float:
    if not 0 < percentile <= 100:
        raise ValueError("Percentiles must be in (0, 100]")
    joined = np.concatenate([value[np.isfinite(value)].reshape(-1) for value in values])
    return max(float(np.percentile(joined, percentile)), 1e-8)


def sampling_maps(flow_target_to_source: np.ndarray):
    height, width = flow_target_to_source.shape[:2]
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = x + flow_target_to_source[..., 0]
    map_y = y + flow_target_to_source[..., 1]
    valid = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    return map_x, map_y, valid


def warp(source: np.ndarray, target_to_source: np.ndarray):
    map_x, map_y, valid = sampling_maps(target_to_source)
    warped = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return warped, valid


def ssim_map(first_rgb: np.ndarray, second_rgb: np.ndarray) -> np.ndarray:
    first = cv2.cvtColor(first_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    second = cv2.cvtColor(second_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mu1, mu2 = cv2.GaussianBlur(first, (11, 11), 1.5), cv2.GaussianBlur(second, (11, 11), 1.5)
    sigma1 = cv2.GaussianBlur(first * first, (11, 11), 1.5) - mu1 * mu1
    sigma2 = cv2.GaussianBlur(second * second, (11, 11), 1.5) - mu2 * mu2
    sigma12 = cv2.GaussianBlur(first * second, (11, 11), 1.5) - mu1 * mu2
    return ((2 * mu1 * mu2 + 0.01**2) * (2 * sigma12 + 0.03**2) /
            ((mu1 * mu1 + mu2 * mu2 + 0.01**2) * (sigma1 + sigma2 + 0.03**2))).clip(-1, 1)


def arrows(image_rgb: np.ndarray, flow: np.ndarray, spacing: int = 32) -> np.ndarray:
    result = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height, width = flow.shape[:2]
    for y in range(spacing // 2, height, spacing):
        for x in range(spacing // 2, width, spacing):
            dx, dy = flow[y, x]
            end = (int(round(x + dx)), int(round(y + dy)))
            cv2.arrowedLine(result, (x, y), end, (40, 255, 40), 1, cv2.LINE_AA, tipLength=0.25)
    return result


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(result, text, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA)
    return result


def montage(tiles: Sequence[Tuple[str, np.ndarray]], tile_size: int) -> np.ndarray:
    rendered = [label(cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA), title)
                for title, image in tiles]
    while len(rendered) % 4:
        rendered.append(np.zeros_like(rendered[0]))
    return np.concatenate([np.concatenate(rendered[i:i + 4], axis=1)
                           for i in range(0, len(rendered), 4)], axis=0)


def masked_mean(value: np.ndarray, valid: np.ndarray) -> float:
    return float(value[valid].mean()) if valid.any() else float("nan")


def region_support(source_roi: np.ndarray, target_roi: np.ndarray,
                   target_to_source: np.ndarray, region: str) -> np.ndarray:
    _, _, valid = sampling_maps(target_to_source)
    if region == "full":
        return valid
    source_region = source_roi if region == "roi" else ~source_roi
    target_region = target_roi if region == "roi" else ~target_roi
    map_x, map_y, _ = sampling_maps(target_to_source)
    warped_source_region = cv2.remap(
        source_region.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    return valid & target_region & warped_source_region


def analyze_temporal_pair(frame_t: np.ndarray, frame_t1: np.ndarray,
                          forward: np.ndarray, backward: np.ndarray,
                          support: np.ndarray) -> Dict[str, np.ndarray]:
    warped_t, valid = warp(frame_t, backward)
    raw_error = np.abs(frame_t.astype(np.float32) - frame_t1).mean(axis=2)
    warp_error = np.abs(warped_t.astype(np.float32) - frame_t1).mean(axis=2)
    raw_ssim = ssim_map(frame_t, frame_t1)
    warp_ssim = ssim_map(warped_t, frame_t1)
    map_x, map_y, _ = sampling_maps(backward)
    sampled_forward = cv2.remap(
        forward, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    consistency = np.linalg.norm(backward + sampled_forward, axis=2)
    return {
        "warped_t": warped_t,
        "valid": valid,
        "support": support,
        "raw_error": raw_error,
        "warp_error": warp_error,
        "raw_ssim": raw_ssim,
        "warp_ssim": warp_ssim,
        "consistency": consistency,
    }


def main() -> None:
    args = parse_args()
    if args.tile_size < 64:
        raise ValueError("--tile_size must be >= 64")
    paths = (args.input_t, args.input_t1, args.output_t, args.output_t1)
    input_t, input_t1, output_t, output_t1 = [read_rgb(path) for path in paths]
    shapes = {image.shape for image in (input_t, input_t1, output_t, output_t1)}
    if len(shapes) != 1:
        raise ValueError(f"All four images must have identical H/W, got {sorted(shapes)}")
    if args.region == "full":
        roi_t = np.zeros(input_t.shape[:2], dtype=bool)
        roi_t1 = np.zeros(input_t.shape[:2], dtype=bool)
    else:
        if args.mask_t is None or args.mask_t1 is None:
            raise ValueError("--mask_t and --mask_t1 are required for --region bg/roi")
        roi_t = read_roi(args.mask_t, input_t.shape[:2])
        roi_t1 = read_roi(args.mask_t1, input_t.shape[:2])
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The fine-tuned RAFT provider requires CUDA")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    student_path = resolve_raft_student_component(args.raft_student_path)
    student_config = json.loads((student_path / "config.json").read_text(encoding="utf-8"))
    architecture = student_config.get("architecture", "propainter_raft_large")
    sea_student = None
    if architecture == "sea_raft":
        sea_student = SEAStudentFlowPredictor.from_pretrained(student_path, map_location="cpu")
        sea_student = sea_student.to(device=device, dtype=torch.float32).eval()
        sea_student.requires_grad_(False)
        student_metadata = {
            "raft_student_component": str(student_path),
            "raft_architecture": architecture,
            "raft_iterations": int(student_config["iterations"]),
            "raft_input_normalization": student_config.get("input_normalization"),
            "flow_convention": student_config.get("flow_convention"),
            "pair_batch_size": 1,
            "frozen": True,
        }
        provider = None
    else:
        provider = FrozenV7RAFTFlowProvider(
            student_path, device=device, pair_batch_size=1, mixed_precision=not args.no_amp,
        )
        student_metadata = provider.metadata()
    sequences = []
    for frame_t, frame_t1 in ((input_t, input_t1), (output_t, output_t1)):
        first, second = pad_pair(to_raft(frame_t, device), to_raft(frame_t1, device))
        sequences.append(torch.stack((first[0], second[0]), dim=0))
    sequence = torch.stack(sequences, dim=0)
    with torch.inference_mode():
        if sea_student is not None:
            forward, backward = sea_student.predict_bidirectional(
                sequence[:, 0], sequence[:, 1], return_all=False, pair_batch_size=1
            )
            # Match the V7 provider's [B, adjacent-pair, 2, H, W] contract.
            prediction_forward, prediction_backward = forward.unsqueeze(1), backward.unsqueeze(1)
        else:
            prediction = provider.predict_sequence(sequence)
            prediction_forward, prediction_backward = prediction.forward, prediction.backward
    height, width = input_t.shape[:2]
    flows = []
    for batch_index in range(2):
        flows.append(tuple(
            value[batch_index, 0, :, :height, :width].permute(1, 2, 0).cpu().numpy()
            for value in (prediction_forward, prediction_backward)
        ))
    (input_forward, input_backward), (output_forward, output_backward) = flows
    backward_support = region_support(roi_t, roi_t1, input_backward, args.region)
    output_backward_support = region_support(roi_t, roi_t1, output_backward, args.region)
    common_backward_support = backward_support & output_backward_support
    forward_support = (
        region_support(roi_t1, roi_t, input_forward, args.region)
        & region_support(roi_t1, roi_t, output_forward, args.region)
    )
    input_result = analyze_temporal_pair(
        input_t, input_t1, input_forward, input_backward, common_backward_support
    )
    output_result = analyze_temporal_pair(
        output_t, output_t1, output_forward, output_backward, common_backward_support
    )
    forward_delta = output_forward - input_forward
    backward_delta = output_backward - input_backward
    forward_epe = np.linalg.norm(forward_delta, axis=2)
    backward_epe = np.linalg.norm(backward_delta, axis=2)
    flow_scale = robust_scale(tuple(
        np.linalg.norm(flow, axis=2)
        for flow in (input_forward, output_forward, input_backward, output_backward)
    ), args.flow_percentile)
    flow_delta_scale = robust_scale(
        (forward_epe[forward_support], backward_epe[common_backward_support]),
        args.map_percentile,
    )
    error_scale = robust_scale(tuple(
        result[key][result["support"]] for result in (input_result, output_result)
        for key in ("raw_error", "warp_error")
    ), args.map_percentile)

    def similarity_image(value: np.ndarray) -> np.ndarray:
        normalized = np.clip((value + 1) * 127.5, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)

    def region_image(image: np.ndarray, support: np.ndarray) -> np.ndarray:
        result = image.copy()
        result[~support] = (70, 70, 70)
        return result

    def bg_only_warp(result: Dict[str, np.ndarray], target: np.ndarray) -> np.ndarray:
        return np.where(result["support"][..., None], result["warped_t"], target)

    tiles = [
        ("Input t", cv2.cvtColor(input_t, cv2.COLOR_RGB2BGR)),
        ("Input t+1", cv2.cvtColor(input_t1, cv2.COLOR_RGB2BGR)),
        ("Output t", cv2.cvtColor(output_t, cv2.COLOR_RGB2BGR)),
        ("Output t+1", cv2.cvtColor(output_t1, cv2.COLOR_RGB2BGR)),
        (f"Input flow t->t+1, p99={flow_scale:.2f}px", flow_color(input_forward, flow_scale)),
        ("Output flow t->t+1", flow_color(output_forward, flow_scale)),
        ("Output-input forward flow", flow_color(forward_delta, flow_delta_scale)),
        (f"Forward flow EPE, {args.region}={masked_mean(forward_epe, forward_support):.3f}px",
         region_image(heatmap(forward_epe, flow_delta_scale), forward_support)),
        ("Input flow t+1->t", flow_color(input_backward, flow_scale)),
        ("Output flow t+1->t", flow_color(output_backward, flow_scale)),
        ("Output-input backward flow", flow_color(backward_delta, flow_delta_scale)),
        (f"Backward flow EPE, {args.region}={masked_mean(backward_epe, common_backward_support):.3f}px",
         region_image(heatmap(backward_epe, flow_delta_scale), common_backward_support)),
        (f"Warp input t to t+1 ({args.region})",
         cv2.cvtColor(bg_only_warp(input_result, input_t1), cv2.COLOR_RGB2BGR)),
        (f"Warp output t to t+1 ({args.region})",
         cv2.cvtColor(bg_only_warp(output_result, output_t1), cv2.COLOR_RGB2BGR)),
        ("Input motion-compensated error",
         region_image(heatmap(input_result["warp_error"], error_scale), common_backward_support)),
        ("Output motion-compensated error",
         region_image(heatmap(output_result["warp_error"], error_scale), common_backward_support)),
        (f"Input raw SSIM ({args.region})={masked_mean(input_result['raw_ssim'], common_backward_support):.4f}",
         region_image(similarity_image(input_result["raw_ssim"]), common_backward_support)),
        (f"Output raw SSIM ({args.region})={masked_mean(output_result['raw_ssim'], common_backward_support):.4f}",
         region_image(similarity_image(output_result["raw_ssim"]), common_backward_support)),
        (f"Input warped SSIM ({args.region})={masked_mean(input_result['warp_ssim'], common_backward_support):.4f}",
         region_image(similarity_image(input_result["warp_ssim"]), common_backward_support)),
        (f"Output warped SSIM ({args.region})={masked_mean(output_result['warp_ssim'], common_backward_support):.4f}",
         region_image(similarity_image(output_result["warp_ssim"]), common_backward_support)),
    ]
    save_dir = args.save_dir.expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    assets: Dict[str, np.ndarray] = {
        "input_forward_flow.png": flow_color(input_forward, flow_scale),
        "output_forward_flow.png": flow_color(output_forward, flow_scale),
        "forward_flow_epe.png": region_image(
            heatmap(forward_epe, flow_delta_scale), forward_support
        ),
        "input_backward_flow.png": flow_color(input_backward, flow_scale),
        "output_backward_flow.png": flow_color(output_backward, flow_scale),
        "backward_flow_epe.png": region_image(
            heatmap(backward_epe, flow_delta_scale), common_backward_support
        ),
        "input_warped_t.png": cv2.cvtColor(bg_only_warp(input_result, input_t1), cv2.COLOR_RGB2BGR),
        "output_warped_t.png": cv2.cvtColor(bg_only_warp(output_result, output_t1), cv2.COLOR_RGB2BGR),
        "input_warp_error.png": region_image(heatmap(input_result["warp_error"], error_scale), common_backward_support),
        "output_warp_error.png": region_image(heatmap(output_result["warp_error"], error_scale), common_backward_support),
        "evaluation_support.png": cv2.cvtColor(common_backward_support.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR),
    }
    assets["montage.png"] = montage(tiles, args.tile_size)
    for name, image in assets.items():
        if not cv2.imwrite(str(save_dir / name), image):
            raise OSError(save_dir / name)
    np.savez_compressed(
        save_dir / "flows.npz", input_forward=input_forward,
        input_backward=input_backward, output_forward=output_forward,
        output_backward=output_backward,
    )

    def pair_metrics(result: Dict[str, np.ndarray], forward: np.ndarray,
                     backward: np.ndarray, forward_valid: np.ndarray) -> Dict[str, float]:
        valid = result["support"]
        return {
            "raw_rgb_mae": masked_mean(result["raw_error"], valid),
            "raw_ssim_mean": masked_mean(result["raw_ssim"], valid),
            "warped_rgb_mae_valid": masked_mean(result["warp_error"], valid),
            "warped_ssim_mean_valid": masked_mean(result["warp_ssim"], valid),
            "forward_flow_magnitude_mean_px": masked_mean(
                np.linalg.norm(forward, axis=2), forward_valid
            ),
            "backward_flow_magnitude_mean_px": masked_mean(
                np.linalg.norm(backward, axis=2), valid
            ),
            "forward_backward_consistency_mean_px": masked_mean(result["consistency"], valid),
            "warp_valid_ratio": float(valid.mean()),
        }

    metrics = {
        "images": {name: str(path.expanduser().resolve()) for name, path in zip(
            ("input_t", "input_t1", "output_t", "output_t1"), paths
        )},
        "interpretation": "V7 student temporal flow is estimated separately for the consecutive input and output pairs.",
        "evaluation_region": args.region,
        "mask_semantics": "White mask pixels are ROI; BG is the inverse.",
        "mask_t": str(args.mask_t.expanduser().resolve()) if args.mask_t else None,
        "mask_t1": str(args.mask_t1.expanduser().resolve()) if args.mask_t1 else None,
        "raft_student": student_metadata,
        "height": height, "width": width,
        "input_pair": pair_metrics(
            input_result, input_forward, input_backward, forward_support
        ),
        "output_pair": pair_metrics(
            output_result, output_forward, output_backward, forward_support
        ),
        "input_output_flow_difference": {
            "forward_epe_mean_px": masked_mean(forward_epe, forward_support),
            "forward_epe_p99_px": float(np.percentile(forward_epe[forward_support], 99)),
            "backward_epe_mean_px": masked_mean(backward_epe, common_backward_support),
            "backward_epe_p99_px": float(np.percentile(
                backward_epe[common_backward_support], 99
            )),
        },
        "display_scales": {"temporal_rgb_mae_p99": error_scale,
                           "flow_p99_px": flow_scale,
                           "flow_difference_p99_px": flow_delta_scale},
        "evaluation_support_pixels": int(common_backward_support.sum()),
        "evaluation_support_ratio": float(common_backward_support.mean()),
    }
    (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
