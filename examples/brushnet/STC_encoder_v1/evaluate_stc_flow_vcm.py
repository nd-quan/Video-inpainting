#!/usr/bin/env python3
"""Evaluate the STC encoder + flow head on VCM clips.

This script evaluates the complete STC flow predictor:

    degraded frames + ROI/BG masks
        -> VAE degraded latents
        -> STC encoder
        -> bidirectional flow head

It supports:
- Stage-1 checkpoints containing ``flow_predictor/``;
- Stage-2/Stage-3 checkpoints containing ``noise_shaper/``;
- a direct ``save_pretrained`` component directory.

Outputs:
- summary.json: dataset-level aggregate metrics;
- per_clip_metrics.csv: one row per evaluated clip;
- per_pair_metrics.csv: one row per adjacent-frame pair;
- visualizations/*.png: qualitative flow/warping montages;
- evaluation.log.

Primary metrics:
- EPE against clean-video teacher flow;
- EPE gain over a zero-flow baseline;
- forward-backward consistency and reliable-pixel ratio;
- BG/ROI EPE;
- GT latent and RGB warping error using predicted, teacher and zero flow;
- flow magnitude and valid-region coverage.

Run from ``examples/brushnet`` or ensure that directory and the repository's custom
Diffusers package are on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusers import AutoencoderKL
from diffusers.models.brushnet_motion_adapter import (
    backward_warp_feature,
)
from diffusers.models.stc_flow_training import (
    compute_flow_training_losses,
    flow_in_bounds,
    prepare_teacher_flow,
    resize_flow_sequence,
)
from diffusers.models.stc_noise_shaper import STCConditionedNoiseShaper
from stc_flow_dataset import STCFlowDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate STC bidirectional optical flow with metrics and visualization."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help=(
            "Stage-1 checkpoint/pointer, Stage-2/3 checkpoint/pointer, "
            "or direct flow_predictor/noise_shaper component."
        ),
    )
    parser.add_argument("--split", default="valid")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)


    parser.add_argument(
        "--max-clips",
        type=int,
        default=0,
        help="0 evaluates the complete split.",
    )
    parser.add_argument(
        "--visualize-clips",
        type=int,
        default=12,
        help="Number of clips for which one informative adjacent pair is visualized.",
    )
    parser.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        help="Use CUDA autocast and FP16 VAE when running on CUDA.",
    )

    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable CUDA autocast and use FP32 VAE.",
    )

    parser.set_defaults(amp=True)
    
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--fb-alpha",
        type=float,
        default=0.01,
        help="Relative term in the forward-backward reliability criterion.",
    )
    parser.add_argument(
        "--fb-beta",
        type=float,
        default=0.5,
        help="Absolute squared-error term in latent-pixel units.",
    )
    parser.add_argument(
        "--charbonnier-eps",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args()


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("evaluate_stc_flow_vcm")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    for handler in (logging.FileHandler(path), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_pointer(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name in {"latest.json", "best.json"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        path = Path(payload["checkpoint"]).expanduser().resolve()
    return path


def resolve_model_component(path: Path) -> Path:
    """Find a save_pretrained STC component under common checkpoint layouts."""
    path = resolve_pointer(path)
    candidates = (
        path / "flow_predictor",
        path / "noise_shaper",
        path,
    )
    for candidate in candidates:
        if (
            candidate.is_dir()
            and (candidate / "config.json").is_file()
            and any(
                (candidate / filename).is_file()
                for filename in (
                    "diffusion_pytorch_model.safetensors",
                    "diffusion_pytorch_model.bin",
                    "pytorch_model.safetensors",
                    "pytorch_model.bin",
                    "model.safetensors",
                    "model.bin",
                )
            )
        ):
            return candidate
    raise FileNotFoundError(
        "Could not find a pretrained STC component. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def make_dataset(config: Mapping, split: str) -> STCFlowDataset:
    data = dict(config["data"])
    if not data.get("teacher_flow_root"):
        raise ValueError(
            "Evaluation requires data.teacher_flow_root for teacher-flow metrics."
        )

    # Evaluation must be deterministic and needs GT frames for warping diagnostics.
    data["load_gt"] = True
    data["random_horizontal_flip"] = False
    data["horizontal_flip_probability"] = 0.0
    data["seed"] = int(config.get("seed", 0))
    return STCFlowDataset(data, split=split)


@torch.inference_mode()
def encode_deterministic_latents(
    vae: AutoencoderKL,
    images: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError(f"Expected [B,T,3,H,W], got {tuple(images.shape)}")
    batch, frames = images.shape[:2]
    flat = images.flatten(0, 1).to(device=device, dtype=dtype)
    latents = vae.encode(flat).latent_dist.mode()
    latents = latents * vae.config.scaling_factor
    return latents.reshape(batch, frames, *latents.shape[1:]).float()


def latent_background_mask(
    roi_masks: torch.Tensor,
    size: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    if roi_masks.ndim != 5 or roi_masks.shape[2] != 1:
        raise ValueError(f"Expected ROI masks [B,T,1,H,W], got {tuple(roi_masks.shape)}")
    batch, frames = roi_masks.shape[:2]
    background = 1.0 - roi_masks.float().to(device)
    return F.interpolate(
        background.flatten(0, 1),
        size=size,
        mode="nearest",
    ).reshape(batch, frames, 1, *size)


def get_optional(
    batch: Mapping,
    primary: str,
    aliases: Iterable[str] = (),
) -> Optional[torch.Tensor]:
    for key in (primary, *aliases):
        if key in batch:
            return batch[key]
    return None


def get_required(
    batch: Mapping,
    primary: str,
    aliases: Iterable[str] = (),
):
    value = get_optional(batch, primary, aliases)
    if value is None:
        raise KeyError(
            f"Batch is missing {primary!r}; aliases={tuple(aliases)}"
        )
    return value


def warp_previous_sequence(
    previous: torch.Tensor,
    backward_flow: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Warp [B,P,C,H,W] previous-frame tensors to current coordinates."""
    if previous.ndim != 5:
        raise ValueError(f"previous must be [B,P,C,H,W], got {tuple(previous.shape)}")
    batch, pairs, channels, height, width = previous.shape
    if backward_flow.shape[:3] != (batch, pairs, 2):
        raise ValueError(
            f"Flow/feature mismatch: previous={tuple(previous.shape)}, "
            f"flow={tuple(backward_flow.shape)}"
        )
    flow = resize_flow_sequence(backward_flow, (height, width))
    warped = backward_warp_feature(
        previous.reshape(batch * pairs, channels, height, width),
        flow.reshape(batch * pairs, 2, height, width),
    ).reshape(batch, pairs, channels, height, width)
    valid = flow_in_bounds(flow)
    return warped, valid


def weighted_scalar(
    value: torch.Tensor,
    weight: torch.Tensor,
) -> float:
    value = value.float()
    weight = weight.float()
    if weight.ndim == value.ndim - 1:
        weight = weight.unsqueeze(2)
    if weight.shape[2] == 1 and value.shape[2] != 1:
        weight = weight.expand(
            *weight.shape[:2], value.shape[2], *weight.shape[-2:]
        )
    denominator = weight.sum()
    if float(denominator) <= 0.0:
        return float("nan")
    return float((value * weight).sum() / denominator)


def masked_endpoint_error(
    predicted: torch.Tensor,
    teacher: torch.Tensor,
    valid: torch.Tensor,
) -> float:
    error = (predicted.float() - teacher.float()).square().sum(
        dim=2, keepdim=True
    ).sqrt()
    return weighted_scalar(error, valid)


def masked_flow_magnitude(
    flow: torch.Tensor,
    valid: torch.Tensor,
) -> float:
    magnitude = flow.float().square().sum(dim=2, keepdim=True).sqrt()
    return weighted_scalar(magnitude, valid)


def masked_charbonnier(
    target: torch.Tensor,
    prediction: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> float:
    error = ((target.float() - prediction.float()).square() + eps**2).sqrt()
    return weighted_scalar(error, mask)


def masked_mse(
    target: torch.Tensor,
    prediction: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    error = (target.float() - prediction.float()).square()
    return weighted_scalar(error, mask)


def psnr_from_mse(mse: float, peak: float = 2.0) -> float:
    if not math.isfinite(mse):
        return float("nan")
    if mse <= 1e-12:
        return 120.0
    return 10.0 * math.log10((peak * peak) / mse)


def forward_backward_diagnostics(
    forward: torch.Tensor,
    backward: torch.Tensor,
    alpha: float,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return FB error, valid mask and reliable mask on current-frame grids.

    ``backward`` is defined on current frame and samples previous frame.
    ``forward`` is defined on previous frame and samples current frame.
    """
    batch, pairs, _, height, width = backward.shape
    warped_forward = backward_warp_feature(
        forward.reshape(batch * pairs, 2, height, width),
        backward.reshape(batch * pairs, 2, height, width),
    ).reshape_as(backward)

    fb_vector = backward.float() + warped_forward.float()
    fb_error_sq = fb_vector.square().sum(dim=2, keepdim=True)
    scale_sq = (
        backward.float().square().sum(dim=2, keepdim=True)
        + warped_forward.float().square().sum(dim=2, keepdim=True)
    )

    backward_valid = flow_in_bounds(backward)
    forward_valid = flow_in_bounds(forward)
    warped_forward_valid = backward_warp_feature(
        forward_valid.reshape(batch * pairs, 1, height, width),
        backward.reshape(batch * pairs, 2, height, width),
    ).reshape(batch, pairs, 1, height, width)
    valid = backward_valid * (warped_forward_valid > 0.999).to(backward.dtype)

    reliable = (
        fb_error_sq <= float(alpha) * scale_sq + float(beta)
    ).to(backward.dtype) * valid
    return fb_error_sq.sqrt(), valid, reliable


def metadata_value(batch: Mapping, key: str, index: int):
    value = batch[key]
    if isinstance(value, torch.Tensor):
        item = value[index]
        return item.detach().cpu().tolist() if item.ndim else item.item()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def sanitize_name(text: str) -> str:
    allowed = []
    for char in str(text):
        allowed.append(char if char.isalnum() or char in "-_." else "_")
    return "".join(allowed)


def tensor_rgb_to_bgr(frame: torch.Tensor) -> np.ndarray:
    array = (
        ((frame.detach().float().cpu() + 1.0) * 127.5)
        .clamp(0, 255)
        .permute(1, 2, 0)
        .byte()
        .numpy()
    )
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def flow_to_bgr(
    flow: torch.Tensor,
    max_magnitude: Optional[float] = None,
) -> np.ndarray:
    array = flow.detach().float().cpu().permute(1, 2, 0).numpy()
    magnitude, angle = cv2.cartToPolar(array[..., 0], array[..., 1])
    if max_magnitude is None or max_magnitude <= 0:
        max_magnitude = float(np.percentile(magnitude, 99.0))
    max_magnitude = max(float(max_magnitude), 1e-6)

    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((angle * 90.0 / np.pi) % 180.0).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude / max_magnitude * 255.0, 0, 255).astype(
        np.uint8
    )
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def scalar_heatmap(
    value: torch.Tensor,
    vmax: Optional[float] = None,
) -> np.ndarray:
    array = value.detach().float().cpu().squeeze().numpy()
    if vmax is None or vmax <= 0:
        vmax = float(np.percentile(array[np.isfinite(array)], 99.0)) if np.isfinite(array).any() else 1.0
    vmax = max(float(vmax), 1e-6)
    normalized = np.clip(array / vmax * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def mask_to_bgr(mask: torch.Tensor) -> np.ndarray:
    array = (
        mask.detach().float().cpu().squeeze().clamp(0, 1).mul(255).byte().numpy()
    )
    return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(
        output,
        label,
        (7, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def resize_tile(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def make_montage(
    tiles: Sequence[Tuple[str, np.ndarray]],
    columns: int = 4,
    tile_size: Tuple[int, int] = (320, 240),
) -> np.ndarray:
    labelled = [
        add_label(resize_tile(image, tile_size), label)
        for label, image in tiles
    ]
    blank = np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
    while len(labelled) % columns:
        labelled.append(blank.copy())
    rows = []
    for start in range(0, len(labelled), columns):
        rows.append(np.concatenate(labelled[start : start + columns], axis=1))
    return np.concatenate(rows, axis=0)


def save_visualization(
    output_path: Path,
    decoded: torch.Tensor,
    gt: torch.Tensor,
    predicted_backward: torch.Tensor,
    teacher_backward: torch.Tensor,
    teacher_valid_backward: torch.Tensor,
    fb_error: torch.Tensor,
    fb_reliable: torch.Tensor,
    pair_index: int,
) -> None:
    """Save one adjacent-pair montage for a single clip."""
    previous_decoded = decoded[pair_index]
    current_decoded = decoded[pair_index + 1]
    previous_gt = gt[pair_index]
    current_gt = gt[pair_index + 1]

    pred_flow = predicted_backward[pair_index : pair_index + 1].unsqueeze(0)
    teacher_flow = teacher_backward[pair_index : pair_index + 1].unsqueeze(0)

    image_size = tuple(current_gt.shape[-2:])
    pred_flow_image = resize_flow_sequence(pred_flow, image_size)[0, 0]
    teacher_flow_image = resize_flow_sequence(teacher_flow, image_size)[0, 0]

    pred_warp = backward_warp_feature(
        previous_gt.unsqueeze(0),
        pred_flow_image.unsqueeze(0),
    )[0]
    teacher_warp = backward_warp_feature(
        previous_gt.unsqueeze(0),
        teacher_flow_image.unsqueeze(0),
    )[0]
    zero_warp = previous_gt

    epe_map = (
        predicted_backward[pair_index].float()
        - teacher_backward[pair_index].float()
    ).square().sum(dim=0).sqrt()

    pred_error = (
        current_gt.float() - pred_warp.float()
    ).abs().mean(dim=0, keepdim=True)
    teacher_error = (
        current_gt.float() - teacher_warp.float()
    ).abs().mean(dim=0, keepdim=True)

    shared_magnitude = max(
        float(pred_flow_image.square().sum(0).sqrt().quantile(0.99)),
        float(teacher_flow_image.square().sum(0).sqrt().quantile(0.99)),
        1e-6,
    )
    shared_error = max(
        float(epe_map.quantile(0.99)),
        1e-6,
    )

    tiles = [
        ("Decoded previous", tensor_rgb_to_bgr(previous_decoded)),
        ("Decoded current", tensor_rgb_to_bgr(current_decoded)),
        ("GT previous", tensor_rgb_to_bgr(previous_gt)),
        ("GT current", tensor_rgb_to_bgr(current_gt)),
        ("Pred backward flow", flow_to_bgr(pred_flow_image, shared_magnitude)),
        ("Teacher backward flow", flow_to_bgr(teacher_flow_image, shared_magnitude)),
        ("Flow EPE map", scalar_heatmap(epe_map, shared_error)),
        ("Teacher valid mask", mask_to_bgr(teacher_valid_backward[pair_index])),
        ("Pred-warped GT prev", tensor_rgb_to_bgr(pred_warp)),
        ("Teacher-warped GT prev", tensor_rgb_to_bgr(teacher_warp)),
        ("Zero-flow warp", tensor_rgb_to_bgr(zero_warp)),
        ("FB reliable mask", mask_to_bgr(fb_reliable[pair_index])),
        ("Pred RGB warp error", scalar_heatmap(pred_error)),
        ("Teacher RGB warp error", scalar_heatmap(teacher_error)),
        ("FB error", scalar_heatmap(fb_error[pair_index])),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), make_montage(tiles))


def finite_or_none(value: float):
    return float(value) if math.isfinite(float(value)) else None


def aggregate_rows(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not rows:
        return {}
    excluded = {
        "clip_index",
        "sequence",
        "class_name",
        "split",
        "frame_start",
        "frame_end",
    }
    result: Dict[str, object] = {"num_clips": len(rows)}
    keys = sorted(set().union(*(row.keys() for row in rows)) - excluded)
    for key in keys:
        values = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        if values:
            result[key] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))
    return result


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pair_weighted_mean(
    value: torch.Tensor,
    weight: torch.Tensor,
    pair: int,
) -> float:
    return weighted_scalar(
        value[:, pair : pair + 1],
        weight[:, pair : pair + 1],
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} is non-empty. Use --overwrite or a new output directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "evaluation.log")

    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    amp = bool(args.amp and device.type == "cuda")
    vae_dtype = torch.float16 if amp else torch.float32

    component = resolve_model_component(args.checkpoint)
    logger.info("Loading STC flow predictor from %s", component)
    model = STCConditionedNoiseShaper.from_pretrained(component)
    if str(model.config.flow_prediction_mode).lower() != "full":
        raise ValueError(
            "This evaluator requires flow_prediction_mode='full'; "
            f"checkpoint has {model.config.flow_prediction_mode!r}."
        )
    model.requires_grad_(False).eval().to(device=device, dtype=torch.float32)

    base_model = config["model"]["base_model"]
    logger.info("Loading VAE from %s", base_model)
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    vae.requires_grad_(False).eval().to(device=device, dtype=vae_dtype)

    dataset = make_dataset(config, args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    logger.info(
        "split=%s clips=%d clip_length=%d batch_size=%d amp=%s device=%s",
        args.split,
        len(dataset),
        dataset.clip_length,
        args.batch_size,
        amp,
        device,
    )

    clip_rows: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []
    visualized = 0
    evaluated_clips = 0

    for batch_index, batch in enumerate(loader):
        if args.max_clips > 0 and evaluated_clips >= args.max_clips:
            break

        decoded_cpu = get_required(batch, "decoded_frames", ("input_frames",))
        gt_cpu = get_required(batch, "gt_frames")
        roi_cpu = get_required(batch, "roi_masks", ("masks",))
        batch_size = int(decoded_cpu.shape[0])

        if args.max_clips > 0:
            keep = min(batch_size, args.max_clips - evaluated_clips)
            if keep < batch_size:
                decoded_cpu = decoded_cpu[:keep]
                gt_cpu = gt_cpu[:keep]
                roi_cpu = roi_cpu[:keep]
                batch_size = keep

        decoded = decoded_cpu.to(device=device, dtype=torch.float32)
        gt = gt_cpu.to(device=device, dtype=torch.float32)
        decoded_latents = encode_deterministic_latents(
            vae, decoded_cpu, device, vae_dtype
        )
        gt_latents = encode_deterministic_latents(
            vae, gt_cpu, device, vae_dtype
        )
        bg_mask = latent_background_mask(
            roi_cpu, decoded_latents.shape[-2:], device
        )

        teacher_forward_raw = get_required(
            batch,
            "teacher_flow_forward",
            ("teacher_forward", "flow_forward"),
        )[:batch_size].to(device=device, dtype=torch.float32)
        teacher_backward_raw = get_required(
            batch,
            "teacher_flow_backward",
            ("teacher_backward", "flow_backward"),
        )[:batch_size].to(device=device, dtype=torch.float32)

        valid_forward_raw = get_optional(
            batch,
            "teacher_valid_forward",
            ("valid_forward", "valid_f"),
        )
        valid_backward_raw = get_optional(
            batch,
            "teacher_valid_backward",
            ("valid_backward", "valid_b"),
        )
        if valid_forward_raw is not None:
            valid_forward_raw = valid_forward_raw[:batch_size].to(
                device=device, dtype=torch.float32
            )
        if valid_backward_raw is not None:
            valid_backward_raw = valid_backward_raw[:batch_size].to(
                device=device, dtype=torch.float32
            )

        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp
            else nullcontext()
        )
        with autocast:
            output = model.predict_flow(
                decoded_latents=decoded_latents,
                bg_mask=bg_mask,
                return_dict=True,
            )
        predicted_forward = output["predicted_flow_forward"].float()
        predicted_backward = output["predicted_flow_backward"].float()

        prediction_size = tuple(predicted_backward.shape[-2:])
        teacher_forward, valid_forward = prepare_teacher_flow(
            teacher_forward_raw,
            prediction_size,
            valid_forward_raw,
        )
        teacher_backward, valid_backward = prepare_teacher_flow(
            teacher_backward_raw,
            prediction_size,
            valid_backward_raw,
        )

        fb_error, fb_valid, fb_reliable = forward_backward_diagnostics(
            predicted_forward,
            predicted_backward,
            alpha=args.fb_alpha,
            beta=args.fb_beta,
        )

        previous_bg = bg_mask[:, :-1]
        current_bg = bg_mask[:, 1:]
        previous_bg_teacher_warp, _ = warp_previous_sequence(
            previous_bg, teacher_backward
        )
        previous_bg_pred_warp, _ = warp_previous_sequence(
            previous_bg, predicted_backward
        )

        predicted_in_bounds = flow_in_bounds(predicted_backward)
        common_teacher_bg = (
            valid_backward
            * predicted_in_bounds
            * current_bg
            * (previous_bg_teacher_warp > 0.5).to(current_bg.dtype)
        )
        predicted_reliable_bg = (
            fb_reliable
            * current_bg
            * (previous_bg_pred_warp > 0.5).to(current_bg.dtype)
        )

        zero_forward = torch.zeros_like(predicted_forward)
        zero_backward = torch.zeros_like(predicted_backward)

        # Warping diagnostics at latent resolution.
        target_latent = gt_latents[:, 1:]
        pred_latent_warp, _ = warp_previous_sequence(
            gt_latents[:, :-1], predicted_backward
        )
        teacher_latent_warp, _ = warp_previous_sequence(
            gt_latents[:, :-1], teacher_backward
        )
        zero_latent_warp, _ = warp_previous_sequence(
            gt_latents[:, :-1], zero_backward
        )

        # Warping diagnostics in RGB space.
        target_rgb = gt[:, 1:]
        pred_rgb_warp, _ = warp_previous_sequence(
            gt[:, :-1], predicted_backward
        )
        teacher_rgb_warp, _ = warp_previous_sequence(
            gt[:, :-1], teacher_backward
        )
        zero_rgb_warp, _ = warp_previous_sequence(
            gt[:, :-1], zero_backward
        )
        common_teacher_bg_rgb = F.interpolate(
            common_teacher_bg.flatten(0, 1),
            size=gt.shape[-2:],
            mode="nearest",
        ).reshape(batch_size, -1, 1, *gt.shape[-2:])
        predicted_reliable_bg_rgb = F.interpolate(
            predicted_reliable_bg.flatten(0, 1),
            size=gt.shape[-2:],
            mode="nearest",
        ).reshape(batch_size, -1, 1, *gt.shape[-2:])

        for sample_index in range(batch_size):
            predicted_f = predicted_forward[sample_index : sample_index + 1]
            predicted_b = predicted_backward[sample_index : sample_index + 1]
            teacher_f = teacher_forward[sample_index : sample_index + 1]
            teacher_b = teacher_backward[sample_index : sample_index + 1]
            valid_f = valid_forward[sample_index : sample_index + 1]
            valid_b = valid_backward[sample_index : sample_index + 1]

            standard = compute_flow_training_losses(
                predicted_f,
                predicted_b,
                teacher_f,
                teacher_b,
                decoded[sample_index : sample_index + 1],
                valid_forward=valid_f,
                valid_backward=valid_b,
                loss_config=config.get("loss", {}),
            )

            zero_epe_f = masked_endpoint_error(
                zero_forward[sample_index : sample_index + 1],
                teacher_f,
                valid_f,
            )
            zero_epe_b = masked_endpoint_error(
                zero_backward[sample_index : sample_index + 1],
                teacher_b,
                valid_b,
            )
            zero_epe = float(np.nanmean([zero_epe_f, zero_epe_b]))

            bg_f = bg_mask[sample_index : sample_index + 1, :-1]
            bg_b = bg_mask[sample_index : sample_index + 1, 1:]
            roi_f = 1.0 - bg_f
            roi_b = 1.0 - bg_b

            bg_epe_f = masked_endpoint_error(
                predicted_f, teacher_f, valid_f * bg_f
            )
            bg_epe_b = masked_endpoint_error(
                predicted_b, teacher_b, valid_b * bg_b
            )
            roi_epe_f = masked_endpoint_error(
                predicted_f, teacher_f, valid_f * roi_f
            )
            roi_epe_b = masked_endpoint_error(
                predicted_b, teacher_b, valid_b * roi_b
            )

            mask_common_latent = common_teacher_bg[
                sample_index : sample_index + 1
            ]
            mask_reliable_latent = predicted_reliable_bg[
                sample_index : sample_index + 1
            ]
            mask_common_rgb = common_teacher_bg_rgb[
                sample_index : sample_index + 1
            ]
            mask_reliable_rgb = predicted_reliable_bg_rgb[
                sample_index : sample_index + 1
            ]

            latent_pred = masked_charbonnier(
                target_latent[sample_index : sample_index + 1],
                pred_latent_warp[sample_index : sample_index + 1],
                mask_common_latent,
                args.charbonnier_eps,
            )
            latent_teacher = masked_charbonnier(
                target_latent[sample_index : sample_index + 1],
                teacher_latent_warp[sample_index : sample_index + 1],
                mask_common_latent,
                args.charbonnier_eps,
            )
            latent_zero = masked_charbonnier(
                target_latent[sample_index : sample_index + 1],
                zero_latent_warp[sample_index : sample_index + 1],
                mask_common_latent,
                args.charbonnier_eps,
            )
            latent_pred_reliable = masked_charbonnier(
                target_latent[sample_index : sample_index + 1],
                pred_latent_warp[sample_index : sample_index + 1],
                mask_reliable_latent,
                args.charbonnier_eps,
            )

            rgb_pred_mae = masked_charbonnier(
                target_rgb[sample_index : sample_index + 1],
                pred_rgb_warp[sample_index : sample_index + 1],
                mask_common_rgb,
                args.charbonnier_eps,
            )
            rgb_teacher_mae = masked_charbonnier(
                target_rgb[sample_index : sample_index + 1],
                teacher_rgb_warp[sample_index : sample_index + 1],
                mask_common_rgb,
                args.charbonnier_eps,
            )
            rgb_zero_mae = masked_charbonnier(
                target_rgb[sample_index : sample_index + 1],
                zero_rgb_warp[sample_index : sample_index + 1],
                mask_common_rgb,
                args.charbonnier_eps,
            )
            rgb_pred_reliable_mae = masked_charbonnier(
                target_rgb[sample_index : sample_index + 1],
                pred_rgb_warp[sample_index : sample_index + 1],
                mask_reliable_rgb,
                args.charbonnier_eps,
            )

            rgb_pred_mse = masked_mse(
                target_rgb[sample_index : sample_index + 1],
                pred_rgb_warp[sample_index : sample_index + 1],
                mask_common_rgb,
            )
            rgb_teacher_mse = masked_mse(
                target_rgb[sample_index : sample_index + 1],
                teacher_rgb_warp[sample_index : sample_index + 1],
                mask_common_rgb,
            )
            rgb_zero_mse = masked_mse(
                target_rgb[sample_index : sample_index + 1],
                zero_rgb_warp[sample_index : sample_index + 1],
                mask_common_rgb,
            )

            epe_value = float(standard["epe"])
            row: Dict[str, object] = {
                "clip_index": evaluated_clips,
                "sequence": metadata_value(batch, "sequence", sample_index),
                "class_name": metadata_value(batch, "class_name", sample_index),
                "split": args.split,
                "frame_start": int(
                    metadata_value(batch, "frame_indices", sample_index)[0]
                ),
                "frame_end": int(
                    metadata_value(batch, "frame_indices", sample_index)[-1]
                ),
                "epe": epe_value,
                "epe_forward": float(standard["epe_forward"]),
                "epe_backward": float(standard["epe_backward"]),
                "zero_flow_epe": zero_epe,
                "epe_gain_vs_zero": (
                    1.0 - epe_value / zero_epe
                    if math.isfinite(zero_epe) and zero_epe > 1e-8
                    else float("nan")
                ),
                "bg_epe_forward": bg_epe_f,
                "bg_epe_backward": bg_epe_b,
                "bg_epe": float(np.nanmean([bg_epe_f, bg_epe_b])),
                "roi_epe_forward": roi_epe_f,
                "roi_epe_backward": roi_epe_b,
                "roi_epe": float(np.nanmean([roi_epe_f, roi_epe_b])),
                "teacher_charbonnier": float(standard["teacher"]),
                "fb_loss": float(standard["fb"]),
                "smoothness": float(standard["smoothness"]),
                "pred_flow_magnitude": float(
                    np.nanmean(
                        [
                            masked_flow_magnitude(predicted_f, valid_f),
                            masked_flow_magnitude(predicted_b, valid_b),
                        ]
                    )
                ),
                "teacher_flow_magnitude": float(
                    np.nanmean(
                        [
                            masked_flow_magnitude(teacher_f, valid_f),
                            masked_flow_magnitude(teacher_b, valid_b),
                        ]
                    )
                ),
                "fb_error_mean": weighted_scalar(
                    fb_error[sample_index : sample_index + 1],
                    fb_valid[sample_index : sample_index + 1],
                ),
                "fb_valid_ratio": float(
                    fb_valid[sample_index].float().mean()
                ),
                "fb_reliable_ratio": weighted_scalar(
                    fb_reliable[sample_index : sample_index + 1],
                    fb_valid[sample_index : sample_index + 1],
                ),
                "common_bg_ratio": float(
                    mask_common_latent.float().mean()
                ),
                "pred_reliable_bg_ratio": float(
                    mask_reliable_latent.float().mean()
                ),
                "latent_warp_mae_pred": latent_pred,
                "latent_warp_mae_teacher": latent_teacher,
                "latent_warp_mae_zero": latent_zero,
                "latent_warp_gain_vs_zero": (
                    1.0 - latent_pred / latent_zero
                    if math.isfinite(latent_zero) and latent_zero > 1e-8
                    else float("nan")
                ),
                "latent_warp_mae_pred_reliable": latent_pred_reliable,
                "rgb_warp_mae_pred": rgb_pred_mae,
                "rgb_warp_mae_teacher": rgb_teacher_mae,
                "rgb_warp_mae_zero": rgb_zero_mae,
                "rgb_warp_gain_vs_zero": (
                    1.0 - rgb_pred_mae / rgb_zero_mae
                    if math.isfinite(rgb_zero_mae) and rgb_zero_mae > 1e-8
                    else float("nan")
                ),
                "rgb_warp_mae_pred_reliable": rgb_pred_reliable_mae,
                "rgb_warp_psnr_pred": psnr_from_mse(rgb_pred_mse),
                "rgb_warp_psnr_teacher": psnr_from_mse(rgb_teacher_mse),
                "rgb_warp_psnr_zero": psnr_from_mse(rgb_zero_mse),
            }
            clip_rows.append(row)

            pairs = int(predicted_b.shape[1])
            flow_error_map = (
                predicted_b - teacher_b
            ).square().sum(dim=2, keepdim=True).sqrt()
            for pair in range(pairs):
                pair_valid = valid_b[:, pair : pair + 1]
                pair_common = mask_common_latent[:, pair : pair + 1]
                pair_reliable = mask_reliable_latent[:, pair : pair + 1]
                pair_rows.append(
                    {
                        "clip_index": evaluated_clips,
                        "sequence": row["sequence"],
                        "class_name": row["class_name"],
                        "frame_from": int(
                            metadata_value(batch, "frame_indices", sample_index)[pair]
                        ),
                        "frame_to": int(
                            metadata_value(batch, "frame_indices", sample_index)[pair + 1]
                        ),
                        "pair_index": pair,
                        "epe_backward": pair_weighted_mean(
                            flow_error_map, pair_valid, pair
                        ),
                        "pred_flow_magnitude": pair_weighted_mean(
                            predicted_b.square().sum(
                                dim=2, keepdim=True
                            ).sqrt(),
                            pair_valid,
                            pair,
                        ),
                        "teacher_flow_magnitude": pair_weighted_mean(
                            teacher_b.square().sum(
                                dim=2, keepdim=True
                            ).sqrt(),
                            pair_valid,
                            pair,
                        ),
                        "fb_error": pair_weighted_mean(
                            fb_error[
                                sample_index : sample_index + 1
                            ],
                            fb_valid[
                                sample_index : sample_index + 1
                            ],
                            pair,
                        ),
                        "fb_reliable_ratio": pair_weighted_mean(
                            fb_reliable[
                                sample_index : sample_index + 1
                            ],
                            fb_valid[
                                sample_index : sample_index + 1
                            ],
                            pair,
                        ),
                        "common_bg_ratio": float(
                            pair_common.float().mean()
                        ),
                        "pred_reliable_bg_ratio": float(
                            pair_reliable.float().mean()
                        ),
                    }
                )

            if visualized < args.visualize_clips:
                teacher_magnitude_per_pair = (
                    teacher_b.square().sum(dim=2, keepdim=True).sqrt()
                    * valid_b
                ).sum(dim=(0, 2, 3, 4)) / valid_b.sum(
                    dim=(0, 2, 3, 4)
                ).clamp_min(1e-6)
                pair_index = int(teacher_magnitude_per_pair.argmax())

                sequence = sanitize_name(str(row["sequence"]))
                class_name = sanitize_name(str(row["class_name"]))
                name = (
                    f"{evaluated_clips:05d}_{class_name}_{sequence}_"
                    f"{int(row['frame_start']):06d}_pair{pair_index:02d}.png"
                )
                save_visualization(
                    visualization_dir / name,
                    decoded=decoded[sample_index].cpu(),
                    gt=gt[sample_index].cpu(),
                    predicted_backward=predicted_b[0].cpu(),
                    teacher_backward=teacher_b[0].cpu(),
                    teacher_valid_backward=valid_b[0].cpu(),
                    fb_error=fb_error[sample_index].cpu(),
                    fb_reliable=fb_reliable[sample_index].cpu(),
                    pair_index=pair_index,
                )
                visualized += 1

            evaluated_clips += 1

        if batch_index % 10 == 0 or (
            args.max_clips > 0 and evaluated_clips >= args.max_clips
        ):
            recent = clip_rows[-min(20, len(clip_rows)) :]
            logger.info(
                "evaluated=%d/%s recent_epe=%.4f recent_bg_epe=%.4f "
                "recent_rgb_warp=%.5f recent_fb_reliable=%.3f",
                evaluated_clips,
                args.max_clips if args.max_clips > 0 else len(dataset),
                float(np.nanmean([float(row["epe"]) for row in recent])),
                float(np.nanmean([float(row["bg_epe"]) for row in recent])),
                float(
                    np.nanmean(
                        [float(row["rgb_warp_mae_pred"]) for row in recent]
                    )
                ),
                float(
                    np.nanmean(
                        [float(row["fb_reliable_ratio"]) for row in recent]
                    )
                ),
            )

    write_csv(output_dir / "per_clip_metrics.csv", clip_rows)
    write_csv(output_dir / "per_pair_metrics.csv", pair_rows)

    summary = aggregate_rows(clip_rows)
    summary.update(
        {
            "checkpoint_component": str(component),
            "config": str(args.config.expanduser().resolve()),
            "split": args.split,
            "fb_alpha": args.fb_alpha,
            "fb_beta": args.fb_beta,
            "epe_units": "latent pixels",
            "metric_notes": {
                "epe_gain_vs_zero": (
                    "Positive means the STC flow is closer to teacher flow than "
                    "a zero-flow baseline; 1.0 is ideal."
                ),
                "latent_warp_gain_vs_zero": (
                    "Positive means predicted flow aligns clean GT latents better "
                    "than zero flow on a common teacher-valid BG mask."
                ),
                "rgb_warp_gain_vs_zero": (
                    "Positive means predicted flow aligns clean GT RGB frames better "
                    "than zero flow on the same common mask."
                ),
                "fb_reliable_ratio": (
                    "Fraction of valid pixels satisfying the configured "
                    "forward-backward consistency criterion."
                ),
                "pred_reliable_bg_ratio": (
                    "Coverage available for downstream temporal latent loss after "
                    "FB reliability and BG-to-BG gating."
                ),
            },
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    logger.info("Finished %d clips", evaluated_clips)
    logger.info(
        "SUMMARY epe=%.4f zero_epe=%.4f gain=%.3f bg_epe=%.4f "
        "latent_warp_pred=%.5f latent_warp_zero=%.5f "
        "rgb_warp_pred=%.5f rgb_warp_zero=%.5f fb_reliable=%.3f",
        float(summary.get("epe", float("nan"))),
        float(summary.get("zero_flow_epe", float("nan"))),
        float(summary.get("epe_gain_vs_zero", float("nan"))),
        float(summary.get("bg_epe", float("nan"))),
        float(summary.get("latent_warp_mae_pred", float("nan"))),
        float(summary.get("latent_warp_mae_zero", float("nan"))),
        float(summary.get("rgb_warp_mae_pred", float("nan"))),
        float(summary.get("rgb_warp_mae_zero", float("nan"))),
        float(summary.get("fb_reliable_ratio", float("nan"))),
    )
    logger.info("Outputs saved to %s", output_dir)


if __name__ == "__main__":
    # logger = setup_logger(output_dir / "evaluation.log")
    main()
