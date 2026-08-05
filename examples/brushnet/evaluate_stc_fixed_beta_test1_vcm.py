#!/usr/bin/env python
"""Evaluate Fixed-beta Test 1 on manifest-defined VCM video clips.

This experiment performs no optimization.  The trained Stage-1 full-flow
predictor shapes the initial latent noise once, after which a frozen SD1.5
U-Net and the frozen V8 BrushNet/IP-Adapter/FGBG-fusion stack run ordinary
DDIM denoising.  No legacy motion adapter or Stage-3 condition adapter is
installed.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from stc_fixed_beta_test1 import (
    deterministic_clip_seed,
    load_fixed_beta_stage1,
    resolve_v8_components,
    validate_fixed_test1_config,
)
from stc_flow_dataset import STCFlowDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fixed-beta STC noise shaping without Stage-3 training"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fixed-beta", type=float, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate paths, checkpoints and split coverage without loading models",
    )
    return parser.parse_args()


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("stc_fixed_beta_test1")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    for handler in (logging.FileHandler(path), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def beta_directory_name(beta: float) -> str:
    return f"beta_{float(beta):.3f}".replace(".", "p")


def tensor_frames_to_pil(frames: torch.Tensor) -> List[Image.Image]:
    arrays = (
        ((frames.float() + 1.0) * 127.5)
        .clamp(0, 255)
        .permute(0, 2, 3, 1)
        .byte()
        .cpu()
        .numpy()
    )
    return [Image.fromarray(array, mode="RGB") for array in arrays]


def roi_masks_to_pil(masks: torch.Tensor) -> List[Image.Image]:
    arrays = (
        (masks.float().clamp(0, 1) * 255)
        .repeat(1, 3, 1, 1)
        .permute(0, 2, 3, 1)
        .byte()
        .cpu()
        .numpy()
    )
    return [Image.fromarray(array, mode="RGB") for array in arrays]


def masked_pil_frames(
    images: Sequence[Image.Image], roi_masks: torch.Tensor
) -> Tuple[List[Image.Image], List[Image.Image]]:
    roi_arrays = roi_masks[:, 0].float().clamp(0, 1).cpu().numpy()[..., None]
    foreground, background = [], []
    for image, roi in zip(images, roi_arrays):
        array = np.asarray(image, dtype=np.float32)
        foreground.append(
            Image.fromarray((array * roi).round().astype(np.uint8), mode="RGB")
        )
        background.append(
            Image.fromarray(
                (array * (1.0 - roi)).round().astype(np.uint8), mode="RGB"
            )
        )
    return foreground, background


@torch.inference_mode()
def build_v8_prompt_embeddings(
    pipe,
    ip_conditioner,
    images: Sequence[Image.Image],
    roi_masks: torch.Tensor,
    prompt: str,
    negative_prompt: str,
    fusion_scale: float,
    v8_mask_order: bool,
):
    """Reproduce V8's null-text + base-image + FGBG image conditioning."""

    device = pipe._execution_device
    prompts = [str(prompt)] * len(images)
    negatives = [str(negative_prompt)] * len(images)
    text, negative_text = pipe.encode_prompt(
        prompts,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negatives,
    )

    foreground, background = masked_pil_frames(images, roi_masks)
    if v8_mask_order:
        # V8 historical convention: branch named "FG" received background,
        # while branch named "BG" received the ROI crop.
        foreground, background = background, foreground
    processor = ip_conditioner.clip_image_processor
    encoder = ip_conditioner.image_encoder
    encoder_device = next(encoder.parameters()).device
    encoder_dtype = next(encoder.parameters()).dtype

    def encode(items):
        pixels = processor(images=list(items), return_tensors="pt").pixel_values
        return encoder(
            pixels.to(device=encoder_device, dtype=encoder_dtype)
        ).image_embeds

    base = encode(images)
    first_branch = encode(foreground)
    second_branch = encode(background)
    fused = ip_conditioner.fusion_module(first_branch, second_branch)
    combined = base + float(fusion_scale) * fused
    image_tokens = ip_conditioner.image_proj_model(combined)
    negative_image_tokens = ip_conditioner.image_proj_model(
        torch.zeros_like(combined)
    )
    image_tokens = image_tokens.to(device=device, dtype=text.dtype)
    negative_image_tokens = negative_image_tokens.to(
        device=device, dtype=negative_text.dtype
    )
    return (
        torch.cat((text, image_tokens), dim=1),
        torch.cat((negative_text, negative_image_tokens), dim=1),
    )


def freeze_module(module):
    module.requires_grad_(False).eval()
    return module


def validate_strict_v8_ip_loading(ip_path: Path, pipe, ip_conditioner):
    """Reject missing, extra or shape-incompatible V8 IP-Adapter tensors.

    The historical loader uses ``strict=False`` for compatibility with older
    checkpoints. Test 1 is a controlled ablation, so silent partial loading is
    not acceptable; validate the exact V8 deployment contract after loading.
    """

    from safetensors import safe_open

    expected = {}
    for key, tensor in ip_conditioner.image_proj_model.state_dict().items():
        expected[f"image_proj_model.{key}"] = tensor
    cross_attention_count = 0
    for name, processor in pipe.unet.attn_processors.items():
        if not name.endswith("attn2.processor"):
            continue
        cross_attention_count += 1
        for key, tensor in processor.state_dict().items():
            expected[f"unet.{name}.{key}"] = tensor
    if cross_attention_count != 16:
        raise RuntimeError(
            f"V8 expects 16 cross-attention processors, found {cross_attention_count}"
        )

    with safe_open(str(ip_path), framework="pt", device="cpu") as handle:
        actual_keys = set(handle.keys())
        expected_keys = set(expected)
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        shape_mismatch = []
        value_mismatch = []
        for key in sorted(expected_keys & actual_keys):
            checkpoint_tensor = handle.get_tensor(key)
            deployed_tensor = expected[key].detach().cpu()
            if tuple(checkpoint_tensor.shape) != tuple(deployed_tensor.shape):
                shape_mismatch.append(
                    f"{key}: checkpoint={tuple(checkpoint_tensor.shape)} "
                    f"model={tuple(deployed_tensor.shape)}"
                )
                continue
            checkpoint_tensor = checkpoint_tensor.to(dtype=deployed_tensor.dtype)
            if not torch.equal(checkpoint_tensor, deployed_tensor):
                value_mismatch.append(key)
    if missing or extra or shape_mismatch or value_mismatch:
        raise RuntimeError(
            "V8 IP-Adapter strict validation failed: "
            f"missing={missing}, extra={extra}, "
            f"shape_mismatch={shape_mismatch}, "
            f"value_mismatch={value_mismatch}"
        )
    return {"tensor_count": len(expected), "cross_attention_processors": 16}


def build_pipeline(config: Mapping, device: torch.device, logger):
    # Keep these imports out of --preflight-only so path validation does not
    # initialize the large inference stack or optional CUDA extensions.
    from diffusers import DDIMScheduler
    from diffusers.models.brushnet import BrushNetModel
    from diffusers.pipelines.brushnet.pipeline_brushnet_sharedNoise_sameBG_v0_0 import (
        StableDiffusionBrushNetPipeline,
    )
    from ip_adapter import FusionIPAdapter

    model_config = config["model"]
    inference = config["inference"]
    ip_config = config["ip_adapter"]
    noise_config = config["noise_shaping"]
    dtype_name = str(inference.get("dtype", "float16")).lower()
    if device.type == "cpu":
        weight_dtype = torch.float32
    elif dtype_name in {"float16", "fp16", "half"}:
        weight_dtype = torch.float16
    elif dtype_name in {"bfloat16", "bf16"}:
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    v8 = resolve_v8_components(Path(model_config["v8_checkpoint"]))
    noise_shaper, stage1_metadata = load_fixed_beta_stage1(
        Path(model_config["stage1_checkpoint"]),
        fixed_beta=float(noise_config["fixed_beta"]),
        torch_dtype=weight_dtype,
    )
    stage1_metadata["stage1_step"] = model_config.get(
        "stage1_step", stage1_metadata.get("stage1_step")
    )
    stage1_metadata["stage1_valid_epe"] = model_config.get(
        "stage1_valid_epe", stage1_metadata.get("stage1_valid_epe")
    )
    brushnet = BrushNetModel.from_pretrained(
        v8["brushnet"], torch_dtype=weight_dtype
    )
    pipe = StableDiffusionBrushNetPipeline.from_pretrained(
        model_config["base_model"],
        brushnet=brushnet,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=False,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_noise_shaper(noise_shaper)

    # FusionIPAdapter installs the V8 gated cross-attention processors into
    # the base U-Net and strictly loads the released image-conditioning stack.
    ip_conditioner = FusionIPAdapter(
        pipe,
        ip_config["image_encoder"],
        str(v8["ip_adapter"]),
        str(v8["fusion"]),
        str(device),
        num_tokens=int(ip_config.get("num_tokens", 4)),
    )
    ip_validation = validate_strict_v8_ip_loading(
        v8["ip_adapter"], pipe, ip_conditioner
    )
    ip_conditioner.set_scale(float(ip_config.get("scale", 1.0)))
    for module in (
        pipe.vae,
        pipe.text_encoder,
        pipe.unet,
        pipe.brushnet,
        noise_shaper,
        ip_conditioner.image_encoder,
        ip_conditioner.image_proj_model,
        ip_conditioner.fusion_module,
    ):
        freeze_module(module)
    trainable = sum(
        parameter.numel()
        for module in (
            pipe.vae,
            pipe.text_encoder,
            pipe.unet,
            pipe.brushnet,
            noise_shaper,
            ip_conditioner.image_encoder,
            ip_conditioner.image_proj_model,
            ip_conditioner.fusion_module,
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    if trainable:
        raise RuntimeError(f"Test 1 unexpectedly has {trainable} trainable parameters")
    if getattr(pipe, "motion_adapter", None) is not None:
        raise RuntimeError("Fixed-beta Test 1 must not install a motion adapter")

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if bool(inference.get("model_cpu_offload", False)) and device.type == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    logger.info(
        "Loaded Stage-1 step=%s valid_epe=%s fixed_beta=%.3f warp=%s",
        stage1_metadata.get("stage1_step"),
        stage1_metadata.get("stage1_valid_epe"),
        stage1_metadata["fixed_beta"],
        stage1_metadata["warp_region"],
    )
    logger.info(
        "Frozen deployment: base SD1.5 U-Net + V8 BrushNet/IP-Adapter/fusion from %s",
        v8["checkpoint"],
    )
    logger.info(
        "Strict V8 IP validation: tensors=%d cross_attention_processors=%d",
        ip_validation["tensor_count"],
        ip_validation["cross_attention_processors"],
    )
    return pipe, ip_conditioner, stage1_metadata, weight_dtype


def image_list_to_tensor(images: Sequence[Image.Image]) -> torch.Tensor:
    arrays = [np.asarray(image, dtype=np.float32) / 255.0 for image in images]
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)


def composite_input_roi(
    generated: Sequence[Image.Image],
    decoded_frames: torch.Tensor,
    roi_masks: torch.Tensor,
) -> List[Image.Image]:
    """Hard composite: decoded input in ROI and generated image in BG."""

    generated_tensor = image_list_to_tensor(generated)
    decoded = ((decoded_frames.float().cpu() + 1.0) * 0.5).clamp(0, 1)
    roi = roi_masks.float().cpu().clamp(0, 1)
    final = decoded * roi + generated_tensor * (1.0 - roi)
    arrays = (
        (final * 255).round().byte().permute(0, 2, 3, 1).numpy()
    )
    return [Image.fromarray(array, mode="RGB") for array in arrays]


def gaussian_soft_composite_input_roi(
    generated: Sequence[Image.Image],
    decoded_frames: torch.Tensor,
    roi_masks: torch.Tensor,
    kernel_size: int = 21,
    sigma: float = 0.0,
) -> List[Image.Image]:
    """Apply the legacy test script's Gaussian soft blending at the ROI seam.

    The generated background is kept unchanged. Deep inside the ROI, pixels
    come from the decoded input. A Gaussian band on the inner ROI boundary
    smoothly blends the input into the generated reconstruction.
    """

    kernel_size = int(kernel_size)
    sigma = float(sigma)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            "gaussian_kernel_size must be a positive odd integer"
        )
    if sigma < 0:
        raise ValueError("gaussian_sigma must be non-negative")

    generated_tensor = image_list_to_tensor(generated)
    decoded = ((decoded_frames.float().cpu() + 1.0) * 0.5).clamp(0, 1)
    roi = roi_masks.float().cpu().clamp(0, 1)
    background = 1.0 - roi
    blurred_background = []
    for frame_background in background:
        blurred = cv2.GaussianBlur(
            frame_background[0].numpy(),
            (kernel_size, kernel_size),
            sigmaX=sigma,
            sigmaY=sigma,
        )
        blurred_background.append(torch.from_numpy(blurred).unsqueeze(0))
    blurred_background = torch.stack(blurred_background).to(
        dtype=generated_tensor.dtype
    )

    # Same asymmetric alpha as the old null-text evaluator:
    #   alpha_generated = 1 - ROI * (1 - GaussianBlur(BG)).
    # It is 1 throughout BG, 0 in the ROI interior and fractional only at the
    # inner ROI seam, so decoded BG pixels never leak into generated BG.
    generated_alpha = 1.0 - roi * (1.0 - blurred_background)
    input_roi = decoded * roi
    final = input_roi * (1.0 - generated_alpha) + generated_tensor * generated_alpha
    arrays = (final.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(array, mode="RGB") for array in arrays]


def postprocess_input_roi(
    generated: Sequence[Image.Image],
    decoded_frames: torch.Tensor,
    roi_masks: torch.Tensor,
    inference_config: Mapping,
) -> List[Image.Image]:
    if not bool(inference_config.get("composite_input_roi", True)):
        return list(generated)
    mode = str(inference_config.get("roi_blending", "gaussian_soft")).lower()
    if mode == "hard":
        return composite_input_roi(generated, decoded_frames, roi_masks)
    if mode == "gaussian_soft":
        return gaussian_soft_composite_input_roi(
            generated,
            decoded_frames,
            roi_masks,
            kernel_size=int(inference_config.get("gaussian_kernel_size", 21)),
            sigma=float(inference_config.get("gaussian_sigma", 0.0)),
        )
    raise ValueError(
        "inference.roi_blending must be 'hard' or 'gaussian_soft', "
        f"got {mode!r}"
    )


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    expanded = mask.expand_as(value)
    denominator = expanded.sum().clamp_min(1.0)
    return float((value * expanded).sum() / denominator)


def compute_metrics(
    generated: Sequence[Image.Image],
    gt_frames: torch.Tensor,
    roi_masks: torch.Tensor,
) -> Dict[str, float]:
    prediction = image_list_to_tensor(generated)
    target = ((gt_frames.float().cpu() + 1.0) * 0.5).clamp(0, 1)
    roi = roi_masks.float().cpu().clamp(0, 1)
    background = 1.0 - roi
    squared = (prediction - target).square()
    absolute = (prediction - target).abs()
    mse = float(squared.mean())
    bg_mse = masked_mean(squared, background)
    roi_mse = masked_mean(squared, roi)

    def psnr(value):
        return 100.0 if value <= 1e-12 else -10.0 * math.log10(value)

    metrics = {
        "mse": mse,
        "psnr": psnr(mse),
        "l1": float(absolute.mean()),
        "bg_mse": bg_mse,
        "bg_psnr": psnr(bg_mse),
        "bg_l1": masked_mean(absolute, background),
        "roi_mse": roi_mse,
        "roi_psnr": psnr(roi_mse),
        "roi_l1": masked_mean(absolute, roi),
    }
    if prediction.shape[0] > 1:
        prediction_delta = prediction[1:] - prediction[:-1]
        target_delta = target[1:] - target[:-1]
        metrics["temporal_delta_l1"] = float(
            (prediction_delta - target_delta).abs().mean()
        )
    return metrics


def save_images(images: Iterable[Image.Image], directory: Path, indices):
    directory.mkdir(parents=True, exist_ok=True)
    for image, index in zip(images, indices):
        image.save(directory / f"{int(index):06d}.png")


def save_video(
    images: Sequence[Image.Image],
    path: Path,
    fps: float,
    codec: str = "mp4v",
):
    """Save one evaluated clip as MP4 alongside the lossless PNG frames."""

    if not images:
        raise ValueError("Cannot save an empty video")
    fps = float(fps)
    if fps <= 0:
        raise ValueError("Video FPS must be positive")
    codec = str(codec)
    if len(codec) != 4:
        raise ValueError("Video codec must be a four-character code")
    width, height = images[0].size
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    try:
        for image in images:
            if image.size != (width, height):
                raise ValueError("All video frames must have the same dimensions")
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def image_file_is_readable(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, ValueError):
        return False
    return True


def video_file_is_readable(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
        return bool(ok and frame is not None and frame.size > 0)
    finally:
        capture.release()


def clip_output_is_complete(
    metrics_path: Path,
    clip_dir: Path,
    frame_indices: Sequence[int],
    require_raw: bool,
    require_video: bool = False,
    require_raw_video: bool = False,
) -> bool:
    if not metrics_path.is_file():
        return False
    try:
        record = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if [int(value) for value in record.get("frame_indices", [])] != [
        int(value) for value in frame_indices
    ]:
        return False
    directories = [clip_dir / "final"]
    if require_raw:
        directories.append(clip_dir / "raw")
    images_complete = all(
        image_file_is_readable(directory / f"{int(index):06d}.png")
        for directory in directories
        for index in frame_indices
    )
    if not images_complete:
        return False
    if require_video and not video_file_is_readable(clip_dir / "final.mp4"):
        return False
    if require_raw_video and not video_file_is_readable(clip_dir / "raw.mp4"):
        return False
    return True


def aggregate_metrics(run_dir: Path) -> Dict:
    records = []
    for path in sorted(run_dir.rglob("clip_metrics.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    scalar_keys = sorted(
        {
            key
            for record in records
            for key, value in record.get("metrics", {}).items()
            if isinstance(value, (int, float))
        }
    )
    mean = {
        key: float(np.mean([record["metrics"][key] for record in records]))
        for key in scalar_keys
        if all(key in record.get("metrics", {}) for record in records)
    }
    summary = {"clips": len(records), "mean": mean}
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def apply_cli_overrides(config: Mapping, args) -> Dict:
    config = copy.deepcopy(dict(config))
    if args.fixed_beta is not None:
        config.setdefault("noise_shaping", {})["fixed_beta"] = args.fixed_beta
    if args.split is not None:
        config.setdefault("data", {})["split"] = args.split
    if args.max_clips is not None:
        config.setdefault("inference", {})["max_clips"] = args.max_clips
    if args.device is not None:
        config.setdefault("inference", {})["device"] = args.device
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.overwrite:
        config.setdefault("inference", {})["overwrite"] = True
    return config


def comparable_run_config(config: Mapping) -> Dict:
    """Remove controls that change run extent, not generated clip content."""

    comparable = copy.deepcopy(dict(config))
    inference = comparable.setdefault("inference", {})
    inference.pop("overwrite", None)
    inference.pop("max_clips", None)
    return comparable


def main():
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = apply_cli_overrides(config, args)
    preflight = validate_fixed_test1_config(config)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return

    # Pin the pointer once at process start. If Stage-1 training updates
    # best.json while evaluation is running, this run remains reproducible.
    config["model"]["stage1_checkpoint"] = preflight["stage1_checkpoint"]
    config["model"]["stage1_step"] = preflight.get("stage1_step")
    config["model"]["stage1_valid_epe"] = preflight.get("stage1_valid_epe")

    seed = int(config.get("seed", 2026))
    beta = float(config["noise_shaping"]["fixed_beta"])
    run_dir = (
        Path(config["output_dir"]).expanduser().resolve()
        / beta_directory_name(beta)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "test1.log")
    resolved_config_path = run_dir / "resolved_config.json"
    if resolved_config_path.is_file():
        previous_config = json.loads(
            resolved_config_path.read_text(encoding="utf-8")
        )
        if comparable_run_config(previous_config) != comparable_run_config(config):
            raise RuntimeError(
                f"Refusing to mix a different configuration into {run_dir}. "
                "Choose a new --output-dir for the new run."
            )
    else:
        resolved_config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (run_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if preflight["coverage"]["empty_sequences"]:
        logger.warning(
            "Split %s has no frames for: %s",
            preflight["coverage"]["split"],
            ", ".join(preflight["coverage"]["empty_sequences"]),
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    requested_device = str(config["inference"].get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(requested_device)
    pipe, ip_conditioner, stage1_metadata, _ = build_pipeline(
        config, device, logger
    )
    metadata = {
        "experiment": "fixed_beta_test1",
        "description": "Stage1 flow + fixed-beta shaped z_T; no Stage3 adapter",
        "stage1": stage1_metadata,
        "preflight": preflight,
        "trainable_parameters": 0,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    data_config = copy.deepcopy(config["data"])
    split = str(data_config.pop("split", "test"))
    data_config.pop("teacher_flow_root", None)
    data_config["load_gt"] = True
    data_config["random_horizontal_flip"] = False
    dataset = STCFlowDataset(data_config, split=split)
    max_clips = int(config["inference"].get("max_clips", 0))
    limit = len(dataset) if max_clips <= 0 else min(max_clips, len(dataset))
    logger.info(
        "split=%s clips=%d/%d T=%d beta=%.3f steps=%d device=%s",
        split,
        limit,
        len(dataset),
        dataset.clip_length,
        beta,
        int(config["inference"].get("num_inference_steps", 50)),
        device,
    )

    overwrite = bool(config["inference"].get("overwrite", False))
    save_raw = bool(config["inference"].get("save_raw_generation", False))
    save_output_video = bool(config["inference"].get("save_video", True))
    save_raw_video = bool(
        config["inference"].get("save_raw_video", False) and save_raw
    )
    video_codec = str(config["inference"].get("video_codec", "mp4v"))
    do_composite = bool(config["inference"].get("composite_input_roi", True))
    roi_blending = str(
        config["inference"].get("roi_blending", "gaussian_soft")
    ).lower()
    started = time.time()
    completed, skipped = 0, 0
    for item_index in range(limit):
        sample = dataset[item_index]
        class_name = str(sample["class_name"])
        sequence = str(sample["sequence"])
        frame_indices = [int(value) for value in sample["frame_indices"].tolist()]
        clip_dir = (
            run_dir
            / "clips"
            / class_name
            / sequence
            / f"clip-{frame_indices[0]:06d}-{frame_indices[-1]:06d}"
        )
        metrics_path = clip_dir / "clip_metrics.json"
        if not overwrite and clip_output_is_complete(
            metrics_path,
            clip_dir,
            frame_indices,
            require_raw=save_raw,
            require_video=save_output_video,
            require_raw_video=save_raw_video,
        ):
            skipped += 1
            continue
        if metrics_path.is_file() and not overwrite:
            logger.warning(
                "Incomplete/corrupt resumed clip will be regenerated: %s",
                clip_dir,
            )

        decoded = sample["decoded_frames"]
        gt = sample["gt_frames"]
        roi = sample["roi_masks"]
        input_images = tensor_frames_to_pil(decoded)
        mask_images = roi_masks_to_pil(roi)
        clip_seed = deterministic_clip_seed(
            seed, class_name, sequence, frame_indices[0]
        )
        torch.manual_seed(clip_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(clip_seed)
        generator = torch.Generator(device=device).manual_seed(clip_seed)
        prompt_embeds, negative_prompt_embeds = build_v8_prompt_embeddings(
            pipe,
            ip_conditioner,
            input_images,
            roi,
            prompt=str(config["data"].get("caption", "")),
            negative_prompt=str(config["inference"].get("negative_prompt", "")),
            fusion_scale=float(config["ip_adapter"].get("fusion_scale", 1.0)),
            v8_mask_order=bool(config["ip_adapter"].get("v8_mask_order", True)),
        )
        result = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=input_images,
            mask=mask_images,
            height=int(config["data"].get("height", 512)),
            width=int(config["data"].get("width", 512)),
            num_inference_steps=int(
                config["inference"].get("num_inference_steps", 50)
            ),
            guidance_scale=float(config["inference"].get("guidance_scale", 7.5)),
            generator=generator,
            num_images_per_prompt=1,
            brushnet_conditioning_scale=float(
                config["inference"].get("brushnet_conditioning_scale", 1.0)
            ),
            use_shared_bg_noise=False,
            use_stc_noise_shaper=True,
            noise_shaper_strength=float(
                config["noise_shaping"].get("strength", 1.0)
            ),
            noise_shaper_clip_length=dataset.clip_length,
        )
        raw_images = result.images
        if len(raw_images) != dataset.clip_length:
            raise RuntimeError(
                f"Pipeline returned {len(raw_images)} images for T={dataset.clip_length}"
            )
        final_images = postprocess_input_roi(
            raw_images,
            decoded,
            roi,
            config["inference"],
        )
        save_images(final_images, clip_dir / "final", frame_indices)
        fps = float(sample["fps"])
        if save_output_video:
            save_video(
                final_images,
                clip_dir / "final.mp4",
                fps=fps,
                codec=video_codec,
            )
        if save_raw:
            save_images(raw_images, clip_dir / "raw", frame_indices)
            if save_raw_video:
                save_video(
                    raw_images,
                    clip_dir / "raw.mp4",
                    fps=fps,
                    codec=video_codec,
                )
        record = {
            "class_name": class_name,
            "sequence": sequence,
            "frame_indices": frame_indices,
            "seed": clip_seed,
            "fixed_beta": beta,
            "composite_input_roi": do_composite,
            "roi_blending": roi_blending if do_composite else "disabled",
            "gaussian_kernel_size": int(
                config["inference"].get("gaussian_kernel_size", 21)
            ),
            "gaussian_sigma": float(
                config["inference"].get("gaussian_sigma", 0.0)
            ),
            "fps": fps,
            "outputs": {
                "final_frames": str(clip_dir / "final"),
                "final_video": (
                    str(clip_dir / "final.mp4") if save_output_video else None
                ),
                "raw_frames": str(clip_dir / "raw") if save_raw else None,
                "raw_video": str(clip_dir / "raw.mp4") if save_raw_video else None,
            },
            "metrics": compute_metrics(final_images, gt, roi),
            "noise_shaper": dict(pipe._last_noise_shaper_stats or {}),
        }
        clip_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        completed += 1
        logger.info(
            "clip=%d/%d %s/%s %d..%d psnr=%.3f bg_psnr=%.3f "
            "temporal_delta_l1=%.6f beta_eff=%.4f",
            item_index + 1,
            limit,
            class_name,
            sequence,
            frame_indices[0],
            frame_indices[-1],
            record["metrics"]["psnr"],
            record["metrics"]["bg_psnr"],
            record["metrics"].get("temporal_delta_l1", float("nan")),
            record["noise_shaper"].get("effective_beta_mean", float("nan")),
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = aggregate_metrics(run_dir)
    logger.info(
        "Finished completed=%d skipped=%d total_metrics=%d elapsed=%.1fs summary=%s",
        completed,
        skipped,
        summary["clips"],
        time.time() - started,
        run_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
