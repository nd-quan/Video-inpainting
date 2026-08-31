#!/usr/bin/env python
"""Evaluate a trained RGB-STC adapter with the frozen V8 BrushNet stack.

This evaluator deliberately does not call FusionIPAdapter.generate_fgbg().
That helper omits V8's base-CLIP embedding, whereas phase-1 RGB-STC was trained
with ``base_CLIP(full_input) + fusion(CLIP(BG_only), CLIP(ROI_only))``.  The
script reproduces that condition and supplies RGB-STC's precomputed five-channel
BrushNet condition to the shared-background-noise DDIM pipeline.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection


BRUSHNET_DIR = Path(__file__).resolve().parent.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from diffusers import DDIMScheduler  # noqa: E402
from diffusers.models.brushnet import BrushNetModel  # noqa: E402
from diffusers.pipelines.brushnet.pipeline_sharedNoiseBG_org import (  # noqa: E402
    StableDiffusionBrushNetPipeline,
)
from ip_adapter.ip_adapter import ImageProjModel  # noqa: E402
from shared_bg_noise_training import (  # noqa: E402
    FlatV8TestClipDataset,
    HierarchicalV8ClipDataset,
)
from STC_encoder_v2_rgb.frozen_v8 import (  # noqa: E402
    install_and_load_ip_adapter,
    load_fusion_module,
)
from STC_encoder_v2_rgb.rgb_stc_adapter import (  # noqa: E402
    RGBSTCConditionAdapter,
    augment_brushnet_condition,
)


PROJECT_ROOT = BRUSHNET_DIR.parent.parent
DEFAULT_BASE_MODEL = (
    BRUSHNET_DIR / "base_model" / "stable-diffusion-v1-5" / "stable-diffusion-v1-5"
)
DEFAULT_BASELINE = (
    PROJECT_ROOT / "experiments" / "train_sharedNoise_sameBG_0.9" / "checkpoint-2000"
)
DEFAULT_ADAPTER = (
    PROJECT_ROOT / "experiments" / "train_rgb_stc_v2_sharedNoise_0.9" / "stc_adapter"
)
DEFAULT_DATASET = Path("/home/cilab/ndquan/videoInpainting/SFU_STC_flow")
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "experiments" / "eval_rgb_stc_v2" / "valid" / "checkpoint-2000-sTCE-0.7-sBG-1"
)
DEFAULT_IMAGE_ENCODER = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

# Optional variant hooks. Existing V2--V5 entry points leave both unset and
# therefore preserve their argument and conditioning contracts.
ADD_EVALUATION_ARGUMENTS_FN: Optional[Callable] = None
CONDITION_EXTRA_KWARGS_FN: Optional[Callable] = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RGB-STC with exact V8 image/fusion conditioning."
    )
    parser.add_argument("--pretrained_model_name_or_path", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--baseline_checkpoint", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--stc_adapter_path", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--split",
        default="valid",
        help="Dataset split directory below --dataset_root (for example valid, test, or long_test).",
    )
    parser.add_argument(
        "--dataset_layout",
        choices=("auto", "hierarchical", "flat_test"),
        default="auto",
        help=(
            "Dataset layout. auto detects either the SFU hierarchy "
            "root/<split>/{GT,input,mask}/... or the legacy BrushNet test "
            "layout root/<sequence>/{gt,inputs,masks}/..."
        ),
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include_branches",
        nargs="+",
        metavar="CLASS/SEQUENCE",
        help=(
            "Evaluate only these branches of a hierarchical dataset, for example "
            "Class_A/Traffic Class_D/BasketballPass."
        ),
    )
    parser.add_argument("--image_encoder_name_or_path", default=DEFAULT_IMAGE_ENCODER)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--clip_length", type=int, default=8)
    parser.add_argument("--clip_stride", type=int, default=6)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--brushnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--stc_injection_scale", type=float, default=1.0)
    parser.add_argument("--fusion_scale", type=float, default=1.0)
    parser.add_argument("--shared_bg_noise_strength", type=float, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shared_bg_seed", type=int, default=6789)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_clips", type=int, default=None)
    parser.add_argument(
        "--roi_composite",
        choices=("none", "hard", "blurred"),
        default="hard",
    )
    parser.add_argument(
        "--roi_blur_kernel_size",
        type=int,
        default=21,
        help=(
            "Odd Gaussian kernel used by --roi_composite blurred. The formula "
            "matches test_brushnet_*_fusion_base.py; default: 21."
        ),
    )
    parser.add_argument("--save_references", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    if ADD_EVALUATION_ARGUMENTS_FN is not None:
        ADD_EVALUATION_ARGUMENTS_FN(parser)
    args = parser.parse_args()
    if args.resolution <= 0 or args.resolution % 8:
        parser.error("--resolution must be positive and divisible by 8")
    if args.clip_length < 2 or args.clip_stride < 1:
        parser.error("clip_length must be >=2 and clip_stride must be positive")
    if args.num_inference_steps < 1:
        parser.error("--num_inference_steps must be positive")
    if args.max_clips is not None and args.max_clips < 1:
        parser.error("--max_clips must be positive")
    if not 0.0 <= args.shared_bg_noise_strength <= 1.0:
        parser.error("--shared_bg_noise_strength must be in [0,1]")
    if args.roi_blur_kernel_size < 1 or args.roi_blur_kernel_size % 2 == 0:
        parser.error("--roi_blur_kernel_size must be a positive odd integer")
    return args


def require_path(path: Path, kind: str):
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")


def paths_for_baseline(root: Path) -> Dict[str, Path]:
    root = root.expanduser().resolve()
    return {
        "root": root,
        "brushnet": root / "brushnet",
        "ip_adapter": root / "ipadapter" / "model.safetensors",
        "fusion": root / "ipadapter" / "fusion_module.safetensors",
    }


def resolve_dataset_layout(args) -> str:
    hierarchical = all(
        (args.dataset_root / args.split / kind).is_dir()
        for kind in ("GT", "input", "mask")
    )
    flat_test = any(
        (sequence / "gt").is_dir()
        and (sequence / "inputs").is_dir()
        and (sequence / "masks").is_dir()
        for sequence in args.dataset_root.iterdir()
        if sequence.is_dir()
    )
    if args.dataset_layout == "auto":
        if hierarchical:
            return "hierarchical"
        if flat_test:
            return "flat_test"
        raise FileNotFoundError(
            "Could not detect a supported dataset layout below "
            f"{args.dataset_root}. Expected either root/{args.split}/{{GT,input,mask}} "
            "or root/<sequence>/{gt,inputs,masks}."
        )
    if args.dataset_layout == "hierarchical" and not hierarchical:
        raise FileNotFoundError(
            "Hierarchical layout requires "
            f"{args.dataset_root / args.split}/{{GT,input,mask}}"
        )
    if args.dataset_layout == "flat_test" and not flat_test:
        raise FileNotFoundError(
            "flat_test layout requires at least one "
            "root/<sequence>/{gt,inputs,masks} triplet"
        )
    return args.dataset_layout


def preflight(args) -> Tuple[HierarchicalV8ClipDataset, Dict[str, Path]]:
    args.pretrained_model_name_or_path = args.pretrained_model_name_or_path.expanduser().resolve()
    args.baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
    args.stc_adapter_path = args.stc_adapter_path.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    paths = paths_for_baseline(args.baseline_checkpoint)
    require_path(args.pretrained_model_name_or_path, "dir")
    for key in ("brushnet", "ip_adapter", "fusion"):
        require_path(paths[key], "dir" if key == "brushnet" else "file")
    require_path(args.stc_adapter_path, "dir")
    require_path(args.stc_adapter_path / "config.json", "file")
    require_path(args.stc_adapter_path / "diffusion_pytorch_model.safetensors", "file")
    require_path(args.dataset_root, "dir")
    dataset_layout = resolve_dataset_layout(args)

    # Dataset needs a tokenizer only to create null-text token IDs. The
    # evaluation pipeline already owns this exact tokenizer.
    from transformers import AutoTokenizer, CLIPImageProcessor

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.pretrained_model_name_or_path), subfolder="tokenizer", use_fast=False
    )
    dataset_class = (
        HierarchicalV8ClipDataset
        if dataset_layout == "hierarchical"
        else FlatV8TestClipDataset
    )
    dataset = dataset_class(
        dataset_root=args.dataset_root,
        split=args.split,
        tokenizer=tokenizer,
        clip_image_processor=CLIPImageProcessor(),
        clip_length=args.clip_length,
        stride=args.clip_stride,
        resolution=args.resolution,
        **(
            {"include_branches": args.include_branches}
            if dataset_layout == "hierarchical"
            else {}
        ),
    )
    adapter = RGBSTCConditionAdapter.from_pretrained(str(args.stc_adapter_path))
    if adapter.config.condition_mode != "full_rgb_bg_mask":
        raise ValueError(
            "This evaluator implements the primary full_rgb_bg_mask experiment, "
            f"not {adapter.config.condition_mode!r}."
        )
    report = {
        "status": "ok",
        "split": args.split,
        "dataset_layout": dataset_layout,
        "clip_count": len(dataset),
        "source_frame_count": dataset.frame_count,
        "covered_frame_count": dataset.covered_frame_count,
        "branch_count": dataset.branch_count,
        "included_branches": list(getattr(dataset, "include_branches", ())),
        "adapter": str(args.stc_adapter_path),
        "baseline": str(paths["root"]),
        "mask_semantics": "raw=0_BG/255_ROI; internal M_BG=1/0",
        "condition": "[z_input + delta_z_BG, M_BG]",
    }
    if hasattr(dataset, "ignored_gt_frame_count"):
        report["ignored_gt_frame_count"] = dataset.ignored_gt_frame_count
        report["total_gt_frame_count"] = dataset.total_gt_frame_count
    print(json.dumps(report, indent=2, sort_keys=True))
    return dataset, paths


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = ":".join([str(int(base_seed))] + [str(part) for part in parts])
    digest = hashlib.blake2b(
        payload.encode("utf-8"), digest_size=8, person=b"RGBSTCEval"
    ).digest()
    return int.from_bytes(digest, byteorder="little") % (2**63 - 1)


def tensor_to_pil(frames: torch.Tensor) -> List[Image.Image]:
    arrays = (
        ((frames.detach().float().cpu() + 1.0) * 127.5)
        .round()
        .clamp(0, 255)
        .byte()
        .permute(0, 2, 3, 1)
        .numpy()
    )
    return [Image.fromarray(array, mode="RGB") for array in arrays]


def masks_to_pil(roi_masks: torch.Tensor) -> List[Image.Image]:
    arrays = (
        (roi_masks.detach().float().cpu().clamp(0, 1)[:, 0] * 255)
        .round()
        .byte()
        .numpy()
    )
    return [Image.fromarray(array, mode="L").convert("RGB") for array in arrays]


def pil_to_tensor(images: Sequence[Image.Image]) -> torch.Tensor:
    arrays = [np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0 for image in images]
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    expanded = mask.expand_as(value)
    return float((value * expanded).sum() / expanded.sum().clamp_min(1.0))


def psnr(mse: float) -> float:
    return 100.0 if mse <= 1e-12 else -10.0 * math.log10(mse)


def compute_metrics(
    output_images: Sequence[Image.Image], gt: torch.Tensor, bg_mask: torch.Tensor
) -> Dict[str, float]:
    prediction = pil_to_tensor(output_images)
    target = ((gt.detach().float().cpu() + 1.0) * 0.5).clamp(0, 1)
    bg = bg_mask.detach().float().cpu().clamp(0, 1)
    roi = 1.0 - bg
    squared = (prediction - target).square()
    absolute = (prediction - target).abs()
    bg_mse = masked_mean(squared, bg)
    roi_mse = masked_mean(squared, roi)
    metrics = {
        "mse": float(squared.mean()),
        "psnr": psnr(float(squared.mean())),
        "l1": float(absolute.mean()),
        "bg_mse": bg_mse,
        "bg_psnr": psnr(bg_mse),
        "bg_l1": masked_mean(absolute, bg),
        "roi_mse": roi_mse,
        "roi_psnr": psnr(roi_mse),
        "roi_l1": masked_mean(absolute, roi),
    }
    if prediction.shape[0] > 1:
        pair_bg = bg[1:] * bg[:-1]
        temporal_error = (prediction[1:] - prediction[:-1] - target[1:] + target[:-1]).abs()
        metrics["temporal_delta_l1_bg"] = masked_mean(temporal_error, pair_bg)
    return metrics


def hard_composite(
    generated: Sequence[Image.Image], input_frames: torch.Tensor, bg_mask: torch.Tensor
) -> List[Image.Image]:
    generated_tensor = pil_to_tensor(generated)
    input_tensor = ((input_frames.detach().float().cpu() + 1.0) * 0.5).clamp(0, 1)
    bg = bg_mask.detach().float().cpu().clamp(0, 1)
    output = generated_tensor * bg + input_tensor * (1.0 - bg)
    arrays = (output.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(array, mode="RGB") for array in arrays]


def blurred_composite(
    generated: Sequence[Image.Image],
    input_frames: torch.Tensor,
    bg_mask: torch.Tensor,
    kernel_size: int = 21,
) -> List[Image.Image]:
    """Blend generated BG into the HQ ROI with the historical soft boundary.

    ``bg_mask`` follows this repo's internal convention: 1 is degraded BG to
    restore and 0 is the high-quality ROI to preserve.  This reproduces the
    active blending path in
    ``test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion_base.py``::

        blurred = GaussianBlur(M_BG, (k, k), sigmaX=0)
        M_soft = 1 - (1 - M_BG) * (1 - blurred)
        output = input_ROI * (1 - M_soft) + generated * M_soft

    OpenCV and uint8 truncation are intentionally retained to match that
    historical inference implementation rather than introducing a different
    torch/PIL blur convention.
    """
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if len(generated) != int(input_frames.shape[0]) or len(generated) != int(
        bg_mask.shape[0]
    ):
        raise ValueError("generated, input_frames, and bg_mask batch sizes must match")

    input_images = tensor_to_pil(input_frames)
    bg_masks = bg_mask.detach().float().cpu().clamp(0, 1)
    outputs: List[Image.Image] = []
    for index, (generated_image, input_image) in enumerate(
        zip(generated, input_images)
    ):
        generated_array = np.asarray(
            generated_image.convert("RGB"), dtype=np.uint8
        ).astype(np.float32)
        input_array = np.asarray(input_image, dtype=np.uint8).astype(np.float32)
        bg = bg_masks[index, 0].numpy().astype(np.float32, copy=False)
        if generated_array.shape[:2] != bg.shape or input_array.shape[:2] != bg.shape:
            raise ValueError(
                "Generated/input/mask spatial sizes must match for blurred composite"
            )

        blurred = cv2.GaussianBlur(
            bg * 255.0,
            (int(kernel_size), int(kernel_size)),
            0,
        ) / 255.0
        soft_bg = 1.0 - (1.0 - bg) * (1.0 - blurred)
        soft_bg = soft_bg[..., None]
        input_roi = input_array * (1.0 - bg[..., None])
        pasted = input_roi * (1.0 - soft_bg) + generated_array * soft_bg
        outputs.append(
            Image.fromarray(np.clip(pasted, 0, 255).astype(np.uint8), mode="RGB")
        )
    return outputs


def composite_images(
    generated: Sequence[Image.Image],
    input_frames: torch.Tensor,
    bg_mask: torch.Tensor,
    mode: str,
    blur_kernel_size: int = 21,
) -> List[Image.Image]:
    if mode == "none":
        return list(generated)
    if mode == "hard":
        return hard_composite(generated, input_frames, bg_mask)
    if mode == "blurred":
        return blurred_composite(
            generated,
            input_frames,
            bg_mask,
            kernel_size=blur_kernel_size,
        )
    raise ValueError(f"Unsupported ROI composite mode: {mode!r}")


def save_images(images: Sequence[Image.Image], directory: Path, frame_ids: Sequence[int]):
    directory.mkdir(parents=True, exist_ok=True)
    for image, frame_id in zip(images, frame_ids):
        image.save(directory / f"{int(frame_id):06d}.png")


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def load_models(args, paths: Mapping[str, Path], device: torch.device):
    if device.type != "cuda":
        raise RuntimeError("RGB-STC evaluation is configured for CUDA inference")
    dtype = torch.float16
    brushnet = BrushNetModel.from_pretrained(str(paths["brushnet"]), torch_dtype=dtype)
    pipe = StableDiffusionBrushNetPipeline.from_pretrained(
        str(args.pretrained_model_name_or_path),
        brushnet=brushnet,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.image_encoder_name_or_path
    ).to(device=device, dtype=dtype)
    image_proj_model = ImageProjModel(
        cross_attention_dim=pipe.unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )
    ip_report = install_and_load_ip_adapter(
        pipe.unet, image_proj_model, paths["ip_adapter"]
    )
    # IP processors are installed after the pipeline has moved to CUDA, so
    # explicitly move the newly created processor modules with the U-Net.
    pipe.unet.to(device=device, dtype=dtype)
    image_proj_model.to(device=device, dtype=dtype)
    fusion_module = load_fusion_module(
        paths["fusion"], embed_dim=image_encoder.config.projection_dim
    ).to(device=device, dtype=dtype)
    adapter = RGBSTCConditionAdapter.from_pretrained(
        str(args.stc_adapter_path)
    ).to(device=device, dtype=torch.float32)

    for module in (
        pipe.vae,
        pipe.text_encoder,
        pipe.unet,
        pipe.brushnet,
        image_encoder,
        image_proj_model,
        fusion_module,
        adapter,
    ):
        module.requires_grad_(False).eval()
    if pipe.brushnet.config.conditioning_channels != 5:
        raise ValueError("RGB-STC requires a five-channel BrushNet condition")
    return pipe, image_encoder, image_proj_model, fusion_module, adapter, ip_report


@torch.inference_mode()
def build_v8_prompt_embeddings(
    pipe,
    image_encoder,
    image_proj_model,
    fusion_module,
    sample: Mapping[str, torch.Tensor],
    device: torch.device,
    fusion_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match train-time base-CLIP plus historical BG/ROI fusion exactly."""
    dtype = next(image_encoder.parameters()).dtype
    base = image_encoder(
        sample["clip_images"].to(device=device, dtype=dtype)
    ).image_embeds
    # Historical names are intentional: fg_clip_images holds BG-only RGB and
    # bg_clip_images holds ROI-only RGB, just as in V8 training.
    historical_fg = image_encoder(
        sample["fg_clip_images"].to(device=device, dtype=dtype)
    ).image_embeds
    historical_bg = image_encoder(
        sample["bg_clip_images"].to(device=device, dtype=dtype)
    ).image_embeds
    with autocast_context(device):
        fused = fusion_module(historical_fg, historical_bg)
        image_tokens = image_proj_model(base + float(fusion_scale) * fused)
        negative_image_tokens = image_proj_model(torch.zeros_like(base))

    prompts = [""] * base.shape[0]
    text, negative_text = pipe.encode_prompt(
        prompts,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=prompts,
    )
    image_tokens = image_tokens.to(device=device, dtype=text.dtype)
    negative_image_tokens = negative_image_tokens.to(
        device=device, dtype=negative_text.dtype
    )
    return (
        torch.cat((text, image_tokens), dim=1),
        torch.cat((negative_text, negative_image_tokens), dim=1),
        text,
        negative_text,
    )


@torch.inference_mode()
def build_stc_condition(
    pipe,
    adapter,
    sample: Mapping[str, torch.Tensor],
    device: torch.device,
    condition_seed: int,
    injection_scale: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    rgb_sequence = sample["conditioning_pixel_values"].unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    bg_mask_sequence = sample["masks"].unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    frames = rgb_sequence.shape[1]
    latent_generator = torch.Generator(device=device).manual_seed(int(condition_seed))
    with autocast_context(device):
        base_condition_latents = pipe.vae.encode(
            rgb_sequence.flatten(0, 1).to(dtype=pipe.vae.dtype)
        ).latent_dist.sample(generator=latent_generator)
        base_condition_latents = base_condition_latents * pipe.vae.config.scaling_factor
        brushnet_condition, output, _ = augment_brushnet_condition(
            adapter=adapter,
            base_condition_latents=base_condition_latents,
            rgb_sequence=rgb_sequence,
            bg_mask_sequence=bg_mask_sequence,
            injection_scale=float(injection_scale),
        )
    if brushnet_condition.shape != (
        frames,
        5,
        base_condition_latents.shape[-2],
        base_condition_latents.shape[-1],
    ):
        raise RuntimeError("Unexpected RGB-STC BrushNet condition layout")
    # Pipeline uses [unconditional frames, conditional frames] under CFG.
    condition_cfg = torch.cat((brushnet_condition, brushnet_condition), dim=0)
    stats = {
        "delta_abs_mean": float(output.delta_bg.detach().float().abs().mean()),
        "latent_bg_ratio": float(output.latent_bg_mask.detach().float().mean()),
        "roi_delta_nonzero": int(
            torch.count_nonzero(
                output.delta_bg.detach()
                * (1.0 - output.latent_bg_mask.detach())
            )
        ),
    }
    if stats["roi_delta_nonzero"]:
        raise RuntimeError("RGB-STC delta leaked outside M_BG")
    return condition_cfg, stats


def clip_directory(output_dir: Path, video: str, frame_ids: Sequence[int]) -> Path:
    safe_video = str(video).replace("/", "__")
    return output_dir / f"{safe_video}__{int(frame_ids[0]):06d}-{int(frame_ids[-1]):06d}"


def aggregate_metrics(output_dir: Path) -> Dict[str, object]:
    records = []
    for path in sorted(output_dir.glob("*/clip_metrics.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    keys = sorted(
        {
            key
            for record in records
            for key, value in record.get("metrics", {}).items()
            if isinstance(value, (int, float))
        }
    )
    return {
        "clips": len(records),
        "mean": {
            key: float(np.mean([record["metrics"][key] for record in records]))
            for key in keys
            if all(key in record.get("metrics", {}) for record in records)
        },
    }


def main():
    args = parse_args()
    dataset, paths = preflight(args)
    if args.preflight_only:
        return

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Use a CUDA device for this full SD1.5 evaluation")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pipe, image_encoder, image_proj_model, fusion_module, adapter, ip_report = load_models(
        args, paths, device
    )
    (args.output_dir / "model_contract.json").write_text(
        json.dumps(
            {
                "ip_loading": ip_report,
                "adapter": str(args.stc_adapter_path),
                "v8_condition": "base_CLIP(full) + fusion(BG_only, ROI_only)",
                "brushnet_condition": "[z_input + delta_z_BG, M_BG]",
                "cfg_condition_duplication": True,
                "mask_semantics": "M_BG=1 degraded BG; M_BG=0 HQ ROI",
                "roi_composite": args.roi_composite,
                "roi_blur_kernel_size": args.roi_blur_kernel_size,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    limit = len(dataset) if args.max_clips is None else min(len(dataset), args.max_clips)
    for index in tqdm(range(limit), desc="RGB-STC evaluation"):
        sample = dataset[index]
        frame_ids = [int(value) for value in sample["frame_ids"].tolist()]
        output = clip_directory(args.output_dir, sample["video"], frame_ids)
        metric_path = output / "clip_metrics.json"
        if metric_path.is_file() and not args.overwrite:
            continue

        condition_seed = stable_seed(args.seed, "condition", sample["video"], frame_ids[0])
        initial_seed = stable_seed(args.seed, "initial", sample["video"], frame_ids[0])
        shared_seed = stable_seed(args.shared_bg_seed, "shared", sample["video"])
        (
            prompt_embeds,
            negative_prompt_embeds,
            brushnet_text,
            negative_brushnet_text,
        ) = build_v8_prompt_embeddings(
            pipe,
            image_encoder,
            image_proj_model,
            fusion_module,
            sample,
            device,
            args.fusion_scale,
        )
        condition_extra_kwargs = {}
        if CONDITION_EXTRA_KWARGS_FN is not None:
            condition_extra_kwargs = CONDITION_EXTRA_KWARGS_FN(args)
        brushnet_condition, condition_stats = build_stc_condition(
            pipe,
            adapter,
            sample,
            device,
            condition_seed,
            args.stc_injection_scale,
            **condition_extra_kwargs,
        )
        input_frames = sample["conditioning_pixel_values"]
        bg_mask = sample["masks"]
        roi_mask = 1.0 - bg_mask
        input_images = tensor_to_pil(input_frames)
        raw_roi_masks = masks_to_pil(roi_mask)
        latent_shape = (
            args.clip_length,
            pipe.unet.config.in_channels,
            args.resolution // pipe.vae_scale_factor,
            args.resolution // pipe.vae_scale_factor,
        )
        independent_noise = torch.randn(
            latent_shape,
            generator=torch.Generator(device=device).manual_seed(initial_seed),
            device=device,
            dtype=pipe.unet.dtype,
        )
        shared_noise = torch.randn(
            (1, *latent_shape[1:]),
            generator=torch.Generator(device=device).manual_seed(shared_seed),
            device=device,
            dtype=pipe.unet.dtype,
        )
        generated = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=input_images,
            mask=raw_roi_masks,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=float(args.guidance_scale),
            latents=independent_noise,
            use_shared_bg_noise=True,
            shared_bg_noise=shared_noise,
            shared_bg_noise_strength=float(args.shared_bg_noise_strength),
            variance_preserving_shared_noise=True,
            brushnet_condition=brushnet_condition,
            brushnet_prompt_embeds=torch.cat(
                (negative_brushnet_text, brushnet_text), dim=0
            ),
            brushnet_conditioning_scale=float(args.brushnet_conditioning_scale),
        ).images
        final = composite_images(
            generated,
            input_frames,
            bg_mask,
            mode=args.roi_composite,
            blur_kernel_size=args.roi_blur_kernel_size,
        )
        metrics = compute_metrics(final, sample["pixel_values"], bg_mask)
        metrics.update({f"raw_{key}": value for key, value in compute_metrics(generated, sample["pixel_values"], bg_mask).items()})
        metrics.update(condition_stats)
        output.mkdir(parents=True, exist_ok=True)
        save_images(generated, output / "raw", frame_ids)
        save_images(final, output / "final", frame_ids)
        if args.save_references:
            save_images(input_images, output / "input", frame_ids)
            save_images(tensor_to_pil(sample["pixel_values"]), output / "gt", frame_ids)
            save_images(raw_roi_masks, output / "mask_roi", frame_ids)
        metric_path.write_text(
            json.dumps(
                {
                    "clip_index": index,
                    "video": sample["video"],
                    "frame_ids": frame_ids,
                    "condition_seeds": {
                        "condition": condition_seed,
                        "initial": initial_seed,
                        "shared": shared_seed,
                    },
                    "metrics": metrics,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        del (
            generated,
            final,
            brushnet_condition,
            prompt_embeds,
            negative_prompt_embeds,
            brushnet_text,
            negative_brushnet_text,
        )
        torch.cuda.empty_cache()

    summary = aggregate_metrics(args.output_dir)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
