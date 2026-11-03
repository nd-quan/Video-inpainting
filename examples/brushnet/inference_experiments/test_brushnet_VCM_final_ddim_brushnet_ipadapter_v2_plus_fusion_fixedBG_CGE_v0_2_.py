"""CGE inference for the null-text shared-noise BrushNet restoration model.

This script uses the same ``dataset/test_1/<sequence>/inputs,masks`` loader
as the null-text test script, but installs ``CustomDDIMScheduler`` and the
VCM-RS dual-region codec required by CGE. It never loads ``caption.txt``:
the text and negative-text conditions both default to ``""``.

For each CGE evaluation, white ROI pixels use VCM-RS QP20 and black background
pixels use QP52. The full-frame reconstructions are hard-composited by the
same nearest-neighbour 512x512 mask used by BrushNet.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Optional, Sequence, Tuple

from sfu_long_test_loader import load_sequences, parse_sequence_names, print_preflight


DEFAULT_TEST_ROOT = Path(
    "/home/cilab/ndquan/videoInpainting/code/BrushNet/examples/brushnet/dataset/test_1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/"
    "eval_sharednoise_cge/test_1_checkpoint-2250"
)
DEFAULT_BASE_MODEL = Path(
    "/home/cilab/ndquan/videoInpainting/code/BrushNet/examples/brushnet/"
    "base_model/stable-diffusion-v1-5/stable-diffusion-v1-5"
)
DEFAULT_CHECKPOINT_DIR = Path(
    "/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/"
    "train_sharedNoise_sameBG_0.95_T8/checkpoint-2250"
)
DEFAULT_IMAGE_ENCODER = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


def run_early_preflight() -> None:
    """Validate the dataset without importing diffusion/VCM-RS dependencies."""

    if "--preflight_only" not in os.sys.argv:
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--long_test_root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sequences", default=None)
    parser.add_argument("--preflight_only", action="store_true")
    args, _ = parser.parse_known_args()
    sequences = load_sequences(
        args.long_test_root, args.output_root, parse_sequence_names(args.sequences)
    )
    print_preflight(sequences)
    raise SystemExit(0)


run_early_preflight()

import cv2
import numpy as np
import torch
from diffusers.models.brushnet import BrushNetModel
from diffusers.pipelines.brushnet.pipeline_sharedNoiseBG_org import (
    StableDiffusionBrushNetPipeline,
)
from diffusers.schedulers.scheduling_ddim_CGE import CustomDDIMScheduler, cond_fn
from ip_adapter import FusionIPAdapter
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from vcmrs_codec_adapter import VCMRSDualRegionCodec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run null-text BrushNet + IP-Adapter + CGE/VCM-RS inference on "
            "dataset/test_1. No caption file is read."
        )
    )
    parser.add_argument(
        "--long_test_root",
        type=Path,
        default=DEFAULT_TEST_ROOT,
        help="Flat dataset/test_1 root, or a supported hierarchical SFU test root.",
    )
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--sequences",
        default=None,
        help="Comma-separated names, e.g. PartyScene,Traffic. Default: all complete sequences.",
    )
    parser.add_argument(
        "--preflight_only",
        action="store_true",
        help="Validate input/mask pairs and print paths without loading weights.",
    )
    parser.add_argument("--base_model_path", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--brushnet_path", type=Path, default=None)
    parser.add_argument("--ip_ckpt", type=Path, default=None)
    parser.add_argument("--fusion_ckpt", type=Path, default=None)
    parser.add_argument("--image_encoder_path", default=DEFAULT_IMAGE_ENCODER)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--ip_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shared_bg_seed", type=int, default=6789)
    parser.add_argument(
        "--shared_bg_noise_strength",
        type=float,
        default=0.95,
        help="Matches the train_sharedNoise_sameBG_0.95_T8 checkpoint.",
    )
    parser.add_argument("--prompt", default="", help="Static prompt; default is null text.")
    parser.add_argument(
        "--negative_prompt", default="", help="Static negative prompt; default is null text."
    )
    parser.add_argument(
        "--no_blend",
        action="store_true",
        help="Save raw generated output instead of pasting the observed ROI back.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--cge_start_step", type=int, default=0)
    parser.add_argument("--cge_end_step", type=int, default=None)
    parser.add_argument("--cge_every_n_steps", type=int, default=1)
    parser.add_argument("--cge_max_evals", type=int, default=-1)
    args = parser.parse_args()
    if args.resolution <= 0 or args.resolution % 8:
        parser.error("--resolution must be a positive multiple of 8")
    if args.num_inference_steps <= 0 or args.num_samples <= 0:
        parser.error("--num_inference_steps and --num_samples must be positive")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max_images must be positive")
    if args.cge_start_step < 0 or (args.cge_end_step is not None and args.cge_end_step < 0):
        parser.error("CGE step limits must be non-negative")
    if args.cge_end_step is not None and args.cge_end_step <= args.cge_start_step:
        parser.error("--cge_end_step must be greater than --cge_start_step")
    if args.cge_every_n_steps <= 0:
        parser.error("--cge_every_n_steps must be positive")
    return args


def set_deterministic(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def require_path(path: Path, kind: str) -> Path:
    path = Path(path).expanduser().resolve()
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def resolve_checkpoint_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    checkpoint_dir = require_path(args.checkpoint_dir, "dir")
    brushnet_path = args.brushnet_path or checkpoint_dir / "brushnet"
    ip_ckpt = args.ip_ckpt or checkpoint_dir / "ipadapter" / "model.safetensors"
    fusion_ckpt = args.fusion_ckpt or checkpoint_dir / "ipadapter" / "fusion_module.safetensors"
    return (
        require_path(brushnet_path, "dir"),
        require_path(ip_ckpt, "file"),
        require_path(fusion_ckpt, "file"),
    )


def configure_codec_descriptors(
    codec: VCMRSDualRegionCodec,
    roi_template: Optional[str],
    bg_template: Optional[str],
    image_path: Path,
    frame_index: int,
) -> None:
    """Use per-frame frozen training descriptors when template env vars are set."""

    if roi_template is None:
        return
    values = {"stem": image_path.stem, "basename": image_path.name, "index": frame_index}
    codec.set_descriptors(
        Path(roi_template.format(**values)), Path(bg_template.format(**values))
    )


def build_pipeline(
    args: argparse.Namespace, device: str
) -> Tuple[StableDiffusionBrushNetPipeline, FusionIPAdapter]:
    brushnet_path, ip_ckpt, fusion_ckpt = resolve_checkpoint_paths(args)
    base_model_path = require_path(args.base_model_path, "dir")
    print(f"[Checkpoint] base model   : {base_model_path}")
    print(f"[Checkpoint] brushnet     : {brushnet_path}")
    print(f"[Checkpoint] ip_adapter   : {ip_ckpt}")
    print(f"[Checkpoint] fusion_module: {fusion_ckpt}")
    print(f"[Text] prompt={args.prompt!r}, negative_prompt={args.negative_prompt!r}")

    brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionBrushNetPipeline.from_pretrained(
        base_model_path,
        brushnet=brushnet,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
        safety_checker=None,
    )
    # This is the same pipeline as the null-text runner; only DDIM is replaced
    # with its CGE-enabled subclass.
    pipe.scheduler = CustomDDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.per_frame_cge = False
    pipe.scheduler.decode_chunk_size = 1
    pipe.scheduler.vae_scaling_factor = float(pipe.vae.config.scaling_factor)
    pipe.scheduler.cge_codec = VCMRSDualRegionCodec.from_env()
    pipe.scheduler.cge_start_step = args.cge_start_step
    pipe.scheduler.cge_end_step = args.cge_end_step
    pipe.scheduler.cge_every_n_steps = args.cge_every_n_steps
    pipe.scheduler.cge_max_evals = args.cge_max_evals
    codec = pipe.scheduler.cge_codec
    print(
        "[CGE codec] profile={}, ROI QP={}, BG QP={}, configuration={}/IP{}, root={}".format(
            codec.profile,
            codec.roi_quality,
            codec.bg_quality,
            codec.configuration,
            codec.intra_period,
            codec.vcmrs_root,
        )
    )

    pipe.enable_model_cpu_offload()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.image_encoder_path
    ).to(pipe.device, dtype=pipe.dtype)
    pipe.register_modules(
        image_encoder=image_encoder,
        feature_extractor=CLIPImageProcessor(),
    )
    ip_model = FusionIPAdapter(
        pipe,
        args.image_encoder_path,
        str(ip_ckpt),
        str(fusion_ckpt),
        device,
    )
    return pipe, ip_model


def _read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return image_bgr[:, :, ::-1]


def _read_hard_roi_mask(path: Path) -> np.ndarray:
    """Load the historical white-ROI mask convention as a boolean HxW mask."""

    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    if mask.ndim == 2:
        return mask > 127
    return mask[:, :, :3].sum(axis=-1) > 255


def prepare_frame(
    image_path: Path,
    mask_path: Path,
    image_resize: transforms.Resize,
    mask_resize: transforms.Resize,
) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image, np.ndarray, np.ndarray]:
    """Resize once and derive all BrushNet and CGE inputs from that exact pair."""

    init_image = image_resize(Image.fromarray(_read_rgb(image_path)).convert("RGB"))
    roi_mask = _read_hard_roi_mask(mask_path)
    mask_image = mask_resize(
        Image.fromarray((roi_mask * 255).astype(np.uint8)).convert("L")
    )
    init_image_np = np.asarray(init_image, dtype=np.uint8)
    roi_mask_np = np.asarray(mask_image, dtype=np.uint8) > 127
    roi_mask_3 = roi_mask_np[:, :, np.newaxis]
    fg_pil = Image.fromarray((init_image_np * roi_mask_3).astype(np.uint8)).convert("RGB")
    bg_pil = Image.fromarray((init_image_np * ~roi_mask_3).astype(np.uint8)).convert("RGB")
    return init_image, mask_image, fg_pil, bg_pil, init_image_np, roi_mask_np


def blend_with_observed_roi(
    generated_image: Image.Image, init_image_np: np.ndarray, roi_mask_np: np.ndarray
) -> Image.Image:
    """Preserve the legacy final compositing: observed ROI plus generated BG."""

    generated_np = np.asarray(generated_image.convert("RGB"), dtype=np.uint8)
    background_mask = 1.0 - roi_mask_np.astype(np.float32)
    observed_roi = init_image_np.astype(np.float32) * roi_mask_np[:, :, None]
    blurred_bg = cv2.GaussianBlur(background_mask * 255.0, (21, 21), 0) / 255.0
    generated_weight = 1.0 - (1.0 - background_mask) * (1.0 - blurred_bg)
    composite = observed_roi * (1.0 - generated_weight[:, :, None])
    composite += generated_np.astype(np.float32) * generated_weight[:, :, None]
    return Image.fromarray(np.clip(composite, 0, 255).astype(np.uint8), mode="RGB")


def configure_scheduler_for_frame(
    scheduler: CustomDDIMScheduler,
    init_image: Image.Image,
    mask_image: Image.Image,
    image_transform: transforms.Compose,
    mask_transform: transforms.Compose,
    device: str,
) -> None:
    scheduler.x_lr = image_transform(init_image).unsqueeze(0).to(device=device)
    scheduler.mask = mask_transform(mask_image).unsqueeze(0).to(device=device)
    scheduler.cge_codec.prepare_region_mask(scheduler.mask[0], scheduler.x_lr[0])
    scheduler.decoder = None
    scheduler.cond_fn = None
    scheduler.cge_codec_eval_count = 0
    scheduler.cge_denoise_step_count = 0


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    sequences = load_sequences(
        args.long_test_root,
        args.output_root,
        parse_sequence_names(args.sequences),
    )
    print_preflight(sequences)
    if args.preflight_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CGE inference requires a CUDA-capable PyTorch installation.")

    roi_descriptor_template = os.environ.get("CGE_VCMRS_ROI_DESCRIPTOR_TEMPLATE")
    bg_descriptor_template = os.environ.get("CGE_VCMRS_BG_DESCRIPTOR_TEMPLATE")
    if bool(roi_descriptor_template) != bool(bg_descriptor_template):
        raise ValueError(
            "Set both CGE_VCMRS_ROI_DESCRIPTOR_TEMPLATE and "
            "CGE_VCMRS_BG_DESCRIPTOR_TEMPLATE, or neither to use static/automatic descriptors."
        )

    device = "cuda"
    pipe, ip_model = build_pipeline(args, device)
    image_resize = transforms.Resize(
        (args.resolution, args.resolution), interpolation=InterpolationMode.BILINEAR
    )
    mask_resize = transforms.Resize(
        (args.resolution, args.resolution), interpolation=InterpolationMode.NEAREST
    )
    image_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda tensor: tensor * 2.0 - 1.0)]
    )
    mask_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda tensor: (tensor > 0.5).to(tensor.dtype)),
        ]
    )
    shared_bg_generator = torch.Generator(device).manual_seed(args.shared_bg_seed)
    shared_bg_noise = torch.randn(
        (
            1,
            pipe.unet.config.in_channels,
            args.resolution // pipe.vae_scale_factor,
            args.resolution // pipe.vae_scale_factor,
        ),
        generator=shared_bg_generator,
        device=device,
        dtype=pipe.dtype,
    )

    for sequence in sequences:
        output_dir = sequence.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: Sequence[Tuple[Path, Path]] = list(zip(sequence.image_paths, sequence.mask_paths))
        if args.max_images is not None:
            frame_paths = frame_paths[: args.max_images]
        existing_basenames = set()
        if not args.overwrite:
            existing_basenames = {path.name for path in output_dir.glob("*.png")}
        indexed_pending = [
            (index, image_path, mask_path)
            for index, (image_path, mask_path) in enumerate(frame_paths)
            if args.overwrite or image_path.name not in existing_basenames
        ]
        print(
            f"[Resume:{sequence.spec.label}] total={len(frame_paths)}, "
            f"done={len(frame_paths) - len(indexed_pending)}, pending={len(indexed_pending)}"
        )
        metadata = {
            "split": sequence.spec.split,
            "label": sequence.spec.label,
            "class": sequence.spec.class_name,
            "source_sequence": sequence.spec.source_name,
            "image_dir": str(sequence.image_dir),
            "mask_dir": str(sequence.mask_dir),
            "output_dir": str(output_dir),
            "frame_count": len(frame_paths),
            "frame_range": [sequence.frame_ids[0], sequence.frame_ids[len(frame_paths) - 1]],
            "checkpoint_dir": str(Path(args.checkpoint_dir).expanduser().resolve()),
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "shared_bg_noise_strength": args.shared_bg_noise_strength,
            "cge_region_quality": {
                "roi": pipe.scheduler.cge_codec.roi_quality,
                "bg": pipe.scheduler.cge_codec.bg_quality,
            },
            "cge_profile": pipe.scheduler.cge_codec.profile,
        }
        (output_dir / "source_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        # This matches the null-text runner: each sequence starts from seed again.
        generator = torch.Generator(device).manual_seed(args.seed)
        progress = tqdm(indexed_pending, total=len(indexed_pending), desc=sequence.spec.label)
        for frame_index, image_path, mask_path in progress:
            configure_codec_descriptors(
                pipe.scheduler.cge_codec,
                roi_descriptor_template,
                bg_descriptor_template,
                image_path,
                frame_index,
            )
            (
                init_image,
                mask_image,
                fg_pil,
                bg_pil,
                init_image_np,
                roi_mask_np,
            ) = prepare_frame(image_path, mask_path, image_resize, mask_resize)
            configure_scheduler_for_frame(
                pipe.scheduler,
                init_image,
                mask_image,
                image_transform,
                mask_transform,
                device,
            )
            pipe.scheduler.decoder = pipe.vae.decode
            pipe.scheduler.cond_fn = cond_fn
            result = ip_model.generate_fgbg(
                fg_pil_image=fg_pil,
                bg_pil_image=bg_pil,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                scale=args.ip_scale,
                image=init_image,
                mask_image=mask_image,
                num_samples=args.num_samples,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                use_shared_bg_noise=True,
                shared_bg_noise=shared_bg_noise,
                shared_bg_noise_strength=args.shared_bg_noise_strength,
                variance_preserving_shared_noise=True,
            )
            image = result[0]
            if not args.no_blend:
                image = blend_with_observed_roi(image, init_image_np, roi_mask_np)
            image.save(output_dir / image_path.name)


if __name__ == "__main__":
    main()
