"""
Inference script for the no-caption / null-text fine-tuned BrushNet + IP-Adapter model.

Main difference from the caption-based test script:
  - No caption file is loaded.
  - Every image uses prompt="" and negative_prompt="" by default.
  - This matches the null-text conditioning used in train_brushnet_VCM_ipadapter_v8_coco_nulltext.py.
"""

import argparse
import os
import random
from glob import glob

import cv2
import numpy as np
import torch
from diffusers import DDIMScheduler
from diffusers.models.brushnet import BrushNetModel
from diffusers.pipelines.brushnet.pipeline_sharedNoiseBG_org import (
    StableDiffusionBrushNetPipeline,
)
from ip_adapter import FusionIPAdapter
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


DEFAULT_BASE_MODEL = (
    "/home/cilab/ndquan/videoInpainting/code/BrushNet/examples/brushnet/"
    "base_model/stable-diffusion-v1-5/stable-diffusion-v1-5"
)
DEFAULT_CHECKPOINT_DIR = (
    "/home/cilab/ndquan/videoInpainting/code/BrushNet/examples/checkpoint_brushnet/train_sharedNoise_sameBG_0.9/checkpoint-2000"
)

DEFAULT_IMAGE_DIR = (
    "/home/cilab/ndquan/videoInpainting/code/BrushNet/examples/brushnet/"
    "dataset/test/BasketballPass/inputs"
)

DEFAULT_MASK_DIR = (
    "/home/cilab/ndquan/videoInpainting/code/BrushNet/examples/brushnet/"
    "dataset/test/BasketballPass/masks"
)
DEFAULT_OUTPUT_DIR = (
    "/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/"
    "Generated_image/BasketballPass/sharedNoise_new_checkpoint2000_09_corr"
)
DEFAULT_IMAGE_ENCODER = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BrushNet + IP-Adapter inference with null-text conditioning."
    )
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--mask_dir", default=DEFAULT_MASK_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base_model_path", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--brushnet_path", default=None)
    parser.add_argument("--ip_ckpt", default=None)
    parser.add_argument("--fusion_ckpt", default=None)
    parser.add_argument("--image_encoder_path", default=DEFAULT_IMAGE_ENCODER)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--ip_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shared_bg_seed", type=int, default=6789)
    parser.add_argument("--shared_bg_noise_strength", type=float, default=0.9)
    parser.add_argument("--prompt", default="", help="Default is null text.")
    parser.add_argument("--negative_prompt", default="", help="Default is null negative text.")
    parser.add_argument("--no_blend", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_images", type=int, default=None)
    return parser.parse_args()


def set_deterministic(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def list_images(directory):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    paths = []
    for ext in exts:
        paths.extend(glob(os.path.join(directory, ext)))
    return sorted(paths)


def require_path(path, kind):
    if kind == "dir" and not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")
    if kind == "file" and not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")


def resolve_checkpoint_paths(args):
    brushnet_path = args.brushnet_path or os.path.join(args.checkpoint_dir, "brushnet")
    ip_ckpt = args.ip_ckpt or os.path.join(args.checkpoint_dir, "ipadapter", "model.safetensors")
    fusion_ckpt = args.fusion_ckpt or os.path.join(
        args.checkpoint_dir, "ipadapter", "fusion_module.safetensors"
    )

    require_path(brushnet_path, "dir")
    require_path(ip_ckpt, "file")
    require_path(fusion_ckpt, "file")
    return brushnet_path, ip_ckpt, fusion_ckpt


def build_pipeline(args, device):
    brushnet_path, ip_ckpt, fusion_ckpt = resolve_checkpoint_paths(args)

    print(f"[Checkpoint] brushnet     : {brushnet_path}")
    print(f"[Checkpoint] ip_adapter   : {ip_ckpt}")
    print(f"[Checkpoint] fusion_module: {fusion_ckpt}")
    print(f"[Text] prompt={args.prompt!r}, negative_prompt={args.negative_prompt!r}")

    brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionBrushNetPipeline.from_pretrained(
        args.base_model_path,
        brushnet=brushnet,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
        safety_checker=None,
    )

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
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
        ip_ckpt,
        fusion_ckpt,
        device,
    )
    return pipe, ip_model


def prepare_fg_bg(init_image_np, mask_np, transform):
    fg_np = init_image_np * mask_np
    bg_np = init_image_np * (1 - mask_np)

    fg_pil = Image.fromarray(fg_np.astype(np.uint8)).convert("RGB")
    bg_pil = Image.fromarray(bg_np.astype(np.uint8)).convert("RGB")

    return transform(fg_pil), transform(bg_pil)


def blend_with_original(generated_image, image_path, mask_path, resolution):
    image_np = np.array(generated_image)
    init_image_np = cv2.imread(image_path)[:, :, ::-1]
    mask_np = 1.0 * (cv2.imread(mask_path).sum(-1) > 255)

    new_size = (resolution, resolution)
    init_image_np = cv2.resize(init_image_np, new_size, interpolation=cv2.INTER_LINEAR)
    mask_np = cv2.resize(mask_np, new_size, interpolation=cv2.INTER_NEAREST)

    mask_np = 1 - mask_np
    mask_np = mask_np[:, :, np.newaxis]
    init_image_np = init_image_np * (1 - mask_np)

    mask_blurred = cv2.GaussianBlur(mask_np * 255, (21, 21), 0) / 255
    mask_blurred = mask_blurred[:, :, np.newaxis]
    mask_np = 1 - (1 - mask_np) * (1 - mask_blurred)

    image_pasted = init_image_np * (1 - mask_np) + image_np * mask_np
    return Image.fromarray(image_pasted.astype(np.uint8))


def main():
    args = parse_args()
    set_deterministic(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    require_path(args.image_dir, "dir")
    require_path(args.mask_dir, "dir")
    require_path(args.base_model_path, "dir")
    os.makedirs(args.output_dir, exist_ok=True)

    pipe, ip_model = build_pipeline(args, device)

    image_paths = list_images(args.image_dir)
    mask_paths = list_images(args.mask_dir)
    if len(image_paths) != len(mask_paths):
        raise ValueError(
            f"Image/mask count mismatch: {len(image_paths)} images vs {len(mask_paths)} masks"
        )

    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
        mask_paths = mask_paths[: args.max_images]

    existing_basenames = set()
    if not args.overwrite:
        existing_basenames = {os.path.basename(p) for p in glob(os.path.join(args.output_dir, "*.png"))}

    indexed_all = list(enumerate(zip(image_paths, mask_paths)))
    indexed_pending = [
        (idx, image_path, mask_path)
        for idx, (image_path, mask_path) in indexed_all
        if args.overwrite or os.path.basename(image_path) not in existing_basenames
    ]

    print(
        f"[Resume] total={len(indexed_all)}, done={len(indexed_all) - len(indexed_pending)}, "
        f"pending={len(indexed_pending)}"
    )

    generator = torch.Generator(device).manual_seed(args.seed)
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

    transform = transforms.Compose([transforms.Resize((args.resolution, args.resolution))])

    for orig_idx, image_path, mask_path in tqdm(indexed_pending, total=len(indexed_pending)):
        init_image_np = cv2.imread(image_path)[:, :, ::-1]
        mask_np = 1.0 * (cv2.imread(mask_path).sum(-1) > 255)[:, :, np.newaxis]

        init_image = Image.fromarray(init_image_np.astype(np.uint8)).convert("RGB")
        mask_image = Image.fromarray((mask_np * 255).astype(np.uint8).repeat(3, -1)).convert("RGB")
        init_image = transform(init_image)
        mask_image = transform(mask_image)

        fg_pil, bg_pil = prepare_fg_bg(init_image_np, mask_np, transform)

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
            print(f"[{orig_idx}] blending...")
            image = blend_with_original(image, image_path, mask_path, args.resolution)

        basename = os.path.basename(image_path)
        image.save(os.path.join(args.output_dir, basename))


if __name__ == "__main__":
    main()
