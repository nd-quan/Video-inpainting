#!/usr/bin/env python
"""Evaluate a complete DGAF-inspired checkpoint without loading an STC model."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from shared_bg_noise_training import mix_shared_background_noise
from STC_encoder_v8_raft_deformable.raft_flow_provider import FrozenV7RAFTFlowProvider
from STC_encoder_v2_rgb.evaluate_rgb_stc_shared_noise import (
    build_v8_prompt_embeddings, stable_seed, composite_images, tensor_to_pil,
    compute_metrics, clip_directory, save_images, aggregate_metrics)
from DGAF_VSR.common import add_common_arguments, resolve_paths, make_dataset, load_stack, json_dump, BooleanOptionalAction
from DGAF_VSR.warping import ClipWarper, GuidanceConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.set_defaults(split="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--temporal_guidance_scale", type=float, default=1.0)
    parser.add_argument("--use_confidence_mask", action=BooleanOptionalAction, default=None,
                        help="Omitted inherits training; explicit override is recorded as an ablation")
    parser.add_argument("--warp_mode", choices=("direct", "dgaf"), default=None,
                        help="Omitted inherits training")
    parser.add_argument("--max_clips", type=int, default=None)
    parser.add_argument("--roi_composite", choices=("hard", "blurred", "none"), default="blurred")
    parser.add_argument("--roi_blur_kernel_size", type=int, default=51)
    parser.add_argument("--shared_bg_seed", type=int, default=6789)
    args = parser.parse_args(argv)
    if args.max_clips is not None and args.max_clips < 1:
        parser.error("max_clips must be positive")
    if args.roi_blur_kernel_size < 1 or args.roi_blur_kernel_size % 2 == 0:
        parser.error("roi_blur_kernel_size must be positive and odd")
    if any(not math.isfinite(v) or v < 0 for v in (args.guidance_scale, args.temporal_guidance_scale)):
        parser.error("Guidance scales must be finite and nonnegative")
    return args


def main(args):
    args.checkpoint = args.checkpoint.expanduser().resolve()
    for name in ("metadata.json", "guidance_config.json", "brushnet_dgaf/config.json",
                 "brushnet_dgaf/diffusion_pytorch_model.safetensors"):
        if not (args.checkpoint / name).is_file():
            raise FileNotFoundError(f"Incomplete DGAF checkpoint: {args.checkpoint / name}")
    metadata = json.loads((args.checkpoint / "metadata.json").read_text())
    contract = metadata["contract"]
    # Model dependencies always follow the checkpoint, including the resolved
    # immutable RAFT component rather than a mutable best/latest pointer.
    for key in ("pretrained_model_name_or_path", "baseline_checkpoint", "raft_student_path"):
        setattr(args, key, Path(contract[key]))
    args.image_encoder_name_or_path = contract["image_encoder_name_or_path"]
    if args.num_inference_steps is None:
        args.num_inference_steps = contract["num_diffusion_steps"]
    if args.num_inference_steps < 2:
        raise ValueError("Need at least two diffusion steps for temporal guidance")
    resolve_paths(args)
    config = GuidanceConfig(**json.loads((args.checkpoint / "guidance_config.json").read_text()))
    if args.use_confidence_mask is not None:
        config = replace(config, use_confidence_mask=args.use_confidence_mask)
    if args.warp_mode is not None:
        config = replace(config, warp_mode=args.warp_mode)
    dataset = make_dataset(args)
    if args.preflight_only:
        print(json.dumps({"status": "ok", "clips": len(dataset), "guidance": config.to_dict(),
            "checkpoint": str(args.checkpoint), "stc_encoder": False}, indent=2))
        return
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Full evaluation requires CUDA for RAFT")
    record = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "guidance": config.to_dict(), "checkpoint_step": metadata["global_step"], "training_contract": contract}
    record["args"].pop("preflight_only")
    record["args"].pop("max_clips")  # Permit extending an identical evaluation.
    manifest = args.output_dir / "run_config.json"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not manifest.is_file() or json.loads(manifest.read_text()) != record:
            raise ValueError("Output contains a different/incomplete run; select a new output_dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_dump(manifest, record)
    torch.manual_seed(args.seed)
    pipe, image_encoder, projection, fusion, ip_report = load_stack(args, device,
        model_path=args.checkpoint / "brushnet_dgaf", inference=True)
    provider = FrozenV7RAFTFlowProvider(args.raft_student_path, device=device,
        pair_batch_size=args.raft_pair_batch_size, mixed_precision=args.raft_mixed_precision)
    json_dump(args.output_dir / "model_contract.json", {**record, "ip_loading": ip_report,
        "stc_encoder": False, "memory_reset": "every clip", "sampling": "synchronous_previous_step_clean_cache",
        "confidence_override": args.use_confidence_mask, "warp_override": args.warp_mode})
    limit = min(len(dataset), args.max_clips) if args.max_clips else len(dataset)
    with torch.inference_mode():
        for index in range(limit):
            sample = dataset[index]
            ids = [int(x) for x in sample["frame_ids"].tolist()]
            output = clip_directory(args.output_dir, sample["video"], ids)
            if (output / "clip_metrics.json").is_file():
                continue
            rgb = sample["conditioning_pixel_values"].to(provider.device)
            prompt, negative, text, negative_text = build_v8_prompt_embeddings(
                pipe, image_encoder, projection, fusion, sample, device, args.fusion_scale)
            generator = torch.Generator(device=device).manual_seed(stable_seed(args.seed, "condition", sample["video"], ids[0]))
            base = pipe.vae.encode(rgb.to(pipe.vae.dtype)).latent_dist.sample(generator=generator).float() * pipe.vae.config.scaling_factor
            frames, _, h, w = base.shape
            bg = (F.interpolate(sample["masks"].to(device), size=(h, w), mode="nearest") >= 0.5).float()
            flows = provider.predict_sequence(rgb[None])
            warper = ClipWarper(flows.forward, flows.backward, bg[None], config)
            generator.manual_seed(stable_seed(args.seed, "initial", sample["video"], ids[0]))
            independent = torch.randn(base.shape, device=device, generator=generator, dtype=pipe.unet.dtype)
            generator.manual_seed(stable_seed(args.shared_bg_seed, "shared", sample["video"]))
            shared = torch.randn((1, 1, 4, h, w), device=device, generator=generator, dtype=pipe.unet.dtype)
            noise = mix_shared_background_noise(independent[None], shared, bg[None],
                strength=args.shared_bg_noise_strength, variance_preserving=True).flatten(0, 1)
            generated = pipe(latents=noise, base_condition=torch.cat((base, bg), dim=1), warper=warper,
                prompt_embeds=prompt, negative_prompt_embeds=negative, brushnet_prompt_embeds=text,
                negative_brushnet_prompt_embeds=negative_text, num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale, temporal_guidance_scale=args.temporal_guidance_scale).images
            final = composite_images(generated, sample["conditioning_pixel_values"], sample["masks"],
                mode=args.roi_composite, blur_kernel_size=args.roi_blur_kernel_size)
            metrics = compute_metrics(final, sample["pixel_values"], sample["masks"])
            metrics.update({f"raw_{k}": v for k, v in compute_metrics(generated, sample["pixel_values"], sample["masks"]).items()})
            save_images(generated, output / "raw", ids)
            save_images(final, output / "final", ids)
            save_images(tensor_to_pil(sample["pixel_values"]), output / "gt", ids)
            save_images(tensor_to_pil(sample["conditioning_pixel_values"]), output / "input", ids)
            json_dump(output / "clip_metrics.json", {"video": sample["video"], "frame_ids": ids, "metrics": metrics})
            print(f"[{index + 1}/{limit}] {sample['video']} {ids[0]}-{ids[-1]} BG PSNR={metrics['bg_psnr']:.3f}", flush=True)
    summary = aggregate_metrics(args.output_dir)
    summary["aggregation"] = "unweighted clip mean; overlap frames are counted per clip"
    json_dump(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main(parse_args())
