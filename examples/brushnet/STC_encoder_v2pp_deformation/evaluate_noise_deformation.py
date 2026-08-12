#!/usr/bin/env python
"""Evaluate STC-v2++ with sequence-level or clip-local deformation noise.

The trained D2 checkpoint contains a clip-local deformation head.  This script
implements the D3 inference contract: clips are processed in source order, one
FP32 lineage is retained per video, and overlap frames reuse both cached noise
and their first generated output exactly.  Only previously unseen frames are
sent through the 2D diffusion pipeline after the first clip.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)


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
from STC_encoder_v2_rgb.evaluate_rgb_stc_shared_noise import (  # noqa: E402
    aggregate_metrics,
    autocast_context,
    build_v8_prompt_embeddings,
    clip_directory,
    composite_images,
    compute_metrics,
    masks_to_pil,
    resolve_dataset_layout,
    save_images,
    stable_seed,
    tensor_to_pil,
)
from STC_encoder_v2_rgb.frozen_v8 import (  # noqa: E402
    install_and_load_ip_adapter,
    load_fusion_module,
)
from STC_encoder_v2_rgb.rgb_stc_adapter import (  # noqa: E402
    RGBSTCConditionAdapter,
    augment_brushnet_condition,
)
from STC_encoder_v2pp_deformation.noise_deformation import (  # noqa: E402
    NoiseDeformationHead,
    build_deformed_clip_noise,
)
from STC_encoder_v2pp_deformation.sequence_noise_state import (  # noqa: E402
    SequenceClipNoiseOutput,
    SequenceNoiseState,
    build_sequence_deformed_noise,
    deterministic_frame_noise,
)


PROJECT_ROOT = BRUSHNET_DIR.parent.parent
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "train_stc_v2pp_T8_cosine_0.9"
    / "checkpoint-2000"
)
DEFAULT_DATASET = BRUSHNET_DIR / "dataset" / "test"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments"
    / "eval_stc_v2pp_T8_cosine_0.9"
    / "test"
    / "checkpoint-2000-sequence-state"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained STC-v2++ checkpoint with exact overlap-noise "
            "and first-owner output reuse."
        )
    )
    parser.add_argument("--checkpoint_path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--baseline_checkpoint", type=Path, default=None)
    parser.add_argument("--pretrained_model_name_or_path", type=Path, default=None)
    parser.add_argument("--image_encoder_name_or_path", default=None)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split directory below --dataset_root (for example test or long_test).",
    )
    parser.add_argument(
        "--dataset_layout",
        choices=("auto", "hierarchical", "flat_test"),
        default="auto",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--clip_length", type=int, default=None)
    parser.add_argument("--clip_stride", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--brushnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--stc_injection_scale", type=float, default=None)
    parser.add_argument("--fusion_scale", type=float, default=1.0)
    parser.add_argument("--transport_alpha", type=float, default=None)
    parser.add_argument("--warp_scope", choices=("full", "bg"), default=None)
    parser.add_argument("--lineage_normalization_eps", type=float, default=None)
    parser.add_argument("--noise_seed", type=int, default=1234)
    parser.add_argument("--condition_seed", type=int, default=2345)
    parser.add_argument("--generation_seed", type=int, default=3456)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--state_scope",
        choices=("sequence", "clip_local"),
        default="sequence",
        help=(
            "sequence keeps one lineage across ordered clips; clip_local "
            "starts a fresh deterministic anchor and generates all T frames "
            "for every clip, matching the training horizon"
        ),
    )
    parser.add_argument(
        "--offset_mode",
        choices=("learned", "zero_grid"),
        default="learned",
        help=(
            "zero_grid passes an all-zero offset through the same bilinear "
            "grid_sample/normalization/fusion path; it does not bypass warp"
        ),
    )
    parser.add_argument(
        "--roi_composite",
        choices=("none", "hard", "blurred"),
        default="hard",
    )
    parser.add_argument("--roi_blur_kernel_size", type=int, default=21)
    parser.add_argument("--video_filter", nargs="+", default=None)
    parser.add_argument("--max_clips", type=int, default=None)
    parser.add_argument(
        "--max_frames_per_sequence",
        type=int,
        default=None,
        help=(
            "Evaluate at most this many consecutive source frames from the "
            "start of every selected sequence. A final shifted tail clip is "
            "included so the selected prefix's last frame is not dropped."
        ),
    )
    parser.add_argument("--save_noise_tensors", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    args = parser.parse_args()
    if args.num_inference_steps < 1:
        parser.error("--num_inference_steps must be positive")
    if args.guidance_scale <= 1.0:
        parser.error("This exact V8 evaluator requires guidance_scale > 1")
    if args.max_clips is not None and args.max_clips < 1:
        parser.error("--max_clips must be positive")
    if args.max_frames_per_sequence is not None and args.max_frames_per_sequence < 1:
        parser.error("--max_frames_per_sequence must be positive")
    if args.roi_blur_kernel_size < 1 or args.roi_blur_kernel_size % 2 == 0:
        parser.error("--roi_blur_kernel_size must be a positive odd integer")
    return args


def require_path(path: Path, kind: str):
    if kind == "dir" and not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "metadata.json").is_file():
        return path
    pointer = path / "latest.json"
    if pointer.is_file():
        record = json.loads(pointer.read_text(encoding="utf-8"))
        candidate = path / str(record["checkpoint"])
        if (candidate / "metadata.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Expected a complete checkpoint-N or experiment root with latest.json: {path}"
    )


def checkpoint_paths(checkpoint: Path) -> Dict[str, Path]:
    return {
        "root": checkpoint,
        "metadata": checkpoint / "metadata.json",
        "brushnet": checkpoint / "brushnet",
        "stc_adapter": checkpoint / "stc_adapter",
        "noise_deformation": checkpoint / "noise_deformation",
    }


def baseline_paths(root: Path) -> Dict[str, Path]:
    root = root.expanduser().resolve()
    return {
        "root": root,
        "ip_adapter": root / "ipadapter" / "model.safetensors",
        "fusion": root / "ipadapter" / "fusion_module.safetensors",
    }


def apply_checkpoint_defaults(args, metadata: Mapping[str, object]):
    if metadata.get("variant") != "noise_deformation_head":
        raise ValueError(
            f"Not an STC-v2++ deformation checkpoint: {metadata.get('variant')!r}"
        )
    args.baseline_checkpoint = (
        args.baseline_checkpoint
        if args.baseline_checkpoint is not None
        else Path(str(metadata["baseline_checkpoint"]))
    ).expanduser().resolve()
    args.pretrained_model_name_or_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_model_name_or_path is not None
        else Path(str(metadata["pretrained_model_name_or_path"]))
    ).expanduser().resolve()
    args.image_encoder_name_or_path = (
        args.image_encoder_name_or_path
        if args.image_encoder_name_or_path is not None
        else str(metadata["image_encoder_name_or_path"])
    )
    args.resolution = (
        int(args.resolution)
        if args.resolution is not None
        else int(metadata["resolution"])
    )
    args.clip_length = (
        int(args.clip_length)
        if args.clip_length is not None
        else int(metadata["clip_length"])
    )
    args.clip_stride = (
        int(args.clip_stride)
        if args.clip_stride is not None
        else int(metadata["clip_stride"])
    )
    args.stc_injection_scale = (
        float(args.stc_injection_scale)
        if args.stc_injection_scale is not None
        else float(metadata["stc_injection_scale"])
    )
    args.transport_alpha = (
        float(args.transport_alpha)
        if args.transport_alpha is not None
        else float(metadata["transport_alpha"])
    )
    args.warp_scope = (
        str(args.warp_scope)
        if args.warp_scope is not None
        else str(metadata["warp_scope"])
    )
    args.lineage_normalization_eps = (
        float(args.lineage_normalization_eps)
        if args.lineage_normalization_eps is not None
        else float(metadata["lineage_normalization_eps"])
    )
    if args.resolution <= 0 or args.resolution % 8:
        raise ValueError("resolution must be positive and divisible by 8")
    if args.clip_length < 2 or not 0 < args.clip_stride < args.clip_length:
        raise ValueError(
            "Sequence-state evaluation requires clip_length>=2 and "
            "0 < clip_stride < clip_length"
        )
    if not 0.0 <= args.transport_alpha <= 1.0:
        raise ValueError("transport_alpha must be in [0,1]")
    if args.lineage_normalization_eps <= 0:
        raise ValueError("lineage_normalization_eps must be positive")


def make_dataset(args):
    dataset_layout = resolve_dataset_layout(args)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.pretrained_model_name_or_path),
        subfolder="tokenizer",
        use_fast=False,
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
    )
    return dataset, dataset_layout


def prefix_clip_records(
    source_frames: Sequence[Tuple[int, Path]],
    *,
    clip_length: int,
    clip_stride: int,
    max_frames: int,
) -> List[Tuple[Tuple[Path, ...], Tuple[int, ...]]]:
    """Build chronological overlapping windows for one contiguous prefix.

    Unlike the historical hierarchical loader, this explicitly appends a
    shifted tail window when the stride misses the prefix end. Thus a request
    for 150 frames with T=8 always evaluates its final frame as part of
    ``142..149`` (relative to that sequence prefix).
    """
    if clip_length < 2 or clip_stride < 1:
        raise ValueError("clip_length must be >=2 and clip_stride must be positive")
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    prefix = list(source_frames[:max_frames])
    if len(prefix) < clip_length:
        return []
    frame_ids = [frame_id for frame_id, _ in prefix]
    if any(current != previous + 1 for previous, current in zip(frame_ids, frame_ids[1:])):
        raise ValueError("max_frames_per_sequence requires a contiguous source prefix")

    last_start = len(prefix) - clip_length
    starts = list(range(0, last_start + 1, clip_stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [
        (
            tuple(path for _, path in prefix[start : start + clip_length]),
            tuple(frame_id for frame_id, _ in prefix[start : start + clip_length]),
        )
        for start in starts
    ]


def source_frames_for_video(
    dataset: HierarchicalV8ClipDataset,
    video: str,
) -> List[Tuple[int, Path]]:
    """Read the full source record list for a dataset video without decoding PNGs."""
    branch = Path(video)
    if isinstance(dataset, FlatV8TestClipDataset):
        gt_root = dataset.dataset_root / branch / "gt"
        relative = lambda path: branch / path.name
    else:
        gt_root = dataset.roots["GT"] / branch
        relative = lambda path: path.relative_to(dataset.roots["GT"])
    frames = []
    for path in gt_root.glob("*.png"):
        try:
            frames.append((int(path.stem), relative(path)))
        except ValueError as error:
            raise ValueError(f"Frame filename must have a numeric stem: {path}") from error
    frames.sort(key=lambda item: item[0])
    if not frames:
        raise ValueError(f"No source PNG frames found for sequence {video!r}")
    return frames


def select_evaluation_indices(
    dataset: HierarchicalV8ClipDataset,
    args,
    *,
    materialize: bool = True,
) -> Tuple[List[int], Dict[str, object]]:
    """Select clips while preserving one ordered, contiguous prefix per video."""
    requested = set(args.video_filter or [])
    available = {str(video) for video, _, _ in dataset.clips}
    if requested:
        missing = sorted(requested - available)
        if missing:
            raise ValueError(
                f"Unknown --video_filter values {missing}; available={sorted(available)}"
            )
    selected_videos = sorted(requested or available)

    if args.max_frames_per_sequence is None:
        indices = [
            index
            for index, (video, _, _) in enumerate(dataset.clips)
            if str(video) in selected_videos
        ]
        report = {
            "max_frames_per_sequence": None,
            "selected_videos": selected_videos,
            "selected_source_frames": None,
            "tail_windows_added": 0,
        }
    else:
        # Rebuild only the selected prefix windows. Appending them to this
        # in-memory dataset keeps its existing __getitem__ implementation and
        # does not alter files or the default (uncapped) evaluation path.
        indices = []
        selected_source_frames: Dict[str, int] = {}
        tail_windows_added = 0
        for video in selected_videos:
            source_frames = source_frames_for_video(dataset, video)
            records = prefix_clip_records(
                source_frames,
                clip_length=args.clip_length,
                clip_stride=args.clip_stride,
                max_frames=args.max_frames_per_sequence,
            )
            capped_count = min(len(source_frames), args.max_frames_per_sequence)
            if capped_count < args.clip_length:
                raise ValueError(
                    f"Sequence {video!r} has only {capped_count} selected frames; "
                    f"need at least clip_length={args.clip_length}"
                )
            selected_source_frames[video] = capped_count
            regular_last_start = (capped_count - args.clip_length) // args.clip_stride * args.clip_stride
            if regular_last_start != capped_count - args.clip_length:
                tail_windows_added += 1
            if materialize:
                for relative_paths, frame_ids in records:
                    dataset.clips.append((video, relative_paths, frame_ids))
                    indices.append(len(dataset.clips) - 1)
            else:
                indices.extend(range(len(records)))
        if materialize:
            indices.sort(key=lambda index: (
                str(dataset.clips[index][0]),
                int(dataset.clips[index][2][0]),
            ))
        report = {
            "max_frames_per_sequence": int(args.max_frames_per_sequence),
            "selected_videos": selected_videos,
            "selected_source_frames": selected_source_frames,
            "tail_windows_added": int(tail_windows_added),
        }

    if args.max_clips is not None:
        indices = indices[: args.max_clips]
    if materialize and not indices:
        raise ValueError("No evaluation clips selected")
    report["selected_clip_count"] = len(indices)
    return indices, report


def preflight(args):
    args.checkpoint_path = resolve_checkpoint(args.checkpoint_path)
    paths = checkpoint_paths(args.checkpoint_path)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    apply_checkpoint_defaults(args, metadata)
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    baseline = baseline_paths(args.baseline_checkpoint)

    require_path(args.pretrained_model_name_or_path, "dir")
    require_path(args.dataset_root, "dir")
    for name in ("brushnet", "stc_adapter", "noise_deformation"):
        require_path(paths[name], "dir")
        require_path(paths[name] / "config.json", "file")
        require_path(paths[name] / "diffusion_pytorch_model.safetensors", "file")
    require_path(baseline["ip_adapter"], "file")
    require_path(baseline["fusion"], "file")

    adapter = RGBSTCConditionAdapter.from_pretrained(str(paths["stc_adapter"]))
    head = NoiseDeformationHead.from_pretrained(str(paths["noise_deformation"]))
    if adapter.config.condition_mode != metadata["condition_mode"]:
        raise ValueError("STC adapter condition mode differs from checkpoint metadata")
    if int(adapter.config.hidden_channels) != int(head.config.feature_channels):
        raise ValueError("STC feature channels do not match deformation head")
    dataset, dataset_layout = make_dataset(args)
    # Persist the resolved layout in the run contract. Recording ``auto``
    # would hide whether this run used the hierarchical training tree or the
    # legacy flat test tree.
    args.dataset_layout = dataset_layout
    sample = dataset[0]
    internal_mask_values = sorted(
        float(value) for value in torch.unique(sample["masks"]).tolist()
    )
    if any(value not in (0.0, 1.0) for value in internal_mask_values):
        raise ValueError(
            "Dataset mask conversion must produce a binary internal M_BG; "
            f"found values={internal_mask_values}"
        )
    report = {
        "status": "ok",
        "checkpoint": str(paths["root"]),
        "checkpoint_step": int(metadata["global_step"]),
        "baseline_checkpoint": str(baseline["root"]),
        "dataset_root": str(args.dataset_root),
        "dataset_layout": dataset_layout,
        "clip_count": len(dataset),
        "source_frame_count": dataset.frame_count,
        "covered_frame_count": dataset.covered_frame_count,
        "branch_count": dataset.branch_count,
        "clip_length": args.clip_length,
        "clip_stride": args.clip_stride,
        "max_frames_per_sequence": args.max_frames_per_sequence,
        "transport_alpha": args.transport_alpha,
        "warp_scope": args.warp_scope,
        "noise_mode": f"{args.state_scope}:{args.offset_mode}",
        "pipeline_shared_bg_mixer": False,
        "sample_video": sample["video"],
        "sample_frame_ids": [int(value) for value in sample["frame_ids"]],
        "internal_bg_mask_values_sample": internal_mask_values,
        "mask_semantics": "raw ROI>=128; internal M_BG=1 degraded BG",
    }
    if hasattr(dataset, "ignored_gt_frame_count"):
        report["ignored_gt_frame_count"] = dataset.ignored_gt_frame_count
        report["total_gt_frame_count"] = dataset.total_gt_frame_count
    if args.max_frames_per_sequence is not None:
        _, selection_report = select_evaluation_indices(dataset, args, materialize=False)
        report["selection"] = selection_report
    print(json.dumps(report, indent=2, sort_keys=True))
    return dataset, paths, baseline, metadata, report


def load_models(args, paths, baseline, device):
    if device.type != "cuda":
        raise RuntimeError("This full SD1.5 evaluator requires a CUDA device")
    dtype = torch.float16
    brushnet = BrushNetModel.from_pretrained(
        str(paths["brushnet"]),
        torch_dtype=dtype,
    )
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
        pipe.unet,
        image_proj_model,
        baseline["ip_adapter"],
    )
    pipe.unet.to(device=device, dtype=dtype)
    image_proj_model.to(device=device, dtype=dtype)
    fusion_module = load_fusion_module(
        baseline["fusion"],
        embed_dim=image_encoder.config.projection_dim,
    ).to(device=device, dtype=dtype)
    adapter = RGBSTCConditionAdapter.from_pretrained(
        str(paths["stc_adapter"])
    ).to(device=device, dtype=torch.float32)
    deformation_head = NoiseDeformationHead.from_pretrained(
        str(paths["noise_deformation"])
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
        deformation_head,
    ):
        module.requires_grad_(False).eval()
    if pipe.brushnet.config.conditioning_channels != 5:
        raise ValueError("STC-v2++ requires five-channel BrushNet condition")
    return (
        pipe,
        image_encoder,
        image_proj_model,
        fusion_module,
        adapter,
        deformation_head,
        ip_report,
    )


@torch.inference_mode()
def build_stc_condition_and_features(
    *,
    pipe,
    adapter,
    sample: Mapping[str, torch.Tensor],
    device: torch.device,
    condition_seed: int,
    injection_scale: float,
):
    rgb_sequence = sample["conditioning_pixel_values"].unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )
    bg_mask_sequence = sample["masks"].unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )
    generator = torch.Generator(device=device).manual_seed(int(condition_seed))
    with autocast_context(device):
        base_condition_latents = pipe.vae.encode(
            rgb_sequence.flatten(0, 1).to(dtype=pipe.vae.dtype)
        ).latent_dist.sample(generator=generator)
        base_condition_latents = (
            base_condition_latents * pipe.vae.config.scaling_factor
        )
        brushnet_condition, output, _ = augment_brushnet_condition(
            adapter=adapter,
            base_condition_latents=base_condition_latents,
            rgb_sequence=rgb_sequence,
            bg_mask_sequence=bg_mask_sequence,
            injection_scale=float(injection_scale),
        )
    expected = (
        rgb_sequence.shape[1],
        5,
        base_condition_latents.shape[-2],
        base_condition_latents.shape[-1],
    )
    if tuple(brushnet_condition.shape) != expected:
        raise RuntimeError(
            f"Unexpected BrushNet condition shape {tuple(brushnet_condition.shape)}; "
            f"expected {expected}"
        )
    roi_leak = int(
        torch.count_nonzero(
            output.delta_bg.detach()
            * (1.0 - output.latent_bg_mask.detach())
        )
    )
    if roi_leak:
        raise RuntimeError("RGB-STC delta leaked outside M_BG")
    stats = {
        "delta_abs_mean": float(output.delta_bg.detach().float().abs().mean()),
        "latent_bg_ratio": float(output.latent_bg_mask.detach().float().mean()),
        "roi_delta_nonzero": roi_leak,
    }
    return brushnet_condition, output, stats


def noise_diagnostics(output) -> Dict[str, object]:
    lineage = output.lineage_noise.float()
    final_noise = output.final_noise.float()

    def adjacent_correlation(tensor: torch.Tensor, axis: int) -> float:
        if tensor.shape[axis] < 2:
            return 0.0
        left = tensor.narrow(axis, 0, tensor.shape[axis] - 1)
        right = tensor.narrow(axis, 1, tensor.shape[axis] - 1)
        left = left - left.mean()
        right = right - right.mean()
        denominator = left.square().mean().sqrt() * right.square().mean().sqrt()
        return float((left * right).mean() / denominator.clamp_min(1e-12))

    def high_frequency_difference_std(tensor: torch.Tensor) -> float:
        differences = []
        if tensor.shape[-1] > 1:
            differences.append(tensor[..., 1:] - tensor[..., :-1])
        if tensor.shape[-2] > 1:
            differences.append(tensor[..., 1:, :] - tensor[..., :-1, :])
        return float(
            torch.stack(
                [value.std(unbiased=False) for value in differences]
            ).mean()
        )

    record = {
        "generated_frame_ids": list(output.generated_frame_ids),
        "reused_frame_ids": list(output.reused_frame_ids),
        "transition_target_frame_ids": list(output.transition_target_frame_ids),
        "overlap_max_abs_difference": output.overlap_max_abs_difference,
        "final_noise_mean": float(final_noise.mean()),
        "final_noise_std": float(final_noise.std(unbiased=False)),
        "lineage_global_mean_abs": float(lineage.mean().abs()),
        "lineage_std": float(lineage.std(unbiased=False)),
        "lineage_neighbor_corr_x": adjacent_correlation(lineage, -1),
        "lineage_neighbor_corr_y": adjacent_correlation(lineage, -2),
        "lineage_high_frequency_difference_std": high_frequency_difference_std(
            lineage
        ),
        "final_noise_neighbor_corr_x": adjacent_correlation(final_noise, -1),
        "final_noise_neighbor_corr_y": adjacent_correlation(final_noise, -2),
        "final_noise_high_frequency_difference_std": (
            high_frequency_difference_std(final_noise)
        ),
    }
    if output.offsets.numel():
        record.update(
            {
                "offset_abs_mean": float(output.offsets.float().abs().mean()),
                "offset_abs_max": float(output.offsets.float().abs().max()),
                "valid_ratio": float(output.valid_masks.float().mean()),
                "pre_normalization_mean_abs": float(
                    output.pre_normalization_mean.float().abs().mean()
                ),
                "pre_normalization_std": float(
                    output.pre_normalization_std.float().mean()
                ),
                "post_normalization_mean_abs": float(
                    output.post_normalization_mean.float().abs().mean()
                ),
                "post_normalization_std": float(
                    output.post_normalization_std.float().mean()
                ),
            }
        )
    else:
        record.update(
            {
                "offset_abs_mean": 0.0,
                "offset_abs_max": 0.0,
                "valid_ratio": 1.0,
            }
        )
    return record


def build_clip_local_deformed_noise(
    *,
    deformation_head: NoiseDeformationHead,
    stc_features: torch.Tensor,
    frame_ids: Sequence[int],
    noise_channels: int,
    noise_seed: int,
    video: str,
    alpha: float,
    warp_scope: str,
    bg_mask: torch.Tensor,
    normalization_eps: float,
    offset_mode: str,
) -> SequenceClipNoiseOutput:
    """Build a fresh T-frame lineage with the exact training recurrence depth."""
    if stc_features.ndim != 5 or stc_features.shape[0] != 1:
        raise ValueError("clip-local STC features must have shape [1,T,C,H,W]")
    _, frames, _, height, width = stc_features.shape
    ids = tuple(int(value) for value in frame_ids)
    if len(ids) != frames:
        raise ValueError("frame_ids and clip-local feature length differ")
    # Keep per-absolute-frame independent/fallback streams identical to the
    # sequence-level evaluator.  Only the anchor is restarted at clip_start,
    # which makes the state-scope comparison paired instead of replacing all
    # random draws with a different realization.
    random_state = SequenceNoiseState(
        sequence_id=str(video),
        seed=int(noise_seed),
        channels=int(noise_channels),
        height=height,
        width=width,
    )
    anchor = deterministic_frame_noise(
        state=random_state,
        frame_id=ids[0],
        purpose="anchor",
        device=stc_features.device,
    ).unsqueeze(0)
    independent = torch.stack(
        [
            deterministic_frame_noise(
                state=random_state,
                frame_id=frame_id,
                purpose="independent",
                device=stc_features.device,
            )
            for frame_id in ids
        ],
        dim=0,
    ).unsqueeze(0)
    fallback = torch.stack(
        [
            deterministic_frame_noise(
                state=random_state,
                frame_id=frame_id,
                purpose="fallback",
                device=stc_features.device,
            )
            for frame_id in ids
        ],
        dim=0,
    ).unsqueeze(0)
    output = build_deformed_clip_noise(
        deformation_head=deformation_head,
        stc_features=stc_features.float(),
        anchor_lineage=anchor,
        independent_noise=independent,
        fallback_noise=fallback,
        alpha=float(alpha),
        warp_scope=warp_scope,
        bg_mask=bg_mask.float(),
        normalization_eps=float(normalization_eps),
        detach_stc_features=True,
        offset_mode=offset_mode,
    )
    return SequenceClipNoiseOutput(
        final_noise=output.final_noise,
        lineage_noise=output.lineage_noise,
        offsets=output.offsets,
        valid_masks=output.valid_masks,
        generated_frame_ids=ids,
        reused_frame_ids=(),
        transition_target_frame_ids=ids[1:],
        pre_normalization_mean=output.pre_normalization_mean,
        pre_normalization_std=output.pre_normalization_std,
        post_normalization_mean=output.post_normalization_mean,
        post_normalization_std=output.post_normalization_std,
        overlap_max_abs_difference=0.0,
    )


def copy_image(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(str(source), str(destination))


def clip_is_complete(clip_dir: Path, frame_ids: Sequence[int]) -> bool:
    if not (clip_dir / "clip_metrics.json").is_file():
        return False
    return all(
        (clip_dir / kind / f"{int(frame_id):06d}.png").is_file()
        for kind in ("raw", "final", "input", "gt", "mask_roi")
        for frame_id in frame_ids
    )


def selected_rows(tensor: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    index = torch.tensor(indices, device=tensor.device, dtype=torch.long)
    return tensor.index_select(0, index)


def run_contract(args, metadata) -> Dict[str, object]:
    return {
        "checkpoint": str(args.checkpoint_path),
        "checkpoint_step": int(metadata["global_step"]),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "pretrained_model_name_or_path": str(args.pretrained_model_name_or_path),
        "image_encoder_name_or_path": args.image_encoder_name_or_path,
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "dataset_layout": args.dataset_layout,
        "resolution": args.resolution,
        "clip_length": args.clip_length,
        "clip_stride": args.clip_stride,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "brushnet_conditioning_scale": args.brushnet_conditioning_scale,
        "stc_injection_scale": args.stc_injection_scale,
        "fusion_scale": args.fusion_scale,
        "transport_alpha": args.transport_alpha,
        "warp_scope": args.warp_scope,
        "lineage_normalization_eps": args.lineage_normalization_eps,
        "noise_seed": args.noise_seed,
        "condition_seed": args.condition_seed,
        "generation_seed": args.generation_seed,
        "roi_composite": args.roi_composite,
        "roi_blur_kernel_size": args.roi_blur_kernel_size,
        "state_scope": args.state_scope,
        "offset_mode": args.offset_mode,
        "sequence_state": (
            "lineage_and_final_noise_by_absolute_frame_id"
            if args.state_scope == "sequence"
            else "fresh_anchor_per_clip"
        ),
        "noise_rng": (
            "CPU FP32 keyed by sequence_id, absolute_frame_id, purpose; "
            "clip_local restarts anchor at each clip_start"
        ),
        "output_owner": (
            "first_clip_occurrence"
            if args.state_scope == "sequence"
            else "every_clip_occurrence"
        ),
        "pipeline_shared_bg_mixer": False,
    }


def main():
    args = parse_args()
    dataset, paths, baseline, metadata, report = preflight(args)
    if args.preflight_only:
        return
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Use a CUDA device for this full evaluation")

    random.seed(args.generation_seed)
    np.random.seed(args.generation_seed)
    torch.manual_seed(args.generation_seed)
    torch.cuda.manual_seed_all(args.generation_seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = run_contract(args, metadata)
    # Materialize the selected windows before writing run_config.  In capped
    # mode this appends the prefix/tail windows to the dataset, and the same
    # report must describe both the saved contract and the evaluation loop.
    indices, selection_report = select_evaluation_indices(dataset, args)
    config_path = args.output_dir / "run_config.json"
    if config_path.is_file() and not args.overwrite:
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("contract") != contract:
            raise ValueError(
                "Existing output has a different evaluation contract. Use a "
                "new --output_dir or pass --overwrite intentionally."
            )
    config_path.write_text(
        json.dumps(
            {
                "contract": contract,
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "preflight": report,
                "selection": selection_report,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        pipe,
        image_encoder,
        image_proj_model,
        fusion_module,
        adapter,
        deformation_head,
        ip_report,
    ) = load_models(args, paths, baseline, device)
    (args.output_dir / "model_contract.json").write_text(
        json.dumps(
            {
                "checkpoint": str(paths["root"]),
                "ip_loading": ip_report,
                "v8_condition": "base_CLIP(full) + fusion(BG_only, ROI_only)",
                "brushnet_condition": "[z_input + delta_z_BG, M_BG]",
                "noise": f"{args.state_scope}:{args.offset_mode} final_noise",
                "lineage_recurrence": (
                    "normalized lineage across sequence; never final fused noise"
                    if args.state_scope == "sequence"
                    else "fresh normalized lineage per clip; never final fused noise"
                ),
                "old_shared_background_mixer": False,
                "overlap_noise_reuse": (
                    "exact" if args.state_scope == "sequence" else "none"
                ),
                "overlap_output_reuse": (
                    "first owner clip"
                    if args.state_scope == "sequence"
                    else "none; every clip occurrence is generated"
                ),
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

    current_video = None
    current_state = None
    previous_start = None
    finished_videos = set()
    owner_paths: Dict[str, Dict[int, Path]] = {
        "raw": {},
        "final": {},
        "input": {},
        "gt": {},
        "mask_roi": {},
    }
    sequence_manifest: Dict[str, Dict[str, object]] = {}
    clip_occurrence_count = 0

    progress_name = f"STC-v2++ {args.state_scope}/{args.offset_mode}"
    for index in tqdm(indices, desc=progress_name):
        sample = dataset[index]
        clip_occurrence_count += len(sample["frame_ids"])
        video = str(sample["video"])
        frame_ids = tuple(int(value) for value in sample["frame_ids"].tolist())
        clip_start = frame_ids[0]
        if video != current_video:
            if video in finished_videos:
                raise RuntimeError(
                    f"Dataset returned non-contiguous clips for sequence {video}"
                )
            if current_video is not None:
                finished_videos.add(current_video)
            current_video = video
            previous_start = None
            current_state = (
                SequenceNoiseState(
                    sequence_id=video,
                    seed=args.noise_seed,
                    channels=pipe.unet.config.in_channels,
                    height=args.resolution // pipe.vae_scale_factor,
                    width=args.resolution // pipe.vae_scale_factor,
                )
                if args.state_scope == "sequence"
                else None
            )
            owner_paths = {kind: {} for kind in owner_paths}
            sequence_manifest[video] = {
                "frame_owners": {},
                "clips": [],
            }
        if previous_start is not None and clip_start <= previous_start:
            raise RuntimeError(
                f"Clips for {video} are not strictly ordered: "
                f"previous start={previous_start}, current={clip_start}"
            )
        previous_start = clip_start

        condition_seed = stable_seed(
            args.condition_seed,
            "condition",
            video,
            clip_start,
        )
        brushnet_condition, stc_output, condition_stats = (
            build_stc_condition_and_features(
                pipe=pipe,
                adapter=adapter,
                sample=sample,
                device=device,
                condition_seed=condition_seed,
                injection_scale=args.stc_injection_scale,
            )
        )
        if args.state_scope == "sequence":
            deformation_output = build_sequence_deformed_noise(
                deformation_head=deformation_head,
                stc_features=stc_output.features.float(),
                frame_ids=frame_ids,
                state=current_state,
                alpha=args.transport_alpha,
                warp_scope=args.warp_scope,
                bg_mask=stc_output.latent_bg_mask.float(),
                normalization_eps=args.lineage_normalization_eps,
                offset_mode=args.offset_mode,
            )
        else:
            deformation_output = build_clip_local_deformed_noise(
                deformation_head=deformation_head,
                stc_features=stc_output.features.float(),
                frame_ids=frame_ids,
                noise_channels=pipe.unet.config.in_channels,
                noise_seed=args.noise_seed,
                video=video,
                alpha=args.transport_alpha,
                warp_scope=args.warp_scope,
                bg_mask=stc_output.latent_bg_mask.float(),
                normalization_eps=args.lineage_normalization_eps,
                offset_mode=args.offset_mode,
            )
        diagnostics = noise_diagnostics(deformation_output)
        clip_dir = clip_directory(args.output_dir, video, frame_ids)
        was_complete = clip_is_complete(clip_dir, frame_ids)

        generated_ids = set(deformation_output.generated_frame_ids)
        if was_complete and not args.overwrite:
            for frame_id in deformation_output.generated_frame_ids:
                if args.state_scope == "sequence":
                    for kind in owner_paths:
                        path = clip_dir / kind / f"{frame_id:06d}.png"
                        if not path.is_file():
                            raise RuntimeError(f"Incomplete resumed owner output: {path}")
                        owner_paths[kind][frame_id] = path
                sequence_manifest[video]["frame_owners"].setdefault(
                    str(frame_id), clip_dir.name
                )
            sequence_manifest[video]["clips"].append(
                {
                    "clip_index": index,
                    "frame_ids": list(frame_ids),
                    "generated_frame_ids": list(deformation_output.generated_frame_ids),
                    "reused_frame_ids": list(deformation_output.reused_frame_ids),
                    "resumed": True,
                }
            )
            # Sequence state is rebuilt deterministically even when image
            # outputs already exist.  Therefore an optional diagnostic tensor
            # requested on a later resume can still be materialized without
            # regenerating any image.
            if args.save_noise_tensors:
                torch.save(
                    {
                        "video": video,
                        "frame_ids": frame_ids,
                        "final_noise": deformation_output.final_noise.detach().cpu(),
                        "lineage_noise": deformation_output.lineage_noise.detach().cpu(),
                        "offsets": deformation_output.offsets.detach().cpu(),
                        "valid_masks": deformation_output.valid_masks.detach().cpu(),
                    },
                    clip_dir / "noise_diagnostics.pt",
                )
            del brushnet_condition, stc_output, deformation_output
            torch.cuda.empty_cache()
            continue

        new_indices = [
            position
            for position, frame_id in enumerate(frame_ids)
            if frame_id in generated_ids
        ]
        input_images = tensor_to_pil(sample["conditioning_pixel_values"])
        roi_masks = masks_to_pil(1.0 - sample["masks"])
        raw_new: List[Image.Image] = []
        final_new: List[Image.Image] = []
        if new_indices:
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
            condition_new = selected_rows(brushnet_condition, new_indices)
            condition_cfg = torch.cat((condition_new, condition_new), dim=0)
            shaped_noise = deformation_output.final_noise[0, new_indices].to(
                device=device,
                dtype=pipe.unet.dtype,
            )
            generation_seed = stable_seed(
                args.generation_seed,
                "generation",
                video,
                clip_start,
            )
            raw_new = list(
                pipe(
                    prompt_embeds=selected_rows(prompt_embeds, new_indices),
                    negative_prompt_embeds=selected_rows(
                        negative_prompt_embeds, new_indices
                    ),
                    image=[input_images[position] for position in new_indices],
                    mask=[roi_masks[position] for position in new_indices],
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=float(args.guidance_scale),
                    generator=torch.Generator(device=device).manual_seed(
                        generation_seed
                    ),
                    latents=shaped_noise,
                    use_shared_bg_noise=False,
                    shared_bg_noise=None,
                    brushnet_condition=condition_cfg,
                    brushnet_prompt_embeds=torch.cat(
                        (
                            selected_rows(negative_brushnet_text, new_indices),
                            selected_rows(brushnet_text, new_indices),
                        ),
                        dim=0,
                    ),
                    brushnet_conditioning_scale=float(
                        args.brushnet_conditioning_scale
                    ),
                ).images
            )
            final_new = composite_images(
                raw_new,
                sample["conditioning_pixel_values"][new_indices],
                sample["masks"][new_indices],
                mode=args.roi_composite,
                blur_kernel_size=args.roi_blur_kernel_size,
            )
            new_frame_ids = [frame_ids[position] for position in new_indices]
            save_images(raw_new, clip_dir / "raw", new_frame_ids)
            save_images(final_new, clip_dir / "final", new_frame_ids)
            for frame_id in new_frame_ids:
                if args.state_scope == "sequence":
                    owner_paths["raw"][frame_id] = (
                        clip_dir / "raw" / f"{frame_id:06d}.png"
                    )
                    owner_paths["final"][frame_id] = (
                        clip_dir / "final" / f"{frame_id:06d}.png"
                    )

        # References are saved for every clip so existing video/temporal tools
        # can consume this evaluator without a flat-dataset fallback.
        save_images(input_images, clip_dir / "input", frame_ids)
        save_images(tensor_to_pil(sample["pixel_values"]), clip_dir / "gt", frame_ids)
        save_images(roi_masks, clip_dir / "mask_roi", frame_ids)
        for frame_id in deformation_output.generated_frame_ids:
            if args.state_scope == "sequence":
                owner_paths["input"][frame_id] = (
                    clip_dir / "input" / f"{frame_id:06d}.png"
                )
                owner_paths["gt"][frame_id] = clip_dir / "gt" / f"{frame_id:06d}.png"
                owner_paths["mask_roi"][frame_id] = (
                    clip_dir / "mask_roi" / f"{frame_id:06d}.png"
                )
            sequence_manifest[video]["frame_owners"].setdefault(
                str(frame_id), clip_dir.name
            )

        # Populate every overlap occurrence from its authoritative first owner.
        if args.state_scope == "sequence":
            for frame_id in frame_ids:
                for kind in owner_paths:
                    if frame_id not in owner_paths[kind]:
                        raise RuntimeError(
                            f"Missing owner for {video}:{frame_id} kind={kind}"
                        )
                    copy_image(
                        owner_paths[kind][frame_id],
                        clip_dir / kind / f"{frame_id:06d}.png",
                    )

        raw_clip = [
            Image.open(clip_dir / "raw" / f"{frame_id:06d}.png")
            .convert("RGB")
            .copy()
            for frame_id in frame_ids
        ]
        final_clip = [
            Image.open(clip_dir / "final" / f"{frame_id:06d}.png")
            .convert("RGB")
            .copy()
            for frame_id in frame_ids
        ]
        metrics = compute_metrics(final_clip, sample["pixel_values"], sample["masks"])
        metrics.update(
            {
                f"raw_{key}": value
                for key, value in compute_metrics(
                    raw_clip,
                    sample["pixel_values"],
                    sample["masks"],
                ).items()
            }
        )
        metrics.update(condition_stats)
        clip_dir.mkdir(parents=True, exist_ok=True)
        (clip_dir / "clip_metrics.json").write_text(
            json.dumps(
                {
                    "clip_index": index,
                    "video": video,
                    "frame_ids": list(frame_ids),
                    "generated_frame_ids": list(
                        deformation_output.generated_frame_ids
                    ),
                    "reused_frame_ids": list(deformation_output.reused_frame_ids),
                    "condition_seed": condition_seed,
                    "metrics": metrics,
                    "noise_diagnostics": diagnostics,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if args.save_noise_tensors:
            torch.save(
                {
                    "video": video,
                    "frame_ids": frame_ids,
                    "final_noise": deformation_output.final_noise.detach().cpu(),
                    "lineage_noise": deformation_output.lineage_noise.detach().cpu(),
                    "offsets": deformation_output.offsets.detach().cpu(),
                    "valid_masks": deformation_output.valid_masks.detach().cpu(),
                },
                clip_dir / "noise_diagnostics.pt",
            )
        sequence_manifest[video]["clips"].append(
            {
                "clip_index": index,
                "frame_ids": list(frame_ids),
                "generated_frame_ids": list(deformation_output.generated_frame_ids),
                "reused_frame_ids": list(deformation_output.reused_frame_ids),
                "resumed": False,
            }
        )
        del brushnet_condition, stc_output, deformation_output, raw_new, final_new
        torch.cuda.empty_cache()

    (args.output_dir / "sequence_manifest.json").write_text(
        json.dumps(sequence_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = aggregate_metrics(args.output_dir)
    summary.update(
        {
            "checkpoint": str(paths["root"]),
            "sequence_state": args.state_scope == "sequence",
            "state_scope": args.state_scope,
            "offset_mode": args.offset_mode,
            "sequences": len(sequence_manifest),
            "unique_frames": sum(
                len(record["frame_owners"])
                for record in sequence_manifest.values()
            ),
            "clip_frame_occurrences": clip_occurrence_count,
            "overlap_output_policy": (
                "first owner; exact copied reuse"
                if args.state_scope == "sequence"
                else "each clip occurrence generated independently"
            ),
            "metric_note": "clip mean includes repeated overlap occurrences",
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
