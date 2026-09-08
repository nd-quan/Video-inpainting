#!/usr/bin/env python
"""Evaluate V8 with composited CGE and V7-student temporal guidance.

This is a dedicated V8 entry point.  It retains the frozen V7 RAFT student
for V8's deformable-condition flow prior, then uses the *same* student on the
degraded RGB clip to construct visible, stable-background temporal pairs for
sampling.  The scheduler composes two independent corrections in each DDIM
transition:

``x_(t-1) = DDIM(x_t) + g_CGE + sqrt(alpha_bar_(t-1)) * Delta_x0_temporal``.

``g_CGE`` is the existing VCM-RS background-only image-space guidance.  The
temporal term is deliberately latent-space only; CGE has already paid for the
differentiable VAE decode at that step, and a second RGB temporal decode would
needlessly increase both memory use and runtime.  Both terms are clip-local.
V8 cross-clip conditioning remains intact because a complete video branch is
assigned to one evaluator shard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from diffusers.schedulers.scheduling_ddim_CGE import cond_fn  # noqa: E402
from diffusers.schedulers.scheduling_ddim_cge_temporal import (  # noqa: E402
    CGETemporalDDIMScheduler,
)
from diffusers.schedulers.scheduling_ddim_temporal import (  # noqa: E402
    build_stable_bg_mask,
)
from STC_encoder_v2_rgb import evaluate_rgb_stc_shared_noise as evaluator  # noqa: E402
from STC_encoder_v4_flow_aligned import (  # noqa: E402
    evaluate_flow_aligned_stc as v4_evaluator,
)
from STC_encoder_v5_relative_crossclip import (  # noqa: E402
    evaluate_v5_relative_crossclip as v5_evaluator,
)
from STC_encoder_v5_relative_crossclip import (  # noqa: E402
    evaluate_v5_relative_crossclip_temporal_guidance as temporal_v5,
)
from STC_encoder_v8_raft_deformable import (  # noqa: E402
    evaluate_v8_raft_deformable as v8_evaluator,
)
from STC_encoder_v8_raft_deformable.raft_guided_deformable_stc_adapter import (  # noqa: E402
    RAFTGuidedDeformableBGSTCAdapter,
)
from vcmrs_codec_adapter import VCMRSBackgroundOnlyCodec  # noqa: E402


_BASE_LOAD_MODELS = evaluator.load_models


def _add_evaluation_arguments(parser) -> None:
    """Add V8, latent-temporal, CGE, and whole-video sharding controls."""

    v8_evaluator._add_evaluation_arguments(parser)
    temporal_v5.add_temporal_guidance_arguments(parser)
    # Mirror the fixedBG combined protocol: both corrections act in the same
    # late denoising window.  ``temporal_loss_scale`` is accepted for CLI
    # compatibility but intentionally unused by the latent-only scheduler.
    parser.set_defaults(
        temporal_start_step=25,
        temporal_end_step=35,
        temporal_guidance_scale=1.0e-4,
    )
    parser.add_argument("--cge_guidance_scale", type=float, default=1.0e-4)
    parser.add_argument(
        "--cge_scale_schedule",
        choices=("fixed", "noise_level"),
        default="fixed",
        help=(
            "CGE gradient scale: fixed=s_tilde, noise_level="
            "sqrt(1-alpha_bar_t)*s_tilde."
        ),
    )
    parser.add_argument("--cge_start_step", type=int, default=25)
    parser.add_argument(
        "--cge_end_step",
        type=int,
        default=35,
        help="Exclusive CGE end step; use --num_inference_steps to run to the end.",
    )
    parser.add_argument("--cge_every_n_steps", type=int, default=1)
    parser.add_argument(
        "--cge_max_evals",
        type=int,
        default=2,
        help="Maximum costly VCM-RS evaluations per clip; -1 means unlimited.",
    )
    parser.add_argument("--cge_decode_chunk_size", type=int, default=1)
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Partition complete video branches across independent GPUs.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Zero-based branch shard index; each shard needs its own output directory.",
    )


def _validate_guidance_arguments(args) -> None:
    if args.temporal_guidance_scale < 0.0:
        raise ValueError("temporal_guidance_scale must be non-negative")
    if args.temporal_start_step < 0 or args.temporal_start_step >= args.temporal_end_step:
        raise ValueError("temporal guidance window must satisfy 0 <= start < end")
    if args.temporal_end_step > args.num_inference_steps:
        raise ValueError("temporal_end_step cannot exceed num_inference_steps")
    if args.temporal_every_n_steps < 1:
        raise ValueError("temporal_every_n_steps must be positive")
    if args.temporal_decode_chunk_size < 1:
        raise ValueError("temporal_decode_chunk_size must be positive")
    if args.temporal_loss_scale <= 0.0:
        raise ValueError("temporal_loss_scale must be positive")
    if args.temporal_flow_batch_size < 1:
        raise ValueError("temporal_flow_batch_size must be positive")
    if args.temporal_visibility_alpha < 0.0 or args.temporal_visibility_beta < 0.0:
        raise ValueError("temporal visibility alpha/beta must be non-negative")

    if args.cge_guidance_scale < 0.0:
        raise ValueError("cge_guidance_scale must be non-negative")
    if args.cge_start_step < 0 or args.cge_start_step >= args.cge_end_step:
        raise ValueError("CGE window must satisfy 0 <= cge_start_step < cge_end_step")
    if args.cge_end_step > args.num_inference_steps:
        raise ValueError("cge_end_step cannot exceed num_inference_steps")
    if args.cge_every_n_steps < 1:
        raise ValueError("cge_every_n_steps must be positive")
    if args.cge_max_evals < -1:
        raise ValueError("cge_max_evals must be -1 or non-negative")
    if args.cge_decode_chunk_size < 1:
        raise ValueError("cge_decode_chunk_size must be positive")
    if args.num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")


def _select_branch_shard(dataset, args) -> Dict[str, object]:
    """Assign each whole video to one process, preserving V8 memory state."""

    branches = sorted({str(video) for video, _, _ in dataset.clips})
    assigned = [
        branch
        for index, branch in enumerate(branches)
        if index % int(args.num_shards) == int(args.shard_index)
    ]
    if not assigned:
        raise ValueError(
            f"Shard {args.shard_index}/{args.num_shards} has no video branches; "
            f"available={branches}"
        )
    selected = set(assigned)
    dataset.clips = [clip for clip in dataset.clips if str(clip[0]) in selected]
    if hasattr(dataset, "rebuild_predecessors"):
        dataset.rebuild_predecessors()
    dataset.covered_frame_count = len(
        {path for _, paths, _ in dataset.clips for path in paths}
    )
    dataset.branch_count = len(assigned)
    return {
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "branches": assigned,
        "clip_count": int(len(dataset.clips)),
        "covered_frame_count": int(dataset.covered_frame_count),
    }


def preflight(args):
    """Validate all V8/CGE resources before allocating diffusion models."""

    _validate_guidance_arguments(args)
    # Resolves the V7 component and makes V8's shared GPU-resident provider
    # contract active for both condition construction and sampler guidance.
    dataset, paths = v8_evaluator.preflight(args)
    shard = _select_branch_shard(dataset, args)
    codec = VCMRSBackgroundOnlyCodec.from_env()
    print(
        json.dumps(
            {
                "v8_cge_temporal": "ok",
                "scheduler": "CGETemporalDDIMScheduler",
                "v8_condition_flow": "frozen_V7_RAFT_student",
                "guidance_flow": "frozen_V7_RAFT_student_bidirectional",
                "guidance_flow_input": "degraded RGB [-1,1] only",
                "temporal_region": "forward_backward_visible_stable_BG",
                "temporal_space": "latent_predicted_x0",
                "temporal_window": [
                    int(args.temporal_start_step),
                    int(args.temporal_end_step),
                ],
                "temporal_scale": float(args.temporal_guidance_scale),
                "cge_region": "background_only_M_BG",
                "cge_operator": str(codec.cge_operator),
                "cge_window": [int(args.cge_start_step), int(args.cge_end_step)],
                "cge_scale": float(args.cge_guidance_scale),
                "cge_scale_schedule": str(args.cge_scale_schedule),
                "cge_max_evals_per_clip": int(args.cge_max_evals),
                "clip_local_scheduler_guidance": True,
                "cross_clip_scheduler_state": False,
                "shard": shard,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return dataset, paths


def load_models_with_cge_temporal_scheduler(args, paths, device: torch.device):
    loaded = _BASE_LOAD_MODELS(args, paths, device)
    pipe = loaded[0]
    scheduler = CGETemporalDDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.per_frame_cge = True
    scheduler.decode_chunk_size = int(args.cge_decode_chunk_size)
    scheduler.vae_scaling_factor = float(pipe.vae.config.scaling_factor)
    scheduler.cge_codec = VCMRSBackgroundOnlyCodec.from_env()
    scheduler.guidance_scale_cge = float(args.cge_guidance_scale)
    scheduler.cge_scale_schedule = str(args.cge_scale_schedule)
    scheduler.cge_start_step = int(args.cge_start_step)
    scheduler.cge_end_step = int(args.cge_end_step)
    scheduler.cge_every_n_steps = int(args.cge_every_n_steps)
    scheduler.cge_max_evals = int(args.cge_max_evals)
    scheduler.direct_cge_guidance = True
    pipe.scheduler = scheduler
    return loaded


@torch.inference_mode()
def build_v8_condition_with_shared_student(
    pipe,
    adapter: RAFTGuidedDeformableBGSTCAdapter,
    sample: Mapping[str, torch.Tensor],
    device: torch.device,
    condition_seed: int,
    injection_scale: float,
    deformable_alignment_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Build the normal V8 condition without offloading the V7 provider."""

    if v8_evaluator._ACTIVE_EVAL_ARGS is None:
        raise RuntimeError("V8 CGE-temporal provider was not initialized")
    return v8_evaluator.build_v8_condition(
        pipe,
        adapter,
        sample,
        device,
        condition_seed,
        injection_scale,
        deformable_alignment_scale=deformable_alignment_scale,
    )


def _configure_cge(
    scheduler: CGETemporalDDIMScheduler,
    pipe,
    sample: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, object]:
    """Attach exact degraded frames and per-frame ROI masks to CGE."""

    # V8/STC masks are M_BG=1, while VCM-RS CGE expects M_ROI=1.
    x_lr = sample["conditioning_pixel_values"].to(
        device=device, dtype=torch.float32
    )
    roi_mask = 1.0 - sample["masks"].to(device=device, dtype=torch.float32)
    roi_mask = (roi_mask > 0.5).to(dtype=torch.float32)
    if x_lr.ndim != 4 or roi_mask.ndim != 4:
        raise ValueError("V8 CGE requires clip tensors with shape [T,C,H,W]")
    if x_lr.shape[0] != roi_mask.shape[0] or x_lr.shape[-2:] != roi_mask.shape[-2:]:
        raise ValueError("V8 CGE input and ROI-mask clip shapes do not match")
    if x_lr.shape[0] > 1 and (
        scheduler.cge_codec.roi_descriptor is not None
        or scheduler.cge_codec.bg_descriptor is not None
    ):
        raise ValueError(
            "V8 clips have a different ROI mask per frame. Do not set "
            "CGE_VCMRS_ROI_DESCRIPTOR/CGE_VCMRS_BG_DESCRIPTOR; use "
            "CGE_VCMRS_AUTO_DESCRIPTORS=1."
        )

    # VCM-RS handles one frame at a time.  Cache/prepare its descriptor for
    # every frame before DDIM; ``cond_fn`` then performs each chosen roundtrip.
    for frame_index in range(int(x_lr.shape[0])):
        scheduler.cge_codec.prepare_region_mask(
            roi_mask[frame_index], x_lr[frame_index]
        )
    scheduler.x_lr = x_lr
    scheduler.mask = roi_mask
    scheduler.decoder = pipe.vae.decode
    scheduler.cond_fn = cond_fn
    scheduler.cge_codec_eval_count = 0
    scheduler.cge_denoise_step_count = 0
    scheduler.last_cge_guidance_scale_base = None
    scheduler.last_cge_guidance_scale_multiplier = None
    scheduler.last_cge_guidance_scale_effective = None
    scheduler.last_cge_scale_schedule = None
    return {
        "cge_enabled": 1,
        "cge_operator": str(scheduler.cge_codec.cge_operator),
        "cge_clip_frame_count": int(x_lr.shape[0]),
        "cge_roi_ratio": float(roi_mask.mean().detach().cpu()),
        "cge_profile": str(scheduler.cge_codec.profile),
        "cge_bg_qp": int(scheduler.cge_codec.bg_quality),
        "cge_roi_qp": None,
        "cge_scale_schedule": str(getattr(scheduler, "cge_scale_schedule", "fixed")),
    }


def _configure_temporal_guidance(
    scheduler: CGETemporalDDIMScheduler,
    sample: Mapping[str, torch.Tensor],
    args,
    device: torch.device,
) -> Dict[str, object]:
    """Build V7-student flow pairs and attach latent temporal guidance."""

    if args.temporal_guidance_scale == 0.0:
        scheduler.clear_temporal_guidance()
        return {"temporal_guidance_enabled": 0}
    if v8_evaluator._ACTIVE_EVAL_ARGS is None:
        raise RuntimeError("V8 CGE-temporal provider was not initialized")

    rgb = sample["conditioning_pixel_values"].to(device=device, dtype=torch.float32)
    bg_masks = sample["masks"].to(device=device, dtype=torch.float32)
    provider = v8_evaluator._provider(v8_evaluator._ACTIVE_EVAL_ARGS, device)
    prediction = provider.predict_sequence(rgb.unsqueeze(0))
    flow_forward = prediction.forward.squeeze(0)
    flow_backward = prediction.backward.squeeze(0)
    visibility = temporal_v5.forward_backward_visibility(
        flow_forward,
        flow_backward,
        alpha=float(args.temporal_visibility_alpha),
        beta=float(args.temporal_visibility_beta),
    )
    stable_bg = build_stable_bg_mask(
        bg_masks=bg_masks,
        flow_backward=flow_backward,
        visibility=visibility,
        threshold=0.5,
    )
    # CGETemporalDDIMScheduler intentionally has a latent-only API: do not
    # pass a VAE decoder, RGB mask, or RGB loss scale here.
    scheduler.set_temporal_guidance(
        flow_backward=flow_backward,
        stable_bg=stable_bg,
        guidance_scale=float(args.temporal_guidance_scale),
        start_step=int(args.temporal_start_step),
        end_step=int(args.temporal_end_step),
        every_n_steps=int(args.temporal_every_n_steps),
        detach_previous=bool(args.temporal_detach_previous),
        enabled=True,
        loss_type=str(args.temporal_loss_type),
    )
    return {
        "temporal_guidance_enabled": 1,
        "temporal_pair_count": int(flow_backward.shape[0]),
        "temporal_stable_bg_ratio": float(stable_bg.mean().detach().cpu()),
        "temporal_visibility_ratio": float(visibility.mean().detach().cpu()),
        "temporal_flow_backend": "v7_student",
        "temporal_flow_backward_magnitude": float(
            flow_backward.square().sum(dim=1).sqrt().mean().detach().cpu()
        ),
    }


def before_pipeline_call(
    *,
    pipe,
    sample: Mapping[str, torch.Tensor],
    args,
    device: torch.device,
) -> Dict[str, object]:
    scheduler = pipe.scheduler
    if not isinstance(scheduler, CGETemporalDDIMScheduler):
        raise TypeError("V8 CGE-temporal evaluator requires CGETemporalDDIMScheduler")
    stats = _configure_cge(scheduler, pipe, sample, device)
    stats.update(_configure_temporal_guidance(scheduler, sample, args, device))
    return stats


def after_pipeline_call(
    *, pipe, sample: Mapping[str, torch.Tensor], args, device: torch.device
) -> Dict[str, object]:
    """Record non-overlapping CGE and latent-temporal scheduler diagnostics."""

    del sample, args, device
    scheduler = pipe.scheduler
    stats: Dict[str, object] = {
        "cge_codec_evaluations": int(scheduler.cge_codec_eval_count),
        "cge_denoise_steps_seen": int(scheduler.cge_denoise_step_count),
        "temporal_guidance_calls": int(scheduler.temporal_guidance_calls),
        "temporal_guidance_applied_steps": int(
            scheduler.temporal_guidance_applied_steps
        ),
        "temporal_guidance_skipped_steps": int(
            scheduler.temporal_guidance_skipped_steps
        ),
    }
    for metric_name, value in (
        ("cge_last_guidance_scale_base", getattr(scheduler, "last_cge_guidance_scale_base", None)),
        ("cge_last_guidance_scale_multiplier", getattr(scheduler, "last_cge_guidance_scale_multiplier", None)),
        ("cge_last_guidance_scale_effective", getattr(scheduler, "last_cge_guidance_scale_effective", None)),
        ("temporal_last_loss", scheduler.last_temporal_loss),
        ("temporal_last_update_norm", scheduler.last_temporal_update_norm),
    ):
        if value is not None:
            stats[metric_name] = float(value)
    if scheduler.last_temporal_skipped_reason is not None:
        stats["temporal_last_skipped_reason"] = str(
            scheduler.last_temporal_skipped_reason
        )
    return stats


def clear_pipeline_call(*, pipe) -> None:
    """Ensure overlapping clips cannot inherit either guidance state."""

    scheduler = pipe.scheduler
    if isinstance(scheduler, CGETemporalDDIMScheduler):
        scheduler.clear_temporal_guidance()
        scheduler.cond_fn = None
        scheduler.decoder = None
        scheduler.x_lr = None
        scheduler.mask = None
        scheduler.cge_codec_eval_count = 0
        scheduler.cge_denoise_step_count = 0


def main() -> None:
    # V5 delegates its cross-clip condition through V4/V2.  Install V8's
    # adapter and our combined scheduler hooks at every delegation point.
    evaluator.ADD_EVALUATION_ARGUMENTS_FN = _add_evaluation_arguments
    evaluator.CONDITION_EXTRA_KWARGS_FN = v8_evaluator._condition_extra_kwargs
    evaluator.BEFORE_PIPELINE_CALL_FN = before_pipeline_call
    evaluator.AFTER_PIPELINE_CALL_FN = after_pipeline_call
    evaluator.CLEAR_PIPELINE_CALL_FN = clear_pipeline_call
    evaluator.load_models = load_models_with_cge_temporal_scheduler

    v5_evaluator.RelativeCrossClipBGSTCAdapter = RAFTGuidedDeformableBGSTCAdapter
    v5_evaluator.preflight = preflight
    v5_evaluator.build_v5_condition = build_v8_condition_with_shared_student
    v4_evaluator.FlowAlignedRGBSTCAdapter = RAFTGuidedDeformableBGSTCAdapter
    v4_evaluator.preflight = preflight
    v4_evaluator.build_flow_aligned_condition = build_v8_condition_with_shared_student
    v5_evaluator.main()


if __name__ == "__main__":
    main()
