#!/usr/bin/env python
"""Evaluate V8 with V7-student flow-guided temporal DDIM sampling.

The frozen V7 RAFT student has two deliberately separate inference roles:

* it supplies V8's RGB-flow prior for deformable STC conditioning; and
* it supplies local bidirectional RGB flow for training-free DDIM temporal
  guidance on visible, stable background regions.

Both roles use degraded RGB in ``[-1, 1]``.  They intentionally share exactly
one GPU-resident V7 instance: moving ProPainter-RAFT parameters between CPU
and GPU inside the inherited ``torch.inference_mode`` condition hook creates
inference tensors that BatchNorm cannot safely reuse.
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

from diffusers.schedulers.scheduling_ddim_temporal import (  # noqa: E402
    TemporalDDIMScheduler,
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


_BASE_LOAD_MODELS = evaluator.load_models


def _add_evaluation_arguments(parser) -> None:
    """Combine V8 condition, temporal-DDIM, and whole-video shard controls."""
    v8_evaluator._add_evaluation_arguments(parser)
    temporal_v5.add_temporal_guidance_arguments(parser)
    parser.add_argument(
        "--temporal_guidance_space",
        choices=("rgb", "latent"),
        default="rgb",
        help=(
            "Compute temporal DDIM guidance after VAE decoding (rgb) or "
            "directly on predicted-clean VAE latents (latent)."
        ),
    )
    # Match the established temporal-student protocol for 50 DDIM steps.
    parser.set_defaults(temporal_start_step=25, temporal_end_step=35)
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


def _select_branch_shard(dataset, args) -> Dict[str, object]:
    """Keep every clip of one video on one GPU, preserving cross-clip context."""
    if args.num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    branches = sorted({str(video) for video, _, _ in dataset.clips})
    assigned = [
        branch
        for index, branch in enumerate(branches)
        if index % args.num_shards == args.shard_index
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
        "clip_count": len(dataset.clips),
        "covered_frame_count": int(dataset.covered_frame_count),
    }


def preflight(args):
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

    # This resolves the V7 component, validates V8 architecture, and sets the
    # V8 module's process-local flow provider contract.
    dataset, paths = v8_evaluator.preflight(args)
    shard = _select_branch_shard(dataset, args)
    print(
        json.dumps(
            {
                "v8_temporal_student": "ok",
                "v8_condition_flow": "frozen_V7_RAFT_student",
                "guidance_flow": "frozen_V7_RAFT_student_bidirectional",
                "guidance_flow_input": "degraded RGB [-1,1] only",
                "guidance_region": "forward_backward_visible_stable_BG",
                "guidance_window": [
                    int(args.temporal_start_step),
                    int(args.temporal_end_step),
                ],
                "guidance_scale": float(args.temporal_guidance_scale),
                "guidance_space": str(args.temporal_guidance_space),
                "clip_local_scheduler_guidance": True,
                "cross_clip_scheduler_state": False,
                "shard": shard,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return dataset, paths


def load_models_with_temporal_scheduler(args, paths, device):
    loaded = _BASE_LOAD_MODELS(args, paths, device)
    pipe = loaded[0]
    pipe.scheduler = TemporalDDIMScheduler.from_config(pipe.scheduler.config)
    return loaded


@torch.inference_mode()
def build_v8_condition_with_temporal_provider(
    pipe,
    adapter: RAFTGuidedDeformableBGSTCAdapter,
    sample: Mapping[str, torch.Tensor],
    device: torch.device,
    condition_seed: int,
    injection_scale: float,
    deformable_alignment_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Build V8 condition while retaining its shared V7 provider on CUDA."""
    if v8_evaluator._ACTIVE_EVAL_ARGS is None:
        raise RuntimeError("V8 temporal evaluator provider was not initialized")
    return v8_evaluator.build_v8_condition(
        pipe,
        adapter,
        sample,
        device,
        condition_seed,
        injection_scale,
        deformable_alignment_scale=deformable_alignment_scale,
    )


def before_pipeline_call(
    *,
    pipe,
    sample: Mapping[str, torch.Tensor],
    args,
    device: torch.device,
) -> Dict[str, object]:
    if not isinstance(pipe.scheduler, TemporalDDIMScheduler):
        raise TypeError("V8 temporal evaluator requires TemporalDDIMScheduler")
    if args.temporal_guidance_scale == 0.0:
        pipe.scheduler.clear_temporal_guidance()
        return {"temporal_guidance_enabled": 0}
    if v8_evaluator._ACTIVE_EVAL_ARGS is None:
        raise RuntimeError("V8 temporal evaluator provider was not initialized")

    # Recompute the local V7 RGB flow for sampler guidance.  This is distinct
    # from V8's feature-grid conversion, but uses the same frozen model and
    # identical degraded-RGB normalization.
    rgb = sample["conditioning_pixel_values"].to(device=device, dtype=torch.float32)
    bg_masks = sample["masks"].to(device=device, dtype=torch.float32)
    provider = v8_evaluator._provider(v8_evaluator._ACTIVE_EVAL_ARGS, device)
    prediction = provider.predict_sequence(rgb.unsqueeze(0))
    flow_forward = prediction.forward.squeeze(0)
    flow_backward = prediction.backward.squeeze(0)

    visibility = temporal_v5.forward_backward_visibility(
        flow_forward,
        flow_backward,
        alpha=args.temporal_visibility_alpha,
        beta=args.temporal_visibility_beta,
    )
    stable_bg = build_stable_bg_mask(
        bg_masks=bg_masks,
        flow_backward=flow_backward,
        visibility=visibility,
        threshold=0.5,
    )
    pipe.scheduler.set_temporal_guidance(
        decoder=(
            pipe.vae.decode
            if str(args.temporal_guidance_space) == "rgb"
            else None
        ),
        flow_backward=flow_backward,
        stable_bg=stable_bg,
        # Keep the per-frame latent update strictly inside each frame's BG.
        # Pairwise stable masks alone mix coordinates from adjacent frames and
        # can otherwise let a moving ROI receive a temporal update.
        bg_masks=bg_masks,
        guidance_scale=float(args.temporal_guidance_scale),
        start_step=int(args.temporal_start_step),
        end_step=int(args.temporal_end_step),
        every_n_steps=int(args.temporal_every_n_steps),
        decode_chunk_size=int(args.temporal_decode_chunk_size),
        vae_scaling_factor=float(pipe.vae.config.scaling_factor),
        loss_scale=float(args.temporal_loss_scale),
        detach_previous=bool(args.temporal_detach_previous),
        enabled=True,
        loss_type=str(args.temporal_loss_type),
        guidance_space=str(args.temporal_guidance_space),
    )
    return {
        "temporal_guidance_enabled": 1,
        "temporal_pair_count": int(flow_backward.shape[0]),
        "temporal_stable_bg_ratio": float(stable_bg.mean().detach().cpu()),
        "temporal_visibility_ratio": float(visibility.mean().detach().cpu()),
        "temporal_flow_backend": "v7_student",
        "temporal_guidance_space": str(args.temporal_guidance_space),
        "temporal_flow_backward_magnitude": float(
            flow_backward.square().sum(dim=1).sqrt().mean().detach().cpu()
        ),
    }


def main() -> None:
    evaluator.ADD_EVALUATION_ARGUMENTS_FN = _add_evaluation_arguments
    evaluator.CONDITION_EXTRA_KWARGS_FN = v8_evaluator._condition_extra_kwargs
    evaluator.BEFORE_PIPELINE_CALL_FN = before_pipeline_call
    evaluator.AFTER_PIPELINE_CALL_FN = temporal_v5.after_pipeline_call
    evaluator.CLEAR_PIPELINE_CALL_FN = temporal_v5.clear_pipeline_call
    evaluator.load_models = load_models_with_temporal_scheduler

    # V5 delegates cross-clip construction through the V4 entrypoint; install
    # the exact V8 adapter/condition builder at both levels.
    v5_evaluator.RelativeCrossClipBGSTCAdapter = RAFTGuidedDeformableBGSTCAdapter
    v5_evaluator.preflight = preflight
    v5_evaluator.build_v5_condition = build_v8_condition_with_temporal_provider
    v4_evaluator.FlowAlignedRGBSTCAdapter = RAFTGuidedDeformableBGSTCAdapter
    v4_evaluator.preflight = preflight
    v4_evaluator.build_flow_aligned_condition = build_v8_condition_with_temporal_provider
    v5_evaluator.main()


if __name__ == "__main__":
    main()
