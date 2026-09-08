#!/usr/bin/env python
"""Evaluate V6 with RGB temporal DDIM guidance from the frozen V7 RAFT student.

The V6 flow head remains part of the learned deformable STC condition.  The
DDIM guidance flow is deliberately independent: it is estimated from degraded
RGB only by the frozen V7 RAFT student, then applied on forward/backward
visible stable background regions during sampling.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Mapping

import torch


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from diffusers.schedulers.scheduling_ddim_temporal import (
    TemporalDDIMScheduler,
    build_stable_bg_mask,
)
from STC_encoder_v2_rgb import evaluate_rgb_stc_shared_noise as evaluator
from STC_encoder_v5_relative_crossclip import (
    evaluate_v5_relative_crossclip as v5_evaluator,
)
from STC_encoder_v5_relative_crossclip import (
    evaluate_v5_relative_crossclip_temporal_guidance as temporal_v5,
)
from STC_encoder_v6_flow_deformable import (
    evaluate_v6_flow_deformable as v6_evaluator,
)
from STC_encoder_v6_flow_deformable.flow_guided_deformable_stc_adapter import (
    FlowGuidedDeformableBGSTCAdapter,
)
from STC_encoder_v8_raft_deformable.raft_flow_provider import (
    FrozenV7RAFTFlowProvider,
    resolve_raft_student_component,
)


_BASE_LOAD_MODELS = evaluator.load_models
_PROVIDERS: Dict[str, FrozenV7RAFTFlowProvider] = {}


def _add_evaluation_arguments(parser) -> None:
    """Register the V6, temporal-DDIM, and frozen-V7 controls together."""
    v6_evaluator._add_evaluation_arguments(parser)
    temporal_v5.add_temporal_guidance_arguments(parser)
    # Match the fixedBG temporal-student evaluation window used by the
    # standalone temporal evaluator; callers can still override either flag.
    parser.set_defaults(temporal_start_step=25, temporal_end_step=35)
    parser.add_argument(
        "--raft_student_path",
        required=True,
        help="V7 best/latest JSON, checkpoint directory, or raft_student directory.",
    )
    parser.add_argument("--raft_pair_batch_size", type=int, default=1)
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Partition complete video branches across this many independent GPUs.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Zero-based branch shard index; use a distinct --output_dir per shard.",
    )
    parser.add_argument(
        "--raft_mixed_precision",
        dest="raft_mixed_precision",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--raft_no_mixed_precision",
        dest="raft_mixed_precision",
        action="store_false",
    )


def _select_branch_shard(dataset, args) -> Dict[str, object]:
    """Assign whole videos, rather than overlapping clips, to each GPU."""
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
        # Cross-clip V6 context indices refer to positions in ``clips``.
        # Rebuild after branch filtering so no predecessor points into another
        # shard's now-removed clip list.
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
def _provider(args, device: torch.device) -> FrozenV7RAFTFlowProvider:
    component = resolve_raft_student_component(args.raft_student_path)
    key = "|".join(
        (
            str(device),
            str(component),
            str(args.raft_pair_batch_size),
            str(args.raft_mixed_precision),
        )
    )
    provider = _PROVIDERS.get(key)
    if provider is None:
        provider = FrozenV7RAFTFlowProvider(
            component,
            device=device,
            pair_batch_size=args.raft_pair_batch_size,
            mixed_precision=args.raft_mixed_precision,
        )
        # The pipeline uses CPU offload.  Keep RAFT off-GPU except for the
        # pre-sampling flow pass so it does not compete with DDIM memory.
        provider.student.to("cpu")
        _PROVIDERS[key] = provider
    return provider


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
    if args.raft_pair_batch_size < 1:
        raise ValueError("raft_pair_batch_size must be positive")
    args.raft_student_path = str(resolve_raft_student_component(args.raft_student_path))
    dataset, paths = v6_evaluator.preflight(args)
    shard = _select_branch_shard(dataset, args)
    print(
        json.dumps(
            {
                "v6_temporal_student": "ok",
                "v7_raft_student_component": args.raft_student_path,
                "guidance_flow": "frozen_V7_RAFT_student_bidirectional",
                "guidance_flow_input": "degraded RGB [-1,1] only",
                "guidance_region": "forward_backward_visible_stable_BG",
                "guidance_window": [
                    int(args.temporal_start_step),
                    int(args.temporal_end_step),
                ],
                "guidance_scale": float(args.temporal_guidance_scale),
                "v6_condition_flow": "V6 learned flow head",
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


def before_pipeline_call(
    *,
    pipe,
    sample: Mapping[str, torch.Tensor],
    args,
    device: torch.device,
) -> Dict[str, object]:
    if not isinstance(pipe.scheduler, TemporalDDIMScheduler):
        raise TypeError("V6 temporal-student evaluator requires TemporalDDIMScheduler")
    if args.temporal_guidance_scale == 0.0:
        pipe.scheduler.clear_temporal_guidance()
        return {"temporal_guidance_enabled": 0}

    # The flat/hierarchical V6 dataset provides degraded condition RGB in
    # [-1,1].  V7 was trained in that same normalization.
    rgb = sample["conditioning_pixel_values"].to(device=device, dtype=torch.float32)
    bg_masks = sample["masks"].to(device=device, dtype=torch.float32)
    provider = _provider(args, device)
    provider.student.to(device)
    try:
        prediction = provider.predict_sequence(rgb.unsqueeze(0))
        flow_forward = prediction.forward.squeeze(0)
        flow_backward = prediction.backward.squeeze(0)
    finally:
        provider.student.to("cpu")
    torch.cuda.empty_cache()

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
        decoder=pipe.vae.decode,
        flow_backward=flow_backward,
        stable_bg=stable_bg,
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


def main() -> None:
    evaluator.ADD_EVALUATION_ARGUMENTS_FN = _add_evaluation_arguments
    evaluator.CONDITION_EXTRA_KWARGS_FN = v6_evaluator._condition_extra_kwargs
    evaluator.BEFORE_PIPELINE_CALL_FN = before_pipeline_call
    evaluator.AFTER_PIPELINE_CALL_FN = temporal_v5.after_pipeline_call
    evaluator.CLEAR_PIPELINE_CALL_FN = temporal_v5.clear_pipeline_call
    evaluator.load_models = load_models_with_temporal_scheduler

    v5_evaluator.RelativeCrossClipBGSTCAdapter = FlowGuidedDeformableBGSTCAdapter
    v5_evaluator.preflight = preflight
    v5_evaluator.build_v5_condition = v6_evaluator.build_v6_condition
    # V5 delegates through the V4 entrypoint, whose globals must also point
    # to V6's full flow-guided deformable condition implementation.
    from STC_encoder_v4_flow_aligned import evaluate_flow_aligned_stc as v4_evaluator

    v4_evaluator.FlowAlignedRGBSTCAdapter = FlowGuidedDeformableBGSTCAdapter
    v4_evaluator.preflight = preflight
    v4_evaluator.build_flow_aligned_condition = v6_evaluator.build_v6_condition
    v5_evaluator.main()


if __name__ == "__main__":
    main()
