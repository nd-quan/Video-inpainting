#!/usr/bin/env python
"""Evaluate V5 with its cross-clip condition and DDIM temporal guidance.

V5's predicted flow remains part of its learned condition encoder.  The
training-free DDIM guidance below intentionally uses a separate frozen RAFT
teacher flow, matching the fixedBG-temporal protocol.  This keeps the sampler
objective independent from V5's learned flow head.
"""

from __future__ import annotations

import json
from typing import Dict, Mapping, Optional, Tuple

import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from diffusers.schedulers.scheduling_ddim_temporal import (
    TemporalDDIMScheduler,
    backward_warp,
    build_stable_bg_mask,
)
from STC_encoder_v2_rgb import evaluate_rgb_stc_shared_noise as evaluator
from STC_encoder_v5_relative_crossclip import (
    evaluate_v5_relative_crossclip as v5_evaluator,
)


_BASE_V5_PREFLIGHT = v5_evaluator.preflight
_BASE_LOAD_MODELS = evaluator.load_models
_FLOW_RUNTIME: Optional["RAFTTemporalFlowRuntime"] = None


class RAFTTemporalFlowRuntime:
    """Frozen RAFT-Large flow, resident on CPU between DDIM clips."""

    def __init__(self, device: torch.device, batch_size: int) -> None:
        self.device = device
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("temporal_flow_batch_size must be positive")
        self.weights = Raft_Large_Weights.DEFAULT
        self.transform = self.weights.transforms()
        self.model = raft_large(weights=self.weights, progress=True).eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def estimate_bidirectional(
        self, normalized_rgb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return adjacent RAFT flow in pixels: t->t+1 and t+1->t."""
        if normalized_rgb.ndim != 4 or normalized_rgb.shape[1] != 3:
            raise ValueError("normalized_rgb must have shape [T,3,H,W]")
        pair_count = int(normalized_rgb.shape[0]) - 1
        if pair_count < 1:
            raise ValueError("Temporal guidance requires at least two frames")

        # V5 dataset RGB uses [-1,1], whereas RAFT weights expect [0,1].
        frames = normalized_rgb.detach().float().add(1.0).mul(0.5).clamp(0.0, 1.0)
        self.model.to(self.device)
        forward, backward = [], []
        try:
            for start in range(0, pair_count, self.batch_size):
                end = min(start + self.batch_size, pair_count)
                previous = frames[start:end]
                current = frames[start + 1 : end + 1]
                previous_input, current_input = self.transform(previous, current)
                previous_input = previous_input.to(
                    device=self.device, dtype=torch.float32
                )
                current_input = current_input.to(
                    device=self.device, dtype=torch.float32
                )
                forward.append(self.model(previous_input, current_input)[-1])
                backward.append(self.model(current_input, previous_input)[-1])
        finally:
            self.model.to("cpu")

        return torch.cat(forward, dim=0), torch.cat(backward, dim=0)


@torch.inference_mode()
def forward_backward_visibility(
    flow_forward: torch.Tensor,
    flow_backward: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Reject occluded/cycle-inconsistent pixels in current-frame coordinates."""
    warped_forward, in_bounds = backward_warp(flow_forward, flow_backward)
    cycle_error = (flow_backward + warped_forward).square().sum(dim=1, keepdim=True)
    flow_magnitude = (
        flow_backward.square().sum(dim=1, keepdim=True)
        + warped_forward.square().sum(dim=1, keepdim=True)
    )
    consistent = cycle_error <= float(alpha) * flow_magnitude + float(beta)
    return consistent.to(dtype=flow_backward.dtype) * in_bounds


def _get_flow_runtime(args, device: torch.device) -> RAFTTemporalFlowRuntime:
    global _FLOW_RUNTIME
    batch_size = int(args.temporal_flow_batch_size)
    if (
        _FLOW_RUNTIME is None
        or _FLOW_RUNTIME.device != device
        or _FLOW_RUNTIME.batch_size != batch_size
    ):
        _FLOW_RUNTIME = RAFTTemporalFlowRuntime(device=device, batch_size=batch_size)
    return _FLOW_RUNTIME


def add_temporal_guidance_arguments(parser) -> None:
    parser.add_argument("--temporal_guidance_scale", type=float, default=1e-4)
    parser.add_argument("--temporal_start_step", type=int, default=15)
    parser.add_argument("--temporal_end_step", type=int, default=35)
    parser.add_argument("--temporal_every_n_steps", type=int, default=1)
    parser.add_argument("--temporal_decode_chunk_size", type=int, default=1)
    parser.add_argument("--temporal_loss_scale", type=float, default=1024.0)
    parser.add_argument(
        "--temporal_loss_type",
        choices=("l1", "l2", "charbonnier"),
        default="l2",
    )
    parser.add_argument("--temporal_flow_batch_size", type=int, default=2)
    parser.add_argument("--temporal_visibility_alpha", type=float, default=0.01)
    parser.add_argument("--temporal_visibility_beta", type=float, default=0.5)
    parser.add_argument(
        "--temporal_detach_previous",
        dest="temporal_detach_previous",
        action="store_true",
    )
    parser.add_argument(
        "--no_temporal_detach_previous",
        dest="temporal_detach_previous",
        action="store_false",
    )
    parser.set_defaults(temporal_detach_previous=True)


def temporal_preflight(args):
    if args.temporal_guidance_scale < 0.0:
        raise ValueError("temporal_guidance_scale must be non-negative")
    if args.temporal_start_step < 0:
        raise ValueError("temporal_start_step must be non-negative")
    if not args.temporal_start_step < args.temporal_end_step:
        raise ValueError("temporal guidance window must satisfy start < end")
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

    dataset, paths = _BASE_V5_PREFLIGHT(args)
    print(
        json.dumps(
            {
                "v5_temporal_guidance": "ok",
                "guidance_flow": "frozen_RAFT_Large_bidirectional",
                "guidance_region": "forward_backward_visible_stable_BG",
                "guidance_window": [
                    int(args.temporal_start_step),
                    int(args.temporal_end_step),
                ],
                "guidance_scale": float(args.temporal_guidance_scale),
                "guidance_loss": str(args.temporal_loss_type),
                "detach_previous": bool(args.temporal_detach_previous),
                "clip_local_scheduler_guidance": True,
                "cross_clip_scheduler_state": False,
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
        raise TypeError("V5 temporal evaluator requires TemporalDDIMScheduler")
    if args.temporal_guidance_scale == 0.0:
        pipe.scheduler.clear_temporal_guidance()
        return {"temporal_guidance_enabled": 0}

    rgb = sample["conditioning_pixel_values"]
    bg_masks = sample["masks"].to(device=device, dtype=torch.float32)
    runtime = _get_flow_runtime(args, device)
    flow_forward, flow_backward = runtime.estimate_bidirectional(rgb)
    visibility = forward_backward_visibility(
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
        "temporal_flow_backward_magnitude": float(
            flow_backward.square().sum(dim=1).sqrt().mean().detach().cpu()
        ),
    }


def after_pipeline_call(
    *, pipe, sample: Mapping[str, torch.Tensor], args, device: torch.device
) -> Dict[str, object]:
    del sample, args, device
    scheduler = pipe.scheduler
    stats: Dict[str, object] = {
        "temporal_guidance_calls": int(scheduler.temporal_guidance_calls),
        "temporal_guidance_applied_steps": int(
            scheduler.temporal_guidance_applied_steps
        ),
        "temporal_guidance_skipped_steps": int(
            scheduler.temporal_guidance_skipped_steps
        ),
        "temporal_guidance_active_frames": int(
            scheduler.last_temporal_active_frames or 0
        ),
    }
    for metric_name, value in (
        ("temporal_last_loss", scheduler.last_temporal_loss),
        ("temporal_last_update_norm", scheduler.last_temporal_update_norm),
        ("temporal_last_grad_norm", scheduler.last_temporal_grad_norm),
    ):
        if value is not None:
            stats[metric_name] = float(value)
    if scheduler.last_temporal_skipped_reason is not None:
        stats["temporal_last_skipped_reason"] = str(
            scheduler.last_temporal_skipped_reason
        )
    return stats


def clear_pipeline_call(*, pipe) -> None:
    if isinstance(pipe.scheduler, TemporalDDIMScheduler):
        pipe.scheduler.clear_temporal_guidance()


def main() -> None:
    # V5 installs its cross-clip adapter/condition builder on the shared V2
    # evaluator.  Install the temporal hooks first, then delegate to V5.
    evaluator.ADD_EVALUATION_ARGUMENTS_FN = add_temporal_guidance_arguments
    evaluator.BEFORE_PIPELINE_CALL_FN = before_pipeline_call
    evaluator.AFTER_PIPELINE_CALL_FN = after_pipeline_call
    evaluator.CLEAR_PIPELINE_CALL_FN = clear_pipeline_call
    evaluator.load_models = load_models_with_temporal_scheduler
    v5_evaluator.preflight = temporal_preflight
    v5_evaluator.main()


if __name__ == "__main__":
    main()
