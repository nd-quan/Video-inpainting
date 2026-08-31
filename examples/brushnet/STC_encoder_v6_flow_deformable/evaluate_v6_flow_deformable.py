#!/usr/bin/env python
"""Evaluate V6 through the exact V5 cross-clip/V8 inference contract."""

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

from STC_encoder_v2_rgb import evaluate_rgb_stc_shared_noise as evaluator
from STC_encoder_v4_flow_aligned import evaluate_flow_aligned_stc as v4_evaluator
from STC_encoder_v5_relative_crossclip import (
    evaluate_v5_relative_crossclip as v5_evaluator,
)
from STC_encoder_v6_flow_deformable.flow_guided_deformable_stc_adapter import (
    FlowGuidedDeformableBGSTCAdapter,
    augment_brushnet_condition_v6,
)


_base_v5_preflight = v5_evaluator.preflight


def _add_evaluation_arguments(parser) -> None:
    parser.add_argument(
        "--deformable_alignment_scale",
        type=float,
        default=1.0,
        help="Scale of the zero-residual V6 fusion; use 0 for exact V5 ablation.",
    )


def _condition_extra_kwargs(args):
    if args.deformable_alignment_scale < 0.0:
        raise ValueError("deformable_alignment_scale must be non-negative")
    return {"deformable_alignment_scale": args.deformable_alignment_scale}


def preflight(args):
    v5_evaluator.RelativeCrossClipBGSTCAdapter = (
        FlowGuidedDeformableBGSTCAdapter
    )
    dataset, paths = _base_v5_preflight(args)
    adapter = FlowGuidedDeformableBGSTCAdapter.from_pretrained(
        str(args.stc_adapter_path)
    )
    report = {
        "v6_preflight": "ok",
        "first_order": True,
        "directions": ["previous_to_current", "next_to_current"],
        "deformable_alignment_scale": float(args.deformable_alignment_scale),
        "deform_kernel_size": int(adapter.config.deform_kernel_size),
        "deform_groups": int(adapter.config.deform_groups),
        "deform_residual_max_displacement": float(
            adapter.config.deform_residual_max_displacement
        ),
        "deformation_scope": "raw adjacent spatial features only",
        "cross_clip_memory_deformed": False,
        "teacher_flow_at_inference": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return dataset, paths


@torch.inference_mode()
def build_v6_condition(
    pipe,
    adapter: FlowGuidedDeformableBGSTCAdapter,
    sample: Mapping[str, torch.Tensor],
    device: torch.device,
    condition_seed: int,
    injection_scale: float,
    deformable_alignment_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    rgb = sample["conditioning_pixel_values"].unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    bg = sample["masks"].unsqueeze(0).to(device=device, dtype=torch.float32)
    previous_rgb = sample["previous_conditioning_pixel_values"].unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    previous_bg = sample["previous_masks"].unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    frame_ids = sample["frame_ids"].unsqueeze(0).to(device=device)
    previous_ids = sample["previous_frame_ids"].unsqueeze(0).to(device=device)
    previous_valid = sample["previous_valid_mask"].unsqueeze(0).to(device=device)
    frames = int(rgb.shape[1])
    generator = torch.Generator(device=device).manual_seed(int(condition_seed))
    with evaluator.autocast_context(device):
        base_condition_latents = pipe.vae.encode(
            rgb.flatten(0, 1).to(dtype=pipe.vae.dtype)
        ).latent_dist.sample(generator=generator)
        base_condition_latents = (
            base_condition_latents * pipe.vae.config.scaling_factor
        )
        brushnet_condition, output, _ = augment_brushnet_condition_v6(
            model=adapter,
            base_condition_latents=base_condition_latents,
            rgb_sequence=rgb,
            bg_mask_sequence=bg,
            injection_scale=float(injection_scale),
            predict_flow=True,
            frame_ids=frame_ids,
            previous_rgb_sequence=previous_rgb,
            previous_bg_mask_sequence=previous_bg,
            previous_frame_ids=previous_ids,
            previous_valid_mask=previous_valid,
            deformable_alignment_scale=float(deformable_alignment_scale),
        )
    expected = (
        frames,
        5,
        base_condition_latents.shape[-2],
        base_condition_latents.shape[-1],
    )
    if tuple(brushnet_condition.shape) != expected:
        raise RuntimeError(
            f"Unexpected V6 BrushNet condition {tuple(brushnet_condition.shape)}; "
            f"expected {expected}"
        )
    condition_cfg = torch.cat((brushnet_condition, brushnet_condition), dim=0)
    delta = output.delta_bg.detach().float()
    latent_bg = output.latent_bg_mask.detach().float()
    offsets = torch.cat(
        (
            output.residual_offset_backward.detach().float().flatten(),
            output.residual_offset_forward.detach().float().flatten(),
        )
    ).abs()
    masks = torch.cat(
        (
            output.modulation_mask_backward.detach().float().flatten(),
            output.modulation_mask_forward.detach().float().flatten(),
        )
    )
    stats = {
        "delta_abs_mean": float(delta.abs().mean()),
        "latent_bg_ratio": float(latent_bg.mean()),
        "roi_delta_nonzero": int(
            torch.count_nonzero(delta * (1.0 - latent_bg))
        ),
        "flow_forward_magnitude": float(
            output.predicted_flow_forward.detach()
            .float()
            .square()
            .sum(dim=2)
            .sqrt()
            .mean()
        ),
        "flow_backward_magnitude": float(
            output.predicted_flow_backward.detach()
            .float()
            .square()
            .sum(dim=2)
            .sqrt()
            .mean()
        ),
        "alignment_confidence_mean": float(
            output.alignment_confidence.detach().float().mean()
        ),
        "has_predecessor": int(previous_valid.any()),
        "dataset_predecessor_overlap": int(sample["predecessor_overlap"]),
        "memory_overlap_count": float(output.memory_overlap_count.float().mean()),
        "relative_bias_abs_mean": float(output.relative_bias_abs_mean),
        "cross_clip_gate_abs_mean": float(output.cross_clip_gate_abs_mean),
        "deformable_alignment_scale": float(deformable_alignment_scale),
        "deform_offset_abs_mean": float(offsets.mean()) if offsets.numel() else 0.0,
        "deform_offset_abs_p95": float(torch.quantile(offsets, 0.95))
        if offsets.numel()
        else 0.0,
        "deform_mask_mean": float(masks.mean()) if masks.numel() else 0.0,
        "deform_mask_saturation": float(
            ((masks < 0.05) | (masks > 0.95)).float().mean()
        )
        if masks.numel()
        else 0.0,
        "deform_reliability_backward": float(
            output.deform_reliability_backward.detach().float().mean()
        ),
        "deform_reliability_forward": float(
            output.deform_reliability_forward.detach().float().mean()
        ),
        "deformation_minus_base_abs_mean": float(
            output.deformation_minus_base_abs_mean
        ),
    }
    if stats["roi_delta_nonzero"]:
        raise RuntimeError("V6 delta leaked outside M_BG")
    if stats["has_predecessor"] and stats["memory_overlap_count"] <= 0:
        raise RuntimeError("V6 predecessor was supplied but no frame IDs overlapped")
    return condition_cfg, stats


def main() -> None:
    evaluator.ADD_EVALUATION_ARGUMENTS_FN = _add_evaluation_arguments
    evaluator.CONDITION_EXTRA_KWARGS_FN = _condition_extra_kwargs
    v5_evaluator.RelativeCrossClipBGSTCAdapter = (
        FlowGuidedDeformableBGSTCAdapter
    )
    v5_evaluator.preflight = preflight
    v5_evaluator.build_v5_condition = build_v6_condition
    v4_evaluator.FlowAlignedRGBSTCAdapter = FlowGuidedDeformableBGSTCAdapter
    v4_evaluator.preflight = preflight
    v4_evaluator.build_flow_aligned_condition = build_v6_condition
    v5_evaluator.main()


if __name__ == "__main__":
    main()
