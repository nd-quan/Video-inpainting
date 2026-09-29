#!/usr/bin/env python
"""Evaluate V5 with stateless predecessor/current cross-clip pairing."""

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
from STC_encoder_v5_relative_crossclip.cross_clip_data import (
    CrossClipFlatEvalDataset,
    CrossClipHierarchicalEvalDataset,
)
from STC_encoder_v5_relative_crossclip.relative_crossclip_stc_adapter import (
    RelativeCrossClipBGSTCAdapter,
    augment_brushnet_condition_v5,
)


_base_v4_preflight = v4_evaluator.preflight


def preflight(args):
    # The reused V2 preflight resolves these globals dynamically.
    evaluator.HierarchicalV8ClipDataset = CrossClipHierarchicalEvalDataset
    evaluator.FlatV8TestClipDataset = CrossClipFlatEvalDataset
    dataset, paths = _base_v4_preflight(args)
    # V4 may append a shifted hierarchical tail after dataset construction.
    dataset.rebuild_predecessors()
    adapter = RelativeCrossClipBGSTCAdapter.from_pretrained(
        str(args.stc_adapter_path)
    )
    overlap = int(args.clip_length) - int(args.clip_stride)
    report = {
        "v5_preflight": "ok",
        "cross_clip_mode": "stateless_predecessor_current_pair",
        "clip_count": len(dataset),
        "cross_clip_transition_count": dataset.cross_clip_transition_count,
        "cross_clip_run_count": dataset.cross_clip_run_count,
        "configured_memory_frames": int(adapter.config.cross_clip_memory_frames),
        "regular_clip_overlap": overlap,
        "relative_position_max_distance": int(
            adapter.config.relative_position_max_distance
        ),
        "absolute_frame_ids": True,
        "resume_safe_without_runtime_cache": True,
    }
    if overlap <= 0:
        raise ValueError("V5 evaluation requires clip_stride < clip_length")
    if int(adapter.config.cross_clip_memory_frames) > overlap:
        raise ValueError("Checkpoint memory exceeds the regular evaluation overlap")
    print(json.dumps(report, indent=2, sort_keys=True))
    return dataset, paths


@torch.inference_mode()
def build_v5_condition(
    pipe,
    adapter: RelativeCrossClipBGSTCAdapter,
    sample: Mapping[str, torch.Tensor],
    device: torch.device,
    condition_seed: int,
    injection_scale: float,
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
    latent_generator = torch.Generator(device=device).manual_seed(int(condition_seed))
    with evaluator.autocast_context(device):
        base_condition_latents = pipe.vae.encode(
            rgb.flatten(0, 1).to(dtype=pipe.vae.dtype)
        ).latent_dist.sample(generator=latent_generator)
        base_condition_latents = (
            base_condition_latents * pipe.vae.config.scaling_factor
        )
        brushnet_condition, output, _ = augment_brushnet_condition_v5(
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
        )
    expected = (
        frames,
        5,
        base_condition_latents.shape[-2],
        base_condition_latents.shape[-1],
    )
    if tuple(brushnet_condition.shape) != expected:
        raise RuntimeError(
            f"Unexpected V5 BrushNet condition {tuple(brushnet_condition.shape)}; "
            f"expected {expected}"
        )
    condition_cfg = torch.cat((brushnet_condition, brushnet_condition), dim=0)
    delta = output.delta_bg.detach().float()
    latent_bg = output.latent_bg_mask.detach().float()
    flow_forward = output.predicted_flow_forward.detach().float()
    flow_backward = output.predicted_flow_backward.detach().float()
    confidence = output.alignment_confidence.detach().float()
    stats = {
        "delta_abs_mean": float(delta.abs().mean()),
        "latent_bg_ratio": float(latent_bg.mean()),
        "roi_delta_nonzero": int(torch.count_nonzero(delta * (1.0 - latent_bg))),
        "flow_forward_magnitude": float(
            flow_forward.square().sum(dim=2).sqrt().mean()
        ),
        "flow_backward_magnitude": float(
            flow_backward.square().sum(dim=2).sqrt().mean()
        ),
        "alignment_confidence_mean": float(confidence.mean()),
        "alignment_confidence_nonzero_ratio": float(
            (confidence > 0).float().mean()
        ),
        "has_predecessor": int(previous_valid.any()),
        "dataset_predecessor_overlap": int(sample["predecessor_overlap"]),
        "memory_overlap_count": float(output.memory_overlap_count.float().mean()),
        "relative_bias_abs_mean": float(output.relative_bias_abs_mean),
        "cross_clip_gate_abs_mean": float(output.cross_clip_gate_abs_mean),
    }
    if stats["roi_delta_nonzero"]:
        raise RuntimeError("V5 delta leaked outside M_BG")
    if stats["has_predecessor"] and stats["memory_overlap_count"] <= 0:
        raise RuntimeError("V5 predecessor was supplied but no frame IDs overlapped")
    return condition_cfg, stats


def main() -> None:
    v4_evaluator.FlowAlignedRGBSTCAdapter = RelativeCrossClipBGSTCAdapter
    v4_evaluator.preflight = preflight
    v4_evaluator.build_flow_aligned_condition = build_v5_condition
    v4_evaluator.main()


if __name__ == "__main__":
    main()

