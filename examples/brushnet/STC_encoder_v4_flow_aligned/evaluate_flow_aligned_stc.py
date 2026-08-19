#!/usr/bin/env python
"""Evaluate V4 flow-aligned RGB-STC with the frozen V8 BrushNet stack.

The mature V2 evaluator already implements the exact V8 CLIP/IP-Adapter/
BrushNet inference contract and the flat/hierarchical datasets.  This entry
point deliberately reuses that contract while replacing the adapter with the
*complete* V4 ``stc_flow_model`` (STC trunk, bidirectional flow head, and
confidence-aware alignment fusion).  Loading the smaller ``stc_adapter``
export here would silently disable the V4 idea.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Tuple

import torch


BRUSHNET_DIR = Path(__file__).resolve().parent.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v2_rgb import evaluate_rgb_stc_shared_noise as evaluator  # noqa: E402
from STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (  # noqa: E402
    FlowAlignedRGBSTCAdapter,
    augment_brushnet_condition,
)


def _select_clips_per_video(dataset, maximum: int):
    """Keep evenly-spaced clips per video for validation-only screening.

    The public test launchers never set this option.  It exists so checkpoint
    selection can use a small, sequence-balanced validation subset instead of
    accidentally selecting a checkpoint on test_1/test_2.
    """
    if maximum < 1:
        raise ValueError("V4_MAX_CLIPS_PER_VIDEO must be positive")
    grouped = defaultdict(list)
    for clip in dataset.clips:
        grouped[str(clip[0])].append(clip)
    selected = []
    for video in sorted(grouped):
        clips = grouped[video]
        if len(clips) <= maximum:
            selected.extend(clips)
            continue
        if maximum == 1:
            selected.append(clips[len(clips) // 2])
            continue
        positions = [
            round(index * (len(clips) - 1) / (maximum - 1))
            for index in range(maximum)
        ]
        selected.extend(clips[position] for position in positions)
    dataset.clips = selected
    dataset.covered_frame_count = len(
        {path for _, paths, _ in selected for path in paths}
    )
    return dataset


def _append_hierarchical_tail_windows(dataset) -> int:
    """Cover hierarchical evaluation tails without changing train windows.

    ``HierarchicalV8ClipDataset`` is shared with training and intentionally
    stops at the last regular stride start.  For evaluation that can omit a
    few final source frames (T=16/S=12 omits up to 11).  The flat-test loader
    already appends its shifted final window; apply the same *evaluation-only*
    policy here without touching the training dataset implementation.
    """
    if not hasattr(dataset, "roots") or "GT" not in dataset.roots:
        return 0
    gt_root = dataset.roots["GT"]
    if not gt_root.is_dir():
        return 0
    selected = set(getattr(dataset, "include_branches", ()))
    grouped = defaultdict(list)
    for path in gt_root.rglob("*.png"):
        if not path.is_file():
            continue
        relative = path.relative_to(gt_root)
        branch = relative.parent.as_posix()
        if selected and branch not in selected:
            continue
        try:
            grouped[branch].append((int(relative.stem), relative))
        except ValueError as error:
            raise ValueError(f"Non-numeric hierarchical frame: {relative}") from error

    existing = {
        (str(branch), tuple(int(frame) for frame in frame_ids))
        for branch, _, frame_ids in dataset.clips
    }
    added = 0
    for branch in sorted(grouped):
        frames = sorted(grouped[branch])
        run = []
        for frame in frames + [(None, None)]:
            if run and (frame[0] is None or frame[0] != run[-1][0] + 1):
                if len(run) >= int(dataset.clip_length):
                    tail = run[-int(dataset.clip_length) :]
                    paths = tuple(item[1] for item in tail)
                    frame_ids = tuple(int(item[0]) for item in tail)
                    key = (branch, frame_ids)
                    if key not in existing:
                        dataset.clips.append((branch, paths, frame_ids))
                        existing.add(key)
                        added += 1
                run = []
            if frame[0] is not None:
                run.append(frame)
    if added:
        dataset.covered_frame_count = len(
            {path for _, paths, _ in dataset.clips for path in paths}
        )
    return added


_base_preflight = evaluator.preflight


def preflight(args):
    dataset, paths = _base_preflight(args)
    if os.environ.get("V4_APPEND_HIERARCHICAL_TAIL", "1") != "0":
        added = _append_hierarchical_tail_windows(dataset)
        if added:
            print(
                "V4 hierarchical evaluation tail policy: "
                f"added={added}, clips={len(dataset)}, "
                f"covered={dataset.covered_frame_count}/{dataset.frame_count}"
            )
    maximum = os.environ.get("V4_MAX_CLIPS_PER_VIDEO")
    if maximum:
        dataset = _select_clips_per_video(dataset, int(maximum))
        print(
            f"V4 validation clip selection: {len(dataset)} clips, "
            f"maximum {int(maximum)} per video"
        )
    return dataset, paths


@torch.inference_mode()
def build_flow_aligned_condition(
    pipe,
    adapter: FlowAlignedRGBSTCAdapter,
    sample: Mapping[str, torch.Tensor],
    device: torch.device,
    condition_seed: int,
    injection_scale: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    rgb_sequence = sample["conditioning_pixel_values"].unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    bg_mask_sequence = sample["masks"].unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    frames = int(rgb_sequence.shape[1])
    latent_generator = torch.Generator(device=device).manual_seed(int(condition_seed))
    with evaluator.autocast_context(device):
        base_condition_latents = pipe.vae.encode(
            rgb_sequence.flatten(0, 1).to(dtype=pipe.vae.dtype)
        ).latent_dist.sample(generator=latent_generator)
        base_condition_latents = (
            base_condition_latents * pipe.vae.config.scaling_factor
        )
        brushnet_condition, output, _ = augment_brushnet_condition(
            model=adapter,
            base_condition_latents=base_condition_latents,
            rgb_sequence=rgb_sequence,
            bg_mask_sequence=bg_mask_sequence,
            injection_scale=float(injection_scale),
            predict_flow=True,
        )
    expected = (
        frames,
        5,
        base_condition_latents.shape[-2],
        base_condition_latents.shape[-1],
    )
    if brushnet_condition.shape != expected:
        raise RuntimeError(
            f"Unexpected V4 BrushNet condition {tuple(brushnet_condition.shape)}; "
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
        "alignment_confidence_nonzero_ratio": float((confidence > 0).float().mean()),
    }
    if stats["roi_delta_nonzero"]:
        raise RuntimeError("V4 delta leaked outside M_BG")
    return condition_cfg, stats


def main():
    # The reused evaluator resolves these names at runtime, so patching the
    # model class and condition builder retains one authoritative V8 pipeline.
    evaluator.RGBSTCConditionAdapter = FlowAlignedRGBSTCAdapter
    evaluator.preflight = preflight
    evaluator.build_stc_condition = build_flow_aligned_condition
    evaluator.main()


if __name__ == "__main__":
    main()
