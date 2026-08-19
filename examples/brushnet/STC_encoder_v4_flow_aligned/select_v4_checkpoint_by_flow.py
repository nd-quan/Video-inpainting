#!/usr/bin/env python
"""Rank V4 checkpoints on cached clean-RAFT flow from the validation split.

This is a lightweight, leakage-free screening stage.  It is intentionally
followed by downstream generation on a small independent validation subset;
flow EPE alone is not claimed to measure restoration quality.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, CLIPImageProcessor


BRUSHNET_DIR = Path(__file__).resolve().parent.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v3_rgb_flow.flow_supervision import compute_teacher_flow_loss  # noqa: E402
from STC_encoder_v3_rgb_flow.teacher_flow_data import TeacherFlowV8ClipDataset  # noqa: E402
from STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (  # noqa: E402
    FlowAlignedRGBSTCAdapter,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--teacher_flow_root", type=Path, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=Path, required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--clip_length", type=int, default=16)
    parser.add_argument("--clip_stride", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--clips_per_video", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match is None:
        raise ValueError(path)
    return int(match.group(1))


def select_indices(dataset, clips_per_video: int):
    grouped = defaultdict(list)
    for index, clip in enumerate(dataset.clips):
        grouped[str(clip[0])].append(index)
    selected = []
    for video in sorted(grouped):
        indices = grouped[video]
        count = min(int(clips_per_video), len(indices))
        if count == 1:
            selected.append(indices[len(indices) // 2])
        else:
            positions = np.linspace(0, len(indices) - 1, count).round().astype(int)
            selected.extend(indices[int(position)] for position in positions)
    return selected


def main():
    args = parse_args()
    if args.clips_per_video < 1:
        raise ValueError("--clips_per_video must be positive")
    checkpoints = sorted(
        (
            path
            for path in args.experiment_root.glob("checkpoint-*")
            if (path / "stc_flow_model" / "config.json").is_file()
        ),
        key=checkpoint_step,
    )
    if not checkpoints:
        raise FileNotFoundError("No complete V4 checkpoints found")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.pretrained_model_name_or_path), subfolder="tokenizer", use_fast=False
    )
    dataset = TeacherFlowV8ClipDataset(
        dataset_root=args.dataset_root,
        split=args.split,
        teacher_flow_root=args.teacher_flow_root,
        tokenizer=tokenizer,
        clip_image_processor=CLIPImageProcessor(),
        clip_length=args.clip_length,
        stride=args.clip_stride,
        resolution=args.resolution,
    )
    selected = select_indices(dataset, args.clips_per_video)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Checkpoint screening requires CUDA")
    rows = []
    with torch.inference_mode():
        for checkpoint in checkpoints:
            model = FlowAlignedRGBSTCAdapter.from_pretrained(
                str(checkpoint / "stc_flow_model")
            ).to(device=device, dtype=torch.float32).eval()
            values = defaultdict(list)
            for index in selected:
                sample = dataset[index]
                rgb = sample["conditioning_pixel_values"].unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                bg = sample["masks"].unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(rgb, bg, predict_flow=True, return_dict=True)
                flow = compute_teacher_flow_loss(
                    output.predicted_flow_forward,
                    output.predicted_flow_backward,
                    sample["teacher_flow_forward"].unsqueeze(0),
                    sample["teacher_flow_backward"].unsqueeze(0),
                    bg,
                    sample["teacher_valid_forward"].unsqueeze(0),
                    sample["teacher_valid_backward"].unsqueeze(0),
                    region="all",
                    charbonnier_eps=1e-3,
                )
                values["flow_loss"].append(float(flow.loss))
                values["flow_epe"].append(float(flow.epe))
                values["predicted_magnitude"].append(float(flow.predicted_magnitude))
                values["confidence"].append(
                    float(output.alignment_confidence.detach().float().mean())
                )
                values["delta_abs_mean"].append(
                    float(output.delta_bg.detach().float().abs().mean())
                )
            row = {
                "step": checkpoint_step(checkpoint),
                "checkpoint": str(checkpoint.resolve()),
                "clip_count": len(selected),
                **{key: float(np.mean(items)) for key, items in values.items()},
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            del model
            torch.cuda.empty_cache()
    ranked = sorted(rows, key=lambda row: (row["flow_epe"], row["flow_loss"]))
    report = {
        "selection_split": args.split,
        "selection_policy": "middle clips, balanced per video; rank by clean-RAFT flow EPE",
        "clips_per_video": args.clips_per_video,
        "selected_clip_count": len(selected),
        "checkpoint_count": len(rows),
        "best_flow_checkpoint": ranked[0],
        "ranking": ranked,
        "note": "Flow EPE is screening only; downstream validation generation decides the final checkpoint.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["best_flow_checkpoint"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
