#!/usr/bin/env python3
"""Evaluate stitched V8 sequences without modifying evaluator PNG outputs.

Overlapping evaluator clips are reduced to one chronological frame per ID using
the same first-occurrence rule as ``imgToVideo_rgb_stc_eval.py``.  A temporary
symlink-only tree is passed to the existing full-reference metric evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from imgToVideo_rgb_stc_eval import discover_sources, select_unique_frames


BRUSHNET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATOR_ROOT = BRUSHNET_ROOT.parent / "video-inpainting-evaluation"
METRIC_SCRIPT = (
    BRUSHNET_ROOT / "examples/brushnet/DGAF_VSR_original/postprocess_eval.py"
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def evaluate_root(
    eval_root: Path,
    evaluator_root: Path,
    output_name: str,
    overwrite: bool,
) -> None:
    eval_root = eval_root.expanduser().resolve()
    if not eval_root.is_dir():
        raise FileNotFoundError(f"Evaluation root not found: {eval_root}")

    output_dir = eval_root / output_name
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Metric output already exists: {output_dir}; pass --overwrite to replace it"
        )

    selected = {}
    for kind in ("final", "gt"):
        discovered = discover_sources(eval_root, kind, recursive=False)
        selected[kind] = {
            sequence: select_unique_frames(candidates, "first")
            for sequence, candidates in discovered.items()
        }

    sequences = sorted(selected["final"])
    if sorted(selected["gt"]) != sequences:
        raise ValueError("Final/GT sequence sets do not match")
    for sequence in sequences:
        final_ids = [frame_id for frame_id, _, _ in selected["final"][sequence]]
        gt_ids = [frame_id for frame_id, _, _ in selected["gt"][sequence]]
        if final_ids != gt_ids:
            raise ValueError(f"Final/GT frame IDs do not match for {sequence}")

    stage = Path(tempfile.mkdtemp(prefix=f"{eval_root.name}_metrics_"))
    try:
        for sequence in sequences:
            frames = selected["final"][sequence]
            sequence_dir = stage / sequence
            for kind in ("final", "gt"):
                kind_dir = sequence_dir / kind
                kind_dir.mkdir(parents=True)
                for frame_id, source_path, _ in selected[kind][sequence]:
                    (kind_dir / f"{frame_id:06d}.png").symlink_to(source_path)
            write_json(
                sequence_dir / "clip_metrics.json",
                {
                    "video": sequence,
                    "frame_ids": [frame_id for frame_id, _, _ in frames],
                },
            )
        write_json(stage / "summary.json", {"clips": len(sequences)})

        print(
            f"Evaluating {eval_root.name}: {len(sequences)} sequences, "
            f"{sum(len(frames) for frames in selected['final'].values())} unique frames",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(METRIC_SCRIPT),
                "--eval_root",
                str(stage),
                "--evaluator_root",
                str(evaluator_root),
            ],
            check=True,
        )

        staged_metrics = stage / "metrics"
        if not (staged_metrics / "metrics.csv").is_file():
            raise RuntimeError(f"Metric evaluator did not create {staged_metrics / 'metrics.csv'}")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(staged_metrics), str(output_dir))
        write_json(
            output_dir / "source.json",
            {
                "eval_root": str(eval_root),
                "selection": "first",
                "sequences": sequences,
                "unique_frame_count": sum(
                    len(frames) for frames in selected["final"].values()
                ),
                "source_pngs_preserved": True,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        )
        print(f"Metrics saved to {output_dir}", flush=True)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_roots", nargs="+", type=Path)
    parser.add_argument("--evaluator-root", type=Path, default=DEFAULT_EVALUATOR_ROOT)
    parser.add_argument("--output-name", default="metrics_full_reference")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    evaluator_root = args.evaluator_root.expanduser().resolve()
    checkpoint = evaluator_root / "pretrained_models/rgb_imagenet.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"I3D checkpoint not found: {checkpoint}")
    for eval_root in args.eval_roots:
        evaluate_root(eval_root, evaluator_root, args.output_name, args.overwrite)


if __name__ == "__main__":
    main()
