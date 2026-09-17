#!/usr/bin/env python3
"""Create like-for-like videos and full-reference metrics for flat PNG cases.

Each input case is a directory containing ``000000.png`` style generated
frames.  The tool selects one explicit contiguous frame range for every case,
uses one shared GT directory, writes equal-length videos, and invokes the
repository's established PSNR/SSIM/LPIPS/WE/FS/VFID evaluator independently
for each case.  Source PNGs are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from imgToVideo_rgb_stc_eval import encode_video


BRUSHNET_ROOT = Path(__file__).resolve().parents[1]
METRIC_SCRIPT = (
    BRUSHNET_ROOT / "examples/brushnet/DGAF_VSR_original/postprocess_eval.py"
)
DEFAULT_EVALUATOR_ROOT = BRUSHNET_ROOT.parent / "video-inpainting-evaluation"
SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("case must use LABEL=/absolute/path")
    label, raw_path = value.split("=", 1)
    if not SAFE_LABEL.fullmatch(label):
        raise argparse.ArgumentTypeError(f"unsafe case label: {label!r}")
    return label, Path(raw_path).expanduser().resolve()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_frames(root: Path, frame_ids: list[int]) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {root}")
    paths = [root / f"{frame_id:06d}.png" for frame_id in frame_ids]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{root} is missing {len(missing)} requested frames: {missing[:10]}"
        )
    return paths


def sha256_manifest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("ascii"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def evaluate_case(
    *,
    label: str,
    generated: list[Path],
    gt: list[Path],
    frame_ids: list[int],
    output_root: Path,
    evaluator_root: Path,
    fps: float,
    overwrite: bool,
) -> dict:
    video_path = output_root / "videos" / f"{label}.mp4"
    encode_video(
        [(frame_id, path, frame_ids[0]) for frame_id, path in zip(frame_ids, generated)],
        video_path,
        fps,
        "mp4v",
        overwrite=overwrite,
    )

    metric_output = output_root / "metrics" / label
    if metric_output.exists() and not overwrite:
        raise FileExistsError(
            f"Metric output exists: {metric_output}; pass --overwrite to replace it"
        )
    stage = Path(tempfile.mkdtemp(prefix=f"compare_{label}_"))
    try:
        sequence_dir = stage / "BasketballPass"
        for kind, paths in (("final", generated), ("gt", gt)):
            kind_dir = sequence_dir / kind
            kind_dir.mkdir(parents=True)
            for frame_id, source in zip(frame_ids, paths):
                (kind_dir / f"{frame_id:06d}.png").symlink_to(source)
        write_json(
            sequence_dir / "clip_metrics.json",
            {"video": "BasketballPass", "frame_ids": frame_ids},
        )
        write_json(stage / "summary.json", {"clips": 1})
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
        staged = stage / "metrics"
        if not (staged / "metrics.json").is_file():
            raise RuntimeError(f"Metric evaluator did not complete for {label}")
        if metric_output.exists():
            shutil.rmtree(metric_output)
        metric_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged), str(metric_output))
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    payload = json.loads((metric_output / "metrics.json").read_text())
    row = next(item for item in payload["results"] if item["sequence"] != "ALL")
    row = {"case": label, **row}
    return {
        "row": row,
        "video": str(video_path),
        "metrics": str(metric_output / "metrics.json"),
        "generated_manifest_sha256": sha256_manifest(generated),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=150)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--evaluator-root", type=Path, default=DEFAULT_EVALUATOR_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.frame_start < 0 or args.frame_count < 2 or args.fps <= 0:
        parser.error("frame-start must be >=0, frame-count >=2, and fps >0")
    labels = [label for label, _ in args.case]
    if len(labels) != len(set(labels)):
        parser.error("case labels must be unique")

    frame_ids = list(range(args.frame_start, args.frame_start + args.frame_count))
    gt_dir = args.gt_dir.expanduser().resolve()
    gt = resolve_frames(gt_dir, frame_ids)
    evaluator_root = args.evaluator_root.expanduser().resolve()
    if not (evaluator_root / "pretrained_models/rgb_imagenet.pt").is_file():
        raise FileNotFoundError("Missing I3D rgb_imagenet checkpoint")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    for label, root in args.case:
        generated = resolve_frames(root, frame_ids)
        records.append(
            {
                "label": label,
                "source": str(root),
                **evaluate_case(
                    label=label,
                    generated=generated,
                    gt=gt,
                    frame_ids=frame_ids,
                    output_root=output_root,
                    evaluator_root=evaluator_root,
                    fps=args.fps,
                    overwrite=args.overwrite,
                ),
            }
        )

    rows = [record["row"] for record in records]
    with (output_root / "comparison_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output_root / "comparison_metrics.json",
        {
            "frame_range": [frame_ids[0], frame_ids[-1]],
            "frame_count": len(frame_ids),
            "pair_count": len(frame_ids) - 1,
            "fps": args.fps,
            "gt_dir": str(gt_dir),
            "gt_manifest_sha256": sha256_manifest(gt),
            "cases": records,
        },
    )
    print(f"Comparison complete: {output_root}")


if __name__ == "__main__":
    main()
