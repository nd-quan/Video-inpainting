#!/usr/bin/env python3
"""Encode GT, input, and mask frame folders in a BrushNet dataset tree.

Expected layout::

    <dataset-root>/<sequence>/gt/000000.png
    <dataset-root>/<sequence>/inputs/000000.png
    <dataset-root>/<sequence>/masks/000000.png

Videos are written to ``<dataset-root>/videos/<kind>/<sequence>.mp4`` by
default.  Frame files may be symlinks.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


DEFAULT_DATASET_ROOT = Path(
    "/home/cilab/ndquan/videoInpainting/code/BrushNet/examples/brushnet/dataset/test_2"
)
SEQUENCE_FPS = {
    "BasketballPass": 50,
    "ParkScene": 24,
    "PartyScene": 50,
    "RaceHorses": 30,
    "Traffic": 30,
    "Cactus": 50,
    "BQMall": 60,
    "BQSquare": 60,
    "BQTerrace": 60,
    "FourPeople": 60,
    "Kimono": 24,
    "KristenAndSara": 60,
    "PeopleOnStreet": 30,
}
FRAME_KINDS = ("gt", "inputs", "masks")


@dataclass(frozen=True)
class VideoJob:
    sequence: str
    kind: str
    frame_dir: Path
    output_path: Path
    fps: int
    first_frame: int
    frame_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert BrushNet dataset GT, input, and mask PNG frames to MP4."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Default: <dataset-root>/videos"
    )
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--frame-kinds", nargs="+", choices=FRAME_KINDS, default=FRAME_KINDS)
    parser.add_argument("--codec", choices=("mpeg4", "png"), default="mpeg4")
    parser.add_argument("--quality", type=int, default=2, help="MPEG-4 qscale (1=best, 31=fastest).")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.quality <= 31:
        parser.error("--quality must be in [1, 31]")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    return args


def numeric_pngs(frame_dir: Path) -> Sequence[Path]:
    paths = [path for path in frame_dir.glob("*.png") if path.is_file()]
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in {frame_dir}")
    try:
        indexed = sorted((int(path.stem), path) for path in paths)
    except ValueError as exc:
        raise ValueError(f"PNG filenames must be numeric in {frame_dir}") from exc
    indices = [index for index, _ in indexed]
    expected = list(range(indices[0], indices[-1] + 1))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        raise ValueError(f"Frames are not contiguous in {frame_dir}; missing={missing[:20]}")
    return [path for _, path in indexed]


def discover_jobs(args: argparse.Namespace) -> List[VideoJob]:
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else dataset_root / "videos"
    sequences = {path.name: path for path in dataset_root.iterdir() if path.is_dir() and path.name != "videos"}
    requested = args.sequences or sorted(sequences)
    unknown = sorted(set(requested) - set(sequences))
    if unknown:
        raise FileNotFoundError(f"Sequences not found: {unknown}")

    jobs = []
    for sequence in requested:
        for kind in args.frame_kinds:
            frame_dir = sequences[sequence] / kind
            frames = numeric_pngs(frame_dir)
            jobs.append(
                VideoJob(
                    sequence=sequence,
                    kind=kind,
                    frame_dir=frame_dir,
                    output_path=output_dir / kind / f"{sequence}.mp4",
                    fps=SEQUENCE_FPS.get(sequence, 30),
                    first_frame=int(frames[0].stem),
                    frame_count=len(frames),
                )
            )
    return jobs


def ffmpeg_command(job: VideoJob, codec: str, quality: int, overwrite: bool) -> List[str]:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y" if overwrite else "-n",
        "-framerate", str(job.fps), "-start_number", str(job.first_frame),
        "-i", str(job.frame_dir / "%06d.png"), "-frames:v", str(job.frame_count), "-an",
    ]
    if codec == "mpeg4":
        command.extend(("-c:v", "mpeg4", "-q:v", str(quality), "-pix_fmt", "yuv420p"))
    else:
        command.extend(("-c:v", "png", "-pix_fmt", "rgb24"))
    return command + [str(job.output_path)]


def encode_job(job: VideoJob, codec: str, quality: int, overwrite: bool) -> VideoJob:
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(ffmpeg_command(job, codec, quality, overwrite), check=True, text=True)
    return job


def main() -> None:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found in PATH")
    jobs = discover_jobs(args)
    pending = [job for job in jobs if args.overwrite or not job.output_path.is_file()]
    for job in jobs:
        status = "pending" if job in pending else "exists"
        print(
            f"[{status}] {job.kind}/{job.sequence}: {job.frame_count} frames "
            f"({job.first_frame:06d}..{job.first_frame + job.frame_count - 1:06d}), "
            f"{job.fps} fps -> {job.output_path}"
        )
    if args.dry_run or not pending:
        return
    workers = args.workers or min(len(pending), os.cpu_count() or 1)
    print(f"Encoding {len(pending)} video(s) with {workers} parallel FFmpeg process(es).")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(encode_job, job, args.codec, args.quality, args.overwrite) for job in pending]
        for future in as_completed(futures):
            job = future.result()
            print(f"[done] {job.kind}/{job.sequence}: {job.output_path}")


if __name__ == "__main__":
    main()
