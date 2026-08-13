#!/usr/bin/env python3
"""Encode all direct sequence-frame folders in a long-test result root.

The expected layout is::

    <dataset_root>/<sequence>/000000.png
    <dataset_root>/<sequence>/000001.png

``ffmpeg`` reads the image sequence directly, avoiding Python/OpenCV's
per-frame decode/write overhead.  Independent sequences are encoded in
parallel by default.
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
    "/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/"
    "/eval_finetune_sharednoise/long_test_fixedBG_temporal_v0"
)

# Match the ten legacy evaluation sequences.  RaceHorses is the output name
# for the Class_C/RaceHorsesC long-test source.
SEQUENCE_FPS = {
    "BasketballPass": 50,
    "ParkScene": 24,
    "PartyScene": 50,
    "RaceHorses": 30,
    "Traffic": 30,
    "BQMall": 60,
    "BQSquare": 60,
    "BQTerrace": 60,
    "FourPeople": 60,
    "PeopleOnStreet": 30,
}


@dataclass(frozen=True)
class SequenceJob:
    name: str
    frame_dir: Path
    output_path: Path
    fps: int
    first_frame: int
    frame_count: int


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fast parallel FFmpeg conversion of direct long-test PNG sequence "
            "folders to MP4."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--sequences",
        nargs="+",
        choices=tuple(SEQUENCE_FPS),
        default=None,
        help="Sequences to encode; default: every detected known sequence.",
    )
    parser.add_argument(
        "--output-name",
        default="output.mp4",
        help="MP4 filename written directly inside each sequence directory.",
    )
    parser.add_argument(
        "--codec",
        choices=("mpeg4", "png"),
        default="mpeg4",
        help="mpeg4 is fastest and compact; png is lossless but much larger.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=2,
        help="FFmpeg MPEG-4 qscale (1=best, 31=fastest/smallest; default: 2).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent FFmpeg processes; default: one per selected sequence.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-encode existing MP4 files. By default they are skipped.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate PNG numbering and print the work without encoding.",
    )
    args = parser.parse_args()
    if not 1 <= args.quality <= 31:
        parser.error("--quality must be in [1, 31]")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    return args


def numeric_pngs(frame_dir: Path) -> Sequence[Path]:
    paths = list(frame_dir.glob("*.png"))
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
        raise ValueError(
            f"Frames must be numerically contiguous in {frame_dir}; "
            f"missing={missing[:20]}"
        )
    return [path for _, path in indexed]


def discover_jobs(args) -> List[SequenceJob]:
    root = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    detected = {path.name: path for path in root.iterdir() if path.is_dir()}
    requested = args.sequences or sorted(set(detected) & set(SEQUENCE_FPS))
    if not requested:
        raise ValueError(f"No known sequence directories found below {root}")

    jobs = []
    for name in requested:
        frame_dir = detected.get(name)
        if frame_dir is None:
            raise FileNotFoundError(f"Sequence directory not found: {root / name}")
        frames = numeric_pngs(frame_dir)
        jobs.append(
            SequenceJob(
                name=name,
                frame_dir=frame_dir,
                output_path=frame_dir / args.output_name,
                fps=SEQUENCE_FPS[name],
                first_frame=int(frames[0].stem),
                frame_count=len(frames),
            )
        )
    return jobs


def ffmpeg_command(job: SequenceJob, codec: str, quality: int, overwrite: bool):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y" if overwrite else "-n",
        "-framerate",
        str(job.fps),
        "-start_number",
        str(job.first_frame),
        "-i",
        str(job.frame_dir / "%06d.png"),
        "-frames:v",
        str(job.frame_count),
        "-an",
    ]
    if codec == "mpeg4":
        command.extend(("-c:v", "mpeg4", "-q:v", str(quality), "-pix_fmt", "yuv420p"))
    else:
        command.extend(("-c:v", "png", "-pix_fmt", "rgb24"))
    command.append(str(job.output_path))
    return command


def encode_job(job: SequenceJob, codec: str, quality: int, overwrite: bool):
    subprocess.run(
        ffmpeg_command(job, codec, quality, overwrite), check=True, text=True
    )
    return job.name, job.output_path


def main():
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found in PATH")
    jobs = discover_jobs(args)
    pending = [job for job in jobs if args.overwrite or not job.output_path.is_file()]

    for job in jobs:
        status = "pending" if job in pending else "exists"
        print(
            f"[{status}] {job.name}: {job.frame_count} frames "
            f"({job.first_frame:06d}..{job.first_frame + job.frame_count - 1:06d}), "
            f"{job.fps} fps -> {job.output_path}"
        )
    if args.dry_run or not pending:
        return

    workers = args.workers or min(len(pending), os.cpu_count() or 1)
    print(f"Encoding {len(pending)} sequence(s) with {workers} parallel FFmpeg process(es).")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(encode_job, job, args.codec, args.quality, args.overwrite): job
            for job in pending
        }
        for future in as_completed(futures):
            name, output_path = future.result()
            print(f"[done] {name}: {output_path}")


if __name__ == "__main__":
    main()
