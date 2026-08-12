#!/usr/bin/env python
"""Create one MP4 per sequence from an SFU input tree.

Expected input layout::

    <input_root>/Class_A/Traffic/000000.png
    <input_root>/Class_B/Cactus/000000.png

The script writes videos to ``<input_root>/videos/<class>/<sequence>.mp4`` by
default.  It accepts symlinked frames, as used by ``SFU_STC_flow/long_test``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2


DEFAULT_INPUT_ROOT = Path("/home/cilab/ndquan/videoInpainting/SFU_STC_flow/long_test/input")
SEQUENCE_FPS = {
    "PeopleOnStreet": 30, "BQTerrace": 60, "BasketballDrive": 50,
    "Cactus": 50, "Kimono": 24, "BQMall": 60, "BasketballDrill": 50,
    "RaceHorsesC": 30, "BQSquare": 60, "BlowingBubbles": 50,
    "RaceHorsesD": 30, "FourPeople": 60, "Johnny": 60,
    "KristenAndSara": 60, "BasketballPass": 50, "ParkScene": 24,
    "PartyScene": 50, "RaceHorses": 30, "Traffic": 30,
}
FRAME_PATTERN = re.compile(r"^(\d+)\.png$", re.IGNORECASE)


class SourceError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SFU input frame folders to MP4 files.")
    parser.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Default: <input_root>/videos")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Optional class directory names, e.g. Class_A Class_B.")
    parser.add_argument("--sequences", nargs="+", default=None,
                        help="Optional sequence names, e.g. Traffic Cactus.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override the per-sequence SFU FPS.")
    parser.add_argument("--fourcc", default="mp4v")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")
    if len(args.fourcc) != 4:
        parser.error("--fourcc must contain exactly four characters")
    return args


def read_frames(sequence_dir: Path) -> List[Tuple[int, Path]]:
    frames = []
    for path in sequence_dir.iterdir():
        if not path.is_file():
            continue
        match = FRAME_PATTERN.fullmatch(path.name)
        if match is None:
            raise SourceError(f"Non-numeric PNG filename in {sequence_dir}: {path.name}")
        frames.append((int(match.group(1)), path))
    if not frames:
        raise SourceError(f"No PNG frames in {sequence_dir}")
    frames.sort()
    frame_ids = [frame_id for frame_id, _ in frames]
    expected = list(range(frame_ids[0], frame_ids[-1] + 1))
    if frame_ids != expected:
        missing = sorted(set(expected) - set(frame_ids))
        raise SourceError(f"Non-contiguous frames in {sequence_dir}; missing {missing[:20]}")
    return frames


def encode(frames: Sequence[Tuple[int, Path]], output: Path, fps: float, fourcc: str,
           overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}. Pass --overwrite to replace it.")
    first = cv2.imread(str(frames[0][1]), cv2.IMREAD_COLOR)
    if first is None:
        raise SourceError(f"Cannot read frame: {frames[0][1]}")
    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open {temporary} with fourcc={fourcc!r}")
    try:
        for frame_id, path in frames:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise SourceError(f"Cannot read frame {frame_id}: {path}")
            if frame.shape[:2] != (height, width):
                raise SourceError(f"Unexpected size for frame {frame_id}: {path}")
            writer.write(frame)
    finally:
        writer.release()
    temporary.replace(output)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else input_root / "videos"
    requested_classes = set(args.classes) if args.classes else None
    requested_sequences = set(args.sequences) if args.sequences else None
    records: List[Dict[str, object]] = []

    class_dirs = sorted(path for path in input_root.iterdir() if path.is_dir() and path.name != "videos")
    if requested_classes is not None:
        available = {path.name for path in class_dirs}
        unknown = sorted(requested_classes - available)
        if unknown:
            raise SourceError(f"Requested classes not found: {unknown}; available={sorted(available)}")
        class_dirs = [path for path in class_dirs if path.name in requested_classes]
    for class_dir in class_dirs:
        for sequence_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            if requested_sequences is not None and sequence_dir.name not in requested_sequences:
                continue
            frames = read_frames(sequence_dir)
            fps = float(args.fps if args.fps is not None else SEQUENCE_FPS.get(sequence_dir.name, 30))
            output = output_dir / class_dir.name / f"{sequence_dir.name}.mp4"
            print(f"{class_dir.name}/{sequence_dir.name:16s} {len(frames):3d} frames "
                  f"({frames[0][0]:06d}..{frames[-1][0]:06d}) at {fps:g} fps -> {output}")
            if not args.dry_run:
                encode(frames, output, fps, args.fourcc, args.overwrite)
            records.append({"class": class_dir.name, "sequence": sequence_dir.name,
                            "fps": fps, "frame_count": len(frames),
                            "frame_range": [frames[0][0], frames[-1][0]], "output": str(output)})
    if requested_sequences is not None and not records:
        raise SourceError(f"Requested sequences not found: {sorted(requested_sequences)}")
    if not records:
        raise SourceError(f"No sequence directories found in {input_root}")
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "video_manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
