#!/usr/bin/env python
"""Stitch RGB-STC evaluator clip outputs into one MP4 per sequence.

The RGB-STC evaluator writes overlapping clip folders such as::

    <eval_root>/BasketballPass__000000-000007/final/000000.png
    <eval_root>/BasketballPass__000006-000013/final/000006.png

This utility reconstructs a single chronological sequence without duplicated
overlap frames.  With ``--selection first`` (the default), each frame uses its
first occurrence in ascending clip-start order.  This is deterministic and
keeps the earliest valid RGB-STC context for every frame.  Encoding uses
OpenCV's ``VideoWriter`` in the same frame-by-frame style as ``imgToVideo.py``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2


DEFAULT_EVAL_ROOT = Path(
    "/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/"
    "eval_rgb_stc_v2/legacy_test/checkpoint-2000-sTCE-0.7-sBG-1.0"
)


# DEFAULT_EVAL_ROOT = Path(
#     "/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/"
#     "eval_rgb_stc_v2/valid/checkpoint-2000-sBG-09"
# )
DEFAULT_FRAME_KINDS = ("final",)
FRAME_KINDS = ("final", "raw", "input", "GT", "mask")
DEFAULT_SEQUENCE_FPS = {
    # SFU validation sequences. The evaluator uses ``Class_X__<name>`` in
    # folder names; fps_for_sequence below strips that prefix before lookup.
    "PeopleOnStreet": 30,
    "BQTerrace": 60,
    "BasketballDrive": 50,
    "Cactus": 50,
    "Kimono": 24,
    "BQMall": 60,
    "BasketballDrill": 50,
    "RaceHorsesC": 30,
    "BQSquare": 60,
    "BlowingBubbles": 50,
    "RaceHorsesD": 30,
    "FourPeople": 60,
    "Johnny": 60,
    "KristenAndSara": 60,
    # Legacy test-layout aliases, retained when --eval_root is overridden.
    "BasketballPass": 50,
    "ParkScene": 24,
    "PartyScene": 50,
    "RaceHorses": 30,
    "Traffic": 30,
}
CLIP_DIRECTORY_PATTERN = re.compile(r"^(?P<sequence>.+)__(?P<start>\d+)-(?P<end>\d+)$")
FRAME_PATTERN = re.compile(r"^(?P<frame>\d+)\.png$")


class SourceError(ValueError):
    """Raised when evaluator clip output is incomplete or inconsistent."""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stitch overlapping RGB-STC evaluation clips into MP4 videos."
    )
    parser.add_argument("--eval_root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Default: <eval_root>/videos",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Optional sequence names; default: every sequence present.",
    )
    parser.add_argument(
        "--frame_kinds",
        nargs="+",
        choices=FRAME_KINDS,
        default=list(DEFAULT_FRAME_KINDS),
        help="Evaluator image kinds to encode.",
    )
    parser.add_argument(
        "--selection",
        choices=("first", "last"),
        default="first",
        help="How to resolve frames repeated by overlapping evaluator clips.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override FPS for every sequence. Defaults to SFU sequence FPS.",
    )
    parser.add_argument(
        "--fourcc",
        default="mp4v",
        help="Four-character OpenCV video codec (default: mp4v).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate and print the planned videos without writing MP4 files.",
    )
    args = parser.parse_args()
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")
    if len(args.fourcc) != 4:
        parser.error("--fourcc must contain exactly four characters")
    return args


def parse_clip_directory(path: Path) -> Tuple[str, int, int]:
    match = CLIP_DIRECTORY_PATTERN.fullmatch(path.name)
    if match is None:
        raise SourceError(
            f"Unexpected clip directory name: {path.name}; expected "
            "<sequence>__<start>-<end>"
        )
    start, end = int(match.group("start")), int(match.group("end"))
    if end < start:
        raise SourceError(f"Invalid clip range in {path.name}: {start}..{end}")
    return match.group("sequence"), start, end


def read_clip_frames(clip_dir: Path, frame_kind: str) -> List[Tuple[int, Path]]:
    frame_dir = clip_dir / frame_kind
    if not frame_dir.is_dir():
        raise SourceError(f"Missing {frame_kind} folder: {frame_dir}")
    frames = []
    for path in frame_dir.iterdir():
        if not path.is_file():
            continue
        match = FRAME_PATTERN.fullmatch(path.name)
        if match is None:
            raise SourceError(f"Non-numeric PNG filename in {frame_dir}: {path.name}")
        frames.append((int(match.group("frame")), path))
    if not frames:
        raise SourceError(f"No PNG frames in {frame_dir}")
    return sorted(frames)


def discover_sources(
    eval_root: Path, frame_kind: str
) -> Mapping[str, List[Tuple[int, int, int, Path]]]:
    """Return sequence -> (clip start, clip end, frame id, image path)."""
    by_sequence: DefaultDict[str, List[Tuple[int, int, int, Path]]] = defaultdict(list)
    for clip_dir in sorted(path for path in eval_root.iterdir() if path.is_dir()):
        if clip_dir.name == "videos":
            continue
        sequence, start, end = parse_clip_directory(clip_dir)
        frames = read_clip_frames(clip_dir, frame_kind)
        actual_ids = [frame_id for frame_id, _ in frames]
        expected_ids = list(range(start, end + 1))
        if actual_ids != expected_ids:
            raise SourceError(
                f"{frame_kind} frames in {clip_dir} do not exactly match its "
                f"clip range {start}..{end}: got {actual_ids[:3]}...{actual_ids[-3:]}"
            )
        by_sequence[sequence].extend(
            (start, end, frame_id, path) for frame_id, path in frames
        )
    if not by_sequence:
        raise SourceError(f"No evaluator clip directories found in {eval_root}")
    return by_sequence


def select_unique_frames(
    candidates: Iterable[Tuple[int, int, int, Path]], selection: str
) -> List[Tuple[int, Path, int]]:
    """Select one image for each frame and verify the result is contiguous."""
    grouped: DefaultDict[int, List[Tuple[int, int, Path]]] = defaultdict(list)
    for clip_start, clip_end, frame_id, path in candidates:
        grouped[frame_id].append((clip_start, clip_end, path))

    chosen = []
    for frame_id in sorted(grouped):
        options = sorted(grouped[frame_id], key=lambda item: (item[0], item[1], str(item[2])))
        if selection == "last":
            options = list(reversed(options))
        clip_start, _, path = options[0]
        chosen.append((frame_id, path, clip_start))

    frame_ids = [frame_id for frame_id, _, _ in chosen]
    expected_ids = list(range(frame_ids[0], frame_ids[-1] + 1))
    if frame_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(frame_ids))
        raise SourceError(
            "Cannot make a continuous video; missing frame IDs: "
            f"{missing[:20]}{'...' if len(missing) > 20 else ''}"
        )
    return chosen


def fps_for_sequence(sequence: str, override_fps: float = None) -> float:
    """Resolve FPS after removing evaluator's optional ``Class_X__`` prefix."""
    if override_fps is not None:
        return float(override_fps)
    leaf_name = sequence.split("__", maxsplit=1)[-1]
    return float(
        DEFAULT_SEQUENCE_FPS.get(sequence, DEFAULT_SEQUENCE_FPS.get(leaf_name, 30))
    )


def encode_video(
    frames: Sequence[Tuple[int, Path, int]],
    output_path: Path,
    fps: float,
    fourcc: str,
    overwrite: bool,
):
    """Write BGR PNG frames with OpenCV, atomically through a temporary MP4."""
    if not frames:
        raise SourceError("Cannot encode an empty frame sequence")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists: {output_path}. Pass --overwrite to replace it."
        )
    first = cv2.imread(str(frames[0][1]), cv2.IMREAD_COLOR)
    if first is None:
        raise SourceError(f"Cannot read first frame: {frames[0][1]}")
    height, width = first.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.tmp{output_path.suffix}"
    )
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*fourcc),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"OpenCV could not open {temporary_path} with fourcc={fourcc!r}"
        )
    try:
        for frame_id, path, _ in frames:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise SourceError(f"Cannot read frame {frame_id}: {path}")
            if frame.shape[:2] != (height, width):
                raise SourceError(
                    f"Frame {frame_id} has {frame.shape[1]}x{frame.shape[0]}, "
                    f"expected {width}x{height}: {path}"
                )
            writer.write(frame)
    finally:
        writer.release()
    temporary_path.replace(output_path)


def main():
    args = parse_args()
    eval_root = args.eval_root.expanduser().resolve()
    if not eval_root.is_dir():
        raise FileNotFoundError(f"Evaluation root not found: {eval_root}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else eval_root / "videos"
    )

    manifest: Dict[str, object] = {
        "eval_root": str(eval_root),
        "output_dir": str(output_dir),
        "selection": args.selection,
        "fourcc": args.fourcc,
        "frame_kinds": args.frame_kinds,
        "videos": [],
    }
    requested_sequences = set(args.sequences) if args.sequences else None
    for frame_kind in args.frame_kinds:
        discovered = discover_sources(eval_root, frame_kind)
        sequence_names = sorted(discovered)
        if requested_sequences is not None:
            unknown = sorted(requested_sequences - set(sequence_names))
            if unknown:
                raise SourceError(
                    f"Requested sequences not found for {frame_kind}: {unknown}; "
                    f"available={sequence_names}"
                )
            sequence_names = [
                sequence for sequence in sequence_names if sequence in requested_sequences
            ]
        for sequence in sequence_names:
            frames = select_unique_frames(discovered[sequence], args.selection)
            fps = fps_for_sequence(sequence, args.fps)
            output_path = output_dir / sequence / f"{frame_kind}.mp4"
            video_record = {
                "sequence": sequence,
                "frame_kind": frame_kind,
                "fps": fps,
                "fourcc": args.fourcc,
                "frame_count": len(frames),
                "frame_range": [frames[0][0], frames[-1][0]],
                "output": str(output_path),
            }
            print(
                f"{sequence:16s} {frame_kind:8s} {len(frames):3d} frames "
                f"({frames[0][0]:06d}..{frames[-1][0]:06d}) at {fps:g} fps -> {output_path}"
            )
            if not args.dry_run:
                encode_video(
                    frames=frames,
                    output_path=output_path,
                    fps=fps,
                    fourcc=args.fourcc,
                    overwrite=args.overwrite,
                )
            manifest["videos"].append(video_record)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "video_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
