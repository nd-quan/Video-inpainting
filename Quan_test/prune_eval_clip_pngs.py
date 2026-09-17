#!/usr/bin/env python3
"""Retain selected evaluator clip PNGs and remove only the rest.

The script preserves videos, JSON, and PNGs outside immediate evaluator clip
folders. Empty image folders can optionally be removed. It defaults to a dry
run; ``--apply`` is required to delete an explicitly inventoried set of PNGs.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple


CLIP_PATTERN = re.compile(r"^(?P<sequence>.+)__(?P<start>\d+)-(?P<end>\d+)$")
KEPT_KINDS = frozenset(("final", "gt", "input", "mask_roi"))
BASKETBALL_KEEP_STARTS = (60, 72)


@dataclass(frozen=True)
class Clip:
    path: Path
    sequence: str
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune evaluator PNGs while retaining two middle consecutive clips."
    )
    parser.add_argument("eval_roots", nargs="+", type=Path)
    parser.add_argument("--recursive", action="store_true", help="Discover sharded clip folders recursively.")
    parser.add_argument("--keep_kinds", nargs="+", default=sorted(KEPT_KINDS),
                        help="Image folders to retain in the two selected clips; use final for output only.")
    parser.add_argument("--middle_all", action="store_true",
                        help="Choose middle clips for every sequence, overriding the legacy BasketballPass ranges.")
    parser.add_argument("--remove_empty_image_dirs", action="store_true",
                        help="Remove empty immediate image folders after deleting their PNGs.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete the planned PNG files. Omit for an inventory-only dry run.",
    )
    return parser.parse_args()


def discover_clips(root: Path, recursive: bool = False) -> List[Clip]:
    clips = []
    candidates = root.rglob("*") if recursive else root.iterdir()
    for path in candidates:
        if not path.is_dir():
            continue
        match = CLIP_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        if path.is_symlink():
            raise ValueError(f"Refusing symlinked clip folder: {path}")
        start, end = int(match.group("start")), int(match.group("end"))
        if end < start:
            raise ValueError(f"Invalid clip range: {path}")
        clips.append(Clip(path, match.group("sequence"), start, end))
    if not clips:
        raise ValueError(f"No evaluator clip directories found in {root}")
    return sorted(clips, key=lambda clip: (clip.sequence, clip.start, clip.end))


def choose_middle_pair(clips: Sequence[Clip]) -> Tuple[Clip, Clip]:
    if len(clips) < 2:
        raise ValueError(f"Need at least two clips for {clips[0].sequence if clips else 'sequence'}")
    coverage_center = (clips[0].start + clips[-1].end) / 2.0
    candidates = []
    for first, second in zip(clips, clips[1:]):
        if second.start <= first.start or second.start > first.end:
            continue
        pair_center = (first.start + second.end) / 2.0
        candidates.append((abs(pair_center - coverage_center), first.start, first, second))
    if not candidates:
        raise ValueError(f"No overlapping consecutive clip pair for {clips[0].sequence}")
    _, _, first, second = min(candidates, key=lambda item: (item[0], item[1]))
    return first, second


def choose_kept_clips(clips: Iterable[Clip], middle_all: bool = False) -> Set[Path]:
    by_sequence = {}
    for clip in clips:
        by_sequence.setdefault(clip.sequence, []).append(clip)
    kept = set()
    for sequence, sequence_clips in by_sequence.items():
        sequence_clips.sort(key=lambda clip: (clip.start, clip.end))
        if sequence == "BasketballPass" and not middle_all:
            selected = [
                next((clip for clip in sequence_clips if clip.start == start), None)
                for start in BASKETBALL_KEEP_STARTS
            ]
            if any(clip is None for clip in selected):
                available = [clip.start for clip in sequence_clips]
                raise ValueError(
                    f"BasketballPass must contain starts {BASKETBALL_KEEP_STARTS}; "
                    f"available={available}"
                )
            first, second = selected
            if second.start > first.end:
                raise ValueError("Requested BasketballPass clips are not consecutive")
        else:
            first, second = choose_middle_pair(sequence_clips)
        kept.update((first.path, second.path))
    return kept


def inventory_pngs(root: Path, clips: Sequence[Clip], kept_clips: Set[Path],
                   kept_kinds: Set[str] = KEPT_KINDS):
    delete_paths = []
    retain_paths = []
    for clip in clips:
        for kind_dir in (path for path in clip.path.iterdir() if path.is_dir()):
            if kind_dir.is_symlink():
                raise ValueError(f"Refusing symlinked image folder: {kind_dir}")
            for png_path in kind_dir.glob("*.png"):
                if png_path.is_symlink() or not png_path.is_file():
                    raise ValueError(f"Refusing non-regular PNG file: {png_path}")
                png_path.resolve().relative_to(root)
                if clip.path in kept_clips and kind_dir.name in kept_kinds:
                    retain_paths.append(png_path)
                else:
                    delete_paths.append(png_path)
    for path in delete_paths + retain_paths:
        relative = path.relative_to(root)
        if relative.is_absolute() or ".." in relative.parts or path.suffix.lower() != ".png":
            raise ValueError(f"Unsafe inventory path: {path}")
    return sorted(delete_paths), sorted(retain_paths)


def describe_kept(root: Path, kept_clips: Set[Path], kept_kinds: Set[str] = KEPT_KINDS) -> List[dict]:
    entries = []
    for path in sorted(kept_clips):
        match = CLIP_PATTERN.fullmatch(path.name)
        assert match is not None
        entries.append(
            {
                "sequence": match.group("sequence"),
                "clip": path.name,
                "relative_path": str(path.relative_to(root)),
                "kept_kinds": sorted(kept_kinds),
            }
        )
    return entries


def process_root(root: Path, apply: bool, recursive: bool = False,
                 kept_kinds: Set[str] = KEPT_KINDS, middle_all: bool = False,
                 remove_empty_image_dirs: bool = False) -> None:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Evaluation root not found: {root}")
    clips = discover_clips(root, recursive=recursive)
    kept_clips = choose_kept_clips(clips, middle_all=middle_all)
    for clip_path in kept_clips:
        clip = next(clip for clip in clips if clip.path == clip_path)
        for kind in kept_kinds:
            expected = {f"{frame:06d}.png" for frame in range(clip.start, clip.end + 1)}
            actual = {path.name for path in (clip_path / kind).glob("*.png")}
            if actual != expected:
                raise ValueError(f"Selected clip has incomplete {kind} PNGs: {clip_path}")
    delete_paths, retain_paths = inventory_pngs(root, clips, kept_clips, kept_kinds)
    image_dirs = sorted({path.parent for path in delete_paths})
    plan = {
        "eval_root": str(root),
        "clip_count": len(clips),
        "kept_clips": describe_kept(root, kept_clips, kept_kinds),
        "kept_kinds": sorted(kept_kinds),
        "recursive": recursive,
        "middle_all": middle_all,
        "remove_empty_image_dirs": remove_empty_image_dirs,
        "retained_png_count": len(retain_paths),
        "deleted_png_count": len(delete_paths),
        "deleted_bytes": sum(path.stat().st_size for path in delete_paths),
        "deleted_png_paths": [str(path.relative_to(root)) for path in delete_paths],
    }
    print(json.dumps({key: value for key, value in plan.items() if key != "deleted_png_paths"}, indent=2))
    if not apply:
        return
    plan_path = root / "png_prune_plan.json"
    temporary_path = plan_path.with_suffix(".tmp.json")
    temporary_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(plan_path)
    for path in delete_paths:
        path.unlink()
    remaining = [path for clip in clips for kind in clip.path.iterdir() if kind.is_dir() for path in kind.glob("*.png")]
    if sorted(remaining) != retain_paths:
        raise RuntimeError(f"Post-delete PNG inventory mismatch in {root}")
    if remove_empty_image_dirs:
        removed_dirs = 0
        for path in image_dirs:
            if not any(path.iterdir()):
                path.rmdir()
                removed_dirs += 1
        print(f"Removed {removed_dirs} empty image folders in {root}")
    print(f"Deleted {len(delete_paths)} PNGs; retained {len(retain_paths)} PNGs in {root}")


def main() -> None:
    args = parse_args()
    for root in args.eval_roots:
        process_root(root, args.apply, recursive=args.recursive,
                     kept_kinds=frozenset(args.keep_kinds), middle_all=args.middle_all,
                     remove_empty_image_dirs=args.remove_empty_image_dirs)


if __name__ == "__main__":
    main()
