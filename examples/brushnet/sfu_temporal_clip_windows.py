"""Plan overlapping temporal evaluation windows and unique frame outputs."""

from pathlib import Path
from typing import Sequence


def build_temporal_windows(frame_count: int, clip_size: int, stride: int):
    """Cover a sequence with full windows and one shifted tail window if needed."""
    if clip_size < 2 or not 1 <= stride <= clip_size:
        raise ValueError("clip_size must be >= 2 and stride must be in [1, clip_size]")
    if frame_count < 0:
        raise ValueError("frame_count must be nonnegative")
    if frame_count == 0:
        return []
    if frame_count <= clip_size:
        return [(0, frame_count)]

    last_start = frame_count - clip_size
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(start, start + clip_size) for start in starts]


def plan_temporal_clips(
    indexed_frames: Sequence, clip_size: int, stride: int, existing_basenames=()
):
    """Run each needed window once and save each frame from its first window."""
    covered_names = set(existing_basenames)
    plan = []
    windows = build_temporal_windows(len(indexed_frames), clip_size, stride)
    for clip_idx, (start, end) in enumerate(windows):
        clip = indexed_frames[start:end]
        new_names = tuple(
            Path(image_path).name
            for _, (image_path, _) in clip
            if Path(image_path).name not in covered_names
        )
        if new_names:
            plan.append((clip_idx, clip, frozenset(new_names)))
            covered_names.update(new_names)
    return windows, plan
