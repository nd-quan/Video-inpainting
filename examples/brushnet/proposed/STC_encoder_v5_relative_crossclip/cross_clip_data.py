"""Self-contained predecessor/current samples for DDP-safe V5 training/eval."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple

import torch

try:
    from shared_bg_noise_training import (
        FlatV8TestClipDataset,
        HierarchicalV8ClipDataset,
    )
    from STC_encoder_v3_rgb_flow.teacher_flow_data import (
        TEACHER_KEYS,
        TeacherFlowV8ClipDataset,
        collate_teacher_flow_clips,
    )
except ModuleNotFoundError:  # Imported through examples.brushnet.
    from ..shared_bg_noise_training import (
        FlatV8TestClipDataset,
        HierarchicalV8ClipDataset,
    )
    from ..STC_encoder_v3_rgb_flow.teacher_flow_data import (
        TEACHER_KEYS,
        TeacherFlowV8ClipDataset,
        collate_teacher_flow_clips,
    )


PREVIOUS_TENSOR_KEYS = (
    "previous_conditioning_pixel_values",
    "previous_masks",
    "previous_frame_ids",
    "previous_valid_mask",
)


def suffix_prefix_overlap(previous_ids: Sequence[int], current_ids: Sequence[int]) -> int:
    """Return the largest exact absolute-ID suffix/prefix overlap."""
    maximum = min(len(previous_ids), len(current_ids))
    for count in range(maximum, 0, -1):
        if tuple(previous_ids[-count:]) == tuple(current_ids[:count]):
            return count
    return 0


def build_predecessor_index(
    clips: Sequence[Tuple[str, Sequence, Sequence[int]]],
    expected_stride: Optional[int] = None,
) -> Tuple[Tuple[Optional[int], ...], Tuple[int, ...]]:
    """Map every clip to the latest valid predecessor in the same branch.

    Training supplies ``expected_stride`` and therefore accepts exactly the
    T/S overlap contract. Evaluation leaves it unset so shifted tail windows
    with a larger overlap remain valid.
    """
    grouped: Dict[str, list] = defaultdict(list)
    for index, (branch, _, frame_ids) in enumerate(clips):
        ids = tuple(int(value) for value in frame_ids)
        grouped[str(branch)].append((ids[0], ids[-1], index, ids))

    predecessors = [None] * len(clips)
    overlaps = [0] * len(clips)
    for entries in grouped.values():
        entries.sort(key=lambda item: (item[0], item[1], item[2]))
        for position, (current_start, _, current_index, current_ids) in enumerate(
            entries
        ):
            best = None
            for previous_start, _, previous_index, previous_ids in entries[:position]:
                if expected_stride is not None and (
                    current_start - previous_start != int(expected_stride)
                ):
                    continue
                overlap = suffix_prefix_overlap(previous_ids, current_ids)
                if overlap <= 0:
                    continue
                candidate = (previous_start, overlap, previous_index)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
            if best is not None:
                predecessors[current_index] = int(best[2])
                overlaps[current_index] = int(best[1])
    return tuple(predecessors), tuple(overlaps)


class _CrossClipContextMixin:
    predecessor_indices: Tuple[Optional[int], ...]
    predecessor_overlaps: Tuple[int, ...]

    def rebuild_predecessors(self, expected_stride: Optional[int] = None) -> None:
        self.predecessor_indices, self.predecessor_overlaps = build_predecessor_index(
            self.clips, expected_stride=expected_stride
        )
        self.cross_clip_transition_count = sum(
            index is not None for index in self.predecessor_indices
        )
        self.cross_clip_run_count = len(self.clips) - self.cross_clip_transition_count

    @staticmethod
    def _attach_previous(current: Dict, previous: Optional[Dict], overlap: int) -> Dict:
        if previous is None:
            previous_rgb = torch.zeros_like(current["conditioning_pixel_values"])
            previous_mask = torch.zeros_like(current["masks"])
            previous_ids = torch.full_like(current["frame_ids"], -1)
            valid = torch.zeros_like(current["frame_ids"], dtype=torch.bool)
            previous_video = ""
        else:
            previous_rgb = previous["conditioning_pixel_values"]
            previous_mask = previous["masks"]
            previous_ids = previous["frame_ids"]
            valid = torch.ones_like(previous_ids, dtype=torch.bool)
            previous_video = str(previous["video"])
        current.update(
            {
                "previous_conditioning_pixel_values": previous_rgb,
                "previous_masks": previous_mask,
                "previous_frame_ids": previous_ids,
                "previous_valid_mask": valid,
                "previous_video": previous_video,
                "predecessor_overlap": int(overlap),
            }
        )
        return current


class CrossClipTeacherFlowV8Dataset(_CrossClipContextMixin, TeacherFlowV8ClipDataset):
    """Teacher-flow current clip plus a read-only predecessor RGB/mask clip."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.stride >= self.clip_length:
            raise ValueError("V5 cross-clip training requires stride < clip_length")
        self.rebuild_predecessors(expected_stride=self.stride)

    def __getitem__(self, item: int):
        item = int(item)
        current = TeacherFlowV8ClipDataset.__getitem__(self, item)
        previous_index = self.predecessor_indices[item]
        previous = None
        if previous_index is not None:
            # The context pass needs no teacher tensors. Avoid a second set of
            # NPZ reads by calling the RGB/mask parent directly.
            previous = HierarchicalV8ClipDataset.__getitem__(self, previous_index)
        return self._attach_previous(
            current, previous, self.predecessor_overlaps[item]
        )


class CrossClipHierarchicalEvalDataset(
    _CrossClipContextMixin, HierarchicalV8ClipDataset
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rebuild_predecessors()

    def __getitem__(self, item: int):
        item = int(item)
        current = HierarchicalV8ClipDataset.__getitem__(self, item)
        previous_index = self.predecessor_indices[item]
        previous = (
            None
            if previous_index is None
            else HierarchicalV8ClipDataset.__getitem__(self, previous_index)
        )
        return self._attach_previous(
            current, previous, self.predecessor_overlaps[item]
        )


class CrossClipFlatEvalDataset(_CrossClipContextMixin, FlatV8TestClipDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rebuild_predecessors()

    def __getitem__(self, item: int):
        item = int(item)
        current = FlatV8TestClipDataset.__getitem__(self, item)
        previous_index = self.predecessor_indices[item]
        previous = (
            None
            if previous_index is None
            else FlatV8TestClipDataset.__getitem__(self, previous_index)
        )
        return self._attach_previous(
            current, previous, self.predecessor_overlaps[item]
        )


def collate_cross_clip_teacher_flow(examples: Sequence[Dict]) -> Dict:
    batch = collate_teacher_flow_clips(examples)
    for key in PREVIOUS_TENSOR_KEYS:
        batch[key] = torch.stack([example[key] for example in examples], dim=0)
    batch["previous_videos"] = [str(example["previous_video"]) for example in examples]
    batch["predecessor_overlaps"] = torch.tensor(
        [int(example["predecessor_overlap"]) for example in examples],
        dtype=torch.long,
    )
    # Guard against accidental loss of current teacher tensors in future edits.
    for key in TEACHER_KEYS:
        if key not in batch:
            raise RuntimeError(f"Cross-clip collation lost teacher tensor {key}")
    return batch

