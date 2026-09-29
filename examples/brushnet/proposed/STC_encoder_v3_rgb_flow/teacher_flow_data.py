"""Attach cached clean-RAFT flow pairs to the existing V8 clip dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from shared_bg_noise_training import (
    HierarchicalV8ClipDataset,
    collate_shared_noise_clips,
)


TEACHER_KEYS = (
    "teacher_flow_forward",
    "teacher_flow_backward",
    "teacher_valid_forward",
    "teacher_valid_backward",
)


def _flow_tensor(array: np.ndarray, path: Path, key: str) -> torch.Tensor:
    value = np.asarray(array)
    if value.ndim != 3:
        raise ValueError(f"{path}:{key} must be a 3D flow array")
    if value.shape[0] == 2:
        pass
    elif value.shape[-1] == 2:
        value = value.transpose(2, 0, 1)
    else:
        raise ValueError(f"{path}:{key} must contain two flow channels")
    return torch.from_numpy(np.ascontiguousarray(value)).float()


def _valid_tensor(array: np.ndarray, path: Path, key: str) -> torch.Tensor:
    value = np.asarray(array)
    if value.ndim == 2:
        value = value[None]
    elif value.ndim == 3 and value.shape[-1] == 1:
        value = value.transpose(2, 0, 1)
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(f"{path}:{key} must have shape HxW or 1xHxW")
    if not np.isfinite(value).all():
        raise FloatingPointError(f"Non-finite teacher validity: {path}:{key}")
    return torch.from_numpy(np.ascontiguousarray(value)).float().clamp_(0.0, 1.0)


class TeacherFlowV8ClipDataset(HierarchicalV8ClipDataset):
    """V8 clip sample plus `[T-1]` clean-video teacher-flow tensors.

    Raw image masks remain handled exclusively by the parent dataset:
    source ``0=BG, 255=ROI`` becomes internal ``M_BG=1, M_ROI=0`` exactly once.
    """

    def __init__(self, *args, teacher_flow_root, validate_teacher_files=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_flow_root = Path(teacher_flow_root).expanduser().resolve()
        if not self.teacher_flow_root.is_dir():
            raise FileNotFoundError(self.teacher_flow_root)
        metadata_path = self.teacher_flow_root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Teacher-flow cache has no metadata.json: {metadata_path}"
            )
        with metadata_path.open("r", encoding="utf-8") as handle:
            self.teacher_metadata: Dict = json.load(handle)
        metadata_dataset = self.teacher_metadata.get("dataset_root")
        if metadata_dataset and Path(metadata_dataset).resolve() != self.dataset_root:
            raise ValueError(
                "Teacher cache belongs to a different dataset: "
                f"{metadata_dataset} != {self.dataset_root}"
            )

        # The original shared-noise loader groups numerically contiguous PNGs,
        # but some prepared sequences contain adjacent indices from distinct
        # source segments (for example Traffic 59 and 60). Clean RAFT teachers
        # intentionally never cross such boundaries. Filter those clips using
        # the authoritative manifest before validating cache completeness.
        manifest_value = self.teacher_metadata.get(
            "manifest", str(self.dataset_root / "manifest.json")
        )
        manifest_path = Path(manifest_value).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Teacher manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        allowed_pairs = set()
        for sequence in manifest.get("sequences", []):
            split_entry = sequence.get("splits", {}).get(self.split)
            if split_entry is None:
                continue
            branch = f"{sequence['class']}/{sequence['name']}"
            for segment in split_entry.get("segments", []):
                start, end = int(segment["start"]), int(segment["end"])
                allowed_pairs.update(
                    (branch, index, index + 1) for index in range(start, end)
                )
        if not allowed_pairs:
            raise ValueError(f"No manifest pairs for split={self.split!r}")
        original_clip_count = len(self.clips)
        self.clips = [
            clip
            for clip in self.clips
            if all(
                (clip[0], int(index0), int(index1)) in allowed_pairs
                for index0, index1 in zip(clip[2][:-1], clip[2][1:])
            )
        ]
        self.dropped_cross_segment_clips = original_clip_count - len(self.clips)
        if not self.clips:
            raise ValueError("Manifest filtering removed every training clip")
        covered_files = {
            path for _, paths, _ in self.clips for path in paths
        }
        self.covered_frame_count = len(covered_files)

        required_paths = set()
        for branch, _, frame_indices in self.clips:
            for index0, index1 in zip(frame_indices[:-1], frame_indices[1:]):
                required_paths.add(self._teacher_path(branch, index0, index1))
        self.required_teacher_pair_count = len(required_paths)
        if validate_teacher_files:
            missing = [path for path in sorted(required_paths) if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Missing cached teacher-flow pairs: "
                    f"count={len(missing)}, first={missing[0]}"
                )

    def _teacher_path(self, branch: str, index0: int, index1: int) -> Path:
        return (
            self.teacher_flow_root
            / self.split
            / Path(branch)
            / f"{int(index0):06d}_{int(index1):06d}.npz"
        )

    def _load_teacher_pair(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            required = {"teacher_f", "teacher_b", "valid_f", "valid_b"}
            missing = sorted(required - set(data.files))
            if missing:
                raise KeyError(f"{path} is missing teacher arrays: {missing}")
            forward = _flow_tensor(data["teacher_f"], path, "teacher_f")
            backward = _flow_tensor(data["teacher_b"], path, "teacher_b")
            valid_forward = _valid_tensor(data["valid_f"], path, "valid_f")
            valid_backward = _valid_tensor(data["valid_b"], path, "valid_b")
        if forward.shape != backward.shape:
            raise ValueError(f"Forward/backward teacher shapes differ: {path}")
        if valid_forward.shape[-2:] != forward.shape[-2:] or (
            valid_backward.shape[-2:] != forward.shape[-2:]
        ):
            raise ValueError(f"Teacher validity does not match flow: {path}")
        return forward, backward, valid_forward, valid_backward

    def __getitem__(self, item: int):
        sample = super().__getitem__(item)
        branch = str(sample["video"])
        frame_indices = [int(value) for value in sample["frame_ids"]]
        forward, backward, valid_forward, valid_backward = [], [], [], []
        for index0, index1 in zip(frame_indices[:-1], frame_indices[1:]):
            values = self._load_teacher_pair(
                self._teacher_path(branch, index0, index1)
            )
            forward.append(values[0])
            backward.append(values[1])
            valid_forward.append(values[2])
            valid_backward.append(values[3])
        sample.update(
            {
                "teacher_flow_forward": torch.stack(forward),
                "teacher_flow_backward": torch.stack(backward),
                "teacher_valid_forward": torch.stack(valid_forward),
                "teacher_valid_backward": torch.stack(valid_backward),
            }
        )
        return sample


def collate_teacher_flow_clips(examples: Sequence[Dict]) -> Dict:
    """Keep V8's flattened image contract and preserve clip-wise flow tensors."""
    batch = collate_shared_noise_clips(examples)
    for key in TEACHER_KEYS:
        batch[key] = torch.stack([example[key] for example in examples], dim=0)
    return batch
