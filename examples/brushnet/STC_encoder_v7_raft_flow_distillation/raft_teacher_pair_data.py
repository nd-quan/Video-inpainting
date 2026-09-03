"""Pair dataset for clean-RAFT teacher / degraded-RGB RAFT-student training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _normalize_branches(branches: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not branches:
        return ()
    normalized = []
    for value in branches:
        branch = Path(str(value).strip())
        if not str(value).strip() or branch.is_absolute() or ".." in branch.parts:
            raise ValueError(f"Invalid Class/sequence branch: {value!r}")
        normalized.append(branch.as_posix())
    if len(set(normalized)) != len(normalized):
        raise ValueError("include_branches must not contain duplicates")
    return tuple(normalized)


def _as_flow(value: np.ndarray, path: Path, key: str) -> torch.Tensor:
    value = np.asarray(value)
    if value.ndim != 3:
        raise ValueError(f"{path}:{key} must have three dimensions")
    if value.shape[0] == 2:
        pass
    elif value.shape[-1] == 2:
        value = value.transpose(2, 0, 1)
    else:
        raise ValueError(f"{path}:{key} must contain [dx,dy]")
    if not np.isfinite(value).all():
        raise FloatingPointError(f"Non-finite teacher flow in {path}:{key}")
    return torch.from_numpy(np.ascontiguousarray(value)).float()


def _as_valid(value: np.ndarray, path: Path, key: str) -> torch.Tensor:
    value = np.asarray(value)
    if value.ndim == 2:
        value = value[None]
    elif value.ndim == 3 and value.shape[-1] == 1:
        value = value.transpose(2, 0, 1)
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(f"{path}:{key} must have shape [1,H,W]")
    if not np.isfinite(value).all():
        raise FloatingPointError(f"Non-finite teacher validity in {path}:{key}")
    return torch.from_numpy(np.ascontiguousarray(value)).float().clamp_(0.0, 1.0)


class RAFTTeacherFlowPairDataset(Dataset):
    """Adjacent pairs which never cross an authoritative manifest segment.

    Source masks use the existing convention ``0=degraded BG, 255=ROI`` and
    are converted once to ``M_BG=1``.  The RAFT input deliberately remains the
    full degraded RGB pair; the mask only selects supervision/metrics.
    """

    def __init__(
        self,
        dataset_root: Path,
        teacher_flow_root: Path,
        split: str,
        resolution: int = 512,
        include_branches: Optional[Sequence[str]] = None,
        validate_files: bool = True,
    ):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.teacher_flow_root = Path(teacher_flow_root).expanduser().resolve()
        self.split = str(split)
        self.resolution = int(resolution)
        self.include_branches = _normalize_branches(include_branches)
        if self.resolution < 8 or self.resolution % 8:
            raise ValueError("resolution must be a positive multiple of 8")
        if self.split not in {"train", "valid", "test"}:
            raise ValueError(f"Unknown split {self.split!r}")
        metadata_path = self.teacher_flow_root / "metadata.json"
        manifest_path = self.dataset_root / "manifest.json"
        if not metadata_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("Dataset manifest or teacher-flow metadata is missing")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_root = self.metadata.get("dataset_root")
        if metadata_root and Path(metadata_root).expanduser().resolve() != self.dataset_root:
            raise ValueError(
                f"Teacher cache belongs to {metadata_root}, not {self.dataset_root}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.entries: List[Tuple[str, int, int, Path]] = []
        available_branches = set()
        for sequence in manifest.get("sequences", []):
            split_entry = sequence.get("splits", {}).get(self.split)
            if split_entry is None:
                continue
            branch = f"{sequence['class']}/{sequence['name']}"
            available_branches.add(branch)
            if self.include_branches and branch not in self.include_branches:
                continue
            for segment in split_entry.get("segments", []):
                start, end = int(segment["start"]), int(segment["end"])
                for index0 in range(start, end):
                    index1 = index0 + 1
                    cache_path = (
                        self.teacher_flow_root
                        / self.split
                        / Path(branch)
                        / f"{index0:06d}_{index1:06d}.npz"
                    )
                    self.entries.append((branch, index0, index1, cache_path))
        missing_requested = set(self.include_branches) - available_branches
        if missing_requested:
            raise FileNotFoundError(
                f"Requested branches are absent from manifest: {sorted(missing_requested)}"
            )
        if not self.entries:
            raise ValueError("No adjacent manifest pairs selected")
        if validate_files:
            for branch, index0, index1, cache_path in self.entries:
                for kind in ("input", "mask"):
                    for index in (index0, index1):
                        path = self.dataset_root / self.split / kind / Path(branch) / f"{index:06d}.png"
                        if not path.is_file():
                            raise FileNotFoundError(path)
                if not cache_path.is_file():
                    raise FileNotFoundError(cache_path)

    def __len__(self) -> int:
        return len(self.entries)

    def _rgb(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (self.resolution, self.resolution), resample=Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).div_(127.5).sub_(1.0)

    def _bg_mask(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("L").resize(
                (self.resolution, self.resolution), resample=Image.Resampling.NEAREST
            )
            roi = np.asarray(image, dtype=np.uint8) >= 128
        return torch.from_numpy(np.ascontiguousarray((~roi).astype(np.float32)[None]))

    @staticmethod
    def _teacher(path: Path):
        with np.load(path, allow_pickle=False) as data:
            required = {"teacher_f", "teacher_b", "valid_f", "valid_b"}
            missing = required - set(data.files)
            if missing:
                raise KeyError(f"{path} is missing {sorted(missing)}")
            return (
                _as_flow(data["teacher_f"], path, "teacher_f"),
                _as_flow(data["teacher_b"], path, "teacher_b"),
                _as_valid(data["valid_f"], path, "valid_f"),
                _as_valid(data["valid_b"], path, "valid_b"),
            )

    def __getitem__(self, index: int) -> Dict:
        branch, index0, index1, cache_path = self.entries[int(index)]
        input_root = self.dataset_root / self.split / "input" / Path(branch)
        mask_root = self.dataset_root / self.split / "mask" / Path(branch)
        teacher_f, teacher_b, valid_f, valid_b = self._teacher(cache_path)
        return {
            "degraded0": self._rgb(input_root / f"{index0:06d}.png"),
            "degraded1": self._rgb(input_root / f"{index1:06d}.png"),
            "bg0": self._bg_mask(mask_root / f"{index0:06d}.png"),
            "bg1": self._bg_mask(mask_root / f"{index1:06d}.png"),
            "teacher_forward": teacher_f,
            "teacher_backward": teacher_b,
            "valid_forward": valid_f,
            "valid_backward": valid_b,
            "video": branch,
            "frame0": index0,
            "frame1": index1,
        }


def evenly_limit_pairs_per_sequence(
    dataset: RAFTTeacherFlowPairDataset, maximum: int
) -> None:
    """Deterministically retain up to ``maximum`` validation pairs per sequence."""
    if maximum <= 0:
        return
    grouped: Dict[str, List[Tuple[str, int, int, Path]]] = {}
    for entry in dataset.entries:
        grouped.setdefault(entry[0], []).append(entry)
    selected = []
    for branch in sorted(grouped):
        entries = grouped[branch]
        if len(entries) <= maximum:
            selected.extend(entries)
            continue
        positions = np.linspace(0, len(entries) - 1, num=maximum).round().astype(int)
        selected.extend(entries[int(position)] for position in positions)
    dataset.entries = selected
