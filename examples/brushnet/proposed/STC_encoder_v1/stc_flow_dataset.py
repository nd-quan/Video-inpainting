"""Split-aware video clips for STC flow prediction.

The dataset prepared by :mod:`prepare_sfu_stc_dataset` is hierarchical::

    <root>/<split>/<kind>/<class>/<sequence>/<source_index>.png

Clips are enumerated from the contiguous segments recorded in ``manifest.json``.
Segment boundaries are deliberately kept even when two segments happen to be
adjacent, so a clip can never bridge a held-out range or a video boundary.

RGB tensors are returned in ``[-1, 1]`` and masks are returned as float tensors
with ``0=background`` and ``1=ROI``.  Optional teacher-flow caches are supported
for the trainer, but are not required for data preparation or data-only tests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


KINDS = ("GT", "input", "mask")


@dataclass(frozen=True)
class ClipRecord:
    """Location and source-frame identity of one contiguous clip."""

    sequence: str
    class_name: str
    fps: float
    segment_index: int
    segment_start: int
    segment_end: int
    frame_indices: Tuple[int, ...]
    roots: Mapping[str, Path]

    @property
    def start(self) -> int:
        return self.frame_indices[0]


def _split_value(value, split, default=None):
    """Resolve either a scalar config value or a ``{split: value}`` mapping."""

    if isinstance(value, Mapping):
        return value.get(split, default)
    return default if value is None else value


def _safe_relative_root(dataset_root: Path, relative_root: str, kind: str) -> Path:
    try:
        rendered = relative_root.format(kind=kind)
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"Invalid relative_root template {relative_root!r}"
        ) from error
    relative = Path(rendered)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"relative_root must stay inside dataset_root: {rendered}")
    if "{" in rendered or "}" in rendered:
        raise ValueError(f"Unresolved field in relative_root: {rendered}")
    return dataset_root / relative


class STCFlowDataset(Dataset):
    """Load aligned STC clips described by the prepared dataset manifest.

    Parameters
    ----------
    config:
        Either the data section itself or a full training config containing a
        ``data`` section.  Recognized data keys are ``dataset_root``,
        ``manifest``, ``clip_length``, ``height``, ``width``, ``stride`` (or
        ``<split>_stride``), ``random_horizontal_flip``,
        ``horizontal_flip_probability``, ``teacher_flow_root`` and
        ``validate_files``. Set ``load_gt=false`` after the clean teacher cache
        has been generated to avoid decoding unused GT images during Stage 1.
    split:
        One of the split names present in the manifest, normally ``train``,
        ``valid`` or ``test``.

    Horizontal-flip decisions are stateless functions of seed, epoch and item
    index.  They are therefore reproducible across repeated reads and DataLoader
    worker counts.  Call :meth:`set_epoch` to obtain a deterministic new set of
    augmentations in a later epoch.
    """

    def __init__(self, config: Mapping, split: str):
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        data = config.get("data", config)
        if not isinstance(data, Mapping):
            raise TypeError("config['data'] must be a mapping")

        self.split = str(split)
        dataset_root = data.get("dataset_root", data.get("root"))
        if not dataset_root:
            raise ValueError("data.dataset_root is required")
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        manifest_path = data.get("manifest", self.dataset_root / "manifest.json")
        self.manifest_path = Path(manifest_path).expanduser().resolve()

        self.clip_length = int(data.get("clip_length", 4))
        self.height = int(data.get("height", 512))
        self.width = int(data.get("width", 512))
        stride_value = data.get(f"{self.split}_stride", data.get("stride", 1))
        self.stride = int(stride_value)
        if self.clip_length < 1:
            raise ValueError("clip_length must be at least 1")
        if self.height < 1 or self.width < 1:
            raise ValueError("height and width must be positive")
        if self.stride < 1:
            raise ValueError("stride must be at least 1")

        seed_default = config.get("seed", 0) if data is not config else 0
        self.seed = int(data.get("seed", seed_default))
        self.epoch = 0

        flip_enabled = bool(
            _split_value(data.get("random_horizontal_flip"), self.split, False)
        )
        probability_default = 0.5 if flip_enabled else 0.0
        self.horizontal_flip_probability = float(
            _split_value(
                data.get("horizontal_flip_probability"),
                self.split,
                probability_default,
            )
        )
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")

        teacher_root = data.get("teacher_flow_root")
        self.teacher_flow_root = (
            Path(teacher_root).expanduser().resolve() if teacher_root else None
        )
        if self.teacher_flow_root is not None and self.clip_length < 2:
            raise ValueError("teacher_flow_root requires clip_length >= 2")
        self.validate_files = bool(data.get("validate_files", True))
        # GT is required on disk for teacher-flow generation and dataset
        # validation, but Stage-1 training consumes the cached teacher instead
        # of RGB GT. Skipping GT decoding avoids repeatedly decompressing large
        # native-resolution PNGs for heavily overlapping training clips.
        self.load_gt = bool(data.get("load_gt", True))

        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset manifest not found: {self.manifest_path}")
        if not isinstance(manifest.get("sequences"), list):
            raise ValueError("manifest must contain a 'sequences' list")
        manifest_splits = manifest.get("splits")
        if manifest_splits is not None and self.split not in manifest_splits:
            raise ValueError(f"Split {self.split!r} is not present in the manifest")

        records = []
        sequence_keys = set()
        for sequence_entry in manifest["sequences"]:
            sequence = str(sequence_entry["name"])
            class_name = str(sequence_entry["class"])
            sequence_key = (class_name, sequence)
            if sequence_key in sequence_keys:
                raise ValueError(f"Duplicate sequence in manifest: {sequence_key}")
            sequence_keys.add(sequence_key)

            split_entry = sequence_entry.get("splits", {}).get(self.split)
            if split_entry is None:
                continue
            segments = split_entry.get("segments")
            if not isinstance(segments, list):
                raise ValueError(
                    f"{class_name}/{sequence}/{self.split} has no segments list"
                )
            relative_root = split_entry.get(
                "relative_root",
                f"{self.split}/{{kind}}/{class_name}/{sequence}",
            )
            roots = {
                kind: _safe_relative_root(
                    self.dataset_root, str(relative_root), kind
                )
                for kind in KINDS
            }

            normalized_segments = []
            for segment_index, segment in enumerate(segments):
                start, end = int(segment["start"]), int(segment["end"])
                if start < 0 or end < start:
                    raise ValueError(
                        f"Invalid segment {class_name}/{sequence}: {start}..{end}"
                    )
                normalized_segments.append((start, end))
                last_start = end - self.clip_length + 1
                for clip_start in range(start, last_start + 1, self.stride):
                    indices = tuple(
                        range(clip_start, clip_start + self.clip_length)
                    )
                    records.append(
                        ClipRecord(
                            sequence=sequence,
                            class_name=class_name,
                            fps=float(sequence_entry["fps"]),
                            segment_index=segment_index,
                            segment_start=start,
                            segment_end=end,
                            frame_indices=indices,
                            roots=roots,
                        )
                    )

            ordered_segments = sorted(normalized_segments)
            for previous, current in zip(
                ordered_segments[:-1], ordered_segments[1:]
            ):
                if current[0] <= previous[1]:
                    raise ValueError(
                        f"Overlapping {self.split} segments for "
                        f"{class_name}/{sequence}: {previous} and {current}"
                    )
            declared_count = split_entry.get("frame_count")
            actual_count = sum(end - start + 1 for start, end in normalized_segments)
            if declared_count is not None and int(declared_count) != actual_count:
                raise ValueError(
                    f"frame_count mismatch for {class_name}/{sequence}/{self.split}: "
                    f"manifest={declared_count}, segments={actual_count}"
                )

        if not records:
            raise ValueError(
                f"No clips of length {self.clip_length} found for split={self.split}"
            )
        self.clips: Tuple[ClipRecord, ...] = tuple(records)

        if self.validate_files:
            self._validate_referenced_files()

    def __len__(self) -> int:
        return len(self.clips)

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic augmentation stream for ``epoch``."""

        self.epoch = int(epoch)

    def _should_flip(self, item: int) -> bool:
        probability = self.horizontal_flip_probability
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        payload = f"{self.seed}:{self.epoch}:{int(item)}".encode("utf-8")
        value = int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(), "big"
        )
        return value < int(probability * (1 << 64))

    @staticmethod
    def _frame_path(record: ClipRecord, kind: str, index: int) -> Path:
        return record.roots[kind] / f"{index:06d}.png"

    def _teacher_path(self, record: ClipRecord, index0: int, index1: int) -> Path:
        root = (
            self.teacher_flow_root
            / self.split
            / record.class_name
            / record.sequence
        )
        padded = root / f"{index0:06d}_{index1:06d}.npz"
        if padded.is_file():
            return padded
        unpadded = root / f"{index0}_{index1}.npz"
        return unpadded if unpadded.is_file() else padded

    def _validate_referenced_files(self) -> None:
        checked_frames = set()
        checked_pairs = set()
        for record in self.clips:
            for index in record.frame_indices:
                key = (record.class_name, record.sequence, index)
                if key in checked_frames:
                    continue
                checked_frames.add(key)
                for kind in KINDS:
                    path = self._frame_path(record, kind, index)
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"Missing aligned {kind} frame: {path}"
                        )
            if self.teacher_flow_root is not None:
                for index0, index1 in zip(
                    record.frame_indices[:-1], record.frame_indices[1:]
                ):
                    key = (record.class_name, record.sequence, index0, index1)
                    if key in checked_pairs:
                        continue
                    checked_pairs.add(key)
                    path = self._teacher_path(record, index0, index1)
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing teacher-flow pair: {path}")

    def _load_triplet(self, record: ClipRecord, index: int):
        gt_path = self._frame_path(record, "GT", index)
        input_path = self._frame_path(record, "input", index)
        mask_path = self._frame_path(record, "mask", index)
        gt = (
            cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
            if self.load_gt
            else None
        )
        decoded = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        roi = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if self.load_gt and gt is None:
            raise FileNotFoundError(f"Unreadable GT frame: {gt_path}")
        if decoded is None:
            raise FileNotFoundError(f"Unreadable input frame: {input_path}")
        if roi is None:
            raise FileNotFoundError(f"Unreadable mask frame: {mask_path}")
        if decoded.shape[:2] != roi.shape[:2] or (
            gt is not None and gt.shape[:2] != decoded.shape[:2]
        ):
            raise ValueError(
                f"Unaligned triplet {record.sequence}/{index}: "
                f"GT={None if gt is None else gt.shape}, "
                f"input={decoded.shape}, mask={roi.shape}"
            )
        original_size = decoded.shape[:2]

        if gt is not None:
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            gt = cv2.resize(
                gt, (self.width, self.height), interpolation=cv2.INTER_LINEAR
            )
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        decoded = cv2.resize(
            decoded, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        roi = cv2.resize(
            roi, (self.width, self.height), interpolation=cv2.INTER_NEAREST
        )

        def rgb_tensor(image):
            array = np.ascontiguousarray(image.transpose(2, 0, 1))
            return torch.from_numpy(array).float().div_(127.5).sub_(1.0)

        roi_array = np.ascontiguousarray((roi >= 128).astype(np.float32)[None])
        gt_tensor = rgb_tensor(gt) if gt is not None else None
        return gt_tensor, rgb_tensor(decoded), torch.from_numpy(roi_array), original_size

    @staticmethod
    def _flow_tensor(array: np.ndarray, path: Path, key: str) -> torch.Tensor:
        flow = np.asarray(array)
        if flow.ndim != 3:
            raise ValueError(f"{path}:{key} must be a 3D flow array")
        if flow.shape[0] == 2:
            pass
        elif flow.shape[-1] == 2:
            flow = flow.transpose(2, 0, 1)
        else:
            raise ValueError(f"{path}:{key} must have two flow channels")
        return torch.from_numpy(np.ascontiguousarray(flow)).float()

    @staticmethod
    def _valid_tensor(array: np.ndarray, path: Path, key: str) -> torch.Tensor:
        valid = np.asarray(array)
        if valid.ndim == 2:
            valid = valid[None]
        elif valid.ndim == 3 and valid.shape[-1] == 1:
            valid = valid.transpose(2, 0, 1)
        if valid.ndim != 3 or valid.shape[0] != 1:
            raise ValueError(f"{path}:{key} must have shape HxW or 1xHxW")
        return torch.from_numpy(np.ascontiguousarray(valid)).float()

    def _load_teacher(self, record: ClipRecord) -> Dict[str, torch.Tensor]:
        forward, backward, valid_forward, valid_backward = [], [], [], []
        has_valid: Optional[bool] = None
        for index0, index1 in zip(
            record.frame_indices[:-1], record.frame_indices[1:]
        ):
            path = self._teacher_path(record, index0, index1)
            if not path.is_file():
                raise FileNotFoundError(f"Missing teacher-flow pair: {path}")
            with np.load(path, allow_pickle=False) as data:
                if "teacher_f" not in data or "teacher_b" not in data:
                    raise KeyError(
                        f"{path} must contain teacher_f and teacher_b"
                    )
                pair_forward = self._flow_tensor(
                    data["teacher_f"], path, "teacher_f"
                )
                pair_backward = self._flow_tensor(
                    data["teacher_b"], path, "teacher_b"
                )
                if pair_forward.shape != pair_backward.shape:
                    raise ValueError(
                        f"{path} teacher_f/teacher_b shapes differ: "
                        f"{tuple(pair_forward.shape)} vs "
                        f"{tuple(pair_backward.shape)}"
                    )
                forward.append(pair_forward)
                backward.append(pair_backward)
                pair_has_valid = "valid_f" in data or "valid_b" in data
                if pair_has_valid and not (
                    "valid_f" in data and "valid_b" in data
                ):
                    raise KeyError(f"{path} must contain both valid_f and valid_b")
                if has_valid is None:
                    has_valid = pair_has_valid
                elif has_valid != pair_has_valid:
                    raise ValueError(
                        "Teacher validity maps must be present for every pair or none"
                    )
                if pair_has_valid:
                    pair_valid_forward = self._valid_tensor(
                        data["valid_f"], path, "valid_f"
                    )
                    pair_valid_backward = self._valid_tensor(
                        data["valid_b"], path, "valid_b"
                    )
                    expected_spatial = pair_forward.shape[-2:]
                    if (
                        pair_valid_forward.shape[-2:] != expected_spatial
                        or pair_valid_backward.shape[-2:] != expected_spatial
                    ):
                        raise ValueError(
                            f"{path} validity maps do not match flow spatial size "
                            f"{tuple(expected_spatial)}"
                        )
                    valid_forward.append(pair_valid_forward)
                    valid_backward.append(pair_valid_backward)

        try:
            result = {
                "teacher_flow_forward": torch.stack(forward),
                "teacher_flow_backward": torch.stack(backward),
            }
            if has_valid:
                result["teacher_valid_forward"] = torch.stack(valid_forward)
                result["teacher_valid_backward"] = torch.stack(valid_backward)
            return result
        except RuntimeError as error:
            raise ValueError(
                f"Teacher flow shapes differ within clip {record.sequence}/"
                f"{record.start}"
            ) from error

    def __getitem__(self, item: int) -> Dict:
        record = self.clips[int(item)]
        gt_frames, decoded_frames, roi_masks, original_sizes = [], [], [], []
        for index in record.frame_indices:
            gt, decoded, roi, original_size = self._load_triplet(record, index)
            if gt is not None:
                gt_frames.append(gt)
            decoded_frames.append(decoded)
            roi_masks.append(roi)
            original_sizes.append(original_size)
        if len(set(original_sizes)) != 1:
            raise ValueError(
                f"Frame sizes change inside {record.sequence}/{record.start}: "
                f"{original_sizes}"
            )

        sample: Dict = {
            "decoded_frames": torch.stack(decoded_frames),
            "roi_masks": torch.stack(roi_masks),
            "sequence": record.sequence,
            "class_name": record.class_name,
            "split": self.split,
            "frame_indices": torch.tensor(record.frame_indices, dtype=torch.long),
            "fps": record.fps,
            "segment_index": record.segment_index,
            "segment_start": record.segment_start,
            "segment_end": record.segment_end,
            "original_size": torch.tensor(original_sizes[0], dtype=torch.long),
        }
        if self.load_gt:
            sample["gt_frames"] = torch.stack(gt_frames)
        if self.teacher_flow_root is not None:
            sample.update(self._load_teacher(record))

        flipped = self._should_flip(int(item))
        if flipped:
            for key in ("gt_frames", "decoded_frames", "roi_masks"):
                if key in sample:
                    sample[key] = torch.flip(sample[key], dims=(-1,))
            for key in ("teacher_flow_forward", "teacher_flow_backward"):
                if key in sample:
                    sample[key] = torch.flip(sample[key], dims=(-1,))
                    sample[key][:, 0].neg_()
            for key in ("teacher_valid_forward", "teacher_valid_backward"):
                if key in sample:
                    sample[key] = torch.flip(sample[key], dims=(-1,))
        sample["flipped"] = flipped
        sample["metadata"] = {
            "sequence": record.sequence,
            "class_name": record.class_name,
            "split": self.split,
            "source_indices": record.frame_indices,
            "segment_index": record.segment_index,
            "segment_range": (record.segment_start, record.segment_end),
            "flipped": flipped,
        }
        return sample
