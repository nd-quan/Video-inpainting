#!/usr/bin/env python3
"""Build the split-aware SFU dataset used by STC full-flow prediction.

The builder assigns approximately 65/15/20 percent of every available video to
train/validation/test. ``BasketballDrill`` is the explicit exception and uses
70/30/0. ``postprocess_remaining/03_frames`` is preferred everywhere. Six
videos whose early frames are absent there use the corresponding ``SFU_train``
prefix for training only; each source boundary remains a separate manifest
segment so clips and teacher-flow pairs never mix the two pipelines. All frame
indices are disjoint across splits.

Output layout::

    <output_root>/<split>/{GT,input,mask}/<class>/<sequence>/<frame>.png

Masks are stored as single-channel binary PNGs with ROI=255 and BG=0.
Input/GT files that already exist as PNGs are hard-linked when possible.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


SplitRanges = Mapping[str, Sequence[Tuple[int, int]]]


@dataclass(frozen=True)
class SFUSource:
    global_start: int
    global_end: int
    source_start: int

    @property
    def source_end(self) -> int:
        return self.source_start + self.global_end - self.global_start


@dataclass(frozen=True)
class SequenceSpec:
    name: str
    class_name: str
    width: int
    height: int
    fps: int
    source_yuv: str
    remaining_dir: str
    descriptor: str
    splits: SplitRanges
    sfu_source: Optional[SFUSource] = None


SEQUENCES: Tuple[SequenceSpec, ...] = (
    SequenceSpec(
        "PeopleOnStreet", "Class_A", 2560, 1600, 30,
        "Class_A/PeopleOnStreet_2560x1600_30.yuv",
        "PeopleOnStreet", "PeopleOnStreet_frames_0_149.txt",
        {"train": ((0, 97),), "valid": ((98, 119),), "test": ((120, 149),)},
    ),
    SequenceSpec(
        "Traffic", "Class_A", 2560, 1600, 30,
        "Class_A/Traffic_2560x1600_30.yuv",
        "Traffic", "Traffic_frames_60_149.txt",
        {
            "train": ((0, 59), (60, 97)),
            "valid": ((98, 119),),
            "test": ((120, 149),),
        },
        SFUSource(1150, 1209, 0),
    ),
    SequenceSpec(
        "BQTerrace", "Class_B", 1920, 1080, 60,
        "Class_B/BQTerrace_1920x1080_60.yuv",
        "BQTerrace", "BQTerrace_frames_0_599.txt",
        {"train": ((0, 389),), "valid": ((390, 479),), "test": ((480, 599),)},
    ),
    SequenceSpec(
        "BasketballDrive", "Class_B", 1920, 1080, 50,
        "Class_B/BasketballDrive_1920x1080_50.yuv",
        "BasketballDrive", "BasketballDrive_frames_0_499.txt",
        {"train": ((0, 324),), "valid": ((325, 399),), "test": ((400, 499),)},
    ),
    SequenceSpec(
        "Cactus", "Class_B", 1920, 1080, 50,
        "Class_B/Cactus_1920x1080_50.yuv",
        "Cactus", "Cactus_frames_0_499.txt",
        {"train": ((0, 324),), "valid": ((325, 399),), "test": ((400, 499),)},
    ),
    SequenceSpec(
        "Kimono", "Class_B", 1920, 1080, 24,
        "Class_B/Kimono1_1920x1080_24.yuv",
        "Kimono", "Kimono_frames_0_239.txt",
        {"train": ((0, 155),), "valid": ((156, 191),), "test": ((192, 239),)},
    ),
    SequenceSpec(
        "ParkScene", "Class_B", 1920, 1080, 24,
        "Class_B/ParkScene_1920x1080_24.yuv",
        "ParkScene", "ParkScene_frames_100_239.txt",
        {
            "train": ((0, 99), (100, 155)),
            "valid": ((156, 191),),
            "test": ((192, 239),),
        },
        SFUSource(810, 909, 0),
    ),
    SequenceSpec(
        "BQMall", "Class_C", 832, 480, 60,
        "Class_C/BQMall_832x480_60.yuv",
        "BQMall", "BQMall_frames_0_599.txt",
        {"train": ((0, 389),), "valid": ((390, 479),), "test": ((480, 599),)},
    ),
    SequenceSpec(
        "BasketballDrill", "Class_C", 832, 480, 50,
        "Class_C/BasketballDrill_832x480_50.yuv",
        "BasketballDrill", "BasketballDrill_frames_0_199.txt",
        {"train": ((0, 139),), "valid": ((140, 199),), "test": ()},
    ),
    SequenceSpec(
        "PartyScene", "Class_C", 832, 480, 50,
        "Class_C/PartyScene_832x480_50.yuv",
        "PartyScene", "PartyScene_frames_150_499.txt",
        {"train": ((150, 377),), "valid": ((378, 429),), "test": ((430, 499),)},
    ),
    SequenceSpec(
        "RaceHorsesC", "Class_C", 832, 480, 30,
        "Class_C/RaceHorses_832x480_30.yuv",
        "RaceHorsesC", "RaceHorsesC_frames_0_299.txt",
        {
            "train": ((0, 194),),
            "valid": ((195, 239),),
            "test": ((240, 299),),
        },
    ),
    SequenceSpec(
        "BQSquare", "Class_D", 416, 240, 60,
        "Class_D/BQSquare_416x240_60.yuv",
        "BQSquare", "BQSquare_frames_180_599.txt",
        {
            "train": ((0, 179), (180, 389)),
            "valid": ((390, 479),),
            "test": ((480, 599),),
        },
        SFUSource(0, 179, 0),
    ),
    SequenceSpec(
        "BasketballPass", "Class_D", 416, 240, 50,
        "Class_D/BasketballPass_416x240_50.yuv",
        "BasketballPass", "BasketballPass_frames_150_499.txt",
        {
            "train": ((0, 149), (150, 324)),
            "valid": ((325, 399),),
            "test": ((400, 499),),
        },
        SFUSource(330, 479, 0),
    ),
    SequenceSpec(
        "BlowingBubbles", "Class_D", 416, 240, 50,
        "Class_D/BlowingBubbles_416x240_50.yuv",
        "BlowingBubbles", "BlowingBubbles_frames_150_499.txt",
        {
            "train": ((0, 149), (150, 324)),
            "valid": ((325, 399),),
            "test": ((400, 499),),
        },
        SFUSource(480, 629, 0),
    ),
    SequenceSpec(
        "RaceHorsesD", "Class_D", 416, 240, 30,
        "Class_D/RaceHorses_416x240_30.yuv",
        "RaceHorsesD", "RaceHorsesD_frames_0_299.txt",
        {"train": ((0, 194),), "valid": ((195, 239),), "test": ((240, 299),)},
    ),
    SequenceSpec(
        "FourPeople", "Class_E", 1280, 720, 60,
        "Class_E/FourPeople_1280x720_60.yuv",
        "FourPeople", "FourPeople_frames_180_599.txt",
        {
            "train": ((0, 179), (180, 389)),
            "valid": ((390, 479),),
            "test": ((480, 599),),
        },
        SFUSource(630, 809, 0),
    ),
    SequenceSpec(
        "Johnny", "Class_E", 1280, 720, 60,
        "Class_E/Johnny_1280x720_60.yuv",
        "Johnny", "Johnny_frames_0_599.txt",
        {"train": ((0, 389),), "valid": ((390, 479),), "test": ((480, 599),)},
    ),
    SequenceSpec(
        "KristenAndSara", "Class_E", 1280, 720, 60,
        "Class_E/KristenAndSara_1280x720_60.yuv",
        "KristenAndSara", "KristenAndSara_frames_0_599.txt",
        {"train": ((0, 389),), "valid": ((390, 479),), "test": ((480, 599),)},
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sfu-test-root", type=Path,
        default=Path("/home/cilab/ndquan/videoInpainting/SFU_train"),
    )
    parser.add_argument(
        "--remaining-root", type=Path,
        default=Path(
            "/home/cilab/ndquan/vcm-rs/output/postprocess_remaining/03_frames"
        ),
    )
    parser.add_argument(
        "--descriptor-root", type=Path,
        default=Path("/home/cilab/ndquan/vcm-rs/output/roi_remaining"),
    )
    parser.add_argument(
        "--clean-yuv-root", type=Path,
        default=Path("/home/cilab/ndquan/vcm-rs/Data/SFU"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/home/cilab/ndquan/videoInpainting/SFU_STC_flow"),
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy existing PNGs instead of using hard links when possible.",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate an already prepared output directory.",
    )
    return parser.parse_args()


def iter_indices(ranges: Sequence[Tuple[int, int]]) -> Iterable[int]:
    for start, end in ranges:
        if end < start:
            raise ValueError(f"Invalid frame range {start}..{end}")
        yield from range(start, end + 1)


def index_to_split(spec: SequenceSpec) -> Dict[int, str]:
    result: Dict[int, str] = {}
    for split in ("train", "valid", "test"):
        for index in iter_indices(spec.splits[split]):
            if index in result:
                raise ValueError(
                    f"{spec.name} frame {index} is shared by {result[index]} and {split}"
                )
            result[index] = split
    return result


def destination(
    output_root: Path,
    split: str,
    kind: str,
    spec: SequenceSpec,
    frame_index: int,
) -> Path:
    return (
        output_root / split / kind / spec.class_name / spec.name
        / f"{frame_index:06d}.png"
    )


def ensure_sequence_directories(output_root: Path, spec: SequenceSpec) -> None:
    """Create every split/modality directory, including intentional empty splits."""
    for split in ("train", "valid", "test"):
        for kind in ("GT", "input", "mask"):
            (
                output_root / split / kind / spec.class_name / spec.name
            ).mkdir(parents=True, exist_ok=True)


def link_or_copy(source: Path, target: Path, force_copy: bool) -> None:
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if not force_copy:
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def load_descriptor(path: Path) -> Dict[int, list]:
    with path.open("r", encoding="utf-8") as descriptor_file:
        value = ast.literal_eval(descriptor_file.read())
    if not isinstance(value, dict):
        raise ValueError(f"Descriptor is not a dictionary: {path}")
    return {int(index): items for index, items in value.items()}


def render_roi_mask(items: list, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for item in items:
        if not isinstance(item, list) or len(item) != 4:
            continue
        x1, y1, x2, y2 = (int(value) for value in item)
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        if x2 >= x1 and y2 >= y1:
            mask[y1:y2 + 1, x1:x2 + 1] = 255
    return mask


def write_mask(path: Path, mask: np.ndarray) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise RuntimeError(f"Failed to write mask: {path}")


def preflight(args: argparse.Namespace) -> None:
    missing: List[Path] = []
    for root in (
        args.sfu_test_root,
        args.remaining_root,
        args.descriptor_root,
        args.clean_yuv_root,
    ):
        if not root.is_dir():
            missing.append(root)

    for spec in SEQUENCES:
        index_to_split(spec)
        video = args.clean_yuv_root / spec.source_yuv
        descriptor = args.descriptor_root / spec.descriptor
        if not video.is_file():
            missing.append(video)
        if not descriptor.is_file():
            missing.append(descriptor)

        remaining_indices = {
            index
            for split in ("train", "valid", "test")
            for index in iter_indices(spec.splits[split])
            if spec.sfu_source is None
            or not (
                spec.sfu_source.source_start
                <= index
                <= spec.sfu_source.source_end
            )
        }
        for index in remaining_indices:
            source = args.remaining_root / spec.remaining_dir / f"{index:06d}.png"
            if not source.is_file():
                missing.append(source)
                break

        if spec.sfu_source is not None:
            for global_index in range(
                spec.sfu_source.global_start, spec.sfu_source.global_end + 1
            ):
                for kind in ("GT", "input", "mask"):
                    source = args.sfu_test_root / kind / f"{global_index:06d}.png"
                    if not source.is_file():
                        missing.append(source)
                        break

    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:30])
        raise FileNotFoundError(f"Missing required sources:\n{preview}")


def build_sfu_source(
    args: argparse.Namespace,
    spec: SequenceSpec,
    logger: logging.Logger,
) -> None:
    source_range = spec.sfu_source
    if source_range is None:
        return
    assignments = index_to_split(spec)
    split_counts = {split: 0 for split in ("train", "valid", "test")}
    for source_index in range(
        source_range.source_start, source_range.source_end + 1
    ):
        if source_index not in assignments:
            raise RuntimeError(
                f"{spec.name} SFU source frame {source_index} is unassigned"
            )
        split = assignments[source_index]
        global_index = (
            source_range.global_start
            + source_index
            - source_range.source_start
        )
        for kind in ("GT", "input"):
            source = args.sfu_test_root / kind / f"{global_index:06d}.png"
            target = destination(
                args.output_root, split, kind, spec, source_index
            )
            link_or_copy(source, target, args.copy)

        source_mask = cv2.imread(
            str(args.sfu_test_root / "mask" / f"{global_index:06d}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if source_mask is None:
            raise RuntimeError(f"Failed to read existing mask {global_index:06d}")
        binary_mask = np.where(source_mask > 127, 255, 0).astype(np.uint8)
        write_mask(
            destination(args.output_root, split, "mask", spec, source_index),
            binary_mask,
        )
        split_counts[split] += 1
    logger.info(
        "%s SFU source: source=%d..%d global=%d..%d assignments=%s",
        spec.name,
        source_range.source_start,
        source_range.source_end,
        source_range.global_start,
        source_range.global_end,
        {key: value for key, value in split_counts.items() if value},
    )


def remaining_assignments(spec: SequenceSpec) -> Dict[int, str]:
    assignments = index_to_split(spec)
    source_range = spec.sfu_source
    if source_range is not None:
        for index in range(
            source_range.source_start, source_range.source_end + 1
        ):
            assignments.pop(index, None)
    return assignments


def build_remaining(
    args: argparse.Namespace,
    spec: SequenceSpec,
    logger: logging.Logger,
) -> None:
    assignments = remaining_assignments(spec)
    descriptor = load_descriptor(args.descriptor_root / spec.descriptor)
    missing_descriptor = sorted(set(assignments) - set(descriptor))
    if missing_descriptor:
        raise KeyError(
            f"{spec.name} descriptor misses frames: {missing_descriptor[:10]}"
        )

    for index, split in assignments.items():
        input_source = args.remaining_root / spec.remaining_dir / f"{index:06d}.png"
        link_or_copy(
            input_source,
            destination(args.output_root, split, "input", spec, index),
            args.copy,
        )
        write_mask(
            destination(args.output_root, split, "mask", spec, index),
            render_roi_mask(descriptor[index], spec.width, spec.height),
        )

    clean_yuv = args.clean_yuv_root / spec.source_yuv
    frame_bytes = spec.width * spec.height * 3 // 2
    selected = set(assignments)
    with clean_yuv.open("rb") as yuv_file:
        for index in sorted(selected):
            target = destination(
                args.output_root, assignments[index], "GT", spec, index
            )
            if target.is_file():
                continue
            yuv_file.seek(index * frame_bytes)
            raw = np.fromfile(yuv_file, dtype=np.uint8, count=frame_bytes)
            if raw.size != frame_bytes:
                raise RuntimeError(
                    f"Clean YUV {spec.name} ended before frame {index}"
                )
            yuv420 = raw.reshape(spec.height * 3 // 2, spec.width)
            frame = cv2.cvtColor(yuv420, cv2.COLOR_YUV2BGR_I420)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), frame):
                raise RuntimeError(f"Failed to write GT: {target}")
    logger.info("%s remaining frames built: %d", spec.name, len(assignments))


def segment_source(spec: SequenceSpec, start: int, end: int) -> str:
    """Resolve one manifest segment to exactly one decoded-input pipeline."""
    source_range = spec.sfu_source
    if source_range is None:
        return "03_frames"
    inside_start = source_range.source_start <= start <= source_range.source_end
    inside_end = source_range.source_start <= end <= source_range.source_end
    if inside_start != inside_end:
        raise ValueError(
            f"{spec.name} segment {start}..{end} crosses the SFU/03 boundary; "
            "split it so clips cannot mix input pipelines"
        )
    return "SFU_train" if inside_start else "03_frames"


def manifest_entry(spec: SequenceSpec) -> dict:
    counts = {
        split: sum(end - start + 1 for start, end in spec.splits[split])
        for split in ("train", "valid", "test")
    }
    test_sources = {
        segment_source(spec, start, end)
        for start, end in spec.splits["test"]
    }
    test_source = (
        "disabled"
        if counts["test"] == 0
        else next(iter(test_sources))
        if len(test_sources) == 1
        else sorted(test_sources)
    )
    entry = {
        "name": spec.name,
        "class": spec.class_name,
        "fps": spec.fps,
        "native_width": spec.width,
        "native_height": spec.height,
        "clean_yuv": spec.source_yuv,
        "mask_semantics": {"background": 0, "roi": 255},
        "splits": {
            split: {
                "segments": [
                    {
                        "start": start,
                        "end": end,
                        "source": segment_source(spec, start, end),
                    }
                    for start, end in spec.splits[split]
                ],
                "frame_count": counts[split],
                "relative_root": (
                    f"{split}/{{kind}}/{spec.class_name}/{spec.name}"
                ),
            }
            for split in ("train", "valid", "test")
        },
        "test_source": test_source,
        "test_duration_seconds": counts["test"] / spec.fps,
    }
    if spec.sfu_source is not None:
        entry["sfu_train_source"] = {
            "global_start": spec.sfu_source.global_start,
            "global_end": spec.sfu_source.global_end,
            "source_start": spec.sfu_source.source_start,
            "source_end": spec.sfu_source.source_end,
        }
    return entry


def validate_dataset(output_root: Path) -> dict:
    report = {
        "output_root": str(output_root),
        "mask_semantics": {"background": 0, "roi": 255},
        "splits": {split: 0 for split in ("train", "valid", "test")},
        "source_frames": {
            split: {"03_frames": 0, "SFU_train": 0}
            for split in ("train", "valid", "test")
        },
        "classes": {},
        "sequences": [],
        "warnings": [],
    }
    for spec in SEQUENCES:
        split_sets: Dict[str, set] = {}
        sequence_report = {
            "name": spec.name,
            "class": spec.class_name,
            "fps": spec.fps,
            "splits": {},
        }
        for split in ("train", "valid", "test"):
            expected = set(iter_indices(spec.splits[split]))
            split_sets[split] = expected
            kind_sets = {}
            for kind in ("GT", "input", "mask"):
                directory = output_root / split / kind / spec.class_name / spec.name
                found = {
                    int(path.stem)
                    for path in directory.glob("*.png")
                    if path.stem.isdigit()
                }
                kind_sets[kind] = found
                if found != expected:
                    missing = sorted(expected - found)[:10]
                    extra = sorted(found - expected)[:10]
                    raise RuntimeError(
                        f"{spec.name}/{split}/{kind} mismatch: "
                        f"missing={missing} extra={extra}"
                    )

            for index in sorted(expected):
                paths = {
                    kind: destination(output_root, split, kind, spec, index)
                    for kind in ("GT", "input", "mask")
                }
                images = {
                    "GT": cv2.imread(str(paths["GT"]), cv2.IMREAD_COLOR),
                    "input": cv2.imread(str(paths["input"]), cv2.IMREAD_COLOR),
                    "mask": cv2.imread(str(paths["mask"]), cv2.IMREAD_GRAYSCALE),
                }
                if any(image is None for image in images.values()):
                    raise RuntimeError(f"Unreadable triplet: {spec.name}/{split}/{index}")
                shapes = {image.shape[:2] for image in images.values()}
                if len(shapes) != 1:
                    raise RuntimeError(
                        f"Shape mismatch {spec.name}/{split}/{index}: "
                        f"{[(kind, image.shape) for kind, image in images.items()]}"
                    )
                mask_values = np.unique(images["mask"])
                if not set(mask_values.tolist()).issubset({0, 255}):
                    raise RuntimeError(
                        f"Non-binary mask {paths['mask']}: {mask_values.tolist()}"
                    )

            count = len(expected)
            report["splits"][split] += count
            split_source_counts = {"03_frames": 0, "SFU_train": 0}
            for start, end in spec.splits[split]:
                source = segment_source(spec, start, end)
                split_source_counts[source] += end - start + 1
                report["source_frames"][split][source] += end - start + 1
            sequence_report["splits"][split] = {
                "frame_count": count,
                "source_frames": split_source_counts,
                "clip_count_t4": sum(
                    max(0, end - start + 1 - 3)
                    for start, end in spec.splits[split]
                ),
                "clip_count_t10_per_segment": sum(
                    max(0, end - start + 1 - 9)
                    for start, end in spec.splits[split]
                ),
            }

        if any(split_sets[left] & split_sets[right] for left, right in (
            ("train", "valid"), ("train", "test"), ("valid", "test")
        )):
            raise RuntimeError(f"Source-frame leakage for {spec.name}")

        class_report = report["classes"].setdefault(
            spec.class_name,
            {"sequence_count": 0, "sequences": []},
        )
        class_report["sequence_count"] += 1
        class_report["sequences"].append(spec.name)
        report["sequences"].append(sequence_report)

    report["total_frames"] = sum(report["splits"].values())
    report["warnings"].append(
        "BasketballDrill intentionally uses 70/30/0: frames 0..139 are train, "
        "140..199 are valid, and test is disabled."
    )
    return report


def configure_logging(output_root: Path) -> logging.Logger:
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("prepare_sfu_stc_dataset")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(log_dir / "prepare.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def main() -> None:
    args = parse_args()
    args.sfu_test_root = args.sfu_test_root.resolve()
    args.remaining_root = args.remaining_root.resolve()
    args.descriptor_root = args.descriptor_root.resolve()
    args.clean_yuv_root = args.clean_yuv_root.resolve()
    args.output_root = args.output_root.resolve()
    logger = configure_logging(args.output_root)

    if not args.validate_only:
        # A failed/interrupted rebuild must never leave an earlier success
        # marker that describes a different physical split.
        for stale_name in ("BUILD_COMPLETE.json", "dataset_report.json"):
            stale_path = args.output_root / stale_name
            if stale_path.is_file():
                stale_path.unlink()
        preflight(args)
        # A previous completion marker must never remain visible while files or
        # split metadata are being mutated. Recreate all generated metadata
        # only after the new physical dataset passes validation.
        for stale_metadata in (
            args.output_root / "BUILD_COMPLETE.json",
            args.output_root / "dataset_report.json",
            args.output_root / "manifest.json",
        ):
            if stale_metadata.is_file():
                stale_metadata.unlink()
        logger.info("Preflight passed for %d sequences", len(SEQUENCES))
        for spec in SEQUENCES:
            ensure_sequence_directories(args.output_root, spec)
            build_sfu_source(args, spec, logger)
            build_remaining(args, spec, logger)

        manifest = {
            "version": 1,
            "description": "SFU VCM dataset for STC full-flow prediction",
            "layout": "<split>/<kind>/<class>/<sequence>/<source_frame>.png",
            "kinds": ["GT", "input", "mask"],
            "splits": ["train", "valid", "test"],
            "mask_semantics": {"background": 0, "roi": 255},
            "sequences": [manifest_entry(spec) for spec in SEQUENCES],
        }
        (args.output_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    report = validate_dataset(args.output_root)
    (args.output_root / "dataset_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_root / "BUILD_COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(args.output_root / "manifest.json"),
                "report": str(args.output_root / "dataset_report.json"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Validation passed: total=%d train=%d valid=%d test=%d",
        report["total_frames"],
        report["splits"]["train"],
        report["splits"]["valid"],
        report["splits"]["test"],
    )


if __name__ == "__main__":
    main()
