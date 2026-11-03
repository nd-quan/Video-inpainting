#!/usr/bin/env python3
"""Paste native input ROI onto the upscaled BG52 evaluation, without touching the source."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import cv2
import numpy as np


DEFAULT_TEST = Path(
    "/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/"
    "eval_bg52_sn2250_latent_tg0p1_v7student_512/test"
)
DEFAULT_DATASET = Path(
    "/home/cilab/ndquan/vcm-rs/output/sfu_qp0_bg_table_window_test/bg52/test"
)


def read_color(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-name", default="resized_to_original")
    parser.add_argument("--output-name", default="resized_to_original_native_roi")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    for name in (args.source_name, args.output_name):
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Source/output names must be single directory names")
    if args.source_name == args.output_name or args.workers < 1:
        raise ValueError("Source and output must differ, and workers must be positive")

    test = args.test_root.resolve()
    dataset = args.dataset_root.resolve()
    source_root = test / args.source_name
    output_root = test / args.output_name
    if not source_root.is_dir() or not (dataset / "input").is_dir() or not (dataset / "mask").is_dir():
        raise FileNotFoundError("Upscaled output or native input/mask root not found")
    cv2.setNumThreads(1)

    cases = []
    for class_dir in sorted(source_root.glob("Class_*")):
        for sequence_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            class_name, sequence = class_dir.name, sequence_dir.name
            sources = {path.name: path for path in sequence_dir.glob("*.png")}
            input_dir = dataset / "input" / class_name / sequence
            mask_dir = dataset / "mask" / class_name / sequence
            inputs = {path.name: path for path in input_dir.glob("*.png")}
            masks = {path.name: path for path in mask_dir.glob("*.png")}
            if not sources or set(sources) != set(inputs) or set(sources) != set(masks):
                raise ValueError("Upscaled/input/mask frame mismatch: " + str(sequence_dir))
            cases.append((class_name, sequence, sources, inputs, masks))
    if not cases:
        raise ValueError("No PNG sequences found under " + str(source_root))

    def composite_case(case):
        class_name, sequence, sources, inputs, masks = case
        dest_dir = output_root / class_name / sequence
        dest_dir.mkdir(parents=True, exist_ok=True)
        roi_pixels = total_pixels = 0
        for filename, background_path in sorted(sources.items()):
            background = read_color(background_path)
            native_input = read_color(inputs[filename])
            native_mask = read_color(masks[filename])
            if background.shape != native_input.shape or native_mask.shape != native_input.shape:
                raise ValueError("Native dimensions differ: " + str(background_path))
            # Match temporal_v0_native_roi.py: the ROI is sum(BGR channels) > 255.
            roi = native_mask.astype(np.uint16).sum(axis=2) > 255
            result = background.copy()
            result[roi] = native_input[roi]
            dest = dest_dir / filename
            if not dest.exists():
                temporary = dest_dir / (dest.stem + ".tmp.png")
                if not cv2.imwrite(str(temporary), result):
                    raise OSError("Failed to write " + str(temporary))
                os.replace(temporary, dest)
            saved = read_color(dest)
            if saved.shape != result.shape or not np.array_equal(saved, result):
                raise ValueError("Output differs from native-ROI composite: " + str(dest))
            roi_pixels += int(roi.sum())
            total_pixels += roi.size
        produced = {path.name for path in dest_dir.glob("*.png")}
        if produced != set(sources):
            raise RuntimeError("Incomplete sequence: " + str(dest_dir))
        print("{} / {}: {} frames".format(class_name, sequence, len(sources)), flush=True)
        return {"class": class_name, "sequence": sequence, "frames": len(sources),
                "roi_pixels": roi_pixels, "total_pixels": total_pixels}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(composite_case, cases))
    summary = {
        "source_root": str(source_root), "output_root": str(output_root),
        "native_input_root": str(dataset / "input"),
        "native_mask_root": str(dataset / "mask"),
        "roi_rule": "sum(BGR mask channels) > 255",
        "composite": "ROI=input native pixels; outside ROI=upsampled evaluation pixels",
        "feather_px": 0,
        "sequence_count": len(records), "frame_count": sum(row["frames"] for row in records),
        "sequences": records,
    }
    (output_root / "roi_composite_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("Completed {} sequences / {} frames: {}".format(
        summary["sequence_count"], summary["frame_count"], output_root), flush=True)


if __name__ == "__main__":
    main()
