#!/usr/bin/env python3
"""Resize completed BG52 512px evaluation PNGs to each GT frame's original size."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from PIL import Image


DEFAULT_EVAL = Path(
    "/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/"
    "eval_bg52_sn2250_latent_tg0p1_v7student_512/test"
)
DEFAULT_DATASET = Path(
    "/home/cilab/ndquan/vcm-rs/output/sfu_qp0_bg_table_window_test/bg52/test"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-name", default="resized_to_original")
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name or args.output_name in {".", ".."}:
        raise ValueError("output-name must be a single directory name")
    root = args.eval_root.resolve()
    dataset = args.dataset_root.resolve()
    output = root / args.output_name
    if not root.is_dir() or not (dataset / "GT").is_dir():
        raise FileNotFoundError("Evaluation frames or BG52 GT root not found")

    cases = []
    for class_dir in sorted(root.glob("Class_*")):
        for sequence_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            gt_dir = dataset / "GT" / class_dir.name / sequence_dir.name
            source = {path.name: path for path in sequence_dir.glob("*.png")}
            reference = {path.name: path for path in gt_dir.glob("*.png")}
            if not source or set(source) != set(reference):
                raise ValueError("Prediction/GT frame mismatch: " + str(sequence_dir))
            cases.append((class_dir.name, sequence_dir.name, source, reference))
    if not cases:
        raise ValueError("No BG52 PNG sequences found under " + str(root))

    def resize_case(case):
        class_name, sequence, source, reference = case
        dest_dir = output / class_name / sequence
        dest_dir.mkdir(parents=True, exist_ok=True)
        native_sizes = set()
        for filename, src in sorted(source.items()):
            dest = dest_dir / filename
            with Image.open(reference[filename]) as gt:
                target_size = gt.size
            native_sizes.add(target_size)
            if dest.is_file():
                with Image.open(dest) as previous:
                    if previous.size == target_size:
                        continue
                raise ValueError("Existing output has wrong size: " + str(dest))
            with Image.open(src) as image:
                resized = image.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
                temporary = dest.with_suffix(".png.tmp")
                resized.save(temporary, format="PNG")
            os.replace(temporary, dest)
        produced = {path.name for path in dest_dir.glob("*.png")}
        if produced != set(source):
            raise RuntimeError("Incomplete resized sequence: " + str(dest_dir))
        print("{} / {}: {} frames".format(class_name, sequence, len(source)), flush=True)
        return {"class": class_name, "sequence": sequence,
                "frames": len(source), "native_sizes_wh": sorted(native_sizes)}

    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(resize_case, cases))

    summary = {
        "source_eval_root": str(root), "dataset_gt_root": str(dataset / "GT"),
        "output_root": str(output), "resampling": "Pillow Lanczos",
        "note": "Upscaling restores original dimensions, not lost native-resolution detail.",
        "sequence_count": len(records), "frame_count": sum(item["frames"] for item in records),
        "sequences": records,
    }
    (output / "resize_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("Completed {} sequences / {} frames: {}".format(
        summary["sequence_count"], summary["frame_count"], output), flush=True)


if __name__ == "__main__":
    main()
