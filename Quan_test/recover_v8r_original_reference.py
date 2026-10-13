#!/usr/bin/env python3
"""Recover original-size frames behind saved V8-R student reference PNGs."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SFU = Path("/home/cilab/ndquan/videoInpainting/SFU_STC_flow")
OUTPUT = ROOT / "experiments/eval_v8r_ckpt2500_temporal_teacher_matched_20260929"


def equivalent(source, saved, kind):
    if kind == "masks":
        with Image.open(source) as image:
            actual = np.asarray(image.convert("L").resize((512, 512), Image.Resampling.NEAREST)) >= 128
        with Image.open(saved) as image:
            expected = np.asarray(image.convert("L")) >= 128
    else:
        with Image.open(source) as image:
            actual = np.asarray(image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR))
        with Image.open(saved) as image:
            expected = np.asarray(image.convert("RGB"))
    return np.array_equal(actual, expected)


def candidate_dirs(sequence, split, kind, first):
    label = {"inputs": "input", "gt": "GT", "masks": "mask"}[kind]
    frame = f"{first:06d}.png"
    for parent in (SFU / "long_test" / label, SFU / "train" / label,
                   SFU / "test" / label, SFU / "valid" / label):
        for file in parent.glob(f"*/{sequence}/{frame}"):
            yield file.parent
    yield ROOT / "examples/brushnet/dataset" / split / sequence / kind


def main():
    records = json.loads((OUTPUT / "reference_manifest.json").read_text())["records"]
    recovered = []
    for record in records:
        seq, split = record["sequence"], record["split"]
        first, last = record["frame_range"]
        sources = {}
        for kind in ("inputs", "gt", "masks"):
            saved = OUTPUT / "frozen_reference" / split / seq / kind
            chosen = None
            for candidate in candidate_dirs(seq, split, kind, first):
                if (candidate / f"{first:06d}.png").is_file() and equivalent(
                    candidate / f"{first:06d}.png", saved / f"{first:06d}.png", kind
                ):
                    chosen = candidate
                    break
            if chosen is None:
                raise FileNotFoundError(f"No original source for {split}/{seq}/{kind}")
            destination = OUTPUT / "original_reference" / split / seq / kind
            destination.mkdir(parents=True, exist_ok=True)
            for frame in range(first, last + 1):
                name = f"{frame:06d}.png"
                source = chosen / name
                if not source.is_file() or not equivalent(source, saved / name, kind):
                    raise ValueError(f"Original source differs from student reference: {source}")
                target = destination / name
                if target.is_symlink():
                    if target.resolve() != source.resolve():
                        target.unlink()
                        target.symlink_to(source.resolve())
                elif target.exists():
                    raise FileExistsError(target)
                else:
                    target.symlink_to(source.resolve())
            sources[kind] = str(chosen)
        recovered.append({"sequence": seq, "split": split, "frame_range": [first, last],
                          "source_directories": sources,
                          "dataset_root": str(OUTPUT / "original_reference" / split)})
        print(f"{split}/{seq}: recovered original dimensions for {last-first+1} frames", flush=True)
    (OUTPUT / "original_reference_manifest.json").write_text(
        json.dumps({"records": recovered, "validation": "every resized RGB image equals the student reference byte-for-byte; every ROI mask matches after the loader's threshold"}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
