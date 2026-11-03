#!/usr/bin/env python3
"""Reuse clean SFU RAFT labels across QPs after verifying exact GT/mask identity."""

import argparse
import filecmp
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_dataset = args.source_dataset.resolve()
    source_cache = args.source_cache.resolve()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    source_manifest = json.loads((source_dataset / "manifest.json").read_text())
    manifest = json.loads((dataset / "manifest.json").read_text())
    source_sequences = {(s["class"], s["name"]): s for s in source_manifest["sequences"]}
    source_metadata = json.loads((source_cache / "metadata.json").read_text())
    if source_metadata.get("dataset_root") != str(source_dataset):
        raise ValueError("Source teacher cache belongs to a different dataset")
    if source_metadata.get("spatial_mode") != "max_side" or source_metadata.get("max_input_side") != 960:
        raise ValueError("Source teacher cache must use max_side=960")

    links = []
    checked_frames = 0
    for sequence in manifest["sequences"]:
        key = (sequence["class"], sequence["name"])
        original = source_sequences.get(key)
        if original is None or original["splits"] != sequence["splits"]:
            raise ValueError(f"Teacher source split differs for {key}")
        branch = Path(*key)
        for split in ("train", "valid"):
            for segment in sequence["splits"][split]["segments"]:
                for index in range(segment["start"], segment["end"] + 1):
                    for kind in ("GT", "mask"):
                        relative = Path(split) / kind / branch / f"{index:06d}.png"
                        if not filecmp.cmp(source_dataset / relative, dataset / relative, shallow=False):
                            raise ValueError(f"Clean frame/mask differs: {relative}")
                    checked_frames += 1
                for index in range(segment["start"], segment["end"]):
                    relative = Path(split) / branch / f"{index:06d}_{index + 1:06d}.npz"
                    source = source_cache / relative
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    links.append((source, output / relative))

    output.mkdir(parents=True, exist_ok=True)
    for source, target in links:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            left, right = source.stat(), target.stat()
            if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
                raise FileExistsError(f"Different cache file already exists: {target}")
        else:
            os.link(source, target)
    metadata = dict(source_metadata)
    metadata.update({
        "dataset_root": str(dataset),
        "manifest": str(dataset / "manifest.json"),
        "reused_from": str(source_cache),
        "reuse_verification": {
            "identical_gt_and_mask_frames": checked_frames,
            "linked_teacher_pairs": len(links),
            "identical_split_segments": True,
        },
    })
    path = output / "metadata.json"
    if path.exists():
        previous = json.loads(path.read_text())
        old_check = previous.pop("reuse_verification", None)
        new_check = metadata["reuse_verification"]
        expected = dict(metadata)
        expected.pop("reuse_verification")
        if (previous != expected or old_check is None
                or old_check.get("identical_split_segments") is not True
                or old_check.get("identical_gt_and_mask_frames", -1)
                > new_check["identical_gt_and_mask_frames"]
                or old_check.get("linked_teacher_pairs", -1)
                > new_check["linked_teacher_pairs"]):
            raise ValueError(f"Existing metadata differs beyond verified expansion: {path}")
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"dataset": str(dataset), "checked_frames": checked_frames,
                      "teacher_pairs": len(links), "output": str(output)}))


if __name__ == "__main__":
    main()
