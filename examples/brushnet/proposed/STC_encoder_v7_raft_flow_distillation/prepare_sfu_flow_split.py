#!/usr/bin/env python3
"""Expose SFU train frames as disjoint RAFT train/validation segments."""

import argparse
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--valid-fraction", type=float, default=0.05)
    parser.add_argument("--expected-qp", type=int, default=46)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not 0 < args.valid_fraction < 0.5:
        raise ValueError("valid-fraction must be in (0,0.5)")
    if not (source / "summary.json").is_file():
        raise FileNotFoundError(source / "summary.json")
    summary = json.loads((source / "summary.json").read_text())
    if summary["bg_qp"] != args.expected_qp:
        raise ValueError(f"Expected SFU QP{args.expected_qp} source")

    sequences = []
    for branch in sorted((source / "train" / "GT").glob("Class_*/*")):
        if not branch.is_dir():
            continue
        relative = branch.relative_to(source / "train" / "GT")
        files = sorted(branch.glob("*.png"))
        count = len(files)
        if count < 4 or [p.stem for p in files] != [f"{i:06d}" for i in range(count)]:
            raise ValueError(f"Missing or noncontiguous GT frames: {branch}")
        valid_count = max(2, round(count * args.valid_fraction))
        cut = count - valid_count
        for split in ("train", "valid"):
            for kind in ("GT", "input", "mask"):
                source_branch = source / "train" / kind / relative
                if len(list(source_branch.glob("*.png"))) != count:
                    raise ValueError(f"Frame-count mismatch: {source_branch}")
                link = output / split / kind / relative
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink():
                    if link.resolve() != source_branch.resolve():
                        raise ValueError(f"Conflicting link: {link}")
                elif link.exists():
                    raise FileExistsError(link)
                else:
                    link.symlink_to(source_branch.resolve(), target_is_directory=True)
        with Image.open(files[0]) as frame:
            width, height = frame.size
        sequences.append({
            "class": relative.parts[0], "name": relative.parts[1],
            "native_width": width, "native_height": height,
            "splits": {
                "train": {"segments": [{"start": 0, "end": cut - 1}], "frame_count": cut},
                "valid": {"segments": [{"start": cut, "end": count - 1}], "frame_count": valid_count},
            },
        })
    if len(sequences) != summary["sequences"]:
        raise ValueError(f"Expected {summary['sequences']} SFU sequences, found {len(sequences)}")
    manifest = {
        "source_dataset": str(source), "source_split": "train",
        "held_out_test": str(source / "test"),
        "validation_policy": f"last {args.valid_fraction:g} of each source train sequence; no crossing pair",
        "sequences": sequences,
    }
    path = output / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        old_sequences = {(s["class"], s["name"]): s for s in previous["sequences"]}
        new_sequences = {(s["class"], s["name"]): s for s in manifest["sequences"]}
        if (previous.get("source_dataset") != manifest["source_dataset"]
                or any(new_sequences.get(key) != value for key, value in old_sequences.items())):
            raise ValueError(f"Existing manifest differs beyond an added-sequence expansion: {path}")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    pairs = {split: sum(s["splits"][split]["frame_count"] - 1 for s in sequences)
             for split in ("train", "valid")}
    print(json.dumps({"dataset": str(output), "sequences": len(sequences), "pairs": pairs}))


if __name__ == "__main__":
    main()
