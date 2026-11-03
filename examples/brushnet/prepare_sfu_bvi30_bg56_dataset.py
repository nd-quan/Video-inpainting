"""Build a reproducible SFU + per-class BVI sequence subset using file symlinks."""

import argparse
import csv
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


BRUSH = Path(__file__).resolve().parent
KINDS = ("GT", "input", "mask")


def index_source(root):
    trees = {
        kind: {p.relative_to(root / "train" / kind)
               for p in (root / "train" / kind).rglob("*.png") if p.is_file()}
        for kind in KINDS
    }
    if not trees["GT"] or any(trees[kind] != trees["GT"] for kind in KINDS):
        raise ValueError(f"Empty or misaligned GT/input/mask trees: {root}")
    branches = {}
    for relative in sorted(trees["GT"]):
        if len(relative.parts) != 3 or not relative.stem.isdigit():
            raise ValueError(f"Expected Class/sequence/numeric.png: {relative}")
        branches.setdefault(relative.parent, []).append(relative)
    for branch, frames in branches.items():
        frames.sort(key=lambda p: int(p.stem))
        ids = [int(p.stem) for p in frames]
        if len(ids) < 8 or ids != list(range(ids[0], ids[-1] + 1)):
            raise ValueError(f"Sequence is too short or not contiguous: {branch}")
    return branches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfu-root", type=Path,
                        default=BRUSH / "dataset/sfu_train_test_80_20/bg56")
    parser.add_argument("--bvi-root", type=Path,
                        default=Path("/home/cilab/ndquan/vcm-rs/output/"
                                     "bvi_dvc_part1_qp0_bg_train/bg56"))
    parser.add_argument("--output-root", type=Path,
                        default=BRUSH / "dataset/sfu_bvi30_bg56")
    parser.add_argument("--bvi-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if not 0 < args.bvi_fraction <= 1:
        parser.error("--bvi-fraction must be in (0, 1]")
    args.sfu_root = args.sfu_root.resolve()
    args.bvi_root = args.bvi_root.resolve()
    destination = args.output_root.absolute()
    staging = destination.with_name(destination.name + ".building")
    if destination.exists() or destination.is_symlink() or staging.exists():
        raise FileExistsError(f"Refusing to replace an existing dataset: {destination}")

    sfu = index_source(args.sfu_root)
    bvi = index_source(args.bvi_root)
    rng = random.Random(args.seed)
    selected_bvi = []
    selection_by_class = {}
    for class_name in sorted({branch.parts[0] for branch in bvi}):
        candidates = sorted(branch for branch in bvi if branch.parts[0] == class_name)
        count = max(1, math.floor(len(candidates) * args.bvi_fraction + 0.5))
        selected = sorted(rng.sample(candidates, count))
        selected_bvi.extend(selected)
        selection_by_class[class_name] = {
            "available_sequences": len(candidates),
            "selected_sequences": count,
            "selected_branches": [p.as_posix() for p in selected],
        }

    sources = [("SFU", args.sfu_root, sfu, sorted(sfu)),
               ("BVI", args.bvi_root, bvi, selected_bvi)]
    staging.mkdir(parents=True)
    records = []
    counts = {name: Counter() for name, *_ in sources}
    pairs_path = staging / "train_pairs.csv"
    with pairs_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["input", "gt", "mask", "class", "sequence", "bg_qp",
                         "source_frame", "source_dataset"])
        for source_name, source_root, branches, selected in sources:
            for branch in selected:
                class_name, sequence_name = branch.parts
                target_name = ("BVI_" + sequence_name
                               if source_name == "BVI" else sequence_name)
                target_branch = Path(class_name) / target_name
                frames = branches[branch]
                for kind in KINDS:
                    (staging / "train" / kind / target_branch).mkdir(parents=True)
                for relative in frames:
                    for kind in KINDS:
                        source_file = source_root / "train" / kind / relative
                        target = staging / "train" / kind / target_branch / relative.name
                        target.symlink_to(source_file)
                    writer.writerow([
                        (Path("train") / kind / target_branch / relative.name).as_posix()
                        for kind in ("input", "GT", "mask")
                    ] + [class_name, target_name, 56, int(relative.stem), source_name])
                clip_count = (len(frames) - 8) // 4 + 1
                counts[source_name].update(sequences=1, frames=len(frames),
                                           clips_T8S4=clip_count)
                records.append({
                    "source_dataset": source_name,
                    "source_root": str(source_root),
                    "source_branch": branch.as_posix(),
                    "target_branch": target_branch.as_posix(),
                    "frames": len(frames),
                    "first_frame": int(frames[0].stem),
                    "last_frame": int(frames[-1].stem),
                    "clips_T8S4": clip_count,
                })

    totals = sum(counts.values(), Counter())
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "bvi_fraction_per_class": args.bvi_fraction,
        "selection_unit": "whole_sequence",
        "selection_rounding": "nearest_integer_half_up_minimum_one",
        "storage": "individual_PNG_symlinks",
        "split": "train_only",
        "source_mask_semantics": "255=ROI, 0=BG",
        "source_counts": {name: dict(count) for name, count in counts.items()},
        "totals": dict(totals),
        "png_symlinks": totals["frames"] * 3,
        "bvi_clip_fraction": counts["BVI"]["clips_T8S4"] / totals["clips_T8S4"],
        "bvi_selection_by_class": selection_by_class,
    }
    for filename, payload in [("summary.json", summary),
                              ("manifest.json", {**summary, "sequences": records})]:
        (staging / filename).write_text(json.dumps(payload, indent=2) + "\n")
    (staging / "README.md").write_text(
        "# SFU + BVI 30% BG56 training set\n\n"
        "All SFU training sequences plus a seeded per-class BVI sequence subset.\n"
        "Each GT/input/mask PNG is a symlink to the source dataset; sources must "
        "remain available. No test split is included.\n\n"
        "manifest.json records source branches and the selected sequences. "
        "The trainer scans train/{GT,input,mask} directly; train_pairs.csv is "
        "an audit list. Source pixels retain their native dimensions; the "
        "proposed training command resizes RGB to 512x512 with bilinear "
        "interpolation and masks with nearest interpolation.\n"
    )
    staging.rename(destination)
    print(json.dumps({"dataset_root": str(destination), **summary}, indent=2))


if __name__ == "__main__":
    main()
