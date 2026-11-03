#!/usr/bin/env python3
"""Compare old student/512 teacher to QP students/960 teacher on overlapping SFU IDs."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


BRUSHNET = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BRUSHNET / "examples/brushnet"))
import visualize_raft_student_flow as vis  # noqa: E402


QPS = (46, 48, 52, 56)
OLD_DATASET = BRUSHNET.parent.parent / "SFU_STC_flow"
OLD_TEACHER = OLD_DATASET / "teacher_flows_512x512"
OLD_CHECKPOINT = BRUSHNET / "experiments/train_v7_raft_student_flow/checkpoint-0004750"


def load_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair_id(row):
    return row["sequence"], int(row["frame_t"]), int(row["frame_t1"])


def mean(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched_960", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = args.output_root.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    old_meta = json.loads((OLD_TEACHER / "metadata.json").read_text())
    if (old_meta["height"], old_meta["width"]) != (512, 512):
        raise ValueError("Old teacher must be the original 512x512 cache")
    newer = {}
    for qp in QPS:
        folder = args.matched_960.expanduser().resolve() / f"qp{qp}"
        summary = json.loads((folder / "summary.json").read_text())
        cache_meta = json.loads((Path(summary["teacher_cache"]) / "metadata.json").read_text())
        if (cache_meta["raft_sha256"] != old_meta["raft_sha256"]
                or cache_meta["spatial_mode"] != "max_side"
                or cache_meta["max_input_side"] != 960):
            raise ValueError(f"QP{qp} uses a different teacher model or geometry")
        records = load_csv(folder / "per_pair_metrics.csv")
        if len(records) != summary["pair_count"] or summary["pair_count"] != 228:
            raise ValueError(f"QP{qp} results are incomplete")
        newer[qp] = (summary, {pair_id(row): row for row in records})
    old_data = vis.RAFTTeacherFlowPairDataset(
        OLD_DATASET, OLD_TEACHER, "valid", 512, spatial_mode="square",
    )
    old_indices = {
        (branch, frame0, frame1): index
        for index, (branch, frame0, frame1, _) in enumerate(old_data.entries)
    }
    intersections = [set(values) & set(old_indices) for _, values in newer.values()]
    if any(ids != intersections[0] for ids in intersections[1:]):
        raise ValueError("QP comparisons have different old-cache pair overlap")
    selected = intersections[0]
    if len(selected) != 209 or len({key[0] for key in selected}) != 13:
        raise ValueError("Expected 209 matching pair indices from 13 sequences")
    old_root, old_component = vis.resolve_checkpoint(OLD_CHECKPOINT)
    if any(summary["old_checkpoint"] != str(old_root) for summary, _ in newer.values()):
        raise ValueError("The previous max-side evaluations used a different old model")
    model = vis.RAFTStudentFlowPredictor.from_pretrained(
        old_component, mixed_precision=False,
    ).to(args.device).eval().requires_grad_(False)
    old_rows = []
    for number, key in enumerate(sorted(selected), 1):
        sample = old_data[old_indices[key]]
        a = sample["degraded0"][None].to(args.device)
        b = sample["degraded1"][None].to(args.device)
        with torch.inference_mode():
            flows = [flow[0].float() for flow in
                     model.predict_bidirectional(a, b, pair_batch_size=1)]
        row = {"sequence": key[0], "frame_t": key[1], "frame_t1": key[2],
               "height": 512, "width": 512}
        for i, direction in enumerate(("forward", "backward")):
            stats = vis.direction_metrics(
                flows[i], sample[f"teacher_{direction}"].to(args.device),
                sample[f"valid_{direction}"].to(args.device),
                sample["bg0" if i == 0 else "bg1"].to(args.device),
            )
            for metric in ("bg_epe", "bg_zero_epe", "bg_zero_gain", "valid_bg_ratio"):
                row[f"{direction}_{metric}"] = stats[metric]
        old_rows.append(row)
        print(f"Old 512 [{number}/{len(selected)}] {key[0]} {key[1]}->{key[2]}", flush=True)
    save_csv(output / "old_512_per_pair_metrics.csv", old_rows)

    results = []
    per_sequence = []
    for qp in QPS:
        summary, qp_rows = newer[qp]
        groups = defaultdict(list)
        for old in old_rows:
            key = pair_id(old)
            new = qp_rows[key]
            diagonal = math.hypot(int(new["height"]), int(new["width"]))
            combined = {"qp": qp, "sequence": key[0], "frame_t": key[1],
                        "frame_t1": key[2]}
            for direction in ("forward", "backward"):
                for label, value in (("old_512", old), ("new_960", new)):
                    prefix = direction if label == "old_512" else f"{direction}_qp_student"
                    for metric in ("bg_epe", "bg_zero_epe", "bg_zero_gain", "valid_bg_ratio"):
                        combined[f"{direction}_{label}_{metric}"] = float(value[f"{prefix}_{metric}"])
                    geometry_diagonal = math.hypot(512, 512) if label == "old_512" else diagonal
                    combined[f"{direction}_{label}_bg_epe_pct_diagonal"] = (
                        100 * combined[f"{direction}_{label}_bg_epe"] / geometry_diagonal)
            groups[key[0]].append(combined)
        sequence_rows = []
        for name, group in sorted(groups.items()):
            metrics = [key for key in group[0] if key not in
                       {"qp", "sequence", "frame_t", "frame_t1"}]
            row = {"qp": qp, "sequence": name, "pair_count": len(group),
                   **{key: mean(group, key) for key in metrics}}
            per_sequence.append(row)
            sequence_rows.append(row)
        row = {"qp": qp, "pair_count": sum(item["pair_count"] for item in sequence_rows),
               "sequence_count": len(sequence_rows),
               "new_checkpoint": summary["qp_checkpoint"]}
        for label in ("old_512", "new_960"):
            for metric in ("bg_epe", "bg_zero_epe", "bg_zero_gain", "bg_epe_pct_diagonal"):
                row[f"{label}_{metric}"] = float(np.mean([
                    mean(sequence_rows, f"{direction}_{label}_{metric}")
                    for direction in ("forward", "backward")
                ]))
        row["relative_diagonal_epe_change_pct"] = (
            100 * (row["new_960_bg_epe_pct_diagonal"] - row["old_512_bg_epe_pct_diagonal"])
            / max(row["old_512_bg_epe_pct_diagonal"], 1e-8))
        results.append(row)
    save_csv(output / "comparison.csv", results)
    save_csv(output / "per_sequence_comparison.csv", per_sequence)

    fig, axis = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
    x = np.arange(len(QPS))
    axis.bar(x - .19, [row["old_512_bg_epe_pct_diagonal"] for row in results],
             .38, label="Old student + old 512 teacher", color="#8392a8")
    axis.bar(x + .19, [row["new_960_bg_epe_pct_diagonal"] for row in results],
             .38, label="QP student + QP 960 teacher", color="#1864aa")
    axis.set_xticks(x, [f"QP{qp}" for qp in QPS])
    axis.set_ylabel("BG EPE / input diagonal (%) ↓")
    axis.set_title("Own-teacher errors; different inputs and teacher targets")
    axis.grid(axis="y", alpha=.25)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"own_teacher_normalized_epe.{extension}", dpi=180)
    plt.close(fig)

    note = {
        "old_checkpoint": str(old_root), "old_dataset": str(OLD_DATASET.resolve()),
        "old_teacher": str(OLD_TEACHER.resolve()),
        "old_geometry": "512x512 square",
        "new_geometry": "pad to 64 then preserve aspect ratio with longest flow side at most 960",
        "same_teacher_weights_sha256": old_meta["raft_sha256"],
        "same_pair_indices": True,
        "pair_count": len(selected), "sequence_count": len({key[0] for key in selected}),
        "excluded": "Class_C/BasketballDrill: its new validation frame indices are absent from old validation/cache",
        "data_difference": "Old and QP datasets have non-identical GT/input PNG pixels even at the same frame index (Traffic frame 114 GT MAE 1.107/255); masks matched in checked examples",
        "interpretation": "Descriptive own-protocol errors; EPE pixels differ by geometry. EPE divided by input diagonal is comparable as a relative image-size measure but teachers and inputs still differ. Not an isolated checkpoint improvement test.",
        "new_metrics_source": str(args.matched_960.expanduser().resolve()),
        "results": results,
    }
    (output / "summary.json").write_text(json.dumps(note, indent=2) + "\n")
    print(json.dumps(note, indent=2), flush=True)


if __name__ == "__main__":
    main()
