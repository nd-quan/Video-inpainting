#!/usr/bin/env python3
"""Summarize matched test_1 coarse RAFT versus V7 student flow evaluation."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    student = json.loads((root / "metrics/summary.json").read_text())
    coarse = json.loads((root / "coarse_metrics/summary.json").read_text())
    for field in ("teacher_cache", "dataset", "split", "flow_resolution",
                  "pair_count", "sequence_count", "validity", "mask"):
        if student[field] != coarse[field]:
            raise ValueError(f"Student/coarse runs differ at {field}")
    if student["pair_count"] != 745 or student["sequence_count"] != 5:
        raise ValueError("Test_1 evaluation is incomplete")
    pair_tables = [load_csv(root / f"{folder}/per_pair_metrics.csv")
                   for folder in ("metrics", "coarse_metrics")]
    pair_ids = [
        {(row["sequence"], row["frame_t"], row["frame_t1"]) for row in table}
        for table in pair_tables
    ]
    if len(pair_tables[0]) != len(pair_tables[1]) or pair_ids[0] != pair_ids[1]:
        raise ValueError("Student/coarse runs used different frame pairs")
    coarse_by_id = {
        (row["sequence"], row["frame_t"], row["frame_t1"]): row
        for row in pair_tables[1]
    }
    for row in pair_tables[0]:
        peer = coarse_by_id[(row["sequence"], row["frame_t"], row["frame_t1"])]
        for direction in ("forward", "backward"):
            for field in ("zero_epe", "bg_zero_epe", "valid_ratio", "valid_bg_ratio",
                          "valid_bg_pixels"):
                key = f"{direction}_{field}"
                if not np.isclose(float(row[key]), float(peer[key]), rtol=0, atol=1e-7):
                    raise ValueError(f"Teacher/valid/BG support mismatch: {row['sequence']} {key}")
    students = {row["sequence"]: row for row in load_csv(root / "metrics/per_sequence_metrics.csv")}
    coarses = {row["sequence"]: row for row in load_csv(root / "coarse_metrics/per_sequence_metrics.csv")}
    if set(students) != set(coarses):
        raise ValueError("Student/coarse sequences differ")

    rows = []
    for name in students:
        s, c = students[name], coarses[name]
        if int(s["pair_count"]) != int(c["pair_count"]):
            raise ValueError(f"Pair count differs for {name}")
        row = {"sequence": name, "pair_count": int(s["pair_count"])}
        for metric in ("epe", "bg_epe", "bg_epe_pixel_weighted"):
            old, new = float(c[metric]), float(s[metric])
            row[f"coarse_{metric}"] = old
            row[f"student_{metric}"] = new
            row[f"{metric}_reduction_pct"] = 100 * (old - new) / max(old, 1e-8)
        row["coarse_bg_zero_gain"] = float(c["bg_zero_gain"])
        row["student_bg_zero_gain"] = float(s["bg_zero_gain"])
        rows.append(row)
    aggregate = {"sequence": "AVERAGE_5_SEQUENCES", "pair_count": sum(r["pair_count"] for r in rows)}
    for metric in ("epe", "bg_epe", "bg_epe_pixel_weighted"):
        old = float(coarse["macro_average"][metric])
        new = float(student["macro_average"][metric])
        aggregate[f"coarse_{metric}"] = old
        aggregate[f"student_{metric}"] = new
        aggregate[f"{metric}_reduction_pct"] = 100 * (old - new) / max(old, 1e-8)
    aggregate["coarse_bg_zero_gain"] = float(coarse["macro_average"]["bg_zero_gain"])
    aggregate["student_bg_zero_gain"] = float(student["macro_average"]["bg_zero_gain"])
    save_csv(root / "comparison.csv", rows + [aggregate])

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    ax.bar(x - .18, [r["coarse_bg_epe"] for r in rows], .36,
           color="#8191a8", label="Coarse pretrained RAFT")
    ax.bar(x + .18, [r["student_bg_epe"] for r in rows], .36,
           color="#1767a8", label="Student step 4750")
    ax.set_xticks(x, [r["sequence"] for r in rows], rotation=15, ha="right")
    ax.set_ylabel("BG EPE to clean teacher (512px) ↓")
    ax.set_title("Matched test_1: same inputs, teacher, masks and 149 pairs per sequence")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    for p, row in zip(x, rows):
        ax.text(p + .18, row["student_bg_epe"] + .018,
                f"{row['bg_epe_reduction_pct']:+.1f}%", ha="center", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(root / f"comparison_bg_epe.{ext}", dpi=180)
    plt.close(fig)
    report = {
        "comparison": "Frozen pretrained ProPainter RAFT versus V7 fine-tuned student at checkpoint-0004750",
        "coarse_checkpoint": coarse["checkpoint"],
        "student_checkpoint": student["checkpoint"],
        "teacher_cache": student["teacher_cache"],
        "dataset": student["dataset"],
        "test": "test_1", "flow_resolution": [512, 512],
        "pair_count": 745, "sequence_count": 5,
        "same_pair_ids_teacher_valid_and_mask_support": True,
        "bg_epe_reduction_macro_pct": aggregate["bg_epe_reduction_pct"],
        "bg_epe_reduction_global_pixel_weighted_pct": 100 * (
            coarse["global_bg_epe_pixel_weighted"] - student["global_bg_epe_pixel_weighted"])
            / max(coarse["global_bg_epe_pixel_weighted"], 1e-8),
        "global_bg_epe_pixel_weighted": {
            "coarse": coarse["global_bg_epe_pixel_weighted"],
            "student": student["global_bg_epe_pixel_weighted"],
        },
        "sequences": rows, "macro_average": aggregate,
    }
    (root / "comparison_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"macro_average": aggregate,
                      "global_bg_epe_pixel_weighted": report["global_bg_epe_pixel_weighted"]},
                     indent=2))


if __name__ == "__main__":
    main()
