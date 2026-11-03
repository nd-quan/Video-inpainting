#!/usr/bin/env python3
"""Summarize matched full-validation old V7 versus QP student flow inference."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


QPS = (46, 48, 52, 56)
DIRECTIONS = ("forward", "backward")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bg_epe(row, model):
    return float(np.mean([float(row[f"{direction}_{model}_bg_epe"])
                          for direction in DIRECTIONS]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    data = {}
    for qp in QPS:
        folder = root / f"qp{qp}"
        summary = json.loads((folder / "summary.json").read_text())
        pairs = read_csv(folder / "per_pair_metrics.csv")
        sequences = read_csv(folder / "per_sequence_metrics.csv")
        if (len(pairs) != summary["pair_count"] or len(sequences) != 14
                or summary["sequence_count"] != 14 or summary["include_branches"] is not None):
            raise ValueError(f"QP{qp} does not have complete validation results")
        if summary["max_input_side"] != 960 or summary["spatial_mode"] != "max_side":
            raise ValueError(f"QP{qp} evaluation geometry differs")
        data[qp] = (summary, pairs, sequences)
    pair_ids = [
        {(row["sequence"], row["frame_t"], row["frame_t1"]) for row in data[qp][1]}
        for qp in QPS
    ]
    if any(ids != pair_ids[0] for ids in pair_ids[1:]):
        raise ValueError("The four QPs do not share the same validation pair indices")
    old_paths = {data[qp][0]["old_checkpoint"] for qp in QPS}
    if len(old_paths) != 1:
        raise ValueError("The four comparisons used different old checkpoints")
    names = sorted(row["sequence"] for row in data[QPS[0]][2])
    if any(sorted(row["sequence"] for row in data[qp][2]) != names for qp in QPS):
        raise ValueError("The four QPs do not share the same validation sequences")

    comparisons = []
    per_sequence = []
    for qp in QPS:
        summary, pairs, sequences = data[qp]
        macro = summary["sequence_macro_average"]
        old = bg_epe(macro, "old_4750")
        student = bg_epe(macro, "qp_student")
        seq_improved = 0
        for row in sequences:
            old_seq = bg_epe(row, "old_4750")
            new_seq = bg_epe(row, "qp_student")
            reduction = 100 * (old_seq - new_seq) / max(old_seq, 1e-8)
            seq_improved += old_seq > new_seq
            per_sequence.append({
                "qp": qp, "sequence": row["sequence"],
                "pair_count": int(row["pair_count"]),
                "old_4750_bg_epe": old_seq,
                "qp_student_bg_epe": new_seq,
                "epe_reduction_pct": reduction,
            })
        comparisons.append({
            "qp": qp, "pair_count": len(pairs), "sequence_count": len(sequences),
            "old_checkpoint": summary["old_checkpoint"],
            "qp_checkpoint": summary["qp_checkpoint"],
            "old_forward_bg_epe": float(macro["forward_old_4750_bg_epe"]),
            "student_forward_bg_epe": float(macro["forward_qp_student_bg_epe"]),
            "old_backward_bg_epe": float(macro["backward_old_4750_bg_epe"]),
            "student_backward_bg_epe": float(macro["backward_qp_student_bg_epe"]),
            "old_4750_bg_epe": old,
            "qp_student_bg_epe": student,
            "epe_reduction_pct": 100 * (old - student) / max(old, 1e-8),
            "sequences_improved": seq_improved,
        })
    write_csv(root / "comparison.csv", comparisons)
    write_csv(root / "per_sequence_comparison.csv", per_sequence)

    x = np.arange(len(QPS))
    fig, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    old_values = [row["old_4750_bg_epe"] for row in comparisons]
    new_values = [row["qp_student_bg_epe"] for row in comparisons]
    axis.bar(x - .19, old_values, .38, label="Old student, step 4750", color="#8492a6")
    axis.bar(x + .19, new_values, .38, label="QP student", color="#1568a8")
    for pos, row in zip(x, comparisons):
        axis.text(pos + .19, row["qp_student_bg_epe"] + .02,
                  f"{row['epe_reduction_pct']:+.1f}%", ha="center", fontsize=9)
    axis.set_xticks(x, [f"QP{qp}" for qp in QPS])
    axis.set_ylim(0, max(old_values + new_values) * 1.18)
    axis.set_ylabel("BG EPE to clean RAFT teacher (pixels) ↓")
    axis.set_title("Same validation pairs and max-side-960 flow input")
    axis.grid(axis="y", alpha=.25)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    for extension in ("png", "pdf"):
        fig.savefig(root / f"comparison_bg_epe.{extension}", dpi=180)
    plt.close(fig)

    matrix = np.asarray([
        [next(row["epe_reduction_pct"] for row in per_sequence
              if row["qp"] == qp and row["sequence"] == name) for qp in QPS]
        for name in names
    ])
    bound = max(10, float(np.ceil(np.abs(matrix).max() / 10) * 10))
    fig, axis = plt.subplots(figsize=(8.5, 7.8), layout="constrained")
    image = axis.imshow(matrix, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
    axis.set_xticks(range(len(QPS)), [f"QP{qp}" for qp in QPS])
    axis.set_yticks(range(len(names)), [name.split("/")[-1] for name in names])
    axis.set_title("BG EPE reduction versus old student by sequence")
    for i in range(len(names)):
        for j in range(len(QPS)):
            axis.text(j, i, f"{matrix[i, j]:+.0f}%", ha="center", va="center",
                      fontsize=8, color="white" if abs(matrix[i, j]) > bound * .55 else "black")
    fig.colorbar(image, ax=axis, label="BG EPE reduction (%) ↑")
    for extension in ("png", "pdf"):
        fig.savefig(root / f"per_sequence_reduction.{extension}", dpi=180)
    plt.close(fig)

    note = {
        "comparison": "Old V7 student checkpoint step 4750 versus QP-specific best student checkpoints",
        "old_checkpoint": next(iter(old_paths)),
        "evaluation": "Matched degraded RGB validation pairs, 14 SFU sequences per QP, max side 960",
        "pair_indices_matched_across_qps": True,
        "metric": "Sequence-macro BG endpoint error averaged over forward/backward directions; pixels at max-side-960 geometry",
        "teacher": "Cached clean-input frozen RAFT pseudo-reference; not physical ground truth",
        "interpretation_limit": "Comparison combines resolution adaptation and QP-specific fine-tuning because old student trained on 512x512",
        "results": comparisons,
    }
    (root / "summary.json").write_text(json.dumps(note, indent=2) + "\n")
    lines = [
        "# SFU student flow: old checkpoint versus QP-specific checkpoints", "",
        "Both students receive identical degraded RGB frame pairs at max-side-960 geometry.",
        "BG EPE uses teacher-valid background pixels and a clean-input frozen RAFT teacher as pseudo-reference.",
        "The old checkpoint was trained at 512×512, so this comparison measures direct use at the current geometry and does not isolate QP adaptation from resolution adaptation.", "",
        "| QP | Pairs | Old 4750 BG EPE ↓ | QP student BG EPE ↓ | Reduction | Improved sequences |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['qp']} | {row['pair_count']} | {row['old_4750_bg_epe']:.4f} | "
            f"{row['qp_student_bg_epe']:.4f} | {row['epe_reduction_pct']:+.1f}% | "
            f"{row['sequences_improved']}/14 |"
        )
    lines.extend(["", "See `comparison.csv`, `per_sequence_comparison.csv`, charts, and `qp*/montages/`."])
    (root / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(note, indent=2))


if __name__ == "__main__":
    main()
