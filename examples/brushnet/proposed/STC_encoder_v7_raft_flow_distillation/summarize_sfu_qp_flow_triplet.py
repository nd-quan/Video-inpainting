#!/usr/bin/env python3
"""Summarize matched SFU coarse versus student flow evaluations by QP."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2


QPS = (46, 48, 52, 56)
DIRECTIONS = ("forward", "backward")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    data = {}
    for qp in QPS:
        folder = root / f"qp{qp}"
        summary = json.loads((folder / "summary.json").read_text())
        if summary["evaluation_scope"] != "all valid pairs" or summary["sequence_count"] != 14:
            raise ValueError(f"QP{qp} lacks all 14 full-validation sequences")
        pairs = read_csv(folder / "per_pair_metrics.csv")
        sequences = read_csv(folder / "per_sequence_metrics.csv")
        if len(pairs) != summary["pair_count"] or len(sequences) != 14:
            raise ValueError(f"QP{qp} CSV/summary count mismatch")
        data[qp] = (summary, pairs, sequences)
    pair_ids = [
        {(row["sequence"], row["frame_t"], row["frame_t1"]) for row in data[qp][1]}
        for qp in QPS
    ]
    if any(ids != pair_ids[0] for ids in pair_ids[1:]):
        raise ValueError("The QPs do not evaluate identical validation pairs")
    names = sorted(row["sequence"] for row in data[QPS[0]][2])
    if any(sorted(row["sequence"] for row in data[qp][2]) != names for qp in QPS):
        raise ValueError("The QPs do not contain identical sequences")

    results = []
    per_sequence = []
    for qp in QPS:
        summary, pairs, sequences = data[qp]
        values = summary["sequence_macro_average"]
        coarse = float(np.mean([float(values[f"{d}_coarse_bg_epe"]) for d in DIRECTIONS]))
        student = float(np.mean([float(values[f"{d}_student_bg_epe"]) for d in DIRECTIONS]))
        reductions = []
        for row in sequences:
            c = float(np.mean([float(row[f"{d}_coarse_bg_epe"]) for d in DIRECTIONS]))
            s = float(np.mean([float(row[f"{d}_student_bg_epe"]) for d in DIRECTIONS]))
            reduction = float(100 * (c - s) / max(c, 1e-8))
            reductions.append(reduction)
            per_sequence.append({"qp": qp, "sequence": row["sequence"],
                                 "pair_count": int(row["pair_count"]),
                                 "coarse_bg_epe": c, "student_bg_epe": s,
                                 "reduction_pct": reduction})
        results.append({
            "qp": qp, "pair_count": len(pairs), "sequence_count": len(sequences),
            "best_checkpoint": summary["checkpoint"],
            "coarse_forward_bg_epe": float(values["forward_coarse_bg_epe"]),
            "student_forward_bg_epe": float(values["forward_student_bg_epe"]),
            "coarse_backward_bg_epe": float(values["backward_coarse_bg_epe"]),
            "student_backward_bg_epe": float(values["backward_student_bg_epe"]),
            "coarse_bg_epe": coarse, "student_bg_epe": student,
            "macro_epe_reduction_pct": 100 * (coarse - student) / max(coarse, 1e-8),
            "sequences_improved": int(sum(value > 0 for value in reductions)),
        })
    write_csv(root / "comparison.csv", list(results[0]), results)
    write_csv(root / "per_sequence_comparison.csv", list(per_sequence[0]), per_sequence)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, layout="constrained")
    x = np.arange(len(QPS))
    colors = ("#767f8c", "#2066aa")
    for axis, direction in zip(axes, DIRECTIONS):
        coarse = [row[f"coarse_{direction}_bg_epe"] for row in results]
        student = [row[f"student_{direction}_bg_epe"] for row in results]
        axis.bar(x - 0.19, coarse, width=0.38, color=colors[0], label="Coarse RAFT")
        axis.bar(x + 0.19, student, width=0.38, color=colors[1], label="QP student")
        axis.axhline(0, color="black", linewidth=1, label="Clean teacher reference")
        axis.set_xticks(x, [f"QP{qp}" for qp in QPS])
        axis.set_title(direction.capitalize())
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        for xpos, c, s in zip(x, coarse, student):
            axis.text(xpos + 0.19, s + max(coarse) * 0.025, f"{s:.2f}",
                      ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("BG endpoint error to clean teacher (pixels) ↓")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("SFU validation flow: coarse vs QP student")
    for extension in ("png", "pdf"):
        fig.savefig(root / f"comparison_bg_epe.{extension}", dpi=180)
    plt.close(fig)

    matrix = np.array([[next(row["reduction_pct"] for row in per_sequence
                             if row["qp"] == qp and row["sequence"] == name)
                        for qp in QPS] for name in names])
    fig, axis = plt.subplots(figsize=(8.2, 7.6), layout="constrained")
    bound = max(50, min(100, float(np.ceil(np.nanmax(np.abs(matrix)) / 10) * 10)))
    image = axis.imshow(matrix, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
    axis.set_xticks(range(len(QPS)), [f"QP{qp}" for qp in QPS])
    axis.set_yticks(range(14), [name.split("/")[-1] for name in names])
    axis.set_title("BG EPE reduction of student versus coarse by sequence")
    for i in range(14):
        for j in range(len(QPS)):
            axis.text(j, i, f"{matrix[i, j]:+.0f}%", ha="center", va="center",
                      fontsize=8, color="white" if abs(matrix[i, j]) > bound * .55 else "black")
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("EPE reduction (%)  ↑ better")
    for extension in ("png", "pdf"):
        fig.savefig(root / f"per_sequence_reduction.{extension}", dpi=180)
    plt.close(fig)

    contact_dir = root / "cross_qp_montages"
    contact_dir.mkdir(exist_ok=True)
    first_pairs = {}
    for item in data[QPS[0]][0]["selected_pairs"]:
        first_pairs.setdefault(item["sequence"], item)
    for name, item in first_pairs.items():
        filename = (name.replace("/", "_")
                    + f"_f{item['frame_t']:06d}_{item['frame_t1']:06d}.png")
        sections = []
        for qp in QPS:
            path = root / f"qp{qp}/montages" / filename
            picture = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if picture is None or picture.shape != (1280, 1280, 3):
                raise ValueError(f"Missing or malformed matched montage: {path}")
            header = np.full((42, 1280, 3), (245, 245, 245), dtype=np.uint8)
            cv2.putText(header, f"QP{qp} | {name} | frames {item['frame_t']}-{item['frame_t1']}",
                        (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2,
                        cv2.LINE_AA)
            # Input row and the two flow rows: coarse, student, teacher, valid BG.
            sections.extend((header, picture[:960]))
        if not cv2.imwrite(str(contact_dir / filename), np.concatenate(sections, axis=0)):
            raise OSError(contact_dir / filename)

    note = {
        "evaluation": "All adjacent validation pairs on the same 14 SFU sequences for QP46/48/52/56",
        "pair_ids_matched": True,
        "flow_types": {
            "coarse": "Frozen pretrained ProPainter RAFT, degraded RGB input",
            "student": "Best fine-tuned RAFT checkpoint for each QP, same degraded RGB input",
            "teacher": "Cached frozen RAFT from clean RGB; pseudo ground truth, not physical GT",
        },
        "geometry": "Pad source to 64, cap longest RAFT side at 960, same as student training",
        "metric": "Sequence-macro mean BG endpoint error in pixels, teacher-valid BG support",
        "montages": "Two matched pairs per sequence and QP; 28 per QP; 14 cross-QP contact sheets",
        "results": results,
    }
    (root / "summary.json").write_text(json.dumps(note, indent=2) + "\n")
    lines = ["# SFU QP flow comparison", "",
             "Coarse = frozen pretrained RAFT on degraded RGB; student = QP-specific fine-tuned RAFT on the same RGB; teacher = cached clean-RAFT flow.",
             "Teacher is a pseudo-reference, not physical ground-truth optical flow.",
             "All valid adjacent pairs from the same 14 sequences are evaluated at the same max-side-960 geometry.",
             "Each QP has 28 matched visualization montages (two pairs per sequence).", "",
             "| QP | Pairs | Coarse BG EPE | Student BG EPE | Reduction | Sequences improved |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['qp']} | {row['pair_count']} | {row['coarse_bg_epe']:.4f} | "
                     f"{row['student_bg_epe']:.4f} | {row['macro_epe_reduction_pct']:.1f}% | "
                     f"{row['sequences_improved']}/14 |")
    lines += ["", "Files: `comparison_bg_epe.pdf`, `per_sequence_reduction.pdf`, "
              "`comparison.csv`, `per_sequence_comparison.csv`, `qp*/montages/*.png`, "
              "and `cross_qp_montages/*.png`."]
    (root / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"pair_count": len(pair_ids[0]), "results": results}, indent=2))


if __name__ == "__main__":
    main()
