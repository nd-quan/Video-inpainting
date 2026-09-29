#!/usr/bin/env python3
"""Render the macro-average comparison table with best values highlighted."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


METRICS = ("PSNR", "SSIM", "LPIPS", "WE", "WE-L2 (×10⁻²)", "FS", "VFID")
HIGHER = {"PSNR", "SSIM", "FS"}
LABELS = {"SN": "SN", "SN+TG": "SN+TG", "R×4": "TCE", "Frame-base": "Frame-base",
          "V8R RGB-TG0.1": "V8R RGB-TG0.1"}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("average_csv", type=Path)
    args = parser.parse_args()
    source = args.average_csv.resolve()
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not {"SN", "SN+TG", "R×4", "Frame-base"}.issubset({row["Method"] for row in rows}):
        raise ValueError("Missing established comparison methods")
    order = tuple(dict.fromkeys(row["Method"] for row in rows))
    rows = [next(row for row in rows if row["Method"] == method) for method in order]
    best = {}
    for metric in METRICS:
        values = {row["Method"]: float(row[metric]) for row in rows}
        target = (max if metric in HIGHER else min)(values.values())
        best[metric] = {method for method, value in values.items() if value == target}
    boundaries = (0, 2.1, 3.4, 4.85, 6.2, 7.55, 8.9, 10.45, 11.8, 13.1)
    centers = [(left + right) / 2 for left, right in zip(boundaries, boundaries[1:])]
    fig = plt.figure(figsize=(13.1, 3.15 + 0.48 * (len(rows) - 4)), facecolor="white")
    ax = fig.add_axes([0.025, 0.17, 0.95, 0.76])
    ax.set_xlim(0, boundaries[-1])
    ax.set_ylim(len(rows) + 0.4, -1.3)
    ax.axis("off")
    headers = ("Method", "Sequences", "PSNR ↑", "SSIM ↑", "LPIPS ↓", "WE ↓",
               r"WE-L2 ($\times10^{-2}$) ↓", "FS ↑", "VFID ↓")
    for x, header in zip(centers, headers):
        ax.text(x, -0.7, header, ha="center", va="center", fontsize=12.5)
    ax.hlines(0.05, 0, boundaries[-1], color="black", linewidth=1.25)
    lines = ["| Method | Sequences | PSNR ↑ | SSIM ↑ | LPIPS ↓ | WE ↓ | WE-L2 (×10⁻²) ↓ | FS ↑ | VFID ↓ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    records = []
    for row_index, row in enumerate(rows):
        y = row_index + 0.6
        ax.text(centers[0], y, LABELS[row["Method"]], ha="center", va="center", fontsize=12.5)
        ax.text(centers[1], y, row["Sequences"], ha="center", va="center", fontsize=12.5)
        rendered = []
        for column, metric in enumerate(METRICS, start=2):
            value = float(row[metric])
            text = f"{value:.4f}" if metric in {"PSNR", "VFID"} else f"{value:.5f}"
            winner = row["Method"] in best[metric]
            if winner:
                ax.add_patch(Rectangle((boundaries[column] + 0.05, y - 0.37),
                                       boundaries[column + 1] - boundaries[column] - 0.1, 0.74,
                                       facecolor="#ddf2e3", edgecolor="none", zorder=0))
                records.append({"metric": metric, "method": LABELS[row["Method"]], "value": value})
            ax.text(centers[column], y, text, ha="center", va="center", fontsize=12.5,
                    fontweight="bold" if winner else "normal", color="#146b35" if winner else "black")
            rendered.append(f"**{text}**" if winner else text)
        lines.append("| " + " | ".join([LABELS[row["Method"]], row["Sequences"], *rendered]) + " |")
    ax.hlines(len(rows) + 0.1, 0, boundaries[-1], color="black", linewidth=1.1)
    footnote = "Green bold values: best macro average; PartyScene is excluded because SN+TG has 32/250 frames."
    fig.text(0.03, 0.06, footnote, fontsize=10.5, va="top")
    output = source.parent / "average_metrics_highlighted"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix("." + extension), dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    output.with_suffix(".md").write_text("\n".join([*lines, "", footnote,
        "WE-L2 actual value = displayed value × 10⁻²."]) + "\n", encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps({"winners": records}, indent=2), encoding="utf-8")
    print(f"Highlighted {len(records)} macro-average values: {output}")


if __name__ == "__main__":
    main()
