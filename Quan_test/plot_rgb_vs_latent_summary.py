#!/usr/bin/env python3
"""Render the 13-sequence RGB/latent temporal-guidance comparison as a PNG table."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.table import Table


METRICS = (
    ("PSNR ↑", "PSNR", True, 3),
    ("SSIM ↑", "SSIM", True, 4),
    ("LPIPS ↓", "LPIPS", False, 4),
    ("WE ↓", "WE", False, 5),
    ("WE-L2 ↓", "WE-L2 (×10⁻²)", False, 4),
    ("FS ↑", "FS", True, 4),
    ("VFID ↓", "VFID", False, 3),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", type=Path)
    args = parser.parse_args()
    source = args.summary_json.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = payload["records"]
    if len(records) != 13:
        raise ValueError(f"Expected 13 sequence records, found {len(records)}")
    rows = [*records, payload["macro_mean"]]

    fig, ax = plt.subplots(figsize=(20.5, 9.2), facecolor="white")
    ax.axis("off")
    fig.text(0.035, 0.955, "RGB vs. latent temporal guidance", fontsize=22,
             fontweight="bold", color="#17324d", va="top")
    fig.text(0.035, 0.916,
             "V7 RAFT student  ·  guidance scale 0.1  ·  same seeds  ·  13 sequences / 1,920 frames per branch",
             fontsize=11.5, color="#54677a", va="top")

    columns = ["Sequence", "Frames", *(item[0] for item in METRICS), "Time (min) ↓"]
    widths = [0.145, 0.052, 0.100, 0.100, 0.100, 0.103, 0.103, 0.095, 0.093, 0.109]
    table = Table(ax, bbox=[0.025, 0.125, 0.95, 0.755])
    cell_height = 1 / (len(rows) + 1)

    for col, (heading, width) in enumerate(zip(columns, widths)):
        cell = table.add_cell(0, col, width, cell_height, text=heading,
                              loc="center", facecolor="#17324d", edgecolor="white")
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
        cell.get_text().set_fontsize(10)

    better_rgb = 0
    for index, row in enumerate(rows, start=1):
        is_mean = index == len(rows)
        base = "#e9eef5" if is_mean else ("#f6f8fb" if index % 2 == 0 else "white")
        values = [row["Sequence"], "—" if is_mean else f'{row["Frames"]:.0f}']
        colors = [base, base]
        for _, metric, higher, digits in METRICS:
            latent, rgb = (float(row[f"{space} {metric}"]) for space in ("Latent", "RGB"))
            values.append(f"{latent:.{digits}f}  →  {rgb:.{digits}f}")
            rgb_wins = rgb > latent if higher else rgb < latent
            better_rgb += int(rgb_wins and not is_mean)
            colors.append("#def2e8" if rgb_wins else "#fbeadf")
        latent_time, rgb_time = (float(row[f"{space} time (s)"]) / 60 for space in ("Latent", "RGB"))
        values.append(f"{latent_time:.1f}  →  {rgb_time:.1f}")
        colors.append("#fbeadf")
        for col, (value, width, color) in enumerate(zip(values, widths, colors)):
            cell = table.add_cell(index, col, width, cell_height, text=value,
                                  loc="center", facecolor=color, edgecolor="white")
            cell.get_text().set_fontsize(9.5 if col >= 2 else 10)
            if is_mean:
                cell.get_text().set_weight("bold")
            if col == 0:
                cell.get_text().set_horizontalalignment("left")
                cell.PAD = 0.08
    ax.add_table(table)

    fig.text(0.035, 0.085,
             "Each cell: latent → RGB.  Green: RGB better; orange: latent better.  WE-L2 values are displayed ×10⁻².",
             fontsize=10.5, color="#32465b")
    fig.text(0.035, 0.055,
             "Macro mean is unweighted over 13 sequences; Kimono has 120 frames, all others 150. "
             "Time includes inference and image writing, not metric evaluation.",
             fontsize=9.5, color="#54677a")
    output = source.with_name("rgb_vs_latent_summary_table.png")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Saved {output} ({len(records)} sequences; RGB wins {better_rgb}/{len(records)*len(METRICS)} quality cells)")


if __name__ == "__main__":
    main()
