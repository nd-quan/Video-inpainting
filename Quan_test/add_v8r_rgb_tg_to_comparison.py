#!/usr/bin/env python3
"""Append the full-sequence V8R RGB-guidance evaluation to the comparison."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "experiments"
TABLE = ROOT / "metrics_three_methods_comparison_20260918"
EVAL = ROOT / "eval_v8r_standalone_ckpt3000_testsets_T24_S20_rgb_tg0p1"
METHOD = "V8R RGB-TG0.1"
METRICS = ("PSNR", "SSIM", "LPIPS", "WE", "WE-L2 (×10⁻²)", "FS", "VFID")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, headers, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    baseline = read_csv(TABLE / "comparison_metrics_four_methods.csv")
    headers = tuple(baseline[0])
    sequence_order = list(dict.fromkeys(row["Sequence"] for row in baseline))
    if len(sequence_order) != 13 or len(baseline) != 52:
        raise ValueError("Expected the preserved four-method, 13-sequence baseline")
    if {row["Method"] for row in baseline} != {"SN", "SN+TG", "R×4", "Frame-base"}:
        raise ValueError("Unexpected methods in preserved baseline")

    metric_paths = [EVAL / split / "metrics_full_reference" / "metrics.csv"
                    for split in ("test_1", "test_2")]
    new = {}
    for path in metric_paths:
        for row in read_csv(path):
            sequence = row["sequence"]
            if sequence == "ALL":
                continue
            if sequence in new:
                raise ValueError(f"Duplicate sequence: {sequence}")
            new[sequence] = row
    if set(new) != set(sequence_order):
        raise ValueError(f"Sequence mismatch: missing={set(sequence_order)-set(new)}, extra={set(new)-set(sequence_order)}")

    rows = []
    for sequence in sequence_order:
        group = [row for row in baseline if row["Sequence"] == sequence]
        expected = int(next(row["Frames"] for row in group if row["Method"] == "SN"))
        record = new[sequence]
        if int(record["frames"]) != expected:
            raise ValueError(f"Incomplete {sequence}: {record['frames']} vs {expected}")
        rows.extend(group)
        rows.append({"Sequence": sequence, "Method": METHOD, "Frames": record["frames"],
                     "PSNR": record["PSNR"], "SSIM": record["SSIM"],
                     "LPIPS": record["LPIPS"], "WE": record["WE"],
                     "WE-L2 (×10⁻²)": str(float(record["WE_L2"]) * 100),
                     "FS": record["FS"], "VFID": record["VFID_Clips"]})
    write_csv(TABLE / "comparison_metrics.csv", headers, rows)
    write_csv(TABLE / "comparison_metrics_five_methods.csv", headers, rows)

    lines = ["| Sequence | Method | Frames | PSNR ↑ | SSIM ↑ | LPIPS ↓ | WE ↓ | WE-L2 (×10⁻²) ↓ | FS ↑ | VFID ↓ |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for index, row in enumerate(rows):
        frame_label = row["Frames"]
        if row["Sequence"] == "PartyScene" and row["Method"] == "SN+TG":
            frame_label += "/250*"
        lines.append("| {} | {} | {} | {:.4f} | {:.5f} | {:.5f} | {:.6f} | {:.5f} | {:.5f} | {:.4f} |".format(
            row["Sequence"] if index % 5 == 0 else "", row["Method"], frame_label,
            *(float(row[metric]) for metric in METRICS)))
    lines += ["", "* PartyScene SN+TG has only 32/250 frames; its scores are not directly comparable to full-length runs.",
              "V8R RGB-TG0.1: standalone V8R checkpoint 3000, 24-frame clips/stride 20, RGB temporal guidance 0.1.",
              "Metrics use original PNGs, not lossy MP4s. WE-L2 actual value = displayed value × 10⁻²."]
    (TABLE / "comparison_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    included = json.loads((TABLE / "average_scope.json").read_text())["included_sequences"]
    methods = ("SN", "SN+TG", "R×4", "Frame-base", METHOD)
    averages = []
    for method in methods:
        selected = [row for row in rows if row["Method"] == method and row["Sequence"] in included]
        if len(selected) != len(included):
            raise ValueError(f"Missing average data for {method}")
        averages.append({"Method": method, "Sequences": len(included),
                         **{metric: sum(float(row[metric]) for row in selected) / len(selected)
                            for metric in METRICS}})
    average_headers = ("Method", "Sequences", *METRICS)
    write_csv(TABLE / "average_metrics.csv", average_headers, averages)
    write_csv(TABLE / "average_metrics_five_methods.csv", average_headers, averages)
    lines = ["| Method | Sequences | PSNR ↑ | SSIM ↑ | LPIPS ↓ | WE ↓ | WE-L2 (×10⁻²) ↓ | FS ↑ | VFID ↓ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in averages:
        lines.append("| {} | {} | {:.4f} | {:.5f} | {:.5f} | {:.6f} | {:.5f} | {:.5f} | {:.4f} |".format(
            row["Method"], row["Sequences"], *(float(row[metric]) for metric in METRICS)))
    lines += ["", "Unweighted mean over 12 complete common sequences; PartyScene is excluded because SN+TG has 32/250 frames.",
              "WE-L2 actual value = displayed value × 10⁻²."]
    (TABLE / "average_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sources = json.loads((TABLE / "sources.json").read_text())
    sources[METHOD] = [str(path) for path in metric_paths]
    write_json(TABLE / "sources.json", sources)
    scope_path = TABLE / "average_scope.json"
    scope = json.loads(scope_path.read_text())
    scope["method_order"] = methods
    scope["added_v8r_method"] = {"name": METHOD, "checkpoint": 3000,
                                 "temporal_guidance": "RGB", "scale": 0.1,
                                 "clip_length": 24, "clip_stride": 20,
                                 "sources": [str(path) for path in metric_paths]}
    write_json(scope_path, scope)
    print(f"Updated {len(rows)} comparison rows and {len(averages)} averages")


if __name__ == "__main__":
    main()
