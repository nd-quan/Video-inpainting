#!/usr/bin/env python3
"""Add independent-noise frame-base metrics to the established method comparison."""
import csv
import json
import shutil
from pathlib import Path


METHOD = "Frame-base"
METRICS = ("PSNR", "SSIM", "LPIPS", "WE", "WE-L2 (×10⁻²)", "FS", "VFID")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, headers, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main():
    root = Path(__file__).resolve().parents[1] / "experiments"
    comparison_root = root / "metrics_three_methods_comparison_20260918"
    comparison = comparison_root / "comparison_metrics.csv"
    baseline = comparison_root / "comparison_metrics_three_methods.csv"
    if not baseline.exists():
        shutil.copy2(comparison, baseline)
    existing = read_csv(baseline)
    headers = list(existing[0])
    required = {"SN", "SN+TG", "R×4"}
    if {row["Method"] for row in existing} != required:
        raise ValueError("The preserved comparison must contain exactly the original three methods")
    sequence_order = list(dict.fromkeys(row["Sequence"] for row in existing))
    grouped = {sequence: [row for row in existing if row["Sequence"] == sequence]
               for sequence in sequence_order}
    if any(len(rows) != 3 for rows in grouped.values()):
        raise ValueError("Expected three original method rows per sequence")

    metric_files = (
        root / "eval_ckpt2250_independent_noise_test1" / "test_1" / "metrics_full_reference" / "metrics.csv",
        root / "eval_ckpt2250_independent_noise_test2" / "test_2" / "metrics_full_reference" / "metrics.csv",
    )
    frame_rows = {}
    for path in metric_files:
        for row in read_csv(path):
            if row["sequence"] == "ALL":
                continue
            if row["sequence"] in frame_rows:
                raise ValueError(f"Duplicate sequence in frame-base metrics: {row['sequence']}")
            frame_rows[row["sequence"]] = row
    if set(frame_rows) != set(sequence_order):
        raise ValueError("Frame-base metric sequences do not match the comparison table")

    output_rows = []
    for sequence in sequence_order:
        output_rows.extend(grouped[sequence])
        row = frame_rows[sequence]
        output_rows.append({
            "Sequence": sequence, "Method": METHOD, "Frames": row["frames"],
            "PSNR": row["PSNR"], "SSIM": row["SSIM"], "LPIPS": row["LPIPS"],
            "WE": row["WE"], "WE-L2 (×10⁻²)": str(float(row["WE_L2"]) * 100),
            "FS": row["FS"], "VFID": row["VFID_Clips"],
        })
    write_csv(comparison, headers, output_rows)
    write_csv(comparison_root / "comparison_metrics_four_methods.csv", headers, output_rows)

    included = sorted(sequence for sequence in sequence_order if sequence != "PartyScene")
    method_order = ("SN", "SN+TG", "R×4", METHOD)
    averages = []
    for method in method_order:
        method_rows = [row for row in output_rows if row["Method"] == method and row["Sequence"] in included]
        if len(method_rows) != len(included):
            raise ValueError(f"Missing aggregate rows for {method}")
        averages.append({"Method": method, "Sequences": len(included),
                         **{metric: sum(float(row[metric]) for row in method_rows) / len(method_rows)
                            for metric in METRICS}})
    average_headers = ("Method", "Sequences", *METRICS)
    for name in ("average_metrics.csv", "average_metrics_four_methods.csv"):
        write_csv(comparison_root / name, average_headers, averages)
    lines = ["| Method | Sequences | PSNR ↑ | SSIM ↑ | LPIPS ↓ | WE ↓ | WE-L2 (×10⁻²) ↓ | FS ↑ | VFID ↓ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in averages:
        lines.append("| {Method} | {Sequences} | {PSNR:.4f} | {SSIM:.5f} | {LPIPS:.5f} | {WE:.6f} | {we:.5f} | {FS:.5f} | {VFID:.4f} |".format(
            **row, we=row["WE-L2 (×10⁻²)"]))
    lines += ["", "Unweighted arithmetic mean across 12 complete common sequences; PartyScene is excluded because SN+TG has 32/250 frames.",
              "WE-L2 actual value = displayed value × 10⁻²."]
    for name in ("average_metrics.md", "average_metrics_four_methods.md"):
        (comparison_root / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    scope = {"aggregation": "unweighted arithmetic mean of full-precision per-sequence scores",
             "included_sequences": included,
             "excluded_sequences": {"PartyScene": "SN+TG has 32/250 frames"},
             "VFID": "mean of sequence VFID, not pooled-feature VFID",
             "WE_L2_display_multiplier": 100, "method_order": method_order,
             "added_method": {"name": METHOD, "description": "independent-noise frame-wise baseline",
                              "sources": [str(path) for path in metric_files]}}
    (comparison_root / "average_scope.json").write_text(json.dumps(scope, indent=2) + "\n", encoding="utf-8")
    source_path = comparison_root / "sources.json"
    sources = json.loads(source_path.read_text()) if source_path.is_file() else {}
    sources[METHOD] = [str(path) for path in metric_files]
    source_path.write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(output_rows)} comparison rows and {len(averages)} aggregate rows to {comparison_root}")


if __name__ == "__main__":
    main()
