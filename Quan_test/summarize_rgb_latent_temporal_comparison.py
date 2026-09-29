#!/usr/bin/env python3
"""Summarize matched RGB- versus latent-space temporal guidance evaluations."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean


METRICS = ("PSNR", "SSIM", "LPIPS", "WE", "WE_L2", "FS", "VFID_Clips")
DISPLAY = {"PSNR": "PSNR", "SSIM": "SSIM", "LPIPS": "LPIPS", "WE": "WE",
           "WE_L2": "WE-L2 (×10⁻²)", "FS": "FS", "VFID_Clips": "VFID"}
HIGHER_IS_BETTER = {"PSNR", "SSIM", "FS"}


def metric_row(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = [row for row in rows if row["sequence"] != "ALL"]
    if len(result) != 1:
        raise ValueError(f"Expected one sequence metric row: {path}")
    return result[0]


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_root", type=Path)
    args = parser.parse_args()
    root = args.comparison_root.resolve()
    records = []
    for sequence_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        run_path = sequence_dir / "comparison_run.json"
        if not run_path.is_file():
            continue
        run = json.loads(run_path.read_text())
        sequence = run["sequence"]
        spaces = {}
        for space in ("latent", "rgb"):
            metric_paths = list((sequence_dir / space).glob("test_*/metrics_full_reference/metrics.csv"))
            if len(metric_paths) != 1:
                raise ValueError(f"Expected one metric file for {sequence}/{space}: {metric_paths}")
            metric = metric_row(metric_paths[0])
            inference = run["runs"][space]["inference"]
            if inference.get("returncode") != 0 or inference.get("elapsed_seconds") is None:
                raise ValueError(f"Incomplete timing metadata for {sequence}/{space}")
            spaces[space] = {"time_seconds": float(inference["elapsed_seconds"]),
                             "frames": int(metric["frames"]),
                             **{metric_name: float(metric[metric_name]) for metric_name in METRICS}}
        if spaces["latent"]["frames"] != spaces["rgb"]["frames"]:
            raise ValueError(f"Frame mismatch for {sequence}")
        record = {"Sequence": sequence, "Frames": spaces["latent"]["frames"],
                  "Latent time (s)": spaces["latent"]["time_seconds"],
                  "RGB time (s)": spaces["rgb"]["time_seconds"],
                  "RGB / latent time": spaces["rgb"]["time_seconds"] / spaces["latent"]["time_seconds"]}
        for metric in METRICS:
            scale = 100 if metric == "WE_L2" else 1
            label = DISPLAY[metric]
            record[f"Latent {label}"] = spaces["latent"][metric] * scale
            record[f"RGB {label}"] = spaces["rgb"][metric] * scale
            record[f"Δ RGB−latent {label}"] = (spaces["rgb"][metric] - spaces["latent"][metric]) * scale
        records.append(record)
    if len(records) < 1:
        raise ValueError(f"No comparison_run.json files found in {root}")

    headers = list(records[0])
    csv_path = root / "rgb_vs_latent_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)

    mean_record = {"Sequence": "Macro mean", "Frames": sum(row["Frames"] for row in records) / len(records)}
    for key in headers[2:]:
        if key == "RGB / latent time":
            mean_record[key] = sum(row["RGB time (s)"] for row in records) / sum(row["Latent time (s)"] for row in records)
        else:
            mean_record[key] = mean(row[key] for row in records)

    lines = ["# RGB vs. latent temporal guidance", "",
             "Matched settings: up to 150 frames per sequence (Kimono: 120); V7 student flow; 50 DDIM steps; temporal window 0–50; "
             "guidance scale 0.1; clip size 4; pair-intersection stable background; identical main/shared noise seeds.", "",
             "## Output quality", "",
             "Δ = RGB − latent. Positive Δ is favorable for PSNR, SSIM, and FS; negative Δ is favorable for LPIPS, WE, WE-L2, and VFID.", "",
             "| Sequence | PSNR L/R/Δ | SSIM L/R/Δ | LPIPS L/R/Δ | WE-L2 (×10⁻²) L/R/Δ | FS L/R/Δ | VFID L/R/Δ |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    quality_metrics = ("PSNR", "SSIM", "LPIPS", "WE_L2", "FS", "VFID_Clips")
    for row in [*records, mean_record]:
        cells = []
        for metric in quality_metrics:
            label = DISPLAY[metric]
            cells.append(" / ".join(fmt(row[f"{space} {label}"], 4 if metric in {"PSNR", "VFID_Clips"} else 5)
                                    for space in ("Latent", "RGB")) + " / " +
                         fmt(row[f"Δ RGB−latent {label}"], 4 if metric in {"PSNR", "VFID_Clips"} else 5))
        lines.append("| " + row["Sequence"] + " | " + " | ".join(cells) + " |")
    lines += ["", "## End-to-end inference time", "",
              "Times include pipeline/model initialization and image writing; metric time is excluded.", "",
              "| Sequence | Frames | Latent | RGB | RGB / latent | RGB − latent |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in [*records, mean_record]:
        delta = row["RGB time (s)"] - row["Latent time (s)"]
        lines.append(f"| {row['Sequence']} | {row['Frames']:.0f} | {row['Latent time (s)']/60:.2f} min | "
                     f"{row['RGB time (s)']/60:.2f} min | {row['RGB / latent time']:.2f}× | {delta/60:.2f} min |")
    (root / "rgb_vs_latent_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "rgb_vs_latent_summary.json").write_text(json.dumps({"records": records, "macro_mean": mean_record,
        "timing_definition": "end-to-end inference subprocess wall time; excludes metric evaluation"}, indent=2) + "\n")
    print(f"Wrote {csv_path} and RGB/latent Markdown and JSON summaries.")


if __name__ == "__main__":
    main()
