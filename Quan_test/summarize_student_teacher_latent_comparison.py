#!/usr/bin/env python3
"""Summarize matched student/teacher latent temporal-guidance runs."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

METRICS = ("PSNR", "SSIM", "LPIPS", "WE", "WE_L2", "FS", "VFID_Clips")
HIGHER = {"PSNR", "SSIM", "FS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--checkpoint-name", default="checkpoint-2250")
    args = parser.parse_args()
    root = args.output_root.resolve() / args.checkpoint_name
    records = []
    for seq_dir in sorted(x for x in root.iterdir() if x.is_dir()):
        flows = {}
        for flow in ("student", "teacher"):
            run = json.loads((seq_dir / flow / "run.json").read_text())
            if not run.get("complete") or run["inference"]["returncode"]:
                raise ValueError(f"Incomplete run: {seq_dir / flow}")
            paths = list((seq_dir / flow).glob("test_*/metrics_full_reference/metrics.csv"))
            if len(paths) != 1:
                raise ValueError(f"Missing metrics: {seq_dir / flow}")
            with paths[0].open(newline="") as file:
                rows = [x for x in csv.DictReader(file) if x["sequence"] != "ALL"]
            if len(rows) != 1:
                raise ValueError(paths[0])
            flows[flow] = {"frames": int(rows[0]["frames"]),
                "time_seconds": float(run["inference"]["elapsed_seconds"]),
                **{key: float(rows[0][key]) for key in METRICS}}
        if flows["student"]["frames"] != flows["teacher"]["frames"]:
            raise ValueError(f"Frame count differs: {seq_dir.name}")
        record = {"Sequence": seq_dir.name, "Frames": flows["student"]["frames"],
                  "Student time (s)": flows["student"]["time_seconds"],
                  "Teacher time (s)": flows["teacher"]["time_seconds"],
                  "Teacher / student time": flows["teacher"]["time_seconds"] / flows["student"]["time_seconds"]}
        for metric in METRICS:
            for flow in ("student", "teacher"):
                record[f"{flow.capitalize()} {metric}"] = flows[flow][metric]
            record[f"Teacher−student {metric}"] = flows["teacher"][metric] - flows["student"][metric]
        records.append(record)
    if not records:
        raise ValueError(f"No results in {root}")
    headers = list(records[0])
    with (root / "student_teacher_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)
    macro = {key: mean(row[key] for row in records) for key in headers[1:]}
    macro["Sequence"] = "Macro mean"
    macro["Teacher / student time"] = sum(x["Teacher time (s)"] for x in records) / sum(x["Student time (s)"] for x in records)
    (root / "student_teacher_summary.json").write_text(json.dumps({"records": records,
        "macro_mean": macro, "delta_definition": "teacher minus student",
        "timing_definition": "end-to-end inference process wall time; excludes cache preparation and metric evaluation",
        "teacher_is_oracle": True}, indent=2) + "\n")
    lines = ["# Student flow vs. teacher flow: latent temporal guidance", "",
             "Matched checkpoint, 50 DDIM steps, window 0–50, scale 0.1, clip size 4, pair-intersection BG mask, identical noise seeds.",
             "Teacher flow is an oracle computed from clean GT using ProPainter RAFT; student flow uses degraded inputs. Teacher-cache preparation is excluded from inference time.", "",
             "| Sequence | Frames | PSNR S/T/Δ | SSIM S/T/Δ | LPIPS S/T/Δ | WE-L2 S/T/Δ | VFID S/T/Δ | Time S/T |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in [*records, macro]:
        values = []
        for metric in ("PSNR", "SSIM", "LPIPS", "WE_L2", "VFID_Clips"):
            values.append(" / ".join(f"{row[key]:.4f}" for key in
                (f"Student {metric}", f"Teacher {metric}", f"Teacher−student {metric}")))
        lines.append("| " + row["Sequence"] + " | " + str(round(row["Frames"])) + " | " +
                     " | ".join(values) + f" | {row['Student time (s)']/60:.2f} / {row['Teacher time (s)']/60:.2f} min |")
    lines += ["", "Δ is teacher minus student. Higher is better for PSNR/SSIM; lower is better for LPIPS/WE-L2/VFID.",
              "Timing includes model initialization, flow loading/estimation, diffusion, and writing images; metric evaluation and teacher cache generation are excluded."]
    (root / "student_teacher_summary.md").write_text("\n".join(lines) + "\n")
    print(f"Summarized {len(records)} paired sequences in {root}")


if __name__ == "__main__":
    main()
