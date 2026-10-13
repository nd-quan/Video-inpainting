#!/usr/bin/env python3
"""Summarize matched student/teacher latent temporal-guidance runs."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

REFERENCE = Path(__file__).resolve().parents[1] / "experiments/eval_temporal_guidance_rgb_vs_latent_student_s0p1_20260918"

METRICS = ("PSNR", "SSIM", "LPIPS", "WE", "WE_L2", "FS", "VFID_Clips")
HIGHER = {"PSNR", "SSIM", "FS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--checkpoint-name", default="checkpoint-2250")
    args = parser.parse_args()
    root = args.output_root.resolve() / args.checkpoint_name
    records = []
    changed_dataset = []
    for seq_dir in sorted(x for x in root.iterdir() if x.is_dir()):
        flows = {}
        for flow in ("student", "teacher"):
            if flow == "student":
                flow_dir = REFERENCE / seq_dir.name / "latent"
                reference_run = json.loads((REFERENCE / seq_dir.name / "comparison_run.json").read_text())
                inference = reference_run["runs"]["latent"]["inference"]
                source = json.loads(next(flow_dir.glob("test_*/*/source_metadata.json")).read_text())
                first = source["frame_range"][0]
                for kind, directory in (("input", Path(source["image_dir"])),
                                        ("GT", Path(source["image_dir"]).parent / "gt")):
                    path = directory / f"{first:06d}.png"
                    if path.stat().st_mtime > inference["started_at_unix"]:
                        changed_dataset.append({"sequence": seq_dir.name, "kind": kind,
                                                "path": str(path), "modified_at_unix": path.stat().st_mtime})
            else:
                flow_dir = seq_dir / flow
                run = json.loads((flow_dir / "run.json").read_text())
                if not run.get("complete"):
                    raise ValueError(f"Incomplete run: {flow_dir}")
                inference = run["inference"]
            if inference["returncode"]:
                raise ValueError(f"Inference failed: {flow_dir}")
            paths = list(flow_dir.glob("test_*/metrics_full_reference/metrics.csv"))
            if len(paths) != 1:
                raise ValueError(f"Missing metrics: {seq_dir / flow}")
            with paths[0].open(newline="") as file:
                rows = [x for x in csv.DictReader(file) if x["sequence"] != "ALL"]
            if len(rows) != 1:
                raise ValueError(paths[0])
            flows[flow] = {"frames": int(rows[0]["frames"]),
                "time_seconds": float(inference["elapsed_seconds"]),
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
        "timing_definition": "end-to-end inference process wall time; excludes cache preparation and metric evaluation; runs occurred on different dates with concurrent GPU workloads",
        "teacher_is_oracle": True, "student_source": str(REFERENCE),
        "comparison_valid": not changed_dataset,
        "dataset_files_modified_after_student_run": changed_dataset}, indent=2) + "\n")
    lines = ["# Student flow vs. teacher flow: latent temporal guidance", "",
             *( [f"**VALIDITY WARNING:** Current dataset inputs/GT were modified after the stored student run for {len(set(x['sequence'] for x in changed_dataset))}/{len(records)} sequences. The table below is not a controlled flow comparison; do not attribute its differences to the flow backend.", ""] if changed_dataset else [] ),
             "Matched checkpoint, 50 DDIM steps, window 0–50, scale 0.1, clip size 4, pair-intersection BG mask, identical noise seeds.",
             "Student metrics and timing come from the prior RGB/latent experiment; teacher flow is an oracle computed from clean GT using ProPainter RAFT. Teacher-cache preparation is excluded from inference time.", "",
             "| Sequence | Frames | PSNR S/T/Δ | SSIM S/T/Δ | LPIPS S/T/Δ | WE-L2 (×10⁻²) S/T/Δ | VFID S/T/Δ | Time S/T |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in [*records, macro]:
        values = []
        for metric in ("PSNR", "SSIM", "LPIPS", "WE_L2", "VFID_Clips"):
            scale = 100 if metric == "WE_L2" else 1
            digits = 5 if metric == "WE_L2" else 4
            values.append(" / ".join(f"{row[key] * scale:.{digits}f}" for key in
                (f"Student {metric}", f"Teacher {metric}", f"Teacher−student {metric}")))
        lines.append("| " + row["Sequence"] + " | " + str(round(row["Frames"])) + " | " +
                     " | ".join(values) + f" | {row['Student time (s)']/60:.2f} / {row['Teacher time (s)']/60:.2f} min |")
    lines += ["", "Δ is teacher minus student. Higher is better for PSNR/SSIM; lower is better for LPIPS/WE-L2/VFID.",
              "Timing includes model initialization, flow loading/estimation, diffusion, and writing images; metric evaluation and teacher cache generation are excluded. Student and teacher ran on different dates with concurrent GPU workloads, so timing ratios are descriptive rather than a controlled speed benchmark."]
    psnr_lower = sum(row["Teacher−student PSNR"] < 0 for row in records)
    ssim_lower = sum(row["Teacher−student SSIM"] < 0 for row in records)
    lines += ["", "## Interpretation", ""]
    if changed_dataset:
        lines += ["The stored student run cannot be compared causally with the teacher run because the dataset changed in between. A four-frame current-code student rerun for BQSquare reproduced the teacher's first frame exactly and differed by only about 0.08–0.11 RGB levels on subsequent frames, while both differed from the stored student output by about 5–6 levels."]
    else:
        lines += [f"Teacher has lower PSNR on {psnr_lower}/{len(records)} sequences and lower SSIM on {ssim_lower}/{len(records)}. "
                  f"Macro differences are {macro['Teacher−student PSNR']:+.4f} dB PSNR, "
                  f"{macro['Teacher−student SSIM']:+.4f} SSIM, "
                  f"and {macro['Teacher−student WE_L2']*100:+.5f} in WE-L2 ×10⁻² units.",
                  "The teacher uses clean GT, so it is an oracle flow input. This result shows how that flow behaves with the fixed student-tuned guidance scale; it does not establish that the teacher flow itself is less accurate or that temporal guidance helps relative to no guidance."]
    (root / "student_teacher_summary.md").write_text("\n".join(lines) + "\n")
    print(f"Summarized {len(records)} paired sequences in {root}")


if __name__ == "__main__":
    main()
