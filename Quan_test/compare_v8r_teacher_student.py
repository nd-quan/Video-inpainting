#!/usr/bin/env python3
"""Compare completed V8-R teacher-flow runs with sources.json student runs."""

import csv
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/eval_v8r_ckpt2500_temporal_teacher_matched_20260929"
METRICS = ("PSNR", "SSIM", "LPIPS", "WE", "WE_L2", "FS", "VFID_Clips")
HIGHER = {"PSNR", "SSIM", "FS"}
sys.path.insert(0, str(ROOT / "Quan_test"))
from compute_vfid_clips_4datasets import frechet_distance_low_rank  # noqa: E402


def one_row(path, sequence):
    with path.open(newline="") as file:
        matches = [r for r in csv.DictReader(file) if r["sequence"] == sequence]
    if len(matches) != 1:
        raise ValueError(f"Expected one {sequence} row in {path}")
    row = matches[0]
    return {k: (int(v) if k in {"frames", "pairs"} else float(v))
            for k, v in row.items() if k not in {"sequence"}}


def feature(path, sequence, kind):
    array = np.load(path / f"{sequence}_{kind}_i3d.npy", allow_pickle=False)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"Invalid I3D features: {path}/{sequence}_{kind}_i3d.npy")
    return array


def aggregate(records, scope):
    selected = [r for r in records if scope == "all" or r["split"] == scope]
    result = {"scope": scope, "sequences": len(selected),
              "frames": sum(r["frames"] for r in selected),
              "pairs": sum(r["pairs"] for r in selected)}
    for method in ("student", "teacher"):
        for metric in METRICS:
            if metric == "VFID_Clips":
                gt = np.concatenate([r[f"{method}_gt_features"] for r in selected])
                final = np.concatenate([r[f"{method}_final_features"] for r in selected])
                score = frechet_distance_low_rank(gt, final)
            else:
                weights = [r["pairs" if metric in {"WE", "WE_L2", "FS"} else "frames"] for r in selected]
                score = float(np.average([r[f"{method}_{metric}"] for r in selected], weights=weights))
            result[f"{method}_{metric}"] = score
    for metric in METRICS:
        result[f"delta_{metric}"] = result[f"teacher_{metric}"] - result[f"student_{metric}"]
        wins = sum((r[f"delta_{metric}"] > 0) if metric in HIGHER else
                   (r[f"delta_{metric}"] < 0) for r in selected)
        result[f"teacher_wins_{metric}"] = wins
    return result


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_root.resolve()
    manifest = json.loads((OUT / "reference_manifest.json").read_text())
    source_index = json.loads(Path(manifest["source_index"]).read_text())["R×4"]
    if len(source_index) != len(manifest["records"]):
        raise ValueError("Student source count changed")
    records = []
    for record, source in zip(manifest["records"], source_index):
        sequence, split = record["sequence"], record["split"]
        if Path(source).parts[-3] != Path(record["source_dir"]).name:
            raise ValueError(f"Student source index changed: {sequence}")
        teacher_dir = OUT / f"{sequence}_teacher"
        original_student_metrics = Path(record["student_metrics"])
        student_metrics = original_student_metrics
        if original_student_metrics.is_file() and one_row(original_student_metrics, sequence)["frames"] != record["unique_frames"]:
            student_metrics = OUT / "matched_student_subset" / sequence / "metrics_full_reference/metrics.csv"
        teacher_metrics = teacher_dir / "metrics_full_reference/metrics.csv"
        if not student_metrics.is_file() or not teacher_metrics.is_file():
            raise FileNotFoundError(f"Missing complete metric pair for {sequence}")
        student = one_row(student_metrics, sequence)
        teacher = one_row(teacher_metrics, sequence)
        if (student["frames"], student["pairs"]) != (record["unique_frames"], record["unique_frames"] - 1):
            raise ValueError(f"Student frame count differs for {sequence}")
        if (teacher["frames"], teacher["pairs"]) != (student["frames"], student["pairs"]):
            raise ValueError(f"Teacher frame count differs for {sequence}")
        student_dir = student_metrics.parent
        teacher_metric_dir = teacher_metrics.parent
        item = {"sequence": sequence, "split": split, "frames": student["frames"], "pairs": student["pairs"],
                "student_metrics": str(student_metrics), "teacher_metrics": str(teacher_metrics)}
        for metric in METRICS:
            a, b = student[metric], teacher[metric]
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError(f"Nonfinite metric: {sequence}/{metric}")
            item[f"student_{metric}"] = a
            item[f"teacher_{metric}"] = b
            item[f"delta_{metric}"] = b - a
        student_gt = feature(student_dir, sequence, "gt")
        teacher_gt = feature(teacher_metric_dir, sequence, "gt")
        if student_gt.shape != teacher_gt.shape or not np.allclose(student_gt, teacher_gt, rtol=1e-5, atol=1e-5):
            raise ValueError(f"GT I3D features differ: {sequence}")
        item["student_gt_features"] = student_gt
        item["student_final_features"] = feature(student_dir, sequence, "final")
        item["teacher_gt_features"] = teacher_gt
        item["teacher_final_features"] = feature(teacher_metric_dir, sequence, "final")
        records.append(item)
    fields = ["sequence", "split", "frames", "pairs"]
    for metric in METRICS:
        fields.extend([f"student_{metric}", f"teacher_{metric}", f"delta_{metric}"])
    with (OUT / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: r[key] for key in fields} for r in records)
    summary = {scope: aggregate(records, scope) for scope in ("all", "test_1", "test_2")}
    summary["method"] = "PSNR/SSIM/LPIPS weighted by frames; WE/WE_L2/FS by adjacent pairs; pooled VFID from saved I3D clip features"
    summary["delta_definition"] = "teacher minus student; teacher_wins is the count of sequences with better teacher score"
    summary["source_index"] = str(manifest["source_index"])
    summary["student_subset_recomputed"] = [r["sequence"] for r in records
        if "matched_student_subset" in r["student_metrics"]]
    summary["comparison_scope"] = "Both methods evaluated on the same saved student frame IDs and GT; condition flow and temporal guidance flow change together"
    experiment_manifest = OUT / "experiment_manifest.json"
    if experiment_manifest.is_file():
        summary["experiment"] = json.loads(experiment_manifest.read_text())
    (OUT / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
