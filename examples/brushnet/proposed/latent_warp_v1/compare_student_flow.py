"""Wait for student inference, submit metrics, and compare with GT flow."""
import csv
import json
from pathlib import Path
import sys
import time

from export_inference_videos import wait_for_inference
from inference_jobs import query_task
from run_full_reference_metrics import submit_metrics

student, reference = map(Path, sys.argv[1:3])
wait_for_inference(student, int(sys.argv[3]))
task = submit_metrics(student)
while True:
    status = query_task(task)["status"]
    if status in {"FAILED", "KILLED", "CANCELLED"}:
        raise RuntimeError(f"Metric job {task}: {status}")
    if status == "COMPLETED":
        break
    time.sleep(30)

assert json.loads((student / "full_reference_metrics_status.json").read_text())["state"] == "complete"

def read_metrics(root):
    with (root / "metrics_full_reference/metrics.csv").open() as file:
        return {row["sequence"]: row for row in csv.DictReader(file)}

gt, predicted = read_metrics(reference), read_metrics(student)
assert gt.keys() == predicted.keys() and len(gt) == 14
metrics = ("PSNR", "SSIM", "LPIPS", "WE", "WE_L2", "FS", "VFID_Clips")
with (student / "comparison_vs_gt_flow.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(("sequence", "metric", "gt", "student", "student_minus_gt"))
    for sequence, row in gt.items():
        other = predicted[sequence]
        assert all(row[key] == other[key] for key in ("frames", "pairs"))
        for metric in metrics:
            delta = float(other[metric]) - float(row[metric])
            writer.writerow((sequence, metric, row[metric], other[metric], delta))
            if sequence == "ALL":
                print(metric, row[metric], other[metric], delta)
