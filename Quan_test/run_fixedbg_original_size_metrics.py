#!/usr/bin/env python3
"""Score native-size fixedBG outputs against native GT on six GPUs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
from threading import Lock
from typing import Optional

import numpy as np
from PIL import Image


BRUSHNET = Path(__file__).resolve().parents[1]
DATASET = BRUSHNET / "examples/brushnet/dataset/original_size"
EVAL = Path(os.environ.get(
    "EVAL_ROOT", str(BRUSHNET / "experiments/eval_fixedBG_temporal_v7_student")
))
METRIC_SCRIPT = BRUSHNET / "examples/brushnet/proposed/DGAF_VSR_original/postprocess_eval.py"
EVALUATOR_ROOT = BRUSHNET.parent / "video-inpainting-evaluation"
PYTHON = Path("/home/cilab/ndquan/envs/guided_diff/bin/python")
OUTPUT_NAME = "metrics_original_size_full_reference"
LOG_DIR = EVAL / os.environ.get("METRIC_LOG_DIR", "logs_original_size_latent_s0p1")
GPUS = tuple(range(6))
FIELDS = ["sequence", "frames", "pairs", "PSNR", "SSIM", "LPIPS", "WE", "WE_L2", "FS", "VFID_Clips"]


def save_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def score_one(job: dict, gpu: int) -> tuple[int, str]:
    split, name = job["split"], job["sequence"]
    pred_dir = EVAL / split / name
    gt_dir = DATASET / split / name / "gt"
    dest = pred_dir / OUTPUT_NAME
    if (dest / "metrics.csv").is_file():
        return 0, "already complete"
    pred = {int(p.stem): p for p in pred_dir.glob("*.png")}
    gt = {int(p.stem): p for p in gt_dir.glob("*.png")}
    if set(pred) != set(gt) or len(pred) != job["frames"]:
        raise ValueError(f"Prediction/GT frame IDs differ for {name}")
    ids = sorted(pred)
    for frame_id in (ids[0], ids[-1]):
        with Image.open(pred[frame_id]) as a, Image.open(gt[frame_id]) as b:
            if a.size != b.size or a.size != (job["width"], job["height"]):
                raise ValueError(f"Prediction/GT dimensions differ for {name}, frame {frame_id}")

    with tempfile.TemporaryDirectory(prefix=f"metric_{name}_") as dirname:
        stage = Path(dirname)
        sequence = stage / name
        for kind, paths in (("final", pred), ("gt", gt)):
            folder = sequence / kind
            folder.mkdir(parents=True)
            for frame_id in ids:
                (folder / f"{frame_id:06d}.png").symlink_to(paths[frame_id].resolve())
        (sequence / "clip_metrics.json").write_text(
            json.dumps({"video": name, "frame_ids": ids}), encoding="utf-8"
        )
        (stage / "summary.json").write_text('{"clips": 1}', encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        log = LOG_DIR / f"metrics_{split}_{name}_gpu{gpu}.log"
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\n=== {datetime.now().isoformat()} GPU {gpu} {name} ===\n")
            stream.flush()
            result = subprocess.run(
                [str(PYTHON), "-u", str(METRIC_SCRIPT), "--eval_root", str(stage),
                 "--evaluator_root", str(EVALUATOR_ROOT)],
                cwd=BRUSHNET, env=env, stdout=stream, stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            return result.returncode, str(log)
        metrics = stage / "metrics"
        if not (metrics / "metrics.csv").is_file():
            return 1, f"No metrics.csv in {metrics}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(metrics), str(dest))
        save_json(dest / "source.json", {
            "sequence": name, "split": split, "frames": len(ids),
            "pred_dir": str(pred_dir), "gt_dir": str(gt_dir),
            "reference_resolution": "native, no resizing",
            "evaluator": str(METRIC_SCRIPT), "cuda_visible_devices": str(gpu),
        })
    return 0, str(dest / "metrics.csv")


def aggregate(jobs: list[dict], split: Optional[str]) -> Path:
    sys.path.insert(0, str(BRUSHNET / "Quan_test"))
    from compute_vfid_clips_4datasets import frechet_distance_low_rank

    selected = [job for job in jobs if split is None or job["split"] == split]
    rows, all_gt, all_pred = [], [], []
    for job in selected:
        name = job["sequence"]
        folder = EVAL / job["split"] / name / OUTPUT_NAME
        with (folder / "metrics.csv").open(newline="") as stream:
            row = next(row for row in csv.DictReader(stream) if row["sequence"] == name)
        row = {key: row[key] if key == "sequence" else
               int(row[key]) if key in ("frames", "pairs") else float(row[key])
               for key in FIELDS}
        rows.append(row)
        all_gt.append(np.load(folder / f"{name}_gt_i3d.npy"))
        all_pred.append(np.load(folder / f"{name}_final_i3d.npy"))

    all_row = {"sequence": "ALL", "frames": sum(r["frames"] for r in rows),
               "pairs": sum(r["pairs"] for r in rows)}
    for key in ("PSNR", "SSIM", "LPIPS", "WE", "WE_L2", "FS"):
        weight = "pairs" if key in ("WE", "WE_L2", "FS") else "frames"
        all_row[key] = float(np.average([r[key] for r in rows],
                                        weights=[r[weight] for r in rows]))
    all_row["VFID_Clips"] = float(frechet_distance_low_rank(
        np.concatenate(all_gt), np.concatenate(all_pred)))
    rows.append(all_row)
    out = (EVAL / split if split else EVAL) / OUTPUT_NAME
    out.mkdir(exist_ok=True)
    with (out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    save_json(out / "source.json", {
        "split": split or "test_1+test_2", "sequences": [x["sequence"] for x in selected],
        "frames": all_row["frames"], "reference_resolution": "native, no resizing",
        "aggregation": "frame weighted image metrics; pair weighted temporal metrics; pooled VFID features",
    })
    return out / "metrics.csv"


def main() -> int:
    jobs = json.loads((DATASET / "manifest.json").read_text())["sequences"]
    if len(jobs) != 13:
        raise ValueError("Expected 13 sequences")
    log_dir = LOG_DIR
    log_dir.mkdir(exist_ok=True)
    status_path = log_dir / "metrics_status.json"
    status = [{"sequence": job["sequence"], "split": job["split"], "state": "queued"}
              for job in jobs]
    save_json(status_path, status)
    pending = queue.Queue()
    for index in sorted(range(len(jobs)),
                        key=lambda i: jobs[i]["width"] * jobs[i]["height"] * jobs[i]["frames"],
                        reverse=True):
        pending.put(index)
    lock = Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                index = pending.get_nowait()
            except queue.Empty:
                return
            with lock:
                status[index].update(state="running", gpu=gpu,
                                     started_at=datetime.now().isoformat(timespec="seconds"))
                save_json(status_path, status)
            try:
                code, detail = score_one(jobs[index], gpu)
            except Exception as error:
                code, detail = 1, repr(error)
            with lock:
                status[index].update(state="complete" if code == 0 else "failed",
                                     exit_code=code, detail=detail,
                                     finished_at=datetime.now().isoformat(timespec="seconds"))
                save_json(status_path, status)
            print(jobs[index]["sequence"], status[index]["state"], detail, flush=True)
            pending.task_done()

    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in GPUS]
        for future in futures:
            future.result()
    if any(item["state"] != "complete" for item in status):
        return 1
    for split in ("test_1", "test_2", None):
        print("Aggregate:", aggregate(jobs, split), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
