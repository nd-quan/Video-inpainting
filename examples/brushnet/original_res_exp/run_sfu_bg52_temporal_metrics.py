#!/usr/bin/env python3
"""Run the comparison-table PNG metrics on all SFU BG52 test sequences."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
from threading import Lock

import numpy as np
from PIL import Image


BRUSHNET = Path(__file__).resolve().parents[2]
DATASET = Path("/home/cilab/ndquan/vcm-rs/output/sfu_qp0_bg_table_window_test/bg52/test")
EVAL = BRUSHNET / "experiments/eval_sfu_bg52_table_window_ckpt1000_rgb_tg"
OUTPUT = EVAL / "metrics_temporal_png"
SCRIPT = BRUSHNET / "examples/brushnet/proposed/DGAF_VSR_original/postprocess_eval.py"
EVALUATOR_ROOT = BRUSHNET.parent / "video-inpainting-evaluation"
PYTHON = Path("/home/cilab/ndquan/envs/guided_diff/bin/python")
GPUS = tuple(range(6))
METRICS = ("PSNR", "SSIM", "LPIPS", "WE", "WE_L2", "FS", "VFID_Clips")
FIELDS = ("sequence", "class", "frames", "pairs", "PSNR", "SSIM", "LPIPS",
          "WE", "WE_L2", "WE-L2 (×10⁻²)", "FS", "VFID_Clips")


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sequences() -> list[dict]:
    status = json.loads((EVAL / "logs/status.json").read_text(encoding="utf-8"))
    if len(status) != 14 or any(item["state"] != "complete" for item in status):
        raise ValueError("Expected 14 completed BG52 evaluations")
    jobs = []
    for item in status:
        name, cls = item["sequence"], item["class"]
        pred_dir = EVAL / "test" / cls / name
        gt_dir = DATASET / "GT" / cls / name
        pred = {int(p.stem): p for p in pred_dir.glob("*.png")}
        gt = {int(p.stem): p for p in gt_dir.glob("*.png")}
        if set(pred) != set(gt) or len(pred) != item["frame_count"]:
            raise ValueError(f"Prediction/GT frame IDs differ for {name}")
        ids = sorted(pred)
        if ids != list(range(item["frame_range"][0], item["frame_range"][1] + 1)):
            raise ValueError(f"Non-contiguous source frames for {name}")
        for frame_id in (ids[0], ids[-1]):
            with Image.open(pred[frame_id]) as a, Image.open(gt[frame_id]) as b:
                if a.size != b.size or list(a.size) != item["size_wh"]:
                    raise ValueError(f"Prediction/GT size differs for {name} frame {frame_id}")
        jobs.append({"sequence": name, "class": cls, "ids": ids,
                     "pred_dir": pred_dir, "gt_dir": gt_dir,
                     "pixels": item["size_wh"][0] * item["size_wh"][1] * len(ids)})
    if sum(len(job["ids"]) for job in jobs) != 1198:
        raise ValueError("Expected 1198 BG52 test frames")
    return jobs


def case_dir(job: dict) -> Path:
    return OUTPUT / "cases" / job["sequence"]


def case_result(job: dict) -> dict | None:
    directory = case_dir(job)
    metric_csv = directory / "metrics.csv"
    if not metric_csv.is_file():
        return None
    try:
        with metric_csv.open(newline="", encoding="utf-8") as stream:
            row = next(row for row in csv.DictReader(stream) if row["sequence"] == job["sequence"])
        if int(row["frames"]) != len(job["ids"]) or int(row["pairs"]) != len(job["ids"]) - 1:
            return None
        if any(not math.isfinite(float(row[key])) for key in METRICS):
            return None
        if not all((directory / f"{job['sequence']}_{kind}_i3d.npy").is_file()
                   for kind in ("gt", "final")):
            return None
        return row
    except (OSError, StopIteration, KeyError, ValueError):
        return None


def run_one(job: dict, gpu: int) -> str:
    if case_result(job) is not None:
        return "already complete"
    name = job["sequence"]
    with tempfile.TemporaryDirectory(prefix=f"sfu_bg52_metric_{name}_") as dirname:
        stage = Path(dirname)
        sequence = stage / name
        for kind, folder in (("final", job["pred_dir"]), ("gt", job["gt_dir"])):
            target = sequence / kind
            target.mkdir(parents=True)
            for frame_id in job["ids"]:
                (target / f"{frame_id:06d}.png").symlink_to(
                    (folder / f"{frame_id:06d}.png").resolve()
                )
        (sequence / "clip_metrics.json").write_text(
            json.dumps({"video": name, "frame_ids": job["ids"]}), encoding="utf-8"
        )
        (stage / "summary.json").write_text('{"clips": 1}', encoding="utf-8")
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4"})
        log = OUTPUT / "logs" / f"{name}_gpu{gpu}.log"
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\n=== {datetime.now().isoformat()} GPU {gpu} {name} ===\n")
            stream.flush()
            result = subprocess.run(
                [str(PYTHON), "-u", str(SCRIPT), "--eval_root", str(stage),
                 "--evaluator_root", str(EVALUATOR_ROOT)],
                cwd=BRUSHNET, env=env, stdout=stream, stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"{name} exited {result.returncode}; see {log}")
        source = stage / "metrics"
        if not (source / "metrics.csv").is_file():
            raise RuntimeError(f"No metric CSV for {name}; see {log}")
        destination = case_dir(job)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(source), str(destination))
        save_json(destination / "source.json", {
            "sequence": name, "class": job["class"], "frames": len(job["ids"]),
            "pred_dir": str(job["pred_dir"]), "gt_dir": str(job["gt_dir"]),
            "evaluator": str(SCRIPT), "metric_domain": "native 8-bit PNG",
            "gpu": gpu,
        })
    if case_result(job) is None:
        raise RuntimeError(f"Invalid metric result for {name}")
    return str(case_dir(job) / "metrics.csv")


def summarize(jobs: list[dict]) -> None:
    sys.path.insert(0, str(BRUSHNET / "Quan_test"))
    from compute_vfid_clips_4datasets import frechet_distance_low_rank

    rows = []
    features = {kind: [] for kind in ("gt", "final")}
    for job in sorted(jobs, key=lambda item: item["sequence"]):
        raw = case_result(job)
        if raw is None:
            raise ValueError(f"Missing valid metrics for {job['sequence']}")
        item = {"sequence": job["sequence"], "class": job["class"],
                "frames": int(raw["frames"]), "pairs": int(raw["pairs"])}
        item.update({key: float(raw[key]) for key in METRICS})
        item["WE-L2 (×10⁻²)"] = item["WE_L2"] * 100
        rows.append(item)
        for kind in features:
            features[kind].append(np.load(case_dir(job) / f"{job['sequence']}_{kind}_i3d.npy"))

    with (OUTPUT / "per_sequence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    mp4_metrics = EVAL / "metrics/rd_metrics.csv"
    with mp4_metrics.open(newline="", encoding="utf-8") as stream:
        mp4_rows = {row["sequence"]: row for row in csv.DictReader(stream)}
    if set(mp4_rows) != {row["sequence"] for row in rows}:
        raise ValueError("MP4 and PNG metric sequences differ")
    combined_fields = (
        "sequence", "class", "frames", "fps", "MP4_PSNR", "MP4_SSIM",
        "MP4_MS-SSIM", "MP4_LPIPS", "MP4_Frame-level_FID", "MP4_FVD",
        "PNG_PSNR", "PNG_SSIM", "PNG_LPIPS", "PNG_WE", "PNG_WE_L2",
        "PNG_WE-L2 (×10⁻²)", "PNG_FS", "PNG_VFID_Clips",
    )
    with (OUTPUT / "combined_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=combined_fields)
        writer.writeheader()
        for row in rows:
            previous = mp4_rows[row["sequence"]]
            if int(previous["frames"]) != row["frames"]:
                raise ValueError(f"Frame count differs for {row['sequence']}")
            writer.writerow({
                "sequence": row["sequence"], "class": row["class"],
                "frames": row["frames"], "fps": previous["fps"],
                **{f"MP4_{metric.replace(' ', '_')}": previous[metric]
                   for metric in ("PSNR", "SSIM", "MS-SSIM", "LPIPS", "Frame-level FID", "FVD")},
                **{f"PNG_{metric}": row[metric]
                   for metric in ("PSNR", "SSIM", "LPIPS", "WE", "WE_L2",
                                  "WE-L2 (×10⁻²)", "FS", "VFID_Clips")},
            })

    macro = {"aggregation": "sequence_macro_mean", "sequences": len(rows),
             "frames": sum(row["frames"] for row in rows),
             "pairs": sum(row["pairs"] for row in rows)}
    weighted = dict(macro, aggregation="frame_pair_weighted_mean")
    for key in METRICS:
        macro[key] = float(np.mean([row[key] for row in rows]))
        if key != "VFID_Clips":
            weight = "pairs" if key in ("WE", "WE_L2", "FS") else "frames"
            weighted[key] = float(np.average([row[key] for row in rows],
                                             weights=[row[weight] for row in rows]))
    macro["WE-L2 (×10⁻²)"] = macro["WE_L2"] * 100
    weighted["WE-L2 (×10⁻²)"] = weighted["WE_L2"] * 100
    weighted["VFID_Clips"] = float(frechet_distance_low_rank(
        np.concatenate(features["gt"]), np.concatenate(features["final"])
    ))
    summary_fields = ("aggregation", "sequences", "frames", "pairs", *FIELDS[4:])
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows([macro, weighted])
    save_json(OUTPUT / "summary.json", {
        "results": [macro, weighted],
        "definitions": {
            "source": str(SCRIPT), "metric_domain": "native 8-bit prediction/GT PNG",
            "WE_WE_L2": "Farneback backward flow; valid-pixel L1 and squared L2 on adjacent predictions",
            "FS": "SSIM between adjacent predicted frames",
            "VFID_Clips": "I3D rgb_imagenet, 10-frame sliding windows, 224x224 bicubic",
            "sequence_macro_mean": "Unweighted arithmetic mean of 14 per-sequence scores, as in the comparison table",
            "frame_pair_weighted_mean": "Image metrics weighted by frames; temporal metrics by pairs; VFID computed on pooled I3D features",
            "WE_L2_display_multiplier": 100,
        },
    })
    print(f"[SUMMARY] {len(rows)} sequences, {macro['frames']} frames, {macro['pairs']} pairs",
          flush=True)


def main() -> int:
    for path in (SCRIPT, EVALUATOR_ROOT / "pretrained_models/rgb_imagenet.pt", PYTHON):
        if not path.exists():
            raise FileNotFoundError(path)
    jobs = sequences()
    (OUTPUT / "logs").mkdir(parents=True, exist_ok=True)
    status_path = OUTPUT / "status.json"
    status = [{"sequence": job["sequence"], "frames": len(job["ids"]),
               "state": "complete" if case_result(job) is not None else "queued"}
              for job in jobs]
    save_json(status_path, status)
    pending: queue.Queue[int] = queue.Queue()
    for index in sorted(range(len(jobs)), key=lambda i: jobs[i]["pixels"], reverse=True):
        if status[index]["state"] == "queued":
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
                detail = run_one(jobs[index], gpu)
                state = "complete"
            except Exception as exc:
                detail, state = repr(exc), "failed"
            with lock:
                status[index].update(state=state, detail=detail,
                                     finished_at=datetime.now().isoformat(timespec="seconds"))
                save_json(status_path, status)
            print(f"[{state.upper()}] GPU {gpu} {jobs[index]['sequence']}: {detail}", flush=True)
            pending.task_done()

    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in GPUS]
        for future in futures:
            future.result()
    if any(item["state"] != "complete" for item in status):
        return 1
    summarize(jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
