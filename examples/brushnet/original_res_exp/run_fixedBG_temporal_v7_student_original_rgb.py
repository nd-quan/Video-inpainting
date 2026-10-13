#!/usr/bin/env python3
"""Run native-size fixedBG + RGB temporal V7 eval across six GPUs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
from datetime import datetime


BRUSHNET = Path(__file__).resolve().parents[2]
DATASET = BRUSHNET / "examples/brushnet/dataset/original_size"
OUTPUT = BRUSHNET / "experiments/eval_fixedBG_temporal_v7_student_original_rgb_s1e-4_steps25-34"
EVALUATOR = Path(__file__).with_name("test_brushnet_fixedBG_temporal_v7_student_original_size.py")
PYTHON = Path("/home/cilab/ndquan/envs/guided_diff/bin/python")
STUDENT = BRUSHNET / "experiments/train_v7_raft_student_flow/checkpoint-0004750/raft_student"
GPUS = tuple(range(6))


def write_status(status_path: Path, records: list[dict]) -> None:
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    temporary.replace(status_path)


def main() -> int:
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    jobs = manifest["sequences"]
    if len(jobs) != 13 or any(job["frames"] != 5 * job["fps"] for job in jobs):
        raise ValueError("Expected thirteen native-size sequences of exactly five seconds")
    if not STUDENT.is_dir() or not EVALUATOR.is_file():
        raise FileNotFoundError("Evaluator or V7 student checkpoint is missing")

    log_dir = OUTPUT / "logs_original_size_rgb_s1e-4_steps25-34"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.json"
    records = []
    for job in jobs:
        output_dir = OUTPUT / job["split"] / job["sequence"]
        existing_frames = len(list(output_dir.glob("*.png")))
        if existing_frames > job["frames"]:
            raise ValueError(f"Too many output frames in {output_dir}: {existing_frames}")
        records.append({
            "sequence": job["sequence"], "split": job["split"],
            "frames": job["frames"], "size_wh": [job["width"], job["height"]],
            "state": "complete" if existing_frames == job["frames"] else "queued",
            "existing_frames": existing_frames,
        })
    write_status(status_path, records)
    work = queue.Queue()
    for index, record in enumerate(records):
        if record["state"] == "queued":
            work.put(index)
    from threading import Lock
    lock = Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                index = work.get_nowait()
            except queue.Empty:
                return
            job = jobs[index]
            label = job["sequence"]
            with lock:
                records[index].update(state="running", gpu=gpu,
                                      started_at=datetime.now().isoformat(timespec="seconds"))
                write_status(status_path, records)
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "LONG_TEST_ROOT": str(DATASET / job["split"]),
                "OUTPUT_ROOT": str(OUTPUT),
                "SEQUENCES": label,
                "MAX_IMAGES": "0",
                "PREFLIGHT_ONLY": "0",
                "FORCE_REGENERATE": "0",
                "GUIDANCE_MODE": "temporal",
                "TEMPORAL_FLOW_BACKEND": "v7_student",
                "V7_RAFT_STUDENT_PATH": str(STUDENT),
                "TEMPORAL_BG_MASK_MODE": "full_stable",
                "TEMPORAL_GUIDANCE_SPACE": "rgb",
                "TEMPORAL_VAE_GRADIENT_CHECKPOINTING": "1" if max(job["width"], job["height"]) >= 2000 else "0",
                "TEMPORAL_VAE_TILING": "1" if max(job["width"], job["height"]) >= 2000 else "0",
                "TEMPORAL_GUIDANCE_SCALE": "0.0001",
                "TEMPORAL_START_STEP": "25",
                "TEMPORAL_END_STEP": "35",
                "TEMPORAL_EVERY_N_STEPS": "1",
                "TEMPORAL_CLIP_SIZE": "4",
                "NUM_INFERENCE_STEPS": "50",
                "SHARED_BG_NOISE_STRENGTH": "1",
                "VARIANCE_PRESERVING_SHARED_NOISE": "0",
                "TEMPORAL_FLOW_BATCH_SIZE": "2",
                "MAIN_NOISE_SEED": "1234",
                "SHARED_NOISE_SEED": "6789",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(BRUSHNET / "examples/brushnet/proposed")
                + os.pathsep + env.get("PYTHONPATH", ""),
            })
            log_path = log_dir / f"{job['split']}_{label}_gpu{gpu}.log"
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"\n=== {datetime.now().isoformat()} GPU {gpu} {label} ===\n")
                stream.flush()
                result = subprocess.run([str(PYTHON), "-u", str(EVALUATOR)],
                                        cwd=BRUSHNET, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, check=False)
            with lock:
                records[index].update(state="complete" if result.returncode == 0 else "failed",
                                      exit_code=result.returncode, log=str(log_path),
                                      finished_at=datetime.now().isoformat(timespec="seconds"))
                write_status(status_path, records)
            print(f"GPU {gpu} {label}: exit {result.returncode}", flush=True)
            work.task_done()

    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in GPUS]
        for future in futures:
            future.result()
    return 0 if all(record["state"] == "complete" for record in records) else 1


if __name__ == "__main__":
    sys.exit(main())
