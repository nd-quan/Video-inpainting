#!/usr/bin/env python3
"""Run the existing temporal-v0 evaluator on the complete flat test_1 split."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "examples/brushnet/dataset/test_1"
CHECKPOINT = ROOT / "experiments/train_sharedNoise_sameBG_0.95_T8/checkpoint-2250"
STUDENT = ROOT / "experiments/train_v7_raft_student_flow/checkpoint-0004750/raft_student"
EVALUATOR = ROOT / "examples/brushnet/test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion_fixedBG_temporal_v0.py"
METRICS = ROOT / "Quan_test/evaluate_flat_sequence_metrics.py"
OUTPUT = ROOT / "experiments/eval_test1_sharednoise2250_latent_tg0p1_v7student_20261021"
GPUS = (3, 4, 5)
SEQUENCES = ("BasketballPass", "PartyScene", "ParkScene", "RaceHorses", "Traffic")


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    for path in (DATASET, CHECKPOINT / "brushnet", CHECKPOINT / "ipadapter",
                 STUDENT, EVALUATOR, METRICS):
        if not path.exists():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logs = OUTPUT / "terminal_logs"
    logs.mkdir(exist_ok=True)
    save_json(OUTPUT / "run_config.json", {
        "dataset": str(DATASET), "checkpoint": str(CHECKPOINT),
        "flow_checkpoint": str(STUDENT), "evaluator": str(EVALUATOR),
        "gpus": list(GPUS), "sequences": list(SEQUENCES), "max_images": 0,
        "guidance_mode": "temporal", "temporal_guidance_space": "latent",
        "temporal_flow_backend": "v7_student", "temporal_guidance_scale": 0.1,
        "temporal_window": [0, 50], "temporal_every_n_steps": 1,
        "temporal_clip_size": 4, "temporal_bg_mask_mode": "pair_intersection",
        "temporal_detach_previous": True, "inference_steps": 50,
        "shared_bg_noise_strength": 1.0,
        "variance_preserving_shared_noise": True,
        "main_noise_seed": 1234, "shared_noise_seed": 6789,
    })
    expected = {}
    for sequence in SEQUENCES:
        source = DATASET / sequence / "inputs"
        masks = DATASET / sequence / "masks"
        image_names = {path.name for path in source.glob("*.png")}
        mask_names = {path.name for path in masks.glob("*.png")}
        if not image_names or image_names != mask_names:
            raise ValueError("Input/mask mismatch: " + sequence)
        expected[sequence] = image_names

    status_path = OUTPUT / "status.json"
    status = {sequence: {"expected_frames": len(expected[sequence]), "state": "queued"}
              for sequence in SEQUENCES}
    save_json(status_path, status)
    jobs = Queue()
    for sequence in SEQUENCES:
        jobs.put(sequence)
    lock = Lock()

    def worker(gpu):
        while True:
            try:
                sequence = jobs.get_nowait()
            except Empty:
                return
            output = OUTPUT / "test_1" / sequence
            existing = {path.name for path in output.glob("*.png")}
            if existing == expected[sequence]:
                with lock:
                    status[sequence].update(state="complete", existing_frames=len(existing),
                                            reused=True)
                    save_json(status_path, status)
                jobs.task_done()
                continue
            if existing - expected[sequence]:
                raise ValueError("Unexpected output frame: " + sequence)
            log_path = logs / (sequence + "_gpu" + str(gpu) + ".log")
            with lock:
                status[sequence].update(state="running", gpu=gpu, log=str(log_path),
                                        started_at=datetime.now().isoformat(timespec="seconds"))
                save_json(status_path, status)
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu), "LONG_TEST_ROOT": str(DATASET),
                "OUTPUT_ROOT": str(OUTPUT), "SEQUENCES": sequence,
                "MAX_IMAGES": "0", "FORCE_REGENERATE": "0",
                "CHECKPOINT_DIR": str(CHECKPOINT), "TEMPORAL_CLIP_SIZE": "4",
                "NUM_INFERENCE_STEPS": "50", "MAIN_NOISE_SEED": "1234",
                "SHARED_NOISE_SEED": "6789", "SHARED_BG_NOISE_STRENGTH": "1",
                "VARIANCE_PRESERVING_SHARED_NOISE": "1",
                "TEMPORAL_BG_MASK_MODE": "pair_intersection",
                "TEMPORAL_FLOW_BACKEND": "v7_student",
                "V7_RAFT_STUDENT_PATH": str(STUDENT),
                "GUIDANCE_MODE": "temporal", "TEMPORAL_GUIDANCE_SPACE": "latent",
                "TEMPORAL_GUIDANCE_SCALE": "0.1", "TEMPORAL_START_STEP": "0",
                "TEMPORAL_END_STEP": "50", "TEMPORAL_EVERY_N_STEPS": "1",
                "TEMPORAL_DETACH_PREVIOUS": "1", "PREFLIGHT_ONLY": "0",
                "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                "PYTHONPATH": str(ROOT / "examples/brushnet/proposed") + os.pathsep
                + os.environ.get("PYTHONPATH", ""),
            })
            with log_path.open("a", encoding="utf-8") as log:
                log.write("\n=== " + datetime.now().isoformat() + " GPU " + str(gpu)
                          + " " + sequence + " ===\n")
                log.flush()
                result = subprocess.run([sys.executable, "-u", str(EVALUATOR)],
                                        cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            existing = {path.name for path in output.glob("*.png")}
            complete = result.returncode == 0 and existing == expected[sequence]
            with lock:
                status[sequence].update(state="complete" if complete else "failed",
                                        exit_code=result.returncode,
                                        existing_frames=len(existing),
                                        finished_at=datetime.now().isoformat(timespec="seconds"))
                save_json(status_path, status)
            print("GPU {} {}: exit={}, frames={}/{}".format(
                gpu, sequence, result.returncode, len(existing), len(expected[sequence])),
                flush=True)
            jobs.task_done()

    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in GPUS]
        for future in futures:
            future.result()
    if not all(item["state"] == "complete" for item in status.values()):
        return 1
    metric_dir = OUTPUT / "test_1/metrics_full_reference"
    if not metric_dir.exists():
        with (logs / "metrics.log").open("w", encoding="utf-8") as log:
            result = subprocess.run([
                sys.executable, str(METRICS), "--eval-root", str(OUTPUT / "test_1"),
                "--gt-root", str(DATASET), "--sequences", *SEQUENCES,
            ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            print("Metric calculation failed; see terminal_logs/metrics.log", flush=True)
            return result.returncode
    print("Complete: " + str(OUTPUT), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
