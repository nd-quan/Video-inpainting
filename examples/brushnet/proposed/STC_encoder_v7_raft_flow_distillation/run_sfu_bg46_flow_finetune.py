#!/usr/bin/env python3
"""Cache SFU QP46 clean RAFT teachers, probe 960-wide training, then finetune."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET = THIS_DIR.parents[3]
EXPERIMENT = BRUSHNET / "experiments/train_v7_raft_student_sfu_bg46_max960_4000"
DATASET = EXPERIMENT / "dataset"
TEACHER = EXPERIMENT / "teacher_flows"
STUDENT = EXPERIMENT / "student"
INITIAL = BRUSHNET / "experiments/train_v7_raft_student_flow_constant_from4750/checkpoint-0007500"
PRECOMPUTE = THIS_DIR.parent / "STC_encoder_v1/precompute_sfu_stc_teacher_flows.py"
GPU_IDS = ("3", "4", "5")
RAFT_CHECKPOINT = Path("/home/cilab/ndquan/videoInpainting/pretrained/ProPainter/weights/raft-things.pth")


def status(phase, **kwargs):
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    payload = {"phase": phase, "updated_at": datetime.now().isoformat(), **kwargs}
    destination = EXPERIMENT / "status.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps(payload), flush=True)


def launch(command, log_path, gpu):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"
    handle = log_path.open("a")
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
    handle.close()
    return process


def wait_success(process, log_path):
    code = process.wait()
    if code:
        raise RuntimeError(f"Command exited {code}; see {log_path}")


def main():
    manifest = json.loads((DATASET / "manifest.json").read_text())
    expected = sum(
        sequence["splits"][split]["frame_count"] - 1
        for sequence in manifest["sequences"] for split in ("train", "valid")
    )
    if not INITIAL.joinpath("raft_student/pytorch_model.bin").is_file():
        raise FileNotFoundError(INITIAL)
    common_teacher = [sys.executable, "-u", str(PRECOMPUTE),
        "--dataset-root", str(DATASET), "--output-root", str(TEACHER),
        "--splits", "train", "valid", "--max-input-side", "960",
        "--iters", "20", "--resume", "--num-shards", str(len(GPU_IDS))]
    cached = sum(1 for _ in (TEACHER / "train").rglob("*.npz")) + sum(
        1 for _ in (TEACHER / "valid").rglob("*.npz")
    )
    if cached != expected:
        status("teacher_running", cached_pairs=cached, expected_pairs=expected, gpus=list(GPU_IDS))
        jobs = []
        for shard, gpu in enumerate(GPU_IDS):
            log = EXPERIMENT / "logs" / f"teacher_gpu{gpu}.log"
            process = launch(common_teacher + ["--shard-index", str(shard)], log, gpu)
            jobs.append((process, log))
        for process, log in jobs:
            wait_success(process, log)
        cached = sum(1 for _ in (TEACHER / "train").rglob("*.npz")) + sum(
            1 for _ in (TEACHER / "valid").rglob("*.npz")
        )
    if cached != expected:
        raise RuntimeError(f"Teacher cache incomplete: {cached}/{expected}")
    status("teacher_complete", cached_pairs=cached, expected_pairs=expected)

    # Traffic is the largest flow input: 2560x1600 -> 960x600.
    probe_log = EXPERIMENT / "logs/probe_960x600.log"
    probe = [sys.executable, "-u", str(THIS_DIR / "train_raft_student_flow.py"),
        "--dataset_root", str(DATASET), "--teacher_flow_root", str(TEACHER),
        "--output_dir", str(EXPERIMENT / "probe_960x600"),
        "--raft_checkpoint", str(RAFT_CHECKPOINT),
        "--spatial_mode", "max_side", "--max_input_side", "960",
        "--include_branches", "Class_A/Traffic", "--valid_pairs_per_sequence", "1",
        "--init_from_checkpoint", str(INITIAL), "--max_train_steps", "2",
        "--memory_probe", "--mixed_precision", "fp16"]
    status("probe_running", cache_pairs=cached, gpu=GPU_IDS[0])
    wait_success(launch(probe, probe_log, GPU_IDS[0]), probe_log)
    status("probe_complete", cache_pairs=cached)

    wrapper = THIS_DIR / "run_train_raft_student_flow.sh"
    log = EXPERIMENT / "logs/train_4000.log"
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(GPU_IDS), "NUM_PROCESSES": str(len(GPU_IDS)),
        "MAIN_PROCESS_PORT": "29646", "DATASET_ROOT": str(DATASET),
        "TEACHER_FLOW_ROOT": str(TEACHER), "OUTPUT_DIR": str(STUDENT),
        "MAX_TRAIN_STEPS": "4000", "CHECKPOINTING_STEPS": "250",
        "VALIDATION_STEPS": "250", "CHECKPOINTS_TOTAL_LIMIT": "16",
        "LEARNING_RATE": "5e-6", "LR_SCHEDULER": "constant",
        "LR_WARMUP_STEPS": "100", "PYTHONUNBUFFERED": "1",
    })
    existing = sorted(STUDENT.glob("checkpoint-*/raft_student/pytorch_model.bin"))
    if existing:
        env["RESUME_FROM_CHECKPOINT"] = "latest"
        additional = ["--spatial_mode", "max_side", "--max_input_side", "960"]
    else:
        additional = ["--spatial_mode", "max_side", "--max_input_side", "960",
                      "--init_from_checkpoint", str(INITIAL)]
    log.parent.mkdir(parents=True, exist_ok=True)
    status("training_running", gpus=list(GPU_IDS), target_steps=4000,
           learning_rate="5e-6", student_dir=str(STUDENT))
    with log.open("a") as handle:
        code = subprocess.call(["bash", str(wrapper), *additional], stdout=handle,
                               stderr=subprocess.STDOUT, env=env)
    if code:
        raise RuntimeError(f"Training exited {code}; see {log}")
    if not (STUDENT / "checkpoint-0004000/raft_student/pytorch_model.bin").is_file():
        raise RuntimeError("Training exited without checkpoint-0004000")
    status("complete", checkpoint=str(STUDENT / "checkpoint-0004000"))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", error=str(error))
        raise
