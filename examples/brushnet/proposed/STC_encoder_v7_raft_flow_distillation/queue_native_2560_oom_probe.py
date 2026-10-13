#!/usr/bin/env python3
"""Wait for GPUs 1 and 2, then run a native Traffic teacher and two-step student probe."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
BRUSHNET = HERE.parents[3]
EXPERIMENT = BRUSHNET / "experiments/v7_student_native_2560_oom_probe"
DATASET = EXPERIMENT / "dataset"
TEACHER = DATASET / "teacher_flows_2560x1600"
TRAIN_OUTPUT = EXPERIMENT / "train"
PYTHON = Path("/home/cilab/ndquan/envs/guided_diff/bin/python")
PRECOMPUTE = HERE.parent / "STC_encoder_v1/precompute_sfu_stc_teacher_flows.py"
RUNNER = HERE / "run_train_raft_student_flow.sh"
INITIAL = BRUSHNET / "experiments/train_v7_raft_student_flow_constant_from4750/checkpoint-0007500"
STATUS = EXPERIMENT / "status.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(state: str, **details) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state": state, "updated_utc": now(), **details}
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATUS)


def gpu_snapshot():
    result = subprocess.run(
        ["nvidia-smi", "-i", "1,2", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    snapshot = {}
    for line in result.stdout.splitlines():
        index, memory, utilization = (int(value.strip()) for value in line.split(","))
        snapshot[index] = {"memory_mib": memory, "utilization_percent": utilization}
    if set(snapshot) != {1, 2}:
        raise RuntimeError(f"Unexpected GPU query: {result.stdout!r}")
    return snapshot


def wait_until_both_idle(poll_seconds: int, stable_checks: int, memory_limit_mib: int):
    stable = 0
    while stable < stable_checks:
        try:
            snapshot = gpu_snapshot()
            idle = all(
                value["memory_mib"] < memory_limit_mib and value["utilization_percent"] <= 5
                for value in snapshot.values()
            )
            stable = stable + 1 if idle else 0
            write_status("waiting_for_gpus", stable_checks=stable, required_checks=stable_checks,
                         gpu_snapshot=snapshot)
        except (OSError, ValueError, subprocess.CalledProcessError, RuntimeError) as exc:
            stable = 0
            write_status("waiting_for_gpus", stable_checks=0, gpu_query_error=str(exc))
        if stable < stable_checks:
            time.sleep(poll_seconds)
    return snapshot


def run_stage(name: str, command, env) -> None:
    log = EXPERIMENT / f"{name}.log"
    write_status(f"running_{name}", command=command, log=str(log), gpu_snapshot=gpu_snapshot())
    print(f"{now()} Starting {name}: {command}", flush=True)
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=str(BRUSHNET), env=env,
                                stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        tail = log.read_text(errors="replace")[-100000:].lower()
        oom = "out of memory" in tail or "cuda error: out of memory" in tail
        write_status(f"{name}_oom" if oom else f"{name}_failed",
                     exit_code=result.returncode, log=str(log))
        raise RuntimeError(f"{name} exited {result.returncode}; see {log}")
    print(f"{now()} Completed {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--memory-limit-mib", type=int, default=1024)
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.stable_checks < 1 or args.memory_limit_mib < 1:
        parser.error("poll-seconds, stable-checks and memory-limit-mib must be positive")
    for path in (PYTHON, PRECOMPUTE, RUNNER, DATASET / "manifest.json",
                 INITIAL / "raft_student/pytorch_model.bin"):
        if not path.is_file():
            raise FileNotFoundError(path)
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    with (EXPERIMENT / "queue.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A native 2560 probe queue is already running")
        prior = json.loads(STATUS.read_text()) if STATUS.is_file() else {}
        if prior.get("state") == "complete":
            raise SystemExit("Probe already completed; refusing to run it twice")
        write_status("queued", watcher_pid=os.getpid(), gpus=[1, 2])
        wait_until_both_idle(args.poll_seconds, args.stable_checks, args.memory_limit_mib)
        teacher_env = os.environ.copy()
        teacher_env["CUDA_VISIBLE_DEVICES"] = "1"
        teacher_command = [
            str(PYTHON), "-u", str(PRECOMPUTE),
            "--dataset-root", str(DATASET), "--output-root", str(TEACHER),
            "--height", "1600", "--width", "2560", "--iters", "20",
            "--device", "cuda", "--splits", "train", "valid", "--resume",
        ]
        run_stage("teacher", teacher_command, teacher_env)
        wait_until_both_idle(args.poll_seconds, args.stable_checks, args.memory_limit_mib)
        train_env = os.environ.copy()
        train_env.update({
            "CUDA_VISIBLE_DEVICES": "1,2", "NUM_PROCESSES": "2",
            "MAIN_PROCESS_PORT": "29671", "DATASET_ROOT": str(DATASET),
            "TEACHER_FLOW_ROOT": str(TEACHER), "OUTPUT_DIR": str(TRAIN_OUTPUT),
            "TRAIN_BATCH_SIZE": "1", "RAFT_PAIR_BATCH_SIZE": "1",
            "MAX_TRAIN_STEPS": "2", "CHECKPOINTING_STEPS": "250",
            "VALIDATION_STEPS": "250", "DATALOADER_NUM_WORKERS": "0",
            "RAFT_ITERATIONS": "20", "RUN_PREFLIGHT": "1",
        })
        train_command = [
            "bash", str(RUNNER), "--spatial_mode", "native", "--memory_probe",
            "--init_from_checkpoint", str(INITIAL),
        ]
        run_stage("student_train", train_command, train_env)
        write_status("complete", teacher_log=str(EXPERIMENT / "teacher.log"),
                     train_log=str(EXPERIMENT / "student_train.log"))
        print(f"{now()} Probe complete", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"{now()} {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
