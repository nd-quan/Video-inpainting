#!/usr/bin/env python3
"""Sequential SFU RAFT-student finetunes; each QP starts from previous best."""

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET = THIS_DIR.parents[3]
EXPERIMENTS = BRUSHNET / "experiments"
QUEUE = EXPERIMENTS / "train_v7_raft_student_sfu_qp48_52_56_queue"
RUNS = {
    46: EXPERIMENTS / "train_v7_raft_student_sfu_bg46_max960_4000",
    48: EXPERIMENTS / "train_v7_raft_student_sfu_bg48_max960_from_bg46_3500_4000",
    52: EXPERIMENTS / "train_v7_raft_student_sfu_bg52_max960_from_previous_best_4000",
    56: EXPERIMENTS / "train_v7_raft_student_sfu_bg56_max960_from_previous_best_4000",
}
GPU_IDS = (4, 5)
TARGET_STEPS = 4000


def write_status(phase, **details):
    QUEUE.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase, "updated_at": datetime.now().isoformat(),
        "order": [48, 52, 56], "gpus": list(GPU_IDS), **details,
    }
    target = QUEUE / "status.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(target)
    print(json.dumps(payload), flush=True)


def best_checkpoint(qp):
    student = RUNS[qp] / "student"
    path = student / "best.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    record = json.loads(path.read_text())
    checkpoint = student / record["checkpoint"]
    if not (checkpoint / "raft_student/pytorch_model.bin").is_file():
        raise FileNotFoundError(checkpoint / "raft_student/pytorch_model.bin")
    return checkpoint, record


def latest_step(student):
    path = student / "latest.json"
    if not path.is_file():
        return 0
    record = json.loads(path.read_text())
    return int(record["global_step"])


def validate_run(qp):
    root = RUNS[qp]
    summary = json.loads((root / "dataset/manifest.json").read_text())
    source = Path(summary["source_dataset"])
    source_summary = json.loads((source / "summary.json").read_text())
    cache = json.loads((root / "teacher_flows/metadata.json").read_text())
    if source_summary["bg_qp"] != qp or len(summary["sequences"]) != source_summary["sequences"]:
        raise ValueError(f"QP{qp} dataset manifest/source mismatch")
    if cache["dataset_root"] != str((root / "dataset").resolve()):
        raise ValueError(f"QP{qp} teacher cache belongs to a different dataset")
    if cache.get("spatial_mode") != "max_side" or cache.get("max_input_side") != 960:
        raise ValueError(f"QP{qp} teacher geometry mismatch")
    for sequence in summary["sequences"]:
        branch = Path(sequence["class"]) / sequence["name"]
        for split in ("train", "valid"):
            for segment in sequence["splits"][split]["segments"]:
                for index in range(segment["start"], segment["end"]):
                    path = root / "teacher_flows" / split / branch / f"{index:06d}_{index + 1:06d}.npz"
                    if not path.is_file():
                        raise FileNotFoundError(path)
    return source_summary["sequences"]


def free_gpus():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    values = [int(value.strip().split()[0]) for value in result.splitlines()]
    return all(values[index] < 2048 for index in GPU_IDS), [values[index] for index in GPU_IDS]


def current_step(log_path):
    if not log_path.is_file():
        return 0
    with log_path.open("rb") as stream:
        stream.seek(max(0, log_path.stat().st_size - 64 * 1024))
        tail = stream.read().decode("utf-8", "replace")
    steps = re.findall(r'^TRAIN .*"step": (\d+)', tail, re.MULTILINE)
    return int(steps[-1]) if steps else 0


def main():
    completed = {}
    for qp in (48, 52, 56):
        sequences = validate_run(qp)
        previous = {48: 46, 52: 48, 56: 52}[qp]
        initial, initial_record = best_checkpoint(previous)
        root = RUNS[qp]
        student = root / "student"
        final_weights = student / "checkpoint-0004000/raft_student/pytorch_model.bin"
        if latest_step(student) == TARGET_STEPS and final_weights.is_file():
            chosen, record = best_checkpoint(qp)
            completed[str(qp)] = {"best_checkpoint": str(chosen),
                                  "bg_flow_epe": record.get("bg_flow_epe")}
            write_status("stage_complete", qp=qp, completed=completed, skipped_existing=True)
            continue
        if latest_step(student) > TARGET_STEPS:
            raise ValueError(f"QP{qp} student already exceeds {TARGET_STEPS} steps")

        ready, memory = free_gpus()
        while not ready:
            write_status("waiting_for_gpus", qp=qp, gpu_memory_mib=memory,
                         completed=completed, initialization_checkpoint=str(initial))
            time.sleep(30)
            ready, memory = free_gpus()

        log = root / "logs/train_4000.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, GPU_IDS)),
            "NUM_PROCESSES": str(len(GPU_IDS)),
            "MAIN_PROCESS_PORT": "29648",
            "DATASET_ROOT": str((root / "dataset").resolve()),
            "TEACHER_FLOW_ROOT": str((root / "teacher_flows").resolve()),
            "OUTPUT_DIR": str(student.resolve()),
            "MAX_TRAIN_STEPS": str(TARGET_STEPS),
            "CHECKPOINTING_STEPS": "250", "VALIDATION_STEPS": "250",
            "CHECKPOINTS_TOTAL_LIMIT": "16",
            "LEARNING_RATE": "5e-6", "LR_SCHEDULER": "constant",
            "LR_WARMUP_STEPS": "100", "PYTHONUNBUFFERED": "1",
        })
        if latest_step(student):
            env["RESUME_FROM_CHECKPOINT"] = "latest"
            extra = []
        else:
            extra = ["--init_from_checkpoint", str(initial)]
        command = ["bash", str(THIS_DIR / "run_train_raft_student_flow.sh"),
                   "--spatial_mode", "max_side", "--max_input_side", "960", *extra]
        write_status("training", qp=qp, step=latest_step(student),
                     sequence_count=sequences, initialization_checkpoint=str(initial),
                     initial_bg_flow_epe=initial_record.get("bg_flow_epe"),
                     completed=completed, log=str(log))
        with log.open("a") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
            while process.poll() is None:
                time.sleep(30)
                write_status("training", qp=qp, step=max(latest_step(student), current_step(log)),
                             sequence_count=sequences, initialization_checkpoint=str(initial),
                             completed=completed, log=str(log))
            code = process.wait()
        if code:
            raise RuntimeError(f"QP{qp} training exited {code}; see {log}")
        if latest_step(student) != TARGET_STEPS or not final_weights.is_file():
            raise RuntimeError(f"QP{qp} exited without final checkpoint")
        chosen, record = best_checkpoint(qp)
        completed[str(qp)] = {"best_checkpoint": str(chosen),
                              "bg_flow_epe": record.get("bg_flow_epe")}
        write_status("stage_complete", qp=qp, completed=completed)
    write_status("complete", completed=completed)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_status("failed", error=str(error))
        raise
