#!/usr/bin/env python3
"""Wait for BVI BG48 training, then train BG46 and BG44 in that order.

The script uses the same six-GPU training configuration as BG48. It only starts
the next job after the previous final checkpoint is complete and its processes
have exited. Optional model-only retention preserves evaluation weights at
intermediate steps while keeping the newest checkpoint fully resumable.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

BRUSHNET = Path(__file__).resolve().parents[2]
EXPERIMENTS = BRUSHNET / "experiments"
DATASET = Path("/home/cilab/ndquan/vcm-rs/output/bvi_dvc_part1_qp0_bg_train")
INITIAL = EXPERIMENTS / "finetune_shared_noise_original_size/checkpoint-1000"
PYTHON = Path("/home/cilab/ndquan/envs/guided_diff/bin/python")
WRAPPER = BRUSHNET / "examples/brushnet/run_finetune_shared_noise_original_size.sh"
STATUS = EXPERIMENTS / "bvi_dvc_bg46_bg44_queue_status.json"
LOG = EXPERIMENTS / "bvi_dvc_bg46_bg44_queue.log"
RETENTION_FILE = EXPERIMENTS / "bvi_dvc_bg46_bg44_retention.json"
BG48 = EXPERIMENTS / "finetune_shared_noise_bvi_dvc_bg48_original_size"
FINAL_STEP = 2000


def say(message):
    print(f"[{datetime.now().astimezone().isoformat()}] {message}", flush=True)


def status(phase, **details):
    payload = {"updated_at": datetime.now().astimezone().isoformat(),
               "phase": phase, **details}
    tmp = STATUS.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, STATUS)


def checkpoint_dir(experiment, step=FINAL_STEP):
    return experiment / f"checkpoint-{step}"


def complete_checkpoint(path, expected_step=None):
    required = ("native_training_state.json", "pytorch_model.bin", "optimizer.bin",
                "brushnet/config.json", "brushnet/diffusion_pytorch_model.safetensors",
                "ipadapter/model.safetensors", "ipadapter/fusion_module.safetensors")
    if any(not (path / name).is_file() or (path / name).stat().st_size == 0
           for name in required):
        return False
    try:
        state = json.loads((path / "native_training_state.json").read_text(encoding="utf-8"))
        return (expected_step is None or state["global_step"] == expected_step)
    except (OSError, ValueError, KeyError):
        return False


def active_training(experiment):
    target = str(experiment)
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = process.info["cmdline"] or []
            if ("finetune_shared_noise_original_size.py" in " ".join(command)
                    and target in command):
                return process.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def prune_intermediate(experiment):
    completed = []
    for path in experiment.glob("checkpoint-*"):
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        if complete_checkpoint(path, step):
            completed.append((step, path))
    if len(completed) < 2:
        return 0
    latest_step = max(step for step, _ in completed)
    released = 0
    for step, path in completed:
        if step >= latest_step:
            continue
        # Eval weights remain in brushnet/ and ipadapter/. Only the newest
        # checkpoint retains Accelerate's optimizer and full DDP state.
        for name in ("optimizer.bin", "pytorch_model.bin"):
            file = path / name
            if file.is_file() and not file.is_symlink():
                size = file.stat().st_size
                file.unlink()
                released += size
                say(f"[RETENTION] {experiment.name}/{path.name}/{name}: freed {size/2**30:.2f} GiB")
    return released


def wait_complete(experiment, retention, launched=None):
    missing_polls = 0
    while True:
        if retention == "models_only":
            prune_intermediate(experiment)
        final = checkpoint_dir(experiment)
        process_id = active_training(experiment)
        if complete_checkpoint(final, FINAL_STEP) and process_id is None:
            say(f"[COMPLETE] {experiment.name}: checkpoint-{FINAL_STEP}")
            return
        if launched is not None and launched.poll() is not None:
            if launched.returncode:
                raise RuntimeError(f"{experiment.name} exited {launched.returncode}; inspect {experiment/'train_queue.log'}")
            if not complete_checkpoint(final, FINAL_STEP):
                raise RuntimeError(f"{experiment.name} exited without complete checkpoint-{FINAL_STEP}")
        if process_id is None and not complete_checkpoint(final, FINAL_STEP):
            missing_polls += 1
            if missing_polls >= 3:
                raise RuntimeError(f"{experiment.name} stopped before checkpoint-{FINAL_STEP}")
        else:
            missing_polls = 0
        status("waiting_training", experiment=str(experiment), process_id=process_id,
               final_checkpoint_complete=complete_checkpoint(final, FINAL_STEP))
        time.sleep(30)


def dataset_ready(qp):
    root = DATASET / f"bg{qp}"
    pairs = root / "train_pairs.csv"
    if not pairs.is_file():
        return False
    try:
        with pairs.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 3648 or len({(r["sequence"], r["source_frame"]) for r in rows}) != 3648:
            return False
        if len({r["sequence"] for r in rows}) != 57 or any(int(r["bg_qp"]) != qp for r in rows):
            return False
        return all((root / row[key]).is_file() for row in rows for key in ("input", "gt", "mask"))
    except (OSError, ValueError, KeyError):
        return False


def wait_dataset(qp):
    while not dataset_ready(qp):
        status("waiting_dataset", bg_qp=qp, dataset=str(DATASET / f"bg{qp}"))
        if qp == 44:
            result = subprocess.run(["tmux", "has-session", "-t", "bvi_bg44_pipeline"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode:
                raise RuntimeError("BG44 dataset pipeline stopped without a complete train dataset")
        say(f"[WAIT DATA] BG{qp}: train dataset not complete")
        time.sleep(60)


def preflight(qp):
    command = [str(PYTHON), str(BRUSHNET / "examples/brushnet/finetune_shared_noise_original_size.py"),
               "--dataset_root", str(DATASET / f"bg{qp}"), "--train_split", "train",
               "--clip_length", "6", "--clip_stride", "4", "--max_height", "1088",
               "--max_width", "1920", "--preflight_only"]
    report = json.loads(subprocess.check_output(command, cwd=BRUSHNET, text=True))
    if report["frames"] != 3648 or report["clips"] != 855 or report["resize"]:
        raise RuntimeError(f"BG{qp} preflight differs from BG48: {report}")
    say(f"[PREFLIGHT] BG{qp}: 3648 frames, 855 clips, native size")


def run_train(qp, retention, min_disk_gib):
    experiment = EXPERIMENTS / f"finetune_shared_noise_bvi_dvc_bg{qp}_original_size"
    if complete_checkpoint(checkpoint_dir(experiment), FINAL_STEP):
        say(f"[SKIP] BG{qp}: already completed")
        return
    if active_training(experiment):
        say(f"[WAIT] BG{qp}: training already running")
        wait_complete(experiment, retention)
        return
    if list(experiment.glob("checkpoint-*")):
        raise RuntimeError(f"BG{qp} has partial checkpoints; resume must be handled explicitly: {experiment}")
    wait_dataset(qp)
    preflight(qp)
    required_disk_gib = max(min_disk_gib, 160.0) if retention == "keep_all" else min_disk_gib
    while shutil.disk_usage(BRUSHNET).free / 2**30 < required_disk_gib:
        free = shutil.disk_usage(BRUSHNET).free / 2**30
        status("waiting_disk", bg_qp=qp, free_gib=round(free, 1), minimum_gib=required_disk_gib)
        say(f"[WAIT DISK] BG{qp}: {free:.1f} GiB free; need {required_disk_gib:.1f} GiB")
        time.sleep(60)
    experiment.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"PYTHON_BIN": str(PYTHON), "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5",
                        "NUM_PROCESSES": "6", "MAIN_PROCESS_PORT": "29640",
                        "DATASET_ROOT": str(DATASET / f"bg{qp}"),
                        "INITIAL_CHECKPOINT": str(INITIAL), "OUTPUT_DIR": str(experiment),
                        "CLIP_LENGTH": "6", "CLIP_STRIDE": "4",
                        "GRADIENT_ACCUMULATION_STEPS": "1", "MAX_TRAIN_STEPS": "2000",
                        "CHECKPOINTING_STEPS": "250", "LEARNING_RATE": "1e-6",
                        "SHARED_BG_NOISE_STRENGTH": "0.95", "MIXED_PRECISION": "fp16",
                        "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    command = ["bash", str(WRAPPER), "--max_height", "1088"]
    log = experiment / "train_queue.log"
    say(f"[START] BG{qp}: {' '.join(command)}; log={log}")
    status("training", bg_qp=qp, dataset=str(DATASET / f"bg{qp}"),
           output_dir=str(experiment), log=str(log))
    with log.open("a", encoding="utf-8") as handle:
        child = subprocess.Popen(command, cwd=BRUSHNET, env=environment,
                                 stdout=handle, stderr=subprocess.STDOUT)
        wait_complete(experiment, retention, child)
    if retention == "models_only":
        prune_intermediate(experiment)


def wait_retention_choice():
    while True:
        if RETENTION_FILE.is_file():
            value = json.loads(RETENTION_FILE.read_text(encoding="utf-8"))
            retention = value.get("retention")
            if retention in ("models_only", "keep_all"):
                say(f"[RETENTION] User choice: {retention}")
                return retention
            raise RuntimeError(f"Invalid retention choice: {RETENTION_FILE}")
        status("waiting_retention_choice", file=str(RETENTION_FILE))
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retention", choices=("models_only", "keep_all", "defer"), default="defer")
    parser.add_argument("--min-disk-gib", type=float, default=45.0)
    args = parser.parse_args()
    if not INITIAL.is_dir() or not dataset_ready(46):
        raise RuntimeError("Missing initial checkpoint or BG46 train dataset")
    try:
        say("[QUEUE] BG48 completion → BG46 train → BG44 train")
        status("waiting_bg48", retention=args.retention)
        wait_complete(BG48, "keep_all" if args.retention == "defer" else args.retention)
        retention = wait_retention_choice() if args.retention == "defer" else args.retention
        if retention == "models_only":
            prune_intermediate(BG48)
        for qp in (46, 44):
            run_train(qp, retention, args.min_disk_gib)
        status("complete", completed_qps=[46, 44], retention=retention)
        say("[FINAL] BG46 and BG44 training complete")
        return 0
    except Exception as error:
        status("failed", error=str(error), retention=args.retention)
        say(f"[FAIL] {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
