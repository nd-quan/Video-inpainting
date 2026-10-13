#!/usr/bin/env python3
"""Evaluate the ten missing BG48 sequences on GPUs 0/1, then finish metrics."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import subprocess
import sys
from threading import Lock, Thread
import time

from PIL import Image

from run_sfu_bg48_table_window_bvi_ckpt2000_rgb_tg import (
    BRUSHNET, CHECKPOINT, DATASET, EVALUATOR, OUTPUT, PYTHON, STUDENT,
    expected_and_existing, gpu_memory, minimum_free_mib, save_json,
)
from sfu_long_test_loader import load_sequences


LOG_DIR = OUTPUT / "logs"
ADDITIONAL_STATUS = LOG_DIR / "additional_status.json"
MAIN_STATUS = LOG_DIR / "status.json"
MEMORY_CEILING_MIB = 47500
CLIP_SIZE = 4
EXPECTED_SEQUENCES = 14
EXPECTED_FRAMES = 1198
lock = Lock()


def traffic_state() -> str:
    records = json.loads(MAIN_STATUS.read_text(encoding="utf-8"))
    return next(row["state"] for row in records if row["sequence"] == "Traffic")


def make_env(sequence, gpu: int, clip_index: int | None = None) -> dict[str, str]:
    with Image.open(sequence.image_paths[0]) as image:
        size = image.size
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "LONG_TEST_ROOT": str(DATASET), "OUTPUT_ROOT": str(OUTPUT),
        "DISCOVER_HIERARCHICAL_SEQUENCES": "1", "SEQUENCES": sequence.spec.label,
        "CHECKPOINT_DIR": str(CHECKPOINT), "PREFLIGHT_ONLY": "0",
        "MAX_IMAGES": "0", "FORCE_REGENERATE": "0",
        "GUIDANCE_MODE": "temporal", "TEMPORAL_GUIDANCE_SPACE": "rgb",
        "TEMPORAL_FLOW_BACKEND": "v7_student",
        "V7_RAFT_STUDENT_PATH": str(STUDENT),
        "TEMPORAL_BG_MASK_MODE": "full_stable",
        "TEMPORAL_VAE_GRADIENT_CHECKPOINTING": "1" if max(size) >= 2000 else "0",
        "TEMPORAL_VAE_TILING": "1" if max(size) >= 2000 else "0",
        "TEMPORAL_GUIDANCE_SCALE": "0.0001",
        "TEMPORAL_START_STEP": "25", "TEMPORAL_END_STEP": "35",
        "TEMPORAL_EVERY_N_STEPS": "1", "TEMPORAL_CLIP_SIZE": str(CLIP_SIZE),
        "NUM_INFERENCE_STEPS": "50", "SHARED_BG_NOISE_STRENGTH": "0.95",
        "VARIANCE_PRESERVING_SHARED_NOISE": "0",
        "TEMPORAL_FLOW_BATCH_SIZE": "2",
        "MAIN_NOISE_SEED": "1234", "SHARED_NOISE_SEED": "6789",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(BRUSHNET / "examples/brushnet/proposed")
        + os.pathsep + env.get("PYTHONPATH", ""),
    })
    if clip_index is not None:
        env["EVAL_CLIP_INDEX"] = str(clip_index)
    else:
        env.pop("EVAL_CLIP_INDEX", None)
    return env


def valid_count(sequence) -> int:
    expected, existing = expected_and_existing(sequence)
    if existing == expected:
        with Image.open(sequence.image_paths[0]) as source:
            expected_size = source.size
        for name in expected:
            output = sequence.output_dir / name
            with Image.open(output) as image:
                image.verify()
            with Image.open(output) as image:
                if image.size != expected_size:
                    raise ValueError(f"Incorrect resolution in {output}")
    return len(existing)


def run_process(sequence, gpu: int, log_path: Path,
                clip_index: int | None = None) -> tuple[int, int]:
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(f"=== {datetime.now().isoformat()} GPU {gpu} "
                     f"{sequence.spec.label}, clip={clip_index} ===\n")
        stream.flush()
        process = subprocess.Popen(
            [str(PYTHON), "-u", str(EVALUATOR)], cwd=BRUSHNET,
            env=make_env(sequence, gpu, clip_index), stdout=stream,
            stderr=subprocess.STDOUT,
        )
        peak = 0
        while process.poll() is None:
            try:
                used, _ = gpu_memory(gpu)
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                stream.write(f"GPU monitoring failed: {exc}\n")
                process.terminate()
                break
            peak = max(peak, used)
            if used >= MEMORY_CEILING_MIB:
                stream.write(f"VRAM guard reached {used} MiB; terminating\n")
                stream.flush()
                process.terminate()
                break
            time.sleep(0.5)
        try:
            exit_code = process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            exit_code = process.wait()
        stream.write(f"\n[RUN] exit={exit_code}, peak={peak} MiB\n")
    return exit_code, peak


def save_status(records: list[dict]) -> None:
    with lock:
        save_json(ADDITIONAL_STATUS, records)


def evaluate_one(sequence, record: dict, gpu: int, records: list[dict]) -> None:
    name = sequence.spec.label
    with lock:
        record.update(state="running", gpu=gpu,
                      started_at=datetime.now().isoformat(timespec="seconds"))
        save_json(ADDITIONAL_STATUS, records)
    required_free = minimum_free_mib(record)
    while True:
        used, free = gpu_memory(gpu)
        if used < MEMORY_CEILING_MIB and free >= required_free:
            break
        time.sleep(10)
    full_log = LOG_DIR / f"{record['class']}_{name}_gpu{gpu}.log"
    exit_code, peak = run_process(sequence, gpu, full_log)
    existing = valid_count(sequence)
    print(f"GPU {gpu} {name}: full process exit={exit_code}, "
          f"frames={existing}/{record['frame_count']}, peak={peak} MiB", flush=True)

    if existing < record["frame_count"]:
        for clip_index, offset in enumerate(range(0, record["frame_count"], CLIP_SIZE)):
            names = {path.name for path in sequence.image_paths[offset:offset + CLIP_SIZE]}
            _, existing_names = expected_and_existing(sequence)
            if names <= existing_names:
                continue
            clip_log = LOG_DIR / f"{record['class']}_{name}_clip{clip_index}_gpu{gpu}.log"
            clip_exit, clip_peak = run_process(sequence, gpu, clip_log, clip_index)
            peak = max(peak, clip_peak)
            with lock:
                record.update(existing_frames=valid_count(sequence),
                              peak_gpu_memory_mib=peak, current_clip=clip_index)
                save_json(ADDITIONAL_STATUS, records)
            if clip_exit or not names <= expected_and_existing(sequence)[1]:
                raise RuntimeError(f"{name} clip {clip_index} failed; see {clip_log}")
    existing = valid_count(sequence)
    if existing != record["frame_count"]:
        raise RuntimeError(f"{name} has {existing}/{record['frame_count']} frames")
    with lock:
        record.update(state="complete", existing_frames=existing, exit_code=0,
                      peak_gpu_memory_mib=peak,
                      finished_at=datetime.now().isoformat(timespec="seconds"))
        save_json(ADDITIONAL_STATUS, records)
    print(f"GPU {gpu} {name}: complete {existing}/{record['frame_count']}", flush=True)


def main() -> int:
    for required in (DATASET, CHECKPOINT / "brushnet", CHECKPOINT / "ipadapter",
                     STUDENT, EVALUATOR, PYTHON, MAIN_STATUS):
        if not required.exists():
            raise FileNotFoundError(required)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    sequences = load_sequences(DATASET, OUTPUT, discover_hierarchical=True)
    if len(sequences) != EXPECTED_SEQUENCES or sum(len(s.frame_ids) for s in sequences) != EXPECTED_FRAMES:
        raise ValueError("Expected 14 BG48 sequences and 1198 frames")
    initial = json.loads(MAIN_STATUS.read_text(encoding="utf-8"))
    initial_names = {row["sequence"] for row in initial}
    if len(initial_names) != 4:
        raise ValueError("Initial status must contain the four previously selected sequences")
    selected = [sequence for sequence in sequences if sequence.spec.label not in initial_names]
    records = []
    for sequence in selected:
        gt = DATASET / "GT" / sequence.spec.class_name / sequence.spec.label
        expected = {path.name for path in sequence.image_paths}
        if {path.name for path in sequence.mask_paths} != expected or {
            path.name for path in gt.glob("*.png")
        } != expected:
            raise ValueError(f"Input, mask, and GT IDs differ for {sequence.spec.label}")
        with Image.open(sequence.image_paths[0]) as image:
            size = image.size
        existing = valid_count(sequence)
        records.append({
            "sequence": sequence.spec.label, "class": sequence.spec.class_name,
            "frame_count": len(sequence.frame_ids),
            "frame_range": [sequence.frame_ids[0], sequence.frame_ids[-1]],
            "size_wh": list(size), "output_dir": str(sequence.output_dir),
            "existing_frames": existing, "state": "complete" if existing == len(expected) else "queued",
        })
    save_status(records)
    if len(records) != 10:
        raise ValueError("Expected ten additional BG48 sequences")
    by_name = {row["sequence"]: row for row in records}
    queue = Queue()
    pending = [sequence for sequence in selected if by_name[sequence.spec.label]["state"] != "complete"]
    pending.sort(key=lambda sequence: -len(sequence.image_paths) *
                 by_name[sequence.spec.label]["size_wh"][0] *
                 by_name[sequence.spec.label]["size_wh"][1])
    for sequence in pending:
        queue.put(sequence)
    failures = []

    def worker(gpu: int) -> None:
        if gpu == 0:
            while traffic_state() != "complete":
                if traffic_state() in {"failed", "stopped_vram"}:
                    failures.append("Traffic stopped before GPU 0 became free")
                    return
                if queue.empty():
                    return
                time.sleep(20)
        while True:
            try:
                sequence = queue.get_nowait()
            except Empty:
                return
            record = by_name[sequence.spec.label]
            try:
                evaluate_one(sequence, record, gpu, records)
            except Exception as exc:
                with lock:
                    record.update(state="failed", error=repr(exc),
                                  finished_at=datetime.now().isoformat(timespec="seconds"))
                    save_json(ADDITIONAL_STATUS, records)
                failures.append(f"{sequence.spec.label}: {exc}")
                print(f"GPU {gpu} {sequence.spec.label}: FAILED: {exc}", flush=True)
            finally:
                queue.task_done()

    workers = [Thread(target=worker, args=(gpu,), daemon=False) for gpu in (0, 1)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    if failures or any(row["state"] != "complete" for row in records):
        raise RuntimeError(f"Additional BG48 evaluation incomplete: {failures}")
    while traffic_state() != "complete":
        if traffic_state() in {"failed", "stopped_vram"}:
            raise RuntimeError("Traffic evaluation stopped")
        time.sleep(20)

    # The Traffic runner has finished writing the initial status now, so merge safely.
    initial = json.loads(MAIN_STATUS.read_text(encoding="utf-8"))
    if len(initial) != 4 or any(row["state"] != "complete" for row in initial):
        raise ValueError("Initial four-sequence status is not complete")
    all_records = sorted(initial + records, key=lambda row: (row["class"], row["sequence"]))
    save_json(MAIN_STATUS, all_records)
    config_path = OUTPUT / "run_config.json"
    backup = OUTPUT / "run_config_initial_4_sequences.json"
    if not backup.exists():
        shutil.copy2(config_path, backup)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(expected_sequences=EXPECTED_SEQUENCES, expected_frames=EXPECTED_FRAMES,
                  gpus=[0, 1], gpu_memory_ceiling_mib=MEMORY_CEILING_MIB,
                  resource_policy="Traffic first on GPU 0; ten additional sequences on GPUs 0 and 1; resume missing clips after a stopped sequence")
    save_json(config_path, config)
    postprocess = Path(__file__).with_name("postprocess_sfu_bg48_bvi.py")
    log_path = OUTPUT / "metrics_full_reference_no_we/postprocess_runner.log"
    log_path.parent.mkdir(exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n=== {datetime.now().isoformat()} all 14 BG48 sequences ===\n")
        stream.flush()
        result = subprocess.run([str(PYTHON), "-u", str(postprocess), "--gpu", "1"],
                                cwd=BRUSHNET, stdout=stream, stderr=subprocess.STDOUT,
                                check=False)
    if result.returncode:
        raise RuntimeError(f"BG48 metrics/video postprocess failed; see {log_path}")
    print("All 14 BG48 sequences, metrics, and videos complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
