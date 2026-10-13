#!/usr/bin/env python3
"""Evaluate SFU BG56 using BVI checkpoint 2000, with a 47 GB GPU guard."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Lock
import time

from PIL import Image

from sfu_long_test_loader import load_sequences


BRUSHNET = Path(__file__).resolve().parents[2]
DATASET = Path("/home/cilab/ndquan/vcm-rs/output/sfu_qp0_bg_table_window_test/bg56/test")
CHECKPOINT = BRUSHNET / "experiments/finetune_shared_noise_bvi_dvc_bg56_original_size/checkpoint-2000"
STUDENT = BRUSHNET / "experiments/train_v7_raft_student_flow/checkpoint-0004750/raft_student"
OUTPUT = BRUSHNET / "experiments/eval_sfu_bg56_table_window_bvi_ckpt2000_rgb_tg"
EVALUATOR = Path(__file__).with_name("test_brushnet_fixedBG_temporal_v7_student_original_size.py")
PYTHON = Path("/home/cilab/ndquan/envs/guided_diff/bin/python")
GPUS = tuple(range(6))
GPU_MEMORY_CEILING_MIB = 47000


def gpu_memory(gpu: int) -> tuple[int, int]:
    output = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,memory.free",
         "--format=csv,noheader,nounits"], text=True,
    ).strip()
    used, free = (int(value.strip()) for value in output.split(","))
    return used, free


def minimum_free_mib(record: dict) -> int:
    width = record["size_wh"][0]
    if width >= 2500:
        return 37000  # Traffic: the prior native-size RGB eval needed about 30 GB.
    if width >= 1900:
        return 22000  # Leave room for a 1920-pixel clip and the running trainer.
    return 14000      # Validated by a four-frame 416x240 smoke eval.


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def expected_and_existing(sequence) -> tuple[set[str], set[str]]:
    expected = {path.name for path in sequence.image_paths}
    existing = {path.name for path in sequence.output_dir.glob("*.png")}
    if not existing <= expected:
        raise ValueError(f"Unexpected output frames in {sequence.output_dir}: {sorted(existing - expected)[:5]}")
    return expected, existing


def main() -> int:
    for path in (DATASET, CHECKPOINT / "brushnet", CHECKPOINT / "ipadapter",
                 STUDENT, EVALUATOR, PYTHON):
        if not path.exists():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log_dir = OUTPUT / "logs"
    log_dir.mkdir(exist_ok=True)
    sequences = load_sequences(DATASET, OUTPUT, discover_hierarchical=True)
    if len(sequences) != 14 or sum(len(seq.frame_ids) for seq in sequences) != 1198:
        raise ValueError("Expected 14 BG56 sequences and 1198 test frames")

    records = []
    for seq in sequences:
        for kind in ("GT", "mask"):
            source_dir = DATASET / kind / seq.spec.class_name / seq.spec.label
            if {p.name for p in source_dir.glob("*.png")} != {p.name for p in seq.image_paths}:
                raise ValueError(f"Mismatched {kind} frame IDs for {seq.spec.label}")
        with Image.open(seq.image_paths[0]) as image:
            width, height = image.size
        expected, existing = expected_and_existing(seq)
        records.append({
            "sequence": seq.spec.label,
            "class": seq.spec.class_name,
            "frame_count": len(expected),
            "frame_range": [seq.frame_ids[0], seq.frame_ids[-1]],
            "size_wh": [width, height],
            "output_dir": str(seq.output_dir),
            "existing_frames": len(existing),
            "minimum_free_mib": 0,
            "state": "complete" if existing == expected else "queued",
        })
        records[-1]["minimum_free_mib"] = minimum_free_mib(records[-1])
    status_path = log_dir / "status.json"
    save_json(status_path, records)
    save_json(OUTPUT / "run_config.json", {
        "dataset": str(DATASET), "checkpoint": str(CHECKPOINT),
        "flow_checkpoint": str(STUDENT), "evaluator": str(EVALUATOR),
        "gpus": list(GPUS), "expected_sequences": 14, "expected_frames": 1198,
        "guidance_mode": "temporal", "temporal_guidance_space": "rgb",
        "temporal_flow_backend": "v7_student", "temporal_guidance_scale": 0.0001,
        "temporal_window": [25, 35], "temporal_clip_size": 4,
        "inference_steps": 50, "shared_bg_noise_strength": 0.95,
        "variance_preserving_shared_noise": False,
        "gpu_memory_ceiling_mib": GPU_MEMORY_CEILING_MIB,
        "resource_policy": "Run 6 GPU workers; defer jobs until VRAM is available; stop all eval subprocesses if any GPU reaches 47000 MiB",
    })

    # Small sequences can run while the existing BG48 training occupies each GPU.
    # Large native-size sequences remain queued until sufficient VRAM is free.
    pending = sorted(
        (index for index, record in enumerate(records) if record["state"] == "queued"),
        key=lambda index: (
            records[index]["minimum_free_mib"],
            -records[index]["frame_count"] * records[index]["size_wh"][0]
            * records[index]["size_wh"][1],
        ),
    )
    lock = Lock()
    stop_event = Event()
    active_processes: dict[int, subprocess.Popen] = {}

    def worker(gpu: int) -> None:
        while True:
            if stop_event.is_set():
                return
            used, free = gpu_memory(gpu)
            if used >= GPU_MEMORY_CEILING_MIB:
                time.sleep(10)
                continue
            index = None
            with lock:
                for position, candidate in enumerate(pending):
                    if free >= records[candidate]["minimum_free_mib"]:
                        index = pending.pop(position)
                        break
                if index is None and not pending and all(
                    record["state"] in {"complete", "failed"} for record in records
                ):
                    return
            if index is None:
                time.sleep(10)
                continue
            seq, record = sequences[index], records[index]
            label = seq.spec.label
            log_path = log_dir / f"{seq.spec.class_name}_{label}_gpu{gpu}.log"
            with lock:
                record.update(state="running", gpu=gpu, log=str(log_path),
                              started_at=datetime.now().isoformat(timespec="seconds"),
                              attempts=record.get("attempts", 0) + 1)
                save_json(status_path, records)
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "LONG_TEST_ROOT": str(DATASET), "OUTPUT_ROOT": str(OUTPUT),
                "DISCOVER_HIERARCHICAL_SEQUENCES": "1", "SEQUENCES": label,
                "CHECKPOINT_DIR": str(CHECKPOINT), "PREFLIGHT_ONLY": "0",
                "MAX_IMAGES": "0", "FORCE_REGENERATE": "0",
                "GUIDANCE_MODE": "temporal", "TEMPORAL_GUIDANCE_SPACE": "rgb",
                "TEMPORAL_FLOW_BACKEND": "v7_student",
                "V7_RAFT_STUDENT_PATH": str(STUDENT),
                "TEMPORAL_BG_MASK_MODE": "full_stable",
                "TEMPORAL_VAE_GRADIENT_CHECKPOINTING": "1" if max(record["size_wh"]) >= 2000 else "0",
                "TEMPORAL_VAE_TILING": "1" if max(record["size_wh"]) >= 2000 else "0",
                "TEMPORAL_GUIDANCE_SCALE": "0.0001",
                "TEMPORAL_START_STEP": "25", "TEMPORAL_END_STEP": "35",
                "TEMPORAL_EVERY_N_STEPS": "1", "TEMPORAL_CLIP_SIZE": "4",
                "NUM_INFERENCE_STEPS": "50", "SHARED_BG_NOISE_STRENGTH": "0.95",
                "VARIANCE_PRESERVING_SHARED_NOISE": "0",
                "TEMPORAL_FLOW_BATCH_SIZE": "2",
                "MAIN_NOISE_SEED": "1234", "SHARED_NOISE_SEED": "6789",
                "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(BRUSHNET / "examples/brushnet/proposed")
                + os.pathsep + env.get("PYTHONPATH", ""),
            })
            memory_limit_hit = False
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"\n=== {datetime.now().isoformat()} GPU {gpu} {label}; "
                             f"before={used} MiB, free={free} MiB ===\n")
                stream.flush()
                process = subprocess.Popen([str(PYTHON), "-u", str(EVALUATOR)],
                                           cwd=BRUSHNET, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT)
                with lock:
                    active_processes[gpu] = process
                peak = used
                while process.poll() is None:
                    if stop_event.is_set():
                        process.terminate()
                        break
                    try:
                        current, _ = gpu_memory(gpu)
                    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
                        stream.write(f"[VRAM GUARD] Cannot read GPU memory: {exc}\n")
                        current = GPU_MEMORY_CEILING_MIB
                    peak = max(peak, current)
                    if current >= GPU_MEMORY_CEILING_MIB:
                        memory_limit_hit = True
                        stream.write(f"[VRAM GUARD] GPU {gpu} reached {current} MiB; "
                                     "stopping all eval subprocesses\n")
                        stream.flush()
                        stop_event.set()
                        with lock:
                            processes_to_stop = list(active_processes.values())
                        for running in processes_to_stop:
                            running.terminate()
                        break
                    time.sleep(0.5)
                try:
                    exit_code = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    exit_code = process.wait()
                with lock:
                    active_processes.pop(gpu, None)
                stream.write(f"[VRAM GUARD] peak={peak} MiB, exit={exit_code}\n")
            expected, existing = expected_and_existing(seq)
            complete = exit_code == 0 and existing == expected and not memory_limit_hit
            with lock:
                if stop_event.is_set() and not complete:
                    record.update(state="stopped_vram", existing_frames=len(existing),
                                  last_error=f"VRAM guard triggered; peak observed {peak} MiB")
                else:
                    record.update(state="complete" if complete else "failed",
                                  exit_code=exit_code, existing_frames=len(existing),
                                  peak_gpu_memory_mib=peak,
                                  finished_at=datetime.now().isoformat(timespec="seconds"))
                save_json(status_path, records)
            print(f"GPU {gpu} {label}: state={record['state']}, exit={exit_code}, "
                  f"frames={len(existing)}/{len(expected)}, peak={peak} MiB", flush=True)

    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in GPUS]
        for future in futures:
            future.result()
    return 0 if all(record["state"] == "complete" for record in records) else 1


if __name__ == "__main__":
    sys.exit(main())
