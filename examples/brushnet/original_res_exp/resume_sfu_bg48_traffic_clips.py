#!/usr/bin/env python3
"""Resume BG48 Traffic one original four-frame clip per evaluator process."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image

from run_sfu_bg48_table_window_bvi_ckpt2000_rgb_tg import (
    BRUSHNET, CHECKPOINT, DATASET, EVALUATOR, OUTPUT, PYTHON, STUDENT,
    gpu_memory, save_json,
)
from sfu_long_test_loader import load_sequences


GPU = 0
CLIP_SIZE = 4
MEMORY_CEILING_MIB = 48000
LOG_DIR = OUTPUT / "logs"
STATUS_PATH = LOG_DIR / "status.json"
RESUME_STATUS_PATH = LOG_DIR / "traffic_resume_status.json"


def count_valid_frames(sequence) -> tuple[int, list[str]]:
    missing = []
    with Image.open(sequence.image_paths[0]) as source:
        expected_size = source.size
    for source_path in sequence.image_paths:
        output_path = sequence.output_dir / source_path.name
        if not output_path.is_file():
            missing.append(source_path.name)
            continue
        with Image.open(output_path) as result:
            result.verify()
        with Image.open(output_path) as result:
            if result.size != expected_size:
                raise ValueError(f"Unexpected image size in {output_path}: {result.size}")
    return len(sequence.image_paths) - len(missing), missing


def update_main_status(state: str, existing: int, **fields) -> None:
    records = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    traffic = next(record for record in records if record["sequence"] == "Traffic")
    traffic.update(state=state, existing_frames=existing, **fields)
    save_json(STATUS_PATH, records)


def evaluator_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(GPU),
        "LONG_TEST_ROOT": str(DATASET), "OUTPUT_ROOT": str(OUTPUT),
        "DISCOVER_HIERARCHICAL_SEQUENCES": "1", "SEQUENCES": "Traffic",
        "CHECKPOINT_DIR": str(CHECKPOINT), "PREFLIGHT_ONLY": "0",
        "MAX_IMAGES": "0", "FORCE_REGENERATE": "0",
        "GUIDANCE_MODE": "temporal", "TEMPORAL_GUIDANCE_SPACE": "rgb",
        "TEMPORAL_FLOW_BACKEND": "v7_student",
        "V7_RAFT_STUDENT_PATH": str(STUDENT),
        "TEMPORAL_BG_MASK_MODE": "full_stable",
        "TEMPORAL_VAE_GRADIENT_CHECKPOINTING": "1",
        "TEMPORAL_VAE_TILING": "1",
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
    return env


def main() -> int:
    for required in (DATASET, CHECKPOINT / "brushnet", CHECKPOINT / "ipadapter",
                     STUDENT, EVALUATOR, PYTHON, STATUS_PATH):
        if not required.exists():
            raise FileNotFoundError(required)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    sequences = load_sequences(DATASET, OUTPUT, ["Traffic"], discover_hierarchical=True)
    if len(sequences) != 1 or len(sequences[0].image_paths) != 33:
        raise ValueError("Expected the 33-frame BG48 Traffic test sequence")
    sequence = sequences[0]
    existing, missing = count_valid_frames(sequence)
    resume = {
        "sequence": "Traffic", "gpu": GPU, "clip_size": CLIP_SIZE,
        "memory_ceiling_mib": MEMORY_CEILING_MIB,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "state": "running", "existing_frames": existing,
        "expected_frames": len(sequence.image_paths), "clips": [],
    }
    save_json(RESUME_STATUS_PATH, resume)
    update_main_status("running", existing, resume_status=str(RESUME_STATUS_PATH))
    env = evaluator_environment()

    for clip_index, offset in enumerate(range(0, len(sequence.image_paths), CLIP_SIZE)):
        names = [path.name for path in sequence.image_paths[offset:offset + CLIP_SIZE]]
        if not any(name in missing for name in names):
            continue
        used, free = gpu_memory(GPU)
        if used >= MEMORY_CEILING_MIB or free < 37000:
            raise RuntimeError(f"GPU {GPU} has insufficient free VRAM: {used} used, {free} free MiB")
        clip_log = LOG_DIR / f"Class_A_Traffic_clip{clip_index}_gpu{GPU}.log"
        clip_record = {
            "clip_index": clip_index, "frames": names, "log": str(clip_log),
            "state": "running", "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        resume["clips"].append(clip_record)
        save_json(RESUME_STATUS_PATH, resume)
        clip_env = dict(env, EVAL_CLIP_INDEX=str(clip_index))
        print(f"Starting Traffic clip {clip_index}: {names}", flush=True)
        with clip_log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                [str(PYTHON), "-u", str(EVALUATOR)], cwd=BRUSHNET,
                env=clip_env, stdout=stream, stderr=subprocess.STDOUT,
            )
            peak = used
            while process.poll() is None:
                used, _ = gpu_memory(GPU)
                peak = max(peak, used)
                if used >= MEMORY_CEILING_MIB:
                    stream.write(f"\n[VRAM GUARD] GPU {GPU} reached {used} MiB\n")
                    stream.flush()
                    process.terminate()
                    break
                time.sleep(0.5)
            try:
                exit_code = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = process.wait()
            stream.write(f"\n[RESUME] exit={exit_code}, peak={peak} MiB\n")
        existing, missing = count_valid_frames(sequence)
        clip_record.update(
            state="complete" if exit_code == 0 and not any(name in missing for name in names) else "failed",
            exit_code=exit_code, peak_gpu_memory_mib=peak,
            finished_at=datetime.now().isoformat(timespec="seconds"),
        )
        resume["existing_frames"] = existing
        save_json(RESUME_STATUS_PATH, resume)
        update_main_status("running" if clip_record["state"] == "complete" else "failed",
                           existing, peak_gpu_memory_mib=peak)
        print(f"Traffic clip {clip_index}: {clip_record['state']}; {existing}/33 frames; peak {peak} MiB", flush=True)
        if clip_record["state"] != "complete":
            resume["state"] = "failed"
            save_json(RESUME_STATUS_PATH, resume)
            return 1

    existing, missing = count_valid_frames(sequence)
    if missing:
        raise RuntimeError(f"Traffic still missing {len(missing)} frames: {missing}")
    resume.update(state="complete", existing_frames=existing,
                  finished_at=datetime.now().isoformat(timespec="seconds"))
    save_json(RESUME_STATUS_PATH, resume)
    update_main_status("complete", existing, exit_code=0,
                       finished_at=resume["finished_at"], last_error=None)
    print(f"Traffic complete: {existing}/33 frames", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        if RESUME_STATUS_PATH.exists():
            resume = json.loads(RESUME_STATUS_PATH.read_text(encoding="utf-8"))
            resume.update(state="failed", error=str(exc))
            save_json(RESUME_STATUS_PATH, resume)
        print(f"Traffic resume failed: {exc}", file=sys.stderr, flush=True)
        raise
