#!/usr/bin/env python3
"""Evaluate the COCO BG52 checkpoint on SFU QP52 without shared noise or guidance."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
from threading import Lock
import time

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sfu_long_test_loader import load_sequences


BRUSHNET = Path(__file__).resolve().parents[3]
DATASET = Path("/home/cilab/ndquan/vcm-rs/output/sfu_qp0_bg_table_window_test/bg52/test")
CHECKPOINT = Path(
    "/user/Cilab/Quan/experiment/"
    "finetune_coco_bg52_native_area_bucket_ckpt200000_3000steps/checkpoint-6000"
)
OUTPUT = BRUSHNET / "experiments/eval_sfu_hw_bg52_coco6000_no_shared_no_temporal_small512"
EVALUATOR = BRUSHNET / "examples/brushnet/inference_experiments/test_brushnet_fixedBG_temporal_v7_student_original_size.py"
PYTHON = Path("/home/cilab/ndquan/envs/guided_diff/bin/python")
GPUS = tuple(int(value) for value in os.environ.get("SFU_EVAL_GPUS", "3,4,5").split(","))
GPU_MEMORY_CEILING_MIB = 47000


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_memory(gpu: int) -> tuple[int, int]:
    output = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,memory.free",
         "--format=csv,noheader,nounits"], text=True,
    ).strip()
    return tuple(int(value.strip()) for value in output.split(","))


def existing_count(sequence) -> int:
    expected = {path.name for path in sequence.image_paths}
    existing = {path.name for path in sequence.output_dir.glob("*.png")}
    if not existing <= expected:
        raise ValueError(f"Unexpected output frame in {sequence.output_dir}")
    return len(existing)


def minimum_free_mib(width: int) -> int:
    if width >= 2500:
        return 45000
    if width >= 1900:
        return 35000
    if width >= 800:
        return 20000
    return 12000


def main() -> int:
    if not GPUS or len(set(GPUS)) != len(GPUS):
        raise ValueError(f"Invalid SFU_EVAL_GPUS={GPUS}")
    for path in (
        DATASET, EVALUATOR, PYTHON, CHECKPOINT / "brushnet/config.json",
        CHECKPOINT / "ipadapter/model.safetensors",
        CHECKPOINT / "ipadapter/fusion_module.safetensors",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log_dir = OUTPUT / "logs"
    log_dir.mkdir(exist_ok=True)
    sequences = load_sequences(DATASET, OUTPUT, discover_hierarchical=True)
    if len(sequences) != 14 or sum(len(seq.frame_ids) for seq in sequences) != 1198:
        raise ValueError("Expected 14 SFU QP52 sequences and 1198 frames")
    records = []
    for sequence in sequences:
        expected = {path.name for path in sequence.image_paths}
        for kind in ("GT", "mask"):
            reference = DATASET / kind / sequence.spec.class_name / sequence.spec.label
            if {path.name for path in reference.glob("*.png")} != expected:
                raise ValueError(f"{sequence.spec.label}: {kind} frame names differ")
        with Image.open(sequence.image_paths[0]) as image:
            width, height = image.size
        count = existing_count(sequence)
        records.append({
            "sequence": sequence.spec.label, "class": sequence.spec.class_name,
            "frame_count": len(sequence.frame_ids), "size_wh": [width, height],
            "minimum_free_mib": minimum_free_mib(width),
            "output_dir": str(sequence.output_dir), "existing_frames": count,
            "state": "complete" if count == len(sequence.frame_ids) else "queued",
        })
    save_json(OUTPUT / "run_config.json", {
        "dataset": str(DATASET), "checkpoint": str(CHECKPOINT),
        "evaluator": str(EVALUATOR), "gpus": list(GPUS),
        "expected_sequences": 14, "expected_frames": 1198,
        "guidance_mode": "none", "shared_bg_noise": False,
        "num_inference_steps": 50, "clip_size": 4,
        "small_frame_upscale_long_side": 512,
        "output_size": "native source size",
        "resource_policy": "Start sequences as soon as a GPU has enough free VRAM for their size",
        "gpu_memory_ceiling_mib": GPU_MEMORY_CEILING_MIB,
    })
    status_path = log_dir / "status.json"
    save_json(status_path, records)
    pending: queue.Queue[int] = queue.Queue()
    for index in sorted(range(len(sequences)), key=lambda i: (
        records[i]["size_wh"][0] * records[i]["size_wh"][1],
        records[i]["frame_count"],
    )):
        if records[index]["state"] == "queued":
            pending.put(index)
    lock = Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                index = pending.get_nowait()
            except queue.Empty:
                return
            sequence = sequences[index]
            record = records[index]
            while True:
                used, free = gpu_memory(gpu)
                if free >= record["minimum_free_mib"] and used < GPU_MEMORY_CEILING_MIB:
                    break
                time.sleep(10)
            log_path = log_dir / f"{sequence.spec.class_name}_{sequence.spec.label}_gpu{gpu}.log"
            with lock:
                record.update(state="running", gpu=gpu, log=str(log_path),
                              started_at=datetime.now().isoformat(timespec="seconds"))
                save_json(status_path, records)
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu), "LONG_TEST_ROOT": str(DATASET),
                "OUTPUT_ROOT": str(OUTPUT), "DISCOVER_HIERARCHICAL_SEQUENCES": "1",
                "SEQUENCES": sequence.spec.label, "CHECKPOINT_DIR": str(CHECKPOINT),
                "GUIDANCE_MODE": "none", "USE_SHARED_BG_NOISE": "0",
                "SHARED_BG_NOISE_STRENGTH": "0", "VARIANCE_PRESERVING_SHARED_NOISE": "0",
                "SMALL_FRAME_UPSCALE_LONG_SIDE": "512", "NUM_INFERENCE_STEPS": "50",
                "TEMPORAL_GUIDANCE_SCALE": "0", "TEMPORAL_CLIP_SIZE": "4",
                "MAIN_NOISE_SEED": "1234", "SHARED_NOISE_SEED": "6789",
                "MAX_IMAGES": "0", "PREFLIGHT_ONLY": "0", "FORCE_REGENERATE": "0",
                "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(BRUSHNET / "examples/brushnet/proposed")
                + os.pathsep + env.get("PYTHONPATH", ""),
            })
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"\n=== {datetime.now().isoformat()} GPU {gpu} "
                             f"{sequence.spec.label}, before={used} MiB, free={free} MiB ===\n")
                stream.flush()
                process = subprocess.Popen(
                    [str(PYTHON), "-u", str(EVALUATOR)], cwd=BRUSHNET,
                    env=env, stdout=stream, stderr=subprocess.STDOUT,
                )
                peak = used
                memory_limit_hit = False
                while process.poll() is None:
                    current, _ = gpu_memory(gpu)
                    peak = max(peak, current)
                    if current >= GPU_MEMORY_CEILING_MIB:
                        memory_limit_hit = True
                        stream.write(f"[VRAM GUARD] GPU {gpu} reached {current} MiB\n")
                        stream.flush()
                        process.terminate()
                        break
                    time.sleep(1)
                try:
                    exit_code = process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    exit_code = process.wait()
            count = existing_count(sequence)
            complete = exit_code == 0 and count == record["frame_count"] and not memory_limit_hit
            with lock:
                record.update(state="complete" if complete else "failed", exit_code=exit_code,
                              existing_frames=count, peak_gpu_memory_mib=peak,
                              finished_at=datetime.now().isoformat(timespec="seconds"))
                save_json(status_path, records)
            print(f"{sequence.spec.label} GPU {gpu}: {record['state']} "
                  f"({count}/{record['frame_count']}), peak={peak} MiB", flush=True)
            pending.task_done()

    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in GPUS]
        for future in futures:
            future.result()
    return 0 if all(record["state"] == "complete" for record in records) else 1


if __name__ == "__main__":
    sys.exit(main())
