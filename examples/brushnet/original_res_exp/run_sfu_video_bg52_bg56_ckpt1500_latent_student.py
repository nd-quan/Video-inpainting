#!/usr/bin/env python3
"""Evaluate BG52/BG56 video checkpoints with latent guidance on GPUs 3, 4, 5."""

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
DATA_ROOT = Path("/home/cilab/ndquan/vcm-rs/output/sfu_qp0_bg_table_window_test")
USER_EXPERIMENTS = Path("/user/Cilab/Quan/experiment")
OUTPUT = BRUSHNET / "experiments/eval_sfu_video_bg52_bg56_ckpt1500_latent_tg0p1_student_small512_flow960"
EVALUATOR = BRUSHNET / "examples/brushnet/test_brushnet_fixedBG_temporal_v7_student_original_size.py"
PYTHON = Path("/home/cilab/ndquan/envs/guided_diff/bin/python")
GPUS = (3, 4, 5)
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


def checkpoint_paths(bg: int) -> tuple[Path, Path]:
    checkpoint = USER_EXPERIMENTS / (
        f"finetune_sfu_video_bg{bg}_from_coco6000_2048x1280_T8S4_1000steps/checkpoint-1500"
    )
    student_run = BRUSHNET / (
        f"experiments/train_v7_raft_student_sfu_bg{bg}_max960_from_previous_best_4000/student"
    )
    best = json.loads((student_run / "best.json").read_text(encoding="utf-8"))
    student = student_run / best["checkpoint"]
    for required in (
        checkpoint / "brushnet/config.json",
        checkpoint / "ipadapter/model.safetensors",
        checkpoint / "ipadapter/fusion_module.safetensors",
        student / "raft_student/config.json",
        student / "raft_student/pytorch_model.bin",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    return checkpoint, student


def existing_count(job: dict) -> int:
    expected = {path.name for path in job["sequence"].image_paths}
    existing = {path.name for path in job["sequence"].output_dir.glob("*.png")}
    if not existing <= expected:
        raise ValueError(f"Unexpected PNG names in {job['sequence'].output_dir}")
    return len(existing)


def main() -> int:
    if not EVALUATOR.is_file() or not PYTHON.is_file():
        raise FileNotFoundError("Evaluator or Python environment is missing")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log_dir = OUTPUT / "logs"
    log_dir.mkdir(exist_ok=True)
    jobs = []
    checkpoint_map = {}
    for bg in (52, 56):
        checkpoint, student = checkpoint_paths(bg)
        checkpoint_map[bg] = (checkpoint, student)
        dataset = DATA_ROOT / f"bg{bg}/test"
        sequences = load_sequences(dataset, OUTPUT / f"bg{bg}", discover_hierarchical=True)
        if len(sequences) != 14 or sum(len(seq.frame_ids) for seq in sequences) != 1198:
            raise ValueError(f"BG{bg}: expected 14 sequences and 1198 frames")
        for seq in sequences:
            expected = {path.name for path in seq.image_paths}
            for kind in ("GT", "mask"):
                reference = dataset / kind / seq.spec.class_name / seq.spec.label
                if {path.name for path in reference.glob("*.png")} != expected:
                    raise ValueError(f"BG{bg} {seq.spec.label}: {kind} frame names differ")
            with Image.open(seq.image_paths[0]) as image:
                width, height = image.size
            if (width, height) == (416, 240):
                working_size = (512, round(240 * 512 / 416))
            else:
                working_size = (width, height)
            padded_size = tuple(((side + 63) // 64) * 64 for side in working_size)
            jobs.append({
                "bg": bg, "sequence": seq, "dataset": dataset,
                "checkpoint": checkpoint, "student": student,
                "size_wh": [width, height], "working_size_wh": list(working_size),
                "padded_size_wh": list(padded_size), "frame_count": len(seq.frame_ids),
            })
    save_json(OUTPUT / "run_config.json", {
        "datasets": {f"bg{bg}": str(DATA_ROOT / f"bg{bg}/test") for bg in (52, 56)},
        "checkpoints": {f"bg{bg}": str(paths[0]) for bg, paths in checkpoint_map.items()},
        "flow_checkpoints": {f"bg{bg}": str(paths[1]) for bg, paths in checkpoint_map.items()},
        "gpus": list(GPUS), "sequences": len(jobs), "frames": sum(job["frame_count"] for job in jobs),
        "small_frame_upscale_long_side": 512, "student_flow_max_input_side": 960,
        "output_size": "native source size", "num_inference_steps": 50,
        "temporal_guidance": {"space": "latent", "scale": 0.1, "steps": [0, 50], "clip_size": 4},
        "shared_bg_noise_strength": 0.95, "variance_preserving_shared_noise": True,
    })
    status_path = log_dir / "status.json"
    records = []
    for job in jobs:
        count = existing_count(job)
        seq = job["sequence"]
        records.append({
            "bg": job["bg"], "sequence": seq.spec.label, "class": seq.spec.class_name,
            "frame_count": job["frame_count"], "existing_frames": count,
            "size_wh": job["size_wh"], "working_size_wh": job["working_size_wh"],
            "padded_size_wh": job["padded_size_wh"], "output_dir": str(seq.output_dir),
            "state": "complete" if count == job["frame_count"] else "queued",
        })
    save_json(status_path, records)
    pending: queue.Queue[int] = queue.Queue()
    for index in sorted(range(len(jobs)), key=lambda i: (
        jobs[i]["padded_size_wh"][0] * jobs[i]["padded_size_wh"][1],
        jobs[i]["frame_count"], jobs[i]["bg"],
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
            job, record = jobs[index], records[index]
            seq = job["sequence"]
            # An inference worker owns its GPU until this sequence finishes.
            minimum_free_mib = 37000 if max(job["size_wh"]) >= 2500 else 21000
            while True:
                used, free = gpu_memory(gpu)
                if free >= minimum_free_mib and used < GPU_MEMORY_CEILING_MIB:
                    break
                time.sleep(10)
            log_path = log_dir / f"bg{job['bg']}_{seq.spec.class_name}_{seq.spec.label}_gpu{gpu}.log"
            with lock:
                record.update(state="running", gpu=gpu, log=str(log_path),
                              started_at=datetime.now().isoformat(timespec="seconds"))
                save_json(status_path, records)
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu), "LONG_TEST_ROOT": str(job["dataset"]),
                "OUTPUT_ROOT": str(OUTPUT / f"bg{job['bg']}"),
                "DISCOVER_HIERARCHICAL_SEQUENCES": "1", "SEQUENCES": seq.spec.label,
                "CHECKPOINT_DIR": str(job["checkpoint"]), "V7_RAFT_STUDENT_PATH": str(job["student"]),
                "GUIDANCE_MODE": "temporal", "TEMPORAL_FLOW_BACKEND": "v7_student",
                "TEMPORAL_GUIDANCE_SPACE": "latent", "TEMPORAL_GUIDANCE_SCALE": "0.1",
                "TEMPORAL_START_STEP": "0", "TEMPORAL_END_STEP": "50",
                "TEMPORAL_EVERY_N_STEPS": "1", "TEMPORAL_CLIP_SIZE": "4",
                "TEMPORAL_FLOW_BATCH_SIZE": "1", "TEMPORAL_FLOW_MAX_INPUT_SIDE": "960",
                "TEMPORAL_BG_MASK_MODE": "full_stable", "SMALL_FRAME_UPSCALE_LONG_SIDE": "512",
                "NUM_INFERENCE_STEPS": "50", "SHARED_BG_NOISE_STRENGTH": "0.95",
                "VARIANCE_PRESERVING_SHARED_NOISE": "1", "MAIN_NOISE_SEED": "1234",
                "SHARED_NOISE_SEED": "6789", "MAX_IMAGES": "0", "PREFLIGHT_ONLY": "0",
                "FORCE_REGENERATE": "0", "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(BRUSHNET / "examples/brushnet/proposed")
                + os.pathsep + env.get("PYTHONPATH", ""),
            })
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"\n=== {datetime.now().isoformat()} GPU {gpu} BG{job['bg']} "
                             f"{seq.spec.label}, before={used} MiB, free={free} MiB ===\n")
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
            count = existing_count(job)
            complete = exit_code == 0 and count == job["frame_count"] and not memory_limit_hit
            with lock:
                record.update(state="complete" if complete else "failed", exit_code=exit_code,
                              existing_frames=count, peak_gpu_memory_mib=peak,
                              finished_at=datetime.now().isoformat(timespec="seconds"))
                save_json(status_path, records)
            print(f"BG{job['bg']} {seq.spec.label} GPU {gpu}: "
                  f"{record['state']} ({count}/{job['frame_count']}), peak={peak} MiB", flush=True)
            pending.task_done()

    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in GPUS]
        for future in futures:
            future.result()
    return 0 if all(record["state"] == "complete" for record in records) else 1


if __name__ == "__main__":
    sys.exit(main())
