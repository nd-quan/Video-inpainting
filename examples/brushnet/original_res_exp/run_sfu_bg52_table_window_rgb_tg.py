#!/usr/bin/env python3
"""Evaluate every BG52 SFU table-window sequence with shared noise and RGB TG."""

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

from PIL import Image

from sfu_long_test_loader import load_sequences


BRUSHNET = Path(__file__).resolve().parents[2]
DATASET = Path("/home/cilab/ndquan/vcm-rs/output/sfu_qp0_bg_table_window_test/bg52/test")
CHECKPOINT = BRUSHNET / "experiments/finetune_shared_noise_original_size/checkpoint-1000"
STUDENT = BRUSHNET / "experiments/train_v7_raft_student_flow/checkpoint-0004750/raft_student"
OUTPUT = BRUSHNET / "experiments/eval_sfu_bg52_table_window_ckpt1000_rgb_tg"
EVALUATOR = Path(__file__).with_name("test_brushnet_fixedBG_temporal_v7_student_original_size.py")
PYTHON = Path("/home/cilab/ndquan/envs/guided_diff/bin/python")
GPUS = tuple(range(6))


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
        raise ValueError("Expected 14 BG52 sequences and 1198 test frames")

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
            "state": "complete" if existing == expected else "queued",
        })
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
    })

    # Start the largest sequences first; each GPU takes another sequence as it frees up.
    pending = sorted(
        (index for index, record in enumerate(records) if record["state"] == "queued"),
        key=lambda index: records[index]["frame_count"]
        * records[index]["size_wh"][0] * records[index]["size_wh"][1],
        reverse=True,
    )
    jobs: queue.Queue[int] = queue.Queue()
    for index in pending:
        jobs.put(index)
    lock = Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                index = jobs.get_nowait()
            except queue.Empty:
                return
            seq, record = sequences[index], records[index]
            label = seq.spec.label
            log_path = log_dir / f"{seq.spec.class_name}_{label}_gpu{gpu}.log"
            with lock:
                record.update(state="running", gpu=gpu, log=str(log_path),
                              started_at=datetime.now().isoformat(timespec="seconds"))
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
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"\n=== {datetime.now().isoformat()} GPU {gpu} {label} ===\n")
                stream.flush()
                result = subprocess.run([str(PYTHON), "-u", str(EVALUATOR)],
                                        cwd=BRUSHNET, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, check=False)
            expected, existing = expected_and_existing(seq)
            complete = result.returncode == 0 and existing == expected
            with lock:
                record.update(state="complete" if complete else "failed",
                              exit_code=result.returncode, existing_frames=len(existing),
                              finished_at=datetime.now().isoformat(timespec="seconds"))
                save_json(status_path, records)
            print(f"GPU {gpu} {label}: exit={result.returncode}, frames={len(existing)}/{len(expected)}",
                  flush=True)
            jobs.task_done()

    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in GPUS]
        for future in futures:
            future.result()
    return 0 if all(record["state"] == "complete" for record in records) else 1


if __name__ == "__main__":
    sys.exit(main())
