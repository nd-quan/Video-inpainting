#!/usr/bin/env python3
"""Run isolated BG52 CGE -> equivalent clean latent -> TGE on four GPUs."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_latent_tg0p1_v7student_512/run_config.json"
EVALUATOR = ROOT / "examples/brushnet/inference_experiments/test_brushnet_fixedBG_cge_then_temporal_equivalent_clean.py"
SCHEDULER = ROOT / "examples/brushnet/diffusers/schedulers/scheduling_ddim_cge_then_temporal.py"
DEFAULT_OUTPUT = ROOT / "experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_cge_then_tg_equivclean_C25-40_T25-40_cge1em4_tg0p1_v7student_512"


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs=4, default=[0, 1, 2, 3])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.gpus)) != 4:
        raise ValueError("Four distinct GPUs required")
    reference = json.loads(REFERENCE.read_text())
    dataset = Path(reference["dataset"])
    output = args.output.resolve()
    if output == REFERENCE.parent:
        raise ValueError("Output must be separate from the baseline")
    expected = {}
    for folder in sorted((dataset / "input").glob("Class_*/*")):
        relative = folder.relative_to(dataset / "input")
        names = {p.name for p in folder.glob("*.png")}
        if not names:
            raise ValueError("Empty sequence: " + str(folder))
        for kind in ("mask", "GT"):
            if names != {p.name for p in (dataset / kind / relative).glob("*.png")}:
                raise ValueError("{} frame mismatch: {}".format(kind, relative))
        expected[relative] = names
    if len(expected) != reference["expected_sequences"] or sum(map(len, expected.values())) != reference["expected_frames"]:
        raise ValueError("Dataset counts differ from reference")
    for path in (EVALUATOR, SCHEDULER, Path(reference["checkpoint"]) / "brushnet",
                 Path(reference["checkpoint"]) / "ipadapter", Path(reference["flow_checkpoint"]),
                 Path("/home/cilab/ndquan/envs/vcm/bin/python")):
        if not path.exists():
            raise FileNotFoundError(path)

    # Assign entire sequences by frame count; workers keep models loaded.
    groups = [[] for _ in args.gpus]
    loads = [0 for _ in args.gpus]
    for relative in sorted(expected, key=lambda p: (-len(expected[p]), str(p))):
        index = min(range(4), key=lambda i: loads[i])
        groups[index].append(relative.name)
        loads[index] += len(expected[relative])
    plan = {str(gpu): {"sequences": sequences, "frames": load}
            for gpu, sequences, load in zip(args.gpus, groups, loads)}
    config = dict(reference)
    config.update({
        "evaluator": str(EVALUATOR.relative_to(ROOT)),
        "launcher": str(Path(__file__).resolve().relative_to(ROOT)),
        "source_config": str(REFERENCE), "gpus": args.gpus,
        "guidance_mode": "combined",
        "combined_guidance_order": "cge_then_temporal_equivalent_clean",
        "scheduler": str(SCHEDULER.relative_to(ROOT)),
        "equivalent_clean_formula": "x0_C = x0 + delta_C / sqrt(alpha_previous)",
        "temporal_window": [25, 40], "cge_window": [25, 40],
        "window_semantics": "zero-based denoising-loop indices, end exclusive: 25..39",
        "temporal_clip_stride": reference.get("temporal_clip_stride", reference["temporal_clip_size"]),
        "cge_guidance_scale": 0.0001, "cge_scale_schedule": "fixed",
        "cge_every_n_steps": 1, "cge_codec_mode": "bg_only", "cge_per_frame": False,
        "cge_bg_quality": 52, "cge_profile": "train_match",
        "cge_configuration": "RandomAccess", "cge_intra_period": 64,
        "cge_nn_intra_qp_offset": -5, "cge_frame_rate": 30,
        "cge_max_evals": -1, "cge_codec_workers": 1, "cge_max_parallel": 1,
        "eta": 0.0, "worker_plan": plan,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (REFERENCE, EVALUATOR, SCHEDULER)},
    })
    logs = output / "terminal_logs"
    logs.mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing experiment config differs: " + str(config_path))
    save_json(config_path, config)

    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("TEMPORAL_", "CGE_", "TEACHER_FLOW_"))
           and k not in ("BASE_MODEL_PATH", "SEQUENCES", "CUDA_VISIBLE_DEVICES")}
    env.update({
        "LONG_TEST_ROOT": str(dataset), "OUTPUT_ROOT": str(output),
        "CHECKPOINT_DIR": reference["checkpoint"], "V7_RAFT_STUDENT_PATH": reference["flow_checkpoint"],
        "DISCOVER_HIERARCHICAL_SEQUENCES": "1", "MAX_IMAGES": str(reference["max_images"]),
        "FORCE_REGENERATE": "0", "GUIDANCE_MODE": "combined",
        "TEMPORAL_FLOW_BACKEND": reference["temporal_flow_backend"],
        "TEMPORAL_GUIDANCE_SPACE": reference["temporal_guidance_space"],
        "TEMPORAL_GUIDANCE_SCALE": str(reference["temporal_guidance_scale"]),
        "TEMPORAL_START_STEP": "25", "TEMPORAL_END_STEP": "40",
        "TEMPORAL_EVERY_N_STEPS": str(reference["temporal_every_n_steps"]),
        "TEMPORAL_CLIP_SIZE": str(reference["temporal_clip_size"]),
        "TEMPORAL_CLIP_STRIDE": str(config["temporal_clip_stride"]),
        "TEMPORAL_BG_MASK_MODE": reference["temporal_bg_mask_mode"],
        "TEMPORAL_DETACH_PREVIOUS": str(int(reference["temporal_detach_previous"])),
        "NUM_INFERENCE_STEPS": str(reference["inference_steps"]),
        "SHARED_BG_NOISE_STRENGTH": str(reference["shared_bg_noise_strength"]),
        "VARIANCE_PRESERVING_SHARED_NOISE": str(int(reference["variance_preserving_shared_noise"])),
        "MAIN_NOISE_SEED": str(reference["main_noise_seed"]),
        "SHARED_NOISE_SEED": str(reference["shared_noise_seed"]),
        "CGE_START_STEP": "25", "CGE_END_STEP": "40", "CGE_EVERY_N_STEPS": "1",
        "CGE_GUIDANCE_SCALE": "0.0001", "CGE_SCALE_SCHEDULE": "fixed",
        "CGE_CODEC_MODE": "bg_only", "CGE_PER_FRAME": "0", "CGE_MAX_EVALS": "-1",
        "CGE_VCMRS_ROOT": "/home/cilab/ndquan/vcm-rs",
        "CGE_VCMRS_PYTHON": "/home/cilab/ndquan/envs/vcm/bin/python",
        "CGE_VCMRS_BG_QUALITY": "52", "CGE_VCMRS_ROI_QUALITY": "20",
        "CGE_VCMRS_PROFILE": "train_match", "CGE_VCMRS_AUTO_DESCRIPTORS": "1",
        "CGE_VCMRS_CONFIGURATION": "RandomAccess", "CGE_VCMRS_INTRA_PERIOD": "64",
        "CGE_VCMRS_NN_INTRA_QP_OFFSET": "-5", "CGE_VCMRS_FRAME_RATE": "30",
        "CGE_VCMRS_MAX_PARALLEL": "1", "CGE_CODEC_WORKERS": "1",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1",
        "TZ": "Asia/Seoul", "PREFLIGHT_ONLY": "0",
        "PYTHONPATH": str(ROOT / "examples/brushnet/proposed") + os.pathsep + os.environ.get("PYTHONPATH", ""),
    })
    command = [sys.executable, "-u", str(EVALUATOR)]
    with (logs / "preflight.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, env=dict(env, PREFLIGHT_ONLY="1"),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print("Preflight passed: {} sequences, {} frames; plan={}".format(
        len(expected), sum(map(len, expected.values())), plan), flush=True)
    if args.preflight_only:
        return 0

    gpu_info = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True)
    available = {int(parts[0]): (int(parts[1]), int(parts[2]))
                 for line in gpu_info.splitlines() for parts in [line.split(",")]}
    for gpu in args.gpus:
        memory, utilization = available[gpu]
        if memory > 500 or utilization > 5:
            raise RuntimeError("GPU {} is no longer idle: {} MiB / {}%".format(gpu, memory, utilization))

    status_path = output / "status.json"
    if status_path.exists():
        old = json.loads(status_path.read_text())
        if old.get("state") in ("inference", "metrics"):
            try:
                os.kill(old["pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("Another launcher is active: PID {}".format(old["pid"]))
    status = {"state": "inference", "pid": os.getpid(), "started_at": now(), "workers": {}}
    workers = []
    save_json(status_path, status)
    try:
        for gpu, sequences in zip(args.gpus, groups):
            log_path = logs / "gpu{}.log".format(gpu)
            with log_path.open("a") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                           env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu), SEQUENCES=",".join(sequences)),
                                           stdout=log, stderr=subprocess.STDOUT)
            workers.append((gpu, process))
            status["workers"][str(gpu)] = {"pid": process.pid, "sequences": sequences, "log": str(log_path)}
        while True:
            running = False
            for gpu, process in workers:
                code = process.poll()
                status["workers"][str(gpu)]["exit_code"] = code
                running = running or code is None
            status.update(generated_frames=len(list((output / "test").glob("Class_*/*/*.png"))), updated_at=now())
            save_json(status_path, status)
            if not running:
                break
            time.sleep(15)
        if any(process.returncode != 0 for _, process in workers):
            raise RuntimeError("Inference worker failed; see terminal_logs/gpu*.log")
        for relative, names in expected.items():
            folder = output / "test" / relative
            if {p.name for p in folder.glob("*.png")} != names:
                raise ValueError("Output frame mismatch: " + str(relative))
            for name in names:
                with Image.open(folder / name) as image:
                    if image.size != tuple(reference["output_size"]):
                        raise ValueError("Output dimensions differ from baseline")
        status.update(state="metrics", verified_frames=sum(map(len, expected.values())))
        save_json(status_path, status)
        metrics = output / "test/metrics_full_reference"
        if not (metrics / "metrics.csv").is_file():
            with (logs / "metrics.log").open("a") as log:
                subprocess.run([
                    sys.executable, "-u", str(ROOT / "Quan_test/evaluate_flat_sequence_metrics.py"),
                    "--eval-root", str(output / "test"), "--gt-root", str(dataset), "--hierarchical",
                    "--sequences", *sorted(p.name for p in expected),
                ], cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=str(args.gpus[-1])),
                    stdout=log, stderr=subprocess.STDOUT, check=True)
        status.update(state="complete", metrics=str(metrics / "metrics.csv"))
        print("Inference and metrics complete: " + str(output), flush=True)
    except Exception as error:
        status.update(state="failed", error=str(error))
        raise
    finally:
        status["updated_at"] = now()
        save_json(status_path, status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
