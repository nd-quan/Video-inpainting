#!/usr/bin/env python3
"""Run CGE x3/x10 on matched Traffic clips, leaving the full evaluation alone."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_cge_then_tg_equivclean_C25-40_T25-40_cge1em4_tg0p1_v7student_512"
OUTPUT = ROOT / "experiments/EVAL_FOR_FINAL/ablation_bg52_cge_x3_x10_tg0p1_C25-40_T25-40_Traffic4"


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs=2, type=int, default=[4, 5])
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.gpus)) != 2:
        raise ValueError("Two distinct GPUs required")
    baseline = json.loads((BASELINE / "run_config.json").read_text())
    evaluator = ROOT / "examples/brushnet/test_brushnet_fixedBG_cge_then_temporal_equivalent_clean_timed.py"
    # All consumers use exactly the same implementation as the x1 experiment.
    for name, digest in baseline["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Source changed since the x1 run: " + name)
    dataset = Path(baseline["dataset"])
    relative = Path("Class_A/Traffic")
    frames = sorted((dataset / "input" / relative).glob("*.png"))[:4]
    names = [p.name for p in frames]
    if names != ["000117.png", "000118.png", "000119.png", "000120.png"]:
        raise ValueError("Unexpected Traffic first clip: " + str(names))
    for kind in ("mask", "GT"):
        for name in names:
            if not (dataset / kind / relative / name).is_file():
                raise FileNotFoundError(dataset / kind / relative / name)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "terminal_logs").mkdir(exist_ok=True)
    plan = []
    for factor, scale, gpu in zip((3, 10), (0.0003, 0.001), args.gpus):
        output = OUTPUT / "cge_x{}".format(factor)
        logs = output / "terminal_logs"
        logs.mkdir(parents=True, exist_ok=True)
        config = dict(baseline)
        config.update({
            "evaluator": str(evaluator.relative_to(ROOT)),
            "launcher": str(Path(__file__).resolve().relative_to(ROOT)),
            "source_config": str(BASELINE / "run_config.json"),
            "gpus": [gpu], "sequences": ["Traffic"], "max_images": 4,
            "expected_sequences": 1, "expected_frames": 4,
            "frame_ids": [117, 118, 119, 120],
            "cge_guidance_scale": scale, "cge_scale_factor": factor,
            "baseline_cge_guidance_scale": baseline["cge_guidance_scale"],
            "generation_timing": "CUDA-synchronized generation_timing.jsonl, batch time / 4 images",
        })
        config["source_sha256"] = dict(baseline["source_sha256"])
        config["source_sha256"][str(evaluator.relative_to(ROOT))] = hashlib.sha256(evaluator.read_bytes()).hexdigest()
        config.pop("worker_plan", None)
        path = output / "run_config.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("Existing output config differs: " + str(path))
        save_json(path, config)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("CGE_", "TEMPORAL_", "TEACHER_FLOW_"))
               and k not in ("BASE_MODEL_PATH", "SEQUENCES", "CUDA_VISIBLE_DEVICES")}
        env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu), "LONG_TEST_ROOT": baseline["dataset"],
            "OUTPUT_ROOT": str(output), "CHECKPOINT_DIR": baseline["checkpoint"],
            "V7_RAFT_STUDENT_PATH": baseline["flow_checkpoint"],
            "SEQUENCES": "Traffic", "MAX_IMAGES": "4", "FORCE_REGENERATE": "0",
            "DISCOVER_HIERARCHICAL_SEQUENCES": "1", "PREFLIGHT_ONLY": "0",
            "GUIDANCE_MODE": "combined", "TEMPORAL_GUIDANCE_SPACE": baseline["temporal_guidance_space"],
            "TEMPORAL_FLOW_BACKEND": baseline["temporal_flow_backend"],
            "TEMPORAL_BG_MASK_MODE": baseline["temporal_bg_mask_mode"],
            "TEMPORAL_GUIDANCE_SCALE": str(baseline["temporal_guidance_scale"]),
            "TEMPORAL_START_STEP": str(baseline["temporal_window"][0]),
            "TEMPORAL_END_STEP": str(baseline["temporal_window"][1]),
            "TEMPORAL_EVERY_N_STEPS": str(baseline["temporal_every_n_steps"]),
            "TEMPORAL_CLIP_SIZE": str(baseline["temporal_clip_size"]),
            "TEMPORAL_CLIP_STRIDE": str(baseline["temporal_clip_stride"]),
            "TEMPORAL_DETACH_PREVIOUS": str(int(baseline["temporal_detach_previous"])),
            "NUM_INFERENCE_STEPS": str(baseline["inference_steps"]),
            "MAIN_NOISE_SEED": str(baseline["main_noise_seed"]),
            "SHARED_NOISE_SEED": str(baseline["shared_noise_seed"]),
            "SHARED_BG_NOISE_STRENGTH": str(baseline["shared_bg_noise_strength"]),
            "VARIANCE_PRESERVING_SHARED_NOISE": str(int(baseline["variance_preserving_shared_noise"])),
            "CGE_GUIDANCE_SCALE": str(scale), "CGE_SCALE_SCHEDULE": baseline["cge_scale_schedule"],
            "CGE_START_STEP": str(baseline["cge_window"][0]),
            "CGE_END_STEP": str(baseline["cge_window"][1]),
            "CGE_EVERY_N_STEPS": str(baseline["cge_every_n_steps"]),
            "CGE_CODEC_MODE": baseline["cge_codec_mode"],
            "CGE_PER_FRAME": str(int(baseline["cge_per_frame"])),
            "CGE_MAX_EVALS": str(baseline["cge_max_evals"]),
            "CGE_VCMRS_ROOT": "/home/cilab/ndquan/vcm-rs",
            "CGE_VCMRS_PYTHON": "/home/cilab/ndquan/envs/vcm/bin/python",
            "CGE_VCMRS_BG_QUALITY": str(baseline["cge_bg_quality"]),
            "CGE_VCMRS_ROI_QUALITY": "20", "CGE_VCMRS_PROFILE": baseline["cge_profile"],
            "CGE_VCMRS_AUTO_DESCRIPTORS": "1",
            "CGE_VCMRS_CONFIGURATION": baseline["cge_configuration"],
            "CGE_VCMRS_INTRA_PERIOD": str(baseline["cge_intra_period"]),
            "CGE_VCMRS_NN_INTRA_QP_OFFSET": str(baseline["cge_nn_intra_qp_offset"]),
            "CGE_VCMRS_FRAME_RATE": str(baseline["cge_frame_rate"]),
            "CGE_VCMRS_MAX_PARALLEL": str(baseline["cge_max_parallel"]),
            "CGE_CODEC_WORKERS": str(baseline["cge_codec_workers"]),
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1",
            "TZ": "Asia/Seoul",
            "PYTHONPATH": str(ROOT / "examples/brushnet/proposed") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        })
        command = [sys.executable, "-u", str(evaluator)]
        with (logs / "preflight.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=dict(env, PREFLIGHT_ONLY="1"),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print("Preflight OK: CGE x{}, scale={}, GPU={}, frames={}".format(factor, scale, gpu, names), flush=True)
        plan.append((factor, gpu, output, command, env))
    if args.preflight_only:
        return 0

    gpu_text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True)
    available = {int(a): (int(b), int(c)) for a, b, c in (line.split(",") for line in gpu_text.splitlines())}
    for gpu in args.gpus:
        if available[gpu][0] > 500 or available[gpu][1] > 5:
            raise RuntimeError("Selected GPU {} is occupied: {}".format(gpu, available[gpu]))

    status_path = OUTPUT / "status.json"
    status = {"state": "inference", "pid": os.getpid(), "started_at": now(), "scope": "Traffic first four frames",
              "temporal_guidance_scale": 0.1, "temporal_window": [25, 40], "cge_window": [25, 40], "runs": {}}
    workers = []
    save_json(status_path, status)
    try:
        for factor, gpu, output, command, env in plan:
            log_path = output / "terminal_logs/evaluate.log"
            with log_path.open("a") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
            workers.append((factor, process, output))
            status["runs"][str(factor)] = {"pid": process.pid, "gpu": gpu, "output": str(output), "log": str(log_path)}
        while True:
            running = False
            for factor, process, output in workers:
                code = process.poll()
                running = running or code is None
                status["runs"][str(factor)].update(exit_code=code,
                    generated_frames=len(list((output / "test/Class_A/Traffic").glob("*.png"))))
            status["updated_at"] = now()
            save_json(status_path, status)
            if not running:
                break
            time.sleep(15)
        if any(process.returncode != 0 for _, process, _ in workers):
            raise RuntimeError("A scale-ablation worker failed; inspect its evaluate.log")
        for factor, _, output in workers:
            folder = output / "test/Class_A/Traffic"
            if sorted(p.name for p in folder.glob("*.png")) != names:
                raise ValueError("Output frame mismatch for x{}".format(factor))
            clips = [json.loads(s) for s in (folder / "guidance_trace.jsonl").read_text().splitlines()]
            trace = clips[-1]
            if trace["frame_ids"] != [117, 118, 119, 120]:
                raise ValueError("Unexpected clip IDs")
            for key in ("cge_applied", "temporal_applied"):
                if [s["step_index"] for s in trace["steps"] if s[key]] != list(range(25, 40)):
                    raise ValueError("Guidance window/application mismatch: " + key)
            timings = [json.loads(s) for s in (folder / "generation_timing.jsonl").read_text().splitlines()]
            status["runs"][str(factor)]["timing"] = timings[-1]
        status["state"] = "comparison"
        save_json(status_path, status)
        with (OUTPUT / "terminal_logs/comparison.log").open("a") as log:
            subprocess.run([sys.executable, "-u", str(ROOT / "Quan_test/compare_bg52_cge_scale_ablation.py")],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        status.update(state="complete", comparison=str(OUTPUT / "comparison/summary.csv"))
        print("CGE x3/x10 complete: " + str(OUTPUT), flush=True)
    except Exception as error:
        status.update(state="failed", error=str(error))
        raise
    finally:
        status["updated_at"] = now()
        save_json(status_path, status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
