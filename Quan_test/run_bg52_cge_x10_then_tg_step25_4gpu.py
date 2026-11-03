#!/usr/bin/env python3
"""Evaluate BG52 with CGE x10 then TGE only at DDIM loop index 25."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_cge_then_tg_equivclean_C25-40_T25-40_cge1em4_tg0p1_v7student_512/run_config.json"
EVALUATOR = ROOT / "examples/brushnet/test_brushnet_fixedBG_cge_then_temporal_equivalent_clean_timed.py"
SCHEDULER = ROOT / "examples/brushnet/diffusers/schedulers/scheduling_ddim_cge_then_temporal.py"
OUTPUT = ROOT / "experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_cge_then_tg_equivclean_C25only_T25only_cge1em3_tg0p1_v7student_512"


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def environment(config, output, gpu, sequences):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CGE_", "TEMPORAL_", "TEACHER_FLOW_"))
           and k not in ("BASE_MODEL_PATH", "SEQUENCES", "CUDA_VISIBLE_DEVICES")}
    mapping = {
        "LONG_TEST_ROOT": "dataset", "CHECKPOINT_DIR": "checkpoint", "V7_RAFT_STUDENT_PATH": "flow_checkpoint",
        "TEMPORAL_GUIDANCE_SPACE": "temporal_guidance_space", "TEMPORAL_FLOW_BACKEND": "temporal_flow_backend",
        "TEMPORAL_BG_MASK_MODE": "temporal_bg_mask_mode", "TEMPORAL_GUIDANCE_SCALE": "temporal_guidance_scale",
        "TEMPORAL_CLIP_SIZE": "temporal_clip_size", "TEMPORAL_CLIP_STRIDE": "temporal_clip_stride",
        "TEMPORAL_EVERY_N_STEPS": "temporal_every_n_steps", "TEMPORAL_DETACH_PREVIOUS": "temporal_detach_previous",
        "NUM_INFERENCE_STEPS": "inference_steps", "MAIN_NOISE_SEED": "main_noise_seed",
        "SHARED_NOISE_SEED": "shared_noise_seed", "SHARED_BG_NOISE_STRENGTH": "shared_bg_noise_strength",
        "VARIANCE_PRESERVING_SHARED_NOISE": "variance_preserving_shared_noise",
        "CGE_GUIDANCE_SCALE": "cge_guidance_scale", "CGE_SCALE_SCHEDULE": "cge_scale_schedule",
        "CGE_EVERY_N_STEPS": "cge_every_n_steps", "CGE_CODEC_MODE": "cge_codec_mode",
        "CGE_PER_FRAME": "cge_per_frame", "CGE_MAX_EVALS": "cge_max_evals",
        "CGE_VCMRS_BG_QUALITY": "cge_bg_quality", "CGE_VCMRS_PROFILE": "cge_profile",
        "CGE_VCMRS_CONFIGURATION": "cge_configuration", "CGE_VCMRS_INTRA_PERIOD": "cge_intra_period",
        "CGE_VCMRS_NN_INTRA_QP_OFFSET": "cge_nn_intra_qp_offset", "CGE_VCMRS_FRAME_RATE": "cge_frame_rate",
        "CGE_VCMRS_MAX_PARALLEL": "cge_max_parallel", "CGE_CODEC_WORKERS": "cge_codec_workers",
        "MAX_IMAGES": "max_images",
    }
    for key, source in mapping.items():
        value = config[source]
        env[key] = str(int(value) if isinstance(value, bool) else value)
    env.update({
        "OUTPUT_ROOT": str(output), "CUDA_VISIBLE_DEVICES": str(gpu), "SEQUENCES": ",".join(sequences),
        "FORCE_REGENERATE": "0", "DISCOVER_HIERARCHICAL_SEQUENCES": "1", "PREFLIGHT_ONLY": "0",
        "GUIDANCE_MODE": "combined", "TEMPORAL_START_STEP": "25", "TEMPORAL_END_STEP": "26",
        "CGE_START_STEP": "25", "CGE_END_STEP": "26",
        "CGE_VCMRS_ROOT": "/home/cilab/ndquan/vcm-rs", "CGE_VCMRS_PYTHON": "/home/cilab/ndquan/envs/vcm/bin/python",
        "CGE_VCMRS_ROI_QUALITY": "20", "CGE_VCMRS_AUTO_DESCRIPTORS": "1",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1", "TZ": "Asia/Seoul",
        "PYTHONPATH": str(ROOT / "examples/brushnet/proposed") + os.pathsep + os.environ.get("PYTHONPATH", ""),
    })
    return env


def validate_outputs(output, config, expected):
    sys.path.insert(0, str(ROOT / "examples/brushnet"))
    from sfu_temporal_clip_windows import build_temporal_windows

    sequences = []
    for relative, names in expected.items():
        folder = output / "test" / relative
        if {p.name for p in folder.glob("*.png")} != names:
            raise ValueError("Incomplete or unexpected frames: " + str(relative))
        for name in names:
            with Image.open(folder / name) as image:
                if image.size != tuple(config["output_size"]):
                    raise ValueError("Output resolution mismatch")
                image.verify()
        ids = sorted(int(Path(name).stem) for name in names)
        windows = build_temporal_windows(len(ids), config["temporal_clip_size"], config["temporal_clip_stride"])
        traces = {c["clip_index"]: c for c in map(json.loads, (folder / "guidance_trace.jsonl").read_text().splitlines())}
        timings = {c["clip_index"]: c for c in map(json.loads, (folder / "generation_timing.jsonl").read_text().splitlines())}
        if set(traces) != set(range(len(windows))) or set(timings) != set(traces):
            raise ValueError("Missing clip audit/timing: " + str(relative))
        for index, (start, end) in enumerate(windows):
            trace = traces[index]
            if trace["frame_ids"] != ids[start:end] or timings[index]["frame_ids"] != ids[start:end]:
                raise ValueError("Frame alignment mismatch")
            if len(trace["steps"]) != config["inference_steps"]:
                raise ValueError("Incorrect total DDIM step count")
            for key in ("cge_applied", "temporal_requested", "temporal_applied"):
                if [s["step_index"] for s in trace["steps"] if s[key]] != [25]:
                    raise ValueError("Guidance must run only at index 25: " + str(relative))
            guided = trace["steps"][25]
            if guided["temporal_input"] != "equivalent_clean_after_cge":
                raise ValueError("Incorrect guidance order")
        seconds = sum(c["generation_seconds"] for c in timings.values())
        total_seconds = sum(c["total_clip_seconds"] for c in timings.values())
        compute_frames = sum(c["batch_frames"] for c in timings.values())
        sequences.append({"sequence": str(relative), "saved_frames": len(names), "completed_clips": len(windows),
                          "generated_clip_frames": compute_frames, "generation_seconds": seconds,
                          "generation_seconds_per_generated_frame": seconds / compute_frames,
                          "generation_seconds_per_saved_frame": seconds / len(names),
                          "total_clip_seconds": total_seconds})
    saved = sum(c["saved_frames"] for c in sequences)
    compute = sum(c["generated_clip_frames"] for c in sequences)
    seconds = sum(c["generation_seconds"] for c in sequences)
    report = {"validated_at": now(), "saved_frames": saved, "generated_clip_frames": compute,
              "completed_clips": sum(c["completed_clips"] for c in sequences),
              "guidance_step_indices": [25], "generation_seconds": seconds,
              "generation_seconds_per_generated_frame": seconds / compute,
              "generation_seconds_per_saved_frame": seconds / saved,
              "semantics": "Summed synchronized generation duration, not parallel wall time; model loading excluded. Tail-window overlap appears in compute frame count.",
              "sequences": sequences}
    save_json(output / "timing_summary.json", report)
    save_json(output / "validation.json", {"passed": True, **report})
    return report


def terminate_processes(processes):
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 8
    for process in processes:
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs=4, type=int, default=[1, 3, 4, 5])
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if len(set(args.gpus)) != 4:
        raise ValueError("Four distinct GPU slots required")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reference = json.loads(REFERENCE.read_text())
    for name, expected_hash in reference["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Reference source changed: " + name)
    dataset = Path(reference["dataset"])
    expected = {}
    for folder in sorted((dataset / "input").glob("Class_*/*")):
        relative = folder.relative_to(dataset / "input")
        names = {p.name for p in folder.glob("*.png")}
        if not names:
            raise ValueError("Empty sequence")
        for kind in ("mask", "GT"):
            if {p.name for p in (dataset / kind / relative).glob("*.png")} != names:
                raise ValueError("Input/GT/mask frame mismatch: " + str(relative))
        expected[relative] = names
    if len(expected) != reference["expected_sequences"] or sum(map(len, expected.values())) != reference["expected_frames"]:
        raise ValueError("Dataset differs from reference")
    for path in (EVALUATOR, SCHEDULER, Path(reference["checkpoint"]) / "brushnet",
                 Path(reference["checkpoint"]) / "ipadapter/model.safetensors", Path(reference["flow_checkpoint"]),
                 Path("/home/cilab/ndquan/envs/vcm/bin/python")):
        if not path.exists():
            raise FileNotFoundError(path)
    groups, loads = [[] for _ in args.gpus], [0 for _ in args.gpus]
    for relative in sorted(expected, key=lambda p: (-len(expected[p]), str(p))):
        index = min(range(4), key=lambda i: loads[i])
        groups[index].append(relative.name)
        loads[index] += len(expected[relative])
    config = dict(reference)
    config.update({
        "source_config": str(REFERENCE), "evaluator": str(EVALUATOR.relative_to(ROOT)),
        "launcher": str(Path(__file__).resolve().relative_to(ROOT)), "gpus": args.gpus,
        "cge_window": [25, 26], "temporal_window": [25, 26], "cge_guidance_scale": 0.001,
        "window_semantics": "zero-based denoising-loop index 25 only; [25,26), end exclusive; total 50 DDIM steps",
        "worker_plan": {str(gpu): {"sequences": seqs, "frames": count}
                        for gpu, seqs, count in zip(args.gpus, groups, loads)},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (REFERENCE, EVALUATOR, SCHEDULER, Path(__file__).resolve())},
        "requested_overrides": {"cge_guidance_scale": 0.001, "cge_window": [25, 26], "temporal_window": [25, 26]},
        "timing": "CUDA-synchronized generate_fgbg duration and total clip duration in per-sequence generation_timing.jsonl",
    })
    assert config["temporal_guidance_scale"] == 0.1 and config["max_images"] == 0
    assert config["combined_guidance_order"] == "cge_then_temporal_equivalent_clean"
    config_path = OUTPUT / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing experiment config differs")
    save_json(config_path, config)
    logs = OUTPUT / "terminal_logs"
    logs.mkdir(exist_ok=True)
    command = [sys.executable, "-u", str(EVALUATOR)]
    with (logs / "preflight.log").open("w") as log:
        subprocess.run(command, cwd=ROOT,
                       env=dict(environment(config, OUTPUT, args.gpus[0], sorted(p.name for p in expected)), PREFLIGHT_ONLY="1"),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print("Preflight passed: {} sequences / {} frames; CGE=0.001 -> TGE=0.1 at step 25 only; plan={}".format(
        len(expected), sum(map(len, expected.values())), config["worker_plan"]), flush=True)
    if args.preflight_only:
        return 0
    gpu_text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                        "--format=csv,noheader,nounits"], text=True)
    available = {int(a): (int(b), int(c)) for a, b, c in (line.split(",") for line in gpu_text.splitlines())}
    for gpu in args.gpus:
        if available[gpu][0] > 500 or available[gpu][1] > 5:
            raise RuntimeError("GPU {} is occupied; no dynamic GPU selection or additional GPUs".format(gpu))

    def stop_requested(signum, frame):
        raise InterruptedError("Stop requested with signal {}".format(signum))

    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    status = {"state": "inference", "pid": os.getpid(), "started_at": now(), "workers": {}}
    status_path = OUTPUT / "status.json"
    processes = []
    workers = []
    save_json(status_path, status)
    try:
        for gpu, sequences in zip(args.gpus, groups):
            log_path = logs / "gpu{}.log".format(gpu)
            with log_path.open("a") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                    env=environment(config, OUTPUT, gpu, sequences), stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
            processes.append(process)
            workers.append((gpu, process))
            status["workers"][str(gpu)] = {"pid": process.pid, "state": "running", "sequences": sequences,
                                           "log": str(log_path), "started_at": now(), "exit_code": None}
            print("Started GPU {} PID {}: {}".format(gpu, process.pid, sequences), flush=True)
        while True:
            running = False
            for gpu, process in workers:
                code = process.poll()
                status["workers"][str(gpu)].update(exit_code=code,
                    state="running" if code is None else "complete" if code == 0 else "failed")
                if code is not None and code != 0:
                    raise RuntimeError("GPU {} worker failed: {}".format(gpu, code))
                running = running or code is None
            status.update(updated_at=now(), generated_frames=len(list((OUTPUT / "test").glob("Class_*/*/*.png"))))
            save_json(status_path, status)
            if not running:
                break
            time.sleep(15)
        status.update(state="validating", inference_completed_at=now())
        save_json(status_path, status)
        status["timing"] = validate_outputs(OUTPUT, config, expected)
        metric_file = OUTPUT / "test/metrics_full_reference/metrics.csv"
        if not metric_file.is_file():
            metric_command = [sys.executable, "-u", str(ROOT / "Quan_test/evaluate_flat_sequence_metrics.py"),
                              "--eval-root", str(OUTPUT / "test"), "--gt-root", str(dataset), "--hierarchical",
                              "--sequences", *sorted(p.name for p in expected)]
            with (logs / "metrics.log").open("a") as log:
                process = subprocess.Popen(metric_command, cwd=ROOT, stdin=subprocess.DEVNULL,
                    env=environment(config, OUTPUT, args.gpus[-1], sorted(p.name for p in expected)),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            status.update(state="metrics", metrics_pid=process.pid)
            save_json(status_path, status)
            while process.poll() is None:
                status["updated_at"] = now()
                save_json(status_path, status)
                time.sleep(15)
            if process.returncode != 0:
                raise RuntimeError("Full-reference metrics failed")
        status.update(state="complete", metrics=str(metric_file), completed_at=now())
        print("Inference, step-25-only audit and metrics complete: " + str(OUTPUT), flush=True)
    except InterruptedError as error:
        status.update(state="stopped_by_user", stop_reason=str(error), stopped_at=now())
    except Exception as error:
        status.update(state="failed", error=str(error))
        raise
    finally:
        if status["state"] in ("stopped_by_user", "failed"):
            terminate_processes(processes)
            for gpu, process in workers:
                status["workers"][str(gpu)].update(exit_code=process.poll(), state=status["state"])
        status.update(updated_at=now(), generated_frames=len(list((OUTPUT / "test").glob("Class_*/*/*.png"))))
        save_json(status_path, status)
    return 0 if status["state"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
