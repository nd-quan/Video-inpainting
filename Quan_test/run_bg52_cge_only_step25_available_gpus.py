#!/usr/bin/env python3
"""Queue CGE-only BG52 workers individually as their assigned GPUs become idle."""
import argparse
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
from run_bg52_cge_x10_then_tg_step25_4gpu import environment as combined_environment
from run_bg52_cge_x10_then_tg_step25_4gpu import now, save_json, terminate_processes

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_cge_then_tg_equivclean_C25only_T25only_cge1em3_tg0p1_v7student_512/run_config.json"
EVALUATOR = ROOT / "examples/brushnet/test_brushnet_fixedBG_cge_only_step25_timed.py"
OUTPUT = ROOT / "experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_cge_only_C25only_cge1em3_tg0_512"


def environment(config, gpu, sequences):
    return dict(combined_environment(config, OUTPUT, gpu, sequences), GUIDANCE_MODE="cge",
                TEMPORAL_GUIDANCE_SCALE="0")


def idle_gpus():
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True)
    return {int(a) for a, b, c in (line.split(",") for line in text.splitlines())
            if int(b) <= 500 and int(c) <= 5}


def process_alive(pid):
    if not pid:
        return False
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text()
        return stat.rsplit(") ", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def validate(config, expected):
    sys.path.insert(0, str(ROOT / "examples/brushnet"))
    from sfu_temporal_clip_windows import build_temporal_windows

    sequences = []
    for relative, names in expected.items():
        folder = OUTPUT / "test" / relative
        if {p.name for p in folder.glob("*.png")} != names:
            raise ValueError("Output filename mismatch: " + str(relative))
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
            raise ValueError("Missing clip trace/timing")
        for index, (start, end) in enumerate(windows):
            trace = traces[index]
            if trace["frame_ids"] != ids[start:end] or timings[index]["frame_ids"] != ids[start:end]:
                raise ValueError("Frame alignment mismatch")
            if len(trace["steps"]) != config["inference_steps"]:
                raise ValueError("Incorrect total DDIM steps")
            if [s["step_index"] for s in trace["steps"] if s["cge_applied"]] != [25]:
                raise ValueError("CGE must apply only at index 25")
            if any(s["temporal_requested"] or s["temporal_applied"] for s in trace["steps"]):
                raise ValueError("Temporal guidance must remain disabled")
        seconds = sum(c["generation_seconds"] for c in timings.values())
        compute = sum(c["batch_frames"] for c in timings.values())
        sequences.append({"sequence": str(relative), "saved_frames": len(names), "completed_clips": len(windows),
                          "generated_clip_frames": compute, "generation_seconds": seconds,
                          "generation_seconds_per_generated_frame": seconds / compute,
                          "generation_seconds_per_saved_frame": seconds / len(names)})
    saved = sum(s["saved_frames"] for s in sequences)
    compute = sum(s["generated_clip_frames"] for s in sequences)
    seconds = sum(s["generation_seconds"] for s in sequences)
    report = {"validated_at": now(), "passed": True, "cge_step_indices": [25], "temporal_step_indices": [],
              "saved_frames": saved, "generated_clip_frames": compute, "generation_seconds": seconds,
              "generation_seconds_per_generated_frame": seconds / compute,
              "generation_seconds_per_saved_frame": seconds / saved,
              "semantics": "Summed synchronized batch generation durations; not parallel wall time. Tail-window overlap included; model loading excluded.",
              "sequences": sequences}
    save_json(OUTPUT / "validation.json", report)
    save_json(OUTPUT / "timing_summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
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
                raise ValueError("Input/GT/mask mismatch")
        expected[relative] = names
    if len(expected) != reference["expected_sequences"] or sum(map(len, expected.values())) != reference["expected_frames"]:
        raise ValueError("Dataset differs from reference")
    # GPU 0 became idle while preparing the queue. GPU 2 stays excluded;
    # GPU 5 retains the reference run's scheduled metrics task.
    gpus = [0, 1, 3, 4]
    prior_plan = [reference["worker_plan"][str(gpu)] for gpu in reference["gpus"]]
    config = dict(reference)
    config.update({
        "source_config": str(REFERENCE), "guidance_mode": "cge", "temporal_guidance_scale": 0.0,
        "temporal_guidance_enabled": False, "combined_guidance_order": "cge_only",
        "evaluator": str(EVALUATOR.relative_to(ROOT)), "launcher": str(Path(__file__).resolve().relative_to(ROOT)),
        "gpus": gpus, "worker_plan": {str(gpu): plan for gpu, plan in zip(gpus, prior_plan)},
        "gpu_launch_policy": "Each assigned GPU starts independently when idle; pool 0,1,3,4, maximum four workers. GPU 2 is excluded; GPU 5 retains reference metrics.",
        "requested_overrides": {"guidance_mode": "cge", "temporal_guidance_scale": 0.0, "temporal_guidance_enabled": False},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (REFERENCE, EVALUATOR, Path(__file__).resolve(),
                                    ROOT / reference["scheduler"], ROOT / reference["launcher"])},
    })
    assert config["cge_guidance_scale"] == 0.001 and config["cge_window"] == [25, 26]
    assert config["inference_steps"] == 50 and config["max_images"] == 0
    assert config["gpus"] == [0, 1, 3, 4]
    config_path = OUTPUT / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing experiment config differs")
    save_json(config_path, config)
    differences = {k: {"reference": reference.get(k), "experiment": config.get(k)}
                   for k in set(reference) | set(config) if reference.get(k) != config.get(k)}
    save_json(OUTPUT / "config_comparison.json", {"source": str(REFERENCE), "differences": differences})
    logs = OUTPUT / "terminal_logs"
    logs.mkdir(exist_ok=True)
    command = [sys.executable, "-u", str(EVALUATOR)]
    with (logs / "preflight.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, env=dict(environment(config, 1, sorted(p.name for p in expected)), PREFLIGHT_ONLY="1"),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print("Preflight passed: 14 sequences / 1198 frames; CGE=0.001 at index 25 only; TGE disabled", flush=True)
    if args.preflight_only:
        return 0

    def stop_requested(signum, frame):
        raise InterruptedError("Stop requested with signal {}".format(signum))

    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    status = {"state": "queued", "pid": os.getpid(), "queued_at": now(), "generated_frames": 0,
              "launch_policy": config["gpu_launch_policy"], "workers": {
                  gpu: {"state": "waiting_for_gpu", "sequences": plan["sequences"], "pid": None, "exit_code": None}
                  for gpu, plan in config["worker_plan"].items()}}
    status_path = OUTPUT / "status.json"
    processes, workers = [], {}
    save_json(status_path, status)
    try:
        while True:
            available = idle_gpus()
            prior = json.loads((REFERENCE.parent / "status.json").read_text())
            for gpu in config["gpus"]:
                item = status["workers"][str(gpu)]
                if gpu in workers:
                    code = workers[gpu].poll()
                    item.update(exit_code=code, state="running" if code is None else "complete" if code == 0 else "failed")
                    if code is not None and code != 0:
                        raise RuntimeError("GPU {} worker failed: {}".format(gpu, code))
                    continue
                prior_worker = prior.get("workers", {}).get(str(gpu), {})
                if process_alive(prior_worker.get("pid")):
                    item.update(state="waiting_for_gpu", waiting_reason="Reference inference worker still running")
                    continue
                # The reference launcher already owns a future metrics task on GPU 5.
                # Other idle GPUs start immediately while that reservation is active.
                if gpu == reference["gpus"][-1] and process_alive(prior.get("pid")):
                    item.update(state="waiting_for_gpu", waiting_reason="Reference metrics reservation on this GPU")
                    continue
                if gpu not in available:
                    item.update(state="waiting_for_gpu", waiting_reason="GPU memory/utilization not idle")
                    continue
                log_path = logs / "gpu{}.log".format(gpu)
                with log_path.open("a") as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                        env=environment(config, gpu, item["sequences"]), stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
                workers[gpu] = process
                processes.append(process)
                item.update(state="running", pid=process.pid, log=str(log_path), started_at=now(), waiting_reason=None)
                status.setdefault("started_at", now())
                print("Started idle GPU {} PID {}: {}".format(gpu, process.pid, item["sequences"]), flush=True)
            status.update(state="inference" if workers else "queued", updated_at=now(),
                          generated_frames=len(list((OUTPUT / "test").glob("Class_*/*/*.png"))))
            save_json(status_path, status)
            if len(workers) == len(config["gpus"]) and all(p.poll() is not None for p in workers.values()):
                break
            time.sleep(15)
        status.update(state="validating", inference_completed_at=now())
        save_json(status_path, status)
        status["timing"] = validate(config, expected)
        metrics = OUTPUT / "test/metrics_full_reference/metrics.csv"
        if not metrics.is_file():
            status["state"] = "waiting_for_metric_gpu"
            while True:
                available = idle_gpus() & set(config["gpus"])
                status["updated_at"] = now()
                save_json(status_path, status)
                if available:
                    gpu = min(available)
                    break
                time.sleep(15)
            metric_command = [sys.executable, "-u", str(ROOT / "Quan_test/evaluate_flat_sequence_metrics.py"),
                              "--eval-root", str(OUTPUT / "test"), "--gt-root", str(dataset), "--hierarchical",
                              "--sequences", *sorted(p.name for p in expected)]
            with (logs / "metrics.log").open("a") as log:
                process = subprocess.Popen(metric_command, cwd=ROOT, stdin=subprocess.DEVNULL,
                    env=environment(config, gpu, sorted(p.name for p in expected)),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            status.update(state="metrics", metrics_pid=process.pid, metrics_gpu=gpu)
            while process.poll() is None:
                status["updated_at"] = now()
                save_json(status_path, status)
                time.sleep(15)
            if process.returncode != 0:
                raise RuntimeError("Full-reference metrics failed")
        status.update(state="complete", completed_at=now(), metrics=str(metrics))
        print("CGE-only inference, zero-TGE audit and metrics complete: " + str(OUTPUT), flush=True)
    except InterruptedError as error:
        status.update(state="stopped_by_user", stopped_at=now(), stop_reason=str(error))
    except Exception as error:
        status.update(state="failed", error=str(error))
        raise
    finally:
        if status["state"] in ("stopped_by_user", "failed"):
            terminate_processes(processes)
            for gpu, process in workers.items():
                status["workers"][str(gpu)].update(state=status["state"], exit_code=process.poll())
        status.update(updated_at=now(), generated_frames=len(list((OUTPUT / "test").glob("Class_*/*/*.png"))))
        save_json(status_path, status)
    return 0 if status["state"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
