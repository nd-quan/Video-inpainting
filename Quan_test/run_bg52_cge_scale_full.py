#!/usr/bin/env python3
"""Full BG52 x3/x10 evaluations with fixed GPU pools, resume and timing."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_cge_then_tg_equivclean_C25-40_T25-40_cge1em4_tg0p1_v7student_512"
PILOT = ROOT / "experiments/EVAL_FOR_FINAL/ablation_bg52_cge_x3_x10_tg0p1_C25-40_T25-40_Traffic4"
OUTPUT = ROOT / "experiments/EVAL_FOR_FINAL/ablation_bg52_cge_x3_x10_tg0p1_C25-40_T25-40_FULL"
EVALUATOR = ROOT / "examples/brushnet/test_brushnet_fixedBG_cge_then_temporal_equivalent_clean_timed.py"


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
    }
    for key, source in mapping.items():
        value = config[source]
        env[key] = str(int(value) if isinstance(value, bool) else value)
    env.update({
        "OUTPUT_ROOT": str(output), "CUDA_VISIBLE_DEVICES": str(gpu), "SEQUENCES": ",".join(sequences),
        "MAX_IMAGES": "0", "FORCE_REGENERATE": "0", "DISCOVER_HIERARCHICAL_SEQUENCES": "1",
        "PREFLIGHT_ONLY": "0", "GUIDANCE_MODE": "combined",
        "TEMPORAL_START_STEP": str(config["temporal_window"][0]), "TEMPORAL_END_STEP": str(config["temporal_window"][1]),
        "CGE_START_STEP": str(config["cge_window"][0]), "CGE_END_STEP": str(config["cge_window"][1]),
        "CGE_VCMRS_ROOT": "/home/cilab/ndquan/vcm-rs", "CGE_VCMRS_PYTHON": "/home/cilab/ndquan/envs/vcm/bin/python",
        "CGE_VCMRS_ROI_QUALITY": "20", "CGE_VCMRS_AUTO_DESCRIPTORS": "1",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1", "TZ": "Asia/Seoul",
        "PYTHONPATH": str(ROOT / "examples/brushnet/proposed") + os.pathsep + os.environ.get("PYTHONPATH", ""),
    })
    return env


def copy_pilot(factor, destination, expected):
    source = PILOT / "cge_x{}".format(factor)
    src = source / "test/Class_A/Traffic"
    dst = destination / "test/Class_A/Traffic"
    provenance = destination / "reused_pilot.json"
    if provenance.exists():
        return
    # Reusing only a completed, matching first window preserves the absolute clip seeds.
    clips = [json.loads(s) for s in (src / "guidance_trace.jsonl").read_text().splitlines()]
    timings = [json.loads(s) for s in (src / "generation_timing.jsonl").read_text().splitlines()]
    if clips[-1]["frame_ids"] != [117, 118, 119, 120] or clips[-1]["clip_index"] != 0:
        raise ValueError("Pilot first-window identity differs")
    if not dst.exists():
        dst.mkdir(parents=True)
    for frame_id in (117, 118, 119, 120):
        name = "{:06d}.png".format(frame_id)
        if name not in expected[Path("Class_A/Traffic")]:
            raise ValueError("Pilot frame missing from full split")
        target = dst / name
        if not target.exists():
            shutil.copy2(src / name, target)
    for filename in ("guidance_trace.jsonl", "generation_timing.jsonl"):
        if not (dst / filename).exists():
            shutil.copy2(src / filename, dst / filename)
    save_json(provenance, {"source": str(source), "reused_frames": [117, 118, 119, 120],
                           "reused_clip_index": 0, "pilot_generation_seconds": timings[-1]["generation_seconds"],
                           "semantics": "First full window is identical; frame-count cap does not alter clip index, seed or context."})


def validate_outputs(output, config, expected):
    sys.path.insert(0, str(ROOT / "examples/brushnet"))
    from sfu_temporal_clip_windows import build_temporal_windows
    total_seconds = total_clip_seconds = 0.0
    generated_frames = 0
    sequence_timings = []
    for relative, names in expected.items():
        folder = output / "test" / relative
        if {p.name for p in folder.glob("*.png")} != names:
            raise ValueError("Incomplete or unexpected frames: " + str(relative))
        for name in names:
            with Image.open(folder / name) as image:
                if image.size != tuple(config["output_size"]):
                    raise ValueError("Prediction resolution mismatch")
        ids = sorted(int(Path(name).stem) for name in names)
        windows = build_temporal_windows(len(ids), config["temporal_clip_size"], config["temporal_clip_stride"])
        traces = {c["clip_index"]: c for c in map(json.loads, (folder / "guidance_trace.jsonl").read_text().splitlines())}
        timings = {c["clip_index"]: c for c in map(json.loads, (folder / "generation_timing.jsonl").read_text().splitlines())}
        if set(traces) != set(range(len(windows))) or set(timings) != set(traces):
            raise ValueError("Missing clip audit or timing: " + str(relative))
        for index, (start, end) in enumerate(windows):
            trace = traces[index]
            if trace["frame_ids"] != ids[start:end] or timings[index]["frame_ids"] != ids[start:end]:
                raise ValueError("Clip frame alignment mismatch")
            for key in ("cge_applied", "temporal_requested"):
                if [s["step_index"] for s in trace["steps"] if s[key]] != list(range(25, 40)):
                    raise ValueError("Guidance window mismatch: " + str(relative))
        seconds = sum(t["generation_seconds"] for t in timings.values())
        clip_seconds = sum(t["total_clip_seconds"] for t in timings.values())
        compute_frames = sum(t["batch_frames"] for t in timings.values())
        sequence_timings.append({"sequence": relative.name, "frames": len(names), "clips": len(windows),
                                 "generated_clip_frames": compute_frames, "generation_seconds": seconds,
                                 "generation_seconds_per_saved_image": seconds / len(names),
                                 "total_clip_seconds": clip_seconds})
        total_seconds += seconds
        total_clip_seconds += clip_seconds
        generated_frames += compute_frames
    frames = sum(map(len, expected.values()))
    report = {"expected_saved_frames": frames, "generated_clip_frames": generated_frames,
              "generation_seconds": total_seconds, "generation_seconds_per_saved_image": total_seconds / frames,
              "generation_seconds_per_generated_clip_frame": total_seconds / generated_frames,
              "total_clip_seconds": total_clip_seconds,
              "total_seconds_per_saved_image": total_clip_seconds / frames,
              "includes_reused_pilot_timing": True,
              "semantics": "Sum of GPU-worker batch durations, not end-to-end parallel wall time. Saved-image denominator includes tail-window overhead; latest successful timing per clip used.",
              "sequences": sequence_timings}
    save_json(output / "timing_summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus-x3", nargs=2, type=int, default=[1, 3])
    parser.add_argument("--gpus-x10", nargs=2, type=int, default=[4, 5])
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    pools = {3: args.gpus_x3, 10: args.gpus_x10}
    if len(set(args.gpus_x3 + args.gpus_x10)) != 4:
        raise ValueError("Four distinct GPU slots required")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "terminal_logs").mkdir(exist_ok=True)
    status_path = OUTPUT / "status.json"
    if not args.preflight_only and status_path.exists():
        previous = json.loads(status_path.read_text())
        try:
            os.kill(previous["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("A full-scale launcher is already active")
    base = json.loads((BASE / "run_config.json").read_text())
    dataset = Path(base["dataset"])
    expected = {}
    for folder in sorted((dataset / "input").glob("Class_*/*")):
        relative = folder.relative_to(dataset / "input")
        names = {p.name for p in folder.glob("*.png")}
        if not names:
            raise ValueError("Empty input sequence")
        for kind in ("mask", "GT"):
            if {p.name for p in (dataset / kind / relative).glob("*.png")} != names:
                raise ValueError("Input/GT/mask mismatch: " + str(relative))
        expected[relative] = names
    if len(expected) != base["expected_sequences"] or sum(map(len, expected.values())) != base["expected_frames"]:
        raise ValueError("Full split differs from baseline counts")
    states = {}
    for factor, gpus in pools.items():
        pilot_config = PILOT / "cge_x{}".format(factor) / "run_config.json"
        config = json.loads(pilot_config.read_text())
        if (config["cge_guidance_scale"] != (0.0003 if factor == 3 else 0.001)
                or config["temporal_guidance_scale"] != 0.1
                or config["cge_window"] != [25, 40] or config["temporal_window"] != [25, 40]):
            raise ValueError("Pilot settings differ from requested scales/windows")
        for name, digest in config["source_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Pilot source changed: " + name)
        output = OUTPUT / "cge_x{}".format(factor)
        logs = output / "terminal_logs"
        logs.mkdir(parents=True, exist_ok=True)
        loads = [0, 0]
        groups = [[], []]
        for relative in sorted(expected, key=lambda p: (-len(expected[p]), str(p))):
            index = min(range(2), key=lambda i: loads[i])
            groups[index].append(relative.name)
            loads[index] += len(expected[relative])
        config.update({"launcher": str(Path(__file__).resolve().relative_to(ROOT)), "source_config": str(pilot_config),
                       "gpus": gpus, "max_images": 0, "expected_sequences": len(expected),
                       "expected_frames": sum(map(len, expected.values())), "sequences": sorted(p.name for p in expected),
                       "worker_plan": {str(gpu): {"sequences": seqs, "frames": count}
                                       for gpu, seqs, count in zip(gpus, groups, loads)}})
        config.pop("frame_ids", None)
        path = output / "run_config.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("Existing full-run config differs")
        save_json(path, config)
        copy_pilot(factor, output, expected)
        env = environment(config, output, gpus[0], sorted(p.name for p in expected))
        with (logs / "preflight.log").open("w") as log:
            subprocess.run([sys.executable, "-u", str(EVALUATOR)], cwd=ROOT, env=dict(env, PREFLIGHT_ONLY="1"),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print("Preflight x{}: 14 sequences / 1198 frames; GPUs {}; plan {}".format(factor, gpus, config["worker_plan"]), flush=True)
        states[factor] = {"config": config, "output": output, "env": env, "groups": groups,
                          "processes": {}, "phase": "inference", "metric_process": None}
    if args.preflight_only:
        return 0
    gpu_text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                         "--format=csv,noheader,nounits"], text=True)
    available = {int(a): (int(b), int(c)) for a, b, c in (line.split(",") for line in gpu_text.splitlines())}
    for gpu in args.gpus_x3 + args.gpus_x10:
        if available[gpu][0] > 500 or available[gpu][1] > 5:
            raise RuntimeError("GPU {} is occupied; this launcher never waits for or adds GPUs".format(gpu))
    status = {"pid": os.getpid(), "state": "inference", "started_at": now(), "runs": {}}
    for factor, state in states.items():
        status["runs"][str(factor)] = {"state": "inference", "output": str(state["output"]), "workers": {
            str(gpu): {"state": "pending_start", "sequences": seqs}
            for gpu, seqs in zip(pools[factor], state["groups"])}}
    save_json(status_path, status)
    try:
        while any(state["phase"] in ("inference", "metrics") for state in states.values()):
            for factor, state in states.items():
                item = status["runs"][str(factor)]
                output = state["output"]
                if state["phase"] == "inference":
                    for gpu, sequences in zip(pools[factor], state["groups"]):
                        worker = item["workers"][str(gpu)]
                        if gpu not in state["processes"]:
                            log_path = output / "terminal_logs/gpu{}.log".format(gpu)
                            with log_path.open("a") as log:
                                process = subprocess.Popen([sys.executable, "-u", str(EVALUATOR)], cwd=ROOT,
                                    env=environment(state["config"], output, gpu, sequences), stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
                            state["processes"][gpu] = process
                            worker.update(state="running", pid=process.pid, log=str(log_path), started_at=now())
                            print("Started x{} GPU {} PID {}: {}".format(factor, gpu, process.pid, sequences), flush=True)
                        else:
                            code = state["processes"][gpu].poll()
                            worker.update(exit_code=code, state="running" if code is None else "complete" if code == 0 else "failed")
                    item["generated_frames"] = len(list((output / "test").glob("Class_*/*/*.png")))
                    if len(state["processes"]) == 2 and all(p.poll() is not None for p in state["processes"].values()):
                        if any(p.returncode != 0 for p in state["processes"].values()):
                            state["phase"] = item["state"] = "failed"
                            item["error"] = "Inference worker failed; successful clips remain resumable"
                            continue
                        item["timing"] = validate_outputs(output, state["config"], expected)
                        metric_dir = output / "test/metrics_full_reference"
                        if (metric_dir / "metrics.csv").is_file():
                            state["phase"] = item["state"] = "complete"
                            continue
                        log_path = output / "terminal_logs/metrics.log"
                        with log_path.open("a") as log:
                            state["metric_process"] = subprocess.Popen([
                                sys.executable, "-u", str(ROOT / "Quan_test/evaluate_flat_sequence_metrics.py"),
                                "--eval-root", str(output / "test"), "--gt-root", str(dataset), "--hierarchical",
                                "--sequences", *sorted(p.name for p in expected)], cwd=ROOT,
                                env=environment(state["config"], output, pools[factor][-1], sorted(p.name for p in expected)),
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                        state["phase"] = item["state"] = "metrics"
                        item["metrics_pid"] = state["metric_process"].pid
                elif state["phase"] == "metrics":
                    code = state["metric_process"].poll()
                    if code is not None:
                        state["phase"] = item["state"] = "complete" if code == 0 else "failed"
                        item["metrics_exit_code"] = code
                item["updated_at"] = now()
            status["updated_at"] = now()
            save_json(status_path, status)
            if any(state["phase"] in ("inference", "metrics") for state in states.values()):
                time.sleep(30)
        if any(state["phase"] == "failed" for state in states.values()):
            raise RuntimeError("A full evaluation failed; see per-run status/logs")
        status["state"] = "comparison"
        save_json(status_path, status)
        with (OUTPUT / "terminal_logs/comparison.log").open("a") as log:
            subprocess.run([sys.executable, "-u", str(ROOT / "Quan_test/compare_bg52_cge_scale_full.py")],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        status.update(state="complete", comparison=str(OUTPUT / "comparison/metrics_summary.csv"))
    except Exception as error:
        status.update(state="failed", error=str(error))
        raise
    finally:
        status["updated_at"] = now()
        save_json(status_path, status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
