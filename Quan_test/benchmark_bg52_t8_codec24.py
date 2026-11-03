#!/usr/bin/env python3
"""Two synchronized QP52 clips, 24 independent codec calls per guidance step.

Run with guided_diff Python. The launcher uses psutil/NVML without loading torch;
workers reuse the deployment engine and install the isolated codec-call pool.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/brushnet"
KST = timezone(timedelta(hours=9))
SEQUENCES = {"Traffic": "Class_A", "BQTerrace": "Class_B"}
DATASET = Path("/home/cilab/ndquan/vcm-rs/output/sfu_qp0_bg_table_window_test/bg52/test")
SOURCE_CONFIG = ROOT / "experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_cge_then_tg_equivclean_C25-40_T25-40_cge1em3_tg0p1_v7student_codecP4_512/run_config.json"


def timestamp():
    return datetime.now(KST).isoformat()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def read_lines(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def worker(args):
    for key in list(os.environ):
        if key.startswith(("CGE_", "TEMPORAL_", "TEACHER_FLOW_")):
            del os.environ[key]
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUBLAS_WORKSPACE_CONFIG=":4096:8",
                      CGE_CODEC_WORKERS="24", TOKENIZERS_PARALLELISM="false")
    sys.path[:0] = [str(EXAMPLES / "inference"), str(EXAMPLES), str(EXAMPLES / "proposed")]
    import torch
    import numpy as np
    import restoration_engine as deployment
    import diffusers.schedulers.scheduling_ddim_CGE as core
    from codec_task_pool_benchmark import CodecTaskPool, EventWriter, track_encoder
    from types import SimpleNamespace

    name = args.worker
    out = args.output / name
    out.mkdir(parents=True, exist_ok=True)
    state = {"sequence": name, "pid": os.getpid(), "gpu": args.gpu, "phase": "loading"}

    def phase(value, **kwargs):
        state.update(phase=value, **kwargs)
        save_json(out / "status.json", dict(state, updated_at=timestamp(), monotonic=time.perf_counter()))

    phase("loading")
    try:
        settings = deployment.settings_from_json(SOURCE_CONFIG)
        models = dict(line.split(": ", 1) for line in (ROOT / "experiments/EVAL_FOR_FINAL/ckpt.yaml").read_text().splitlines() if line.strip())
        settings.checkpoint = models["ckpt_bg52"]
        settings.flow_checkpoint = models["student_flow_bg52"]
        settings.guidance = "cge_then_tg"
        settings.temporal_clip_size = settings.temporal_clip_stride = 8
        settings.cge_start = settings.temporal_start = 25
        settings.cge_end = settings.temporal_end = 35
        settings.cge_codec_workers = settings.cge_max_parallel = 24
        deployment.validate_settings(settings)
        input_dir, mask_dir = (DATASET / kind / SEQUENCES[name] / name for kind in ("input", "mask"))
        discovery = SimpleNamespace(input_dir=input_dir, mask_dir=mask_dir, dataset_root=None, sequences=None,
                                    max_frames=8, output_dir=out)
        relative, pairs = deployment.discover_sequences(discovery)[0]
        if len(pairs) != 8:
            raise ValueError("This benchmark requires exactly eight frames")
        save_json(out / "run_config.json", {
            "settings": asdict(settings), "frames": [str(p) for p, _ in pairs],
            "masks": [str(p) for _, p in pairs], "gpu": args.gpu, "source_config": str(SOURCE_CONFIG),
            "codec_task_layout": "one independent task per (frame, base/plus/minus)",
            "omp_threads": os.environ.get("OMP_NUM_THREADS"), "mkl_threads": os.environ.get("MKL_NUM_THREADS"),
            "clip_stride_note": "Only one complete clip; stride=clip-size keeps the baseline batch noise sampling rule.",
        })
        loaded_start = time.perf_counter()
        engine = deployment.RestorationEngine(settings)
        model_load_seconds = time.perf_counter() - loaded_start
        context = {"step_index": -1}
        codec_events = EventWriter(out / "codec_events.jsonl")
        counts = track_encoder(engine.codec, codec_events, lambda: context["step_index"])
        pool = CodecTaskPool(core, 24, out / "codec_task_timings.jsonl", verify_first=True)
        pool.install()
        scheduler = engine.pipe.scheduler
        component_times, steps, generation = {}, [], {}
        step_writer = EventWriter(out / "step_timing.jsonl")
        original_cond = scheduler.cond_fn
        original_temporal = scheduler._apply_latent_temporal_guidance
        original_step = scheduler.step
        original_generate = engine.ip_model.generate_fgbg
        original_configure = engine.configure_guidance
        barrier_wait = [0.0]

        @wraps(original_cond)
        def timed_cge(*positional, **keyword):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = original_cond(*positional, **keyword)
            torch.cuda.synchronize()
            component_times.setdefault(context["step_index"], {})["cge_seconds"] = time.perf_counter() - start
            return output

        @wraps(original_temporal)
        def timed_temporal(*positional, **keyword):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = original_temporal(*positional, **keyword)
            torch.cuda.synchronize()
            component_times.setdefault(context["step_index"], {})["tg_seconds"] = time.perf_counter() - start
            return output

        @wraps(original_step)
        def timed_step(*positional, **keyword):
            index = len(steps)
            context["step_index"] = index
            phase("guidance" if 25 <= index < 35 else "denoise", step_index=index)
            torch.cuda.synchronize()
            scheduler_start = time.perf_counter()
            output = original_step(*positional, **keyword)
            torch.cuda.synchronize()
            ended = time.perf_counter()
            transition_start = generation.get("previous_step_end", generation["start_monotonic"])
            record = dict(scheduler.sequential_step_reports[-1])
            record.update(component_times.get(index, {}))
            record.update(transition_seconds=ended - transition_start,
                          scheduler_seconds=ended - scheduler_start,
                          transition_start_monotonic=transition_start, scheduler_start_monotonic=scheduler_start,
                          end_monotonic=ended, encoder_completed_total=counts["completed"],
                          peak_encoder_concurrency_so_far=counts["peak"])
            steps.append(record)
            step_writer.write(record)
            generation["previous_step_end"] = ended
            if record["cge_applied"]:
                print("[Measured {} step {}] transition={:.3f}s cge={:.3f}s tg={:.3f}s peak_encoders={}".format(
                    name, index, record["transition_seconds"], record.get("cge_seconds", 0),
                    record.get("tg_seconds", 0), counts["peak"]), flush=True)
            return output

        @wraps(original_generate)
        def timed_generate(*positional, **keyword):
            torch.cuda.synchronize()
            generation["start_monotonic"] = time.perf_counter()
            generation["started_at"] = timestamp()
            phase("generation", step_index=0, generation_start=generation["start_monotonic"])
            result = original_generate(*positional, **keyword)
            torch.cuda.synchronize()
            generation["end_monotonic"] = time.perf_counter()
            generation["seconds"] = generation["end_monotonic"] - generation["start_monotonic"]
            return result

        def synchronized_configure(*positional, **keyword):
            original_configure(*positional, **keyword)
            torch.cuda.synchronize()
            phase("ready")
            save_json(args.output / (name + ".ready.json"), {"pid": os.getpid(), "ready_at": timestamp()})
            waiting = time.perf_counter()
            while not (args.output / "start_gate.json").is_file():
                if time.perf_counter() - waiting > 600:
                    raise TimeoutError("The second worker did not reach the start gate")
                time.sleep(.05)
            barrier_wait[0] = time.perf_counter() - waiting
            phase("generation")

        scheduler.cond_fn = timed_cge
        scheduler._apply_latent_temporal_guidance = timed_temporal
        scheduler.step = timed_step
        engine.ip_model.generate_fgbg = timed_generate
        engine.configure_guidance = synchronized_configure
        phase("preparation")
        clip_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        try:
            images, trace, engine_seconds = engine.generate_clip(pairs, 0, range(8))
        finally:
            pool.close()
        phase("postprocessing")
        checks = []
        for image, (source, mask) in zip(images, pairs):
            native, roi = deployment.read_native(source, mask)
            blended = deployment.evaluation_blend_512(image, native, roi)
            upscaled, final = deployment.restore_native(blended, native, roi)
            for folder, result in (("512", blended), ("resized", upscaled), ("final", final)):
                deployment.save_png(result, out / folder / source.name)
            exact_roi = bool(np.array_equal(np.asarray(final)[roi], native[roi]))
            checks.append({"frame": source.name, "native_size": list(final.size), "roi_exact": exact_roi})
        finished = time.perf_counter()
        guided = [s for s in steps if s["cge_applied"]]
        if (len(steps) != 50 or [s["step_index"] for s in guided] != list(range(25, 35))
                or not all(s["temporal_applied"] for s in guided)
                or counts["started"] != 240 or counts["completed"] != 240
                or counts["peak"] > 24 or counts["active"] != 0
                or not pool.replay_validation["passed"] or not all(c["roi_exact"] for c in checks)):
            raise AssertionError("Benchmark output/guidance/codec audit failed")
        save_json(out / "validation.json", {"passed": True, "frames": checks,
                  "equivalence": pool.replay_validation, "encoder_counts": counts,
                  "actual_cge_steps": [s["step_index"] for s in guided],
                  "actual_tg_steps": [s["step_index"] for s in steps if s["temporal_applied"]],
                  "task_counts": [r["tasks"] for r in pool.reports]})
        result = {"sequence": name, "gpu": args.gpu, "frames": 8, "ddim_steps": 50, "guidance_steps": 10,
                  "generation_started_at": generation["started_at"],
                  "generation_start_monotonic": generation["start_monotonic"],
                  "generation_end_monotonic": generation["end_monotonic"],
                  "generation_seconds": generation["seconds"],
                  "generation_seconds_per_frame": generation["seconds"] / 8,
                  "engine_generation_seconds": engine_seconds,
                  "clip_seconds_excluding_model_load_and_barrier": finished - clip_started - barrier_wait[0],
                  "model_load_seconds": model_load_seconds, "barrier_wait_seconds": barrier_wait[0],
                  "guidance_transition_seconds_mean": statistics.mean(s["transition_seconds"] for s in guided),
                  "guidance_transition_seconds_min": min(s["transition_seconds"] for s in guided),
                  "guidance_transition_seconds_max": max(s["transition_seconds"] for s in guided),
                  "guidance_scheduler_seconds_mean": statistics.mean(s["scheduler_seconds"] for s in guided),
                  "cge_seconds_mean": statistics.mean(s["cge_seconds"] for s in guided),
                  "tg_seconds_mean": statistics.mean(s["tg_seconds"] for s in guided),
                  "codec_pool_seconds_mean": statistics.mean(r["seconds"] for r in pool.reports),
                  "first_step_replay_validation_seconds": pool.reports[0]["validation_seconds"],
                  "encoder_counts": counts, "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                  "finished_at": timestamp(), "timing_note": "Transition includes denoiser since previous scheduler completion; "
                  "scheduler includes CGE then TGE; generation includes all 50 steps and final decode. "
                  "Per-frame time is batch duration / 8. No warm-up rollout or extra real codec validation calls."}
        save_json(out / "summary.json", result)
        phase("complete")
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except BaseException as error:
        phase("failed", error=repr(error))
        save_json(out / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise


def resource_sample(processes, gpu_handles, nvml, previous_cpu, dt):
    import psutil
    rows = {}
    for name, info in processes.items():
        members = []
        try:
            root = psutil.Process(info["process"].pid)
            members = [root] + root.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
        cpu_delta, rss, threads, pids = 0.0, 0, 0, []
        for process in members:
            try:
                with process.oneshot():
                    key = (process.pid, process.create_time())
                    c = process.cpu_times()
                    now = c.user + c.system
                    cpu_delta += max(0.0, now - previous_cpu.get(key, 0.0))
                    previous_cpu[key] = now
                    rss += process.memory_info().rss
                    threads += process.num_threads()
                    pids.append(process.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        state = read_json(info["folder"] / "status.json", {})
        rows[name] = {"pid": info["process"].pid, "gpu": info["gpu"], "phase": state.get("phase"),
                      "step_index": state.get("step_index"), "process_count": len(pids), "thread_count": threads,
                      "cpu_core_percent": cpu_delta / dt * 100, "rss_sum_mib": rss / 2**20,
                      "pids": pids}
    gpus = {}
    for index, handle in gpu_handles.items():
        util, memory = nvml.nvmlDeviceGetUtilizationRates(handle), nvml.nvmlDeviceGetMemoryInfo(handle)
        gpus[str(index)] = {"utilization_percent": util.gpu, "memory_utilization_percent": util.memory,
                            "used_mib": memory.used / 2**20}
    vm = psutil.virtual_memory()
    return {"time": timestamp(), "monotonic": time.perf_counter(), "sample_dt_seconds": dt,
            "system_cpu_percent": psutil.cpu_percent(interval=None),
            "system_memory_used_gib": vm.used / 2**30, "system_memory_available_gib": vm.available / 2**30,
            "workers": rows, "gpus": gpus}


def summarize_resources(samples, start, end, name=None):
    selected = [r for r in samples if start <= r["monotonic"] <= end]
    if not selected:
        return {}

    def stats(values):
        return {"mean": statistics.mean(values), "peak": max(values)}

    report = {"sample_count": len(selected),
              "system_cpu_percent": stats([r["system_cpu_percent"] for r in selected]),
              "system_memory_used_gib": stats([r["system_memory_used_gib"] for r in selected])}
    if name is not None:
        rows = [r["workers"][name] for r in selected]
        gpu = str(rows[0]["gpu"])
        report.update(process_tree_cpu_core_percent=stats([r["cpu_core_percent"] for r in rows]),
                      process_tree_rss_sum_mib=stats([r["rss_sum_mib"] for r in rows]),
                      process_tree_process_count=stats([r["process_count"] for r in rows]),
                      process_tree_thread_count=stats([r["thread_count"] for r in rows]),
                      gpu_utilization_percent=stats([r["gpus"][gpu]["utilization_percent"] for r in selected]),
                      gpu_used_mib=stats([r["gpus"][gpu]["used_mib"] for r in selected]))
    return report


def launcher(args):
    import psutil
    import pynvml as nvml
    from codec_task_pool_benchmark import EventWriter
    nvml.nvmlInit()
    handles = {gpu: nvml.nvmlDeviceGetHandleByIndex(gpu) for gpu in args.gpus}
    for gpu, handle in handles.items():
        compute = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
        used = nvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20
        if compute or used > 512:
            raise RuntimeError("GPU {} is occupied; no worker was launched".format(gpu))
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Benchmark output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__).resolve(), ROOT / "Quan_test/codec_task_pool_benchmark.py",
             EXAMPLES / "inference/restoration_engine.py", EXAMPLES / "diffusers/schedulers/scheduling_ddim_CGE.py",
             EXAMPLES / "diffusers/schedulers/scheduling_ddim_cge_then_temporal.py", EXAMPLES / "vcmrs_codec_adapter.py"]
    launch = {"started_at": timestamp(), "pid": os.getpid(), "gpus": args.gpus,
              "sequences": list(SEQUENCES), "clip_frames": 8, "guidance_window": [25, 35], "codec_workers_per_process": 24,
              "max_parallel_per_process": 24, "global_possible_codec_concurrency": 48,
              "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
              "system_ram_gib": psutil.virtual_memory().total / 2**30,
              "sampling_interval_seconds": args.interval,
              "cpu_semantics": "Tree core percent: 100%=one logical CPU, may exceed 100%; system CPU percent normalized to all logical CPUs. "
                               "RSS is summed across descendants and includes shared pages. Process CPU counters sampled, not exact child-exit accounting.",
              "gpu_semantics": "Physical GPU utilization/memory sampled via NVML; codec children may also use this GPU. Includes background display memory.",
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    save_json(args.output / "launch.json", launch)
    processes, streams = {}, []
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", CUDA_DEVICE_ORDER="PCI_BUS_ID", TZ="Asia/Seoul")
    for name, gpu in zip(SEQUENCES, args.gpus):
        folder = args.output / name
        folder.mkdir()
        stream = (folder / "worker.log").open("w")
        streams.append(stream)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", name, "--gpu", str(gpu),
                   "--output", str(args.output)]
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = {"process": process, "gpu": gpu, "folder": folder}
        launch.setdefault("workers", {})[name] = {"pid": process.pid, "gpu": gpu, "command": command}
    save_json(args.output / "launch.json", launch)
    print("Launched:", json.dumps(launch["workers"]), flush=True)
    writer = EventWriter(args.output / "resource_samples.jsonl")
    previous_cpu, samples = {}, []
    psutil.cpu_percent(interval=None)
    previous_time, started = time.perf_counter(), time.perf_counter()
    last_progress = 0.0
    while any(info["process"].poll() is None for info in processes.values()):
        now = time.perf_counter()
        sample = resource_sample(processes, handles, nvml, previous_cpu, max(.001, now - previous_time))
        previous_time = now
        writer.write(sample)
        samples.append(sample)
        if not (args.output / "start_gate.json").exists() and all((args.output / (name + ".ready.json")).is_file() for name in SEQUENCES):
            save_json(args.output / "start_gate.json", {"released_at": timestamp(), "monotonic": time.perf_counter()})
            print("Both clips prepared; simultaneous generation released.", flush=True)
        failures = {name: info["process"].poll() for name, info in processes.items() if info["process"].poll() not in (None, 0)}
        if failures or time.perf_counter() - started > 3600:
            # Stop only descendants of the benchmark workers; never unrelated jobs.
            for info in processes.values():
                try:
                    parent = psutil.Process(info["process"].pid)
                    children = parent.children(recursive=True)
                    for proc in reversed(children):
                        try: proc.terminate()
                        except psutil.NoSuchProcess: pass
                    parent.terminate()
                except psutil.NoSuchProcess:
                    pass
            save_json(args.output / "failure.json", {"failures": failures, "timeout": not bool(failures), "at": timestamp()})
            raise RuntimeError("Benchmark worker failed: " + str(failures))
        if now - last_progress > 20:
            status = {name: {"phase": row["phase"], "step": row.get("step_index"),
                            "cpu_cores": round(row["cpu_core_percent"] / 100, 2), "rss_gib": round(row["rss_sum_mib"] / 1024, 2)}
                      for name, row in sample["workers"].items()}
            print("[Resources]", json.dumps({"status": status, "system_cpu_percent": sample["system_cpu_percent"], "gpus": sample["gpus"]}), flush=True)
            last_progress = now
        time.sleep(max(.05, args.interval - (time.perf_counter() - now)))
    for stream in streams:
        stream.close()
    results, encoder_events = {}, []
    for name, info in processes.items():
        report = read_json(info["folder"] / "summary.json")
        if report is None:
            raise RuntimeError("Missing completed worker report: " + name)
        report["resources_during_generation"] = summarize_resources(samples, report["generation_start_monotonic"], report["generation_end_monotonic"], name)
        for step in read_lines(info["folder"] / "step_timing.jsonl"):
            if step["cge_applied"]:
                step["resources"] = summarize_resources(samples, step["transition_start_monotonic"], step["end_monotonic"], name)
                EventWriter(info["folder"] / "guidance_steps_with_resources.jsonl").write(step)
        events = read_lines(info["folder"] / "codec_events.jsonl")
        for event in events:
            encoder_events.append(dict(event, sequence=name))
        results[name] = report
    active, peak = 0, 0
    for event in sorted(encoder_events, key=lambda e: e["monotonic"]):
        active += 1 if event["event"] == "encoder_start" else -1
        peak = max(peak, active)
        if active < 0:
            raise AssertionError("Unbalanced encoder events")
    if active != 0:
        raise AssertionError("Some encoder jobs did not finish")
    start = min(r["generation_start_monotonic"] for r in results.values())
    end = max(r["generation_end_monotonic"] for r in results.values())
    result = {"completed_at": timestamp(), "passed": True, "global_peak_encoder_concurrency": peak,
              "generation_start_skew_seconds": max(r["generation_start_monotonic"] for r in results.values()) - start,
              "combined_generation_wall_seconds": end - start,
              "system_resources_during_generation": summarize_resources(samples, start, end),
              "sequences": results, "launch": launch}
    save_json(args.output / "summary.json", result)
    lines = ["# QP52: T8, 10 CGE→TG steps, codec task pool 24 × 2 GPUs", "",
             "Both sequences generated simultaneously after model/flow preparation. Per-frame time is clip duration / 8.", "",
             "| Sequence | GPU | CGE→TG step mean (s) | CGE mean (s) | TG mean (s) | Generate clip (s) | Generate/frame (s) | Encoder peak |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        lines.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
            name, r["gpu"], r["guidance_transition_seconds_mean"], r["cge_seconds_mean"], r["tg_seconds_mean"],
            r["generation_seconds"], r["generation_seconds_per_frame"], r["encoder_counts"]["peak"]))
    lines.extend(["", "Global peak simultaneous encoders: {}. Generation start skew: {:.3f}s.".format(peak, result["generation_start_skew_seconds"]),
                  "", "Detailed resource metrics: summary.json; raw 0.5-second samples: resource_samples.jsonl.",
                  "Step transition includes the denoiser; scheduler timing isolates the scheduler call. CGE includes VAE decode/backprop and codec; TG is measured separately.",
                  "CPU tree percent uses 100% per logical CPU; system CPU percent is normalized to all {} logical CPUs.".format(psutil.cpu_count()),
                  "RSS sums include shared pages. One first-step equivalence replay per sequence adds no real codec calls; its duration is recorded separately."])
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    nvml.nvmlShutdown()
    print("COMPLETE:", args.output / "REPORT.md", flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", choices=list(SEQUENCES))
    p.add_argument("--gpu", type=int)
    p.add_argument("--gpus", type=int, nargs=2, default=[0, 1])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--interval", type=float, default=.5)
    args = p.parse_args()
    args.output = args.output.resolve()
    if len(set(args.gpus)) != 2 or args.interval <= 0:
        p.error("Use two distinct GPU indices and a positive sampling interval")
    return worker(args) if args.worker else launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
