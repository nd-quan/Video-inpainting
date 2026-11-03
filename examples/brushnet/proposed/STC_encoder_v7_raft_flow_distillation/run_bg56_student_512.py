#!/usr/bin/env python3
"""Prepare BG56, build new square-512 teachers, select batch, and train/resume."""
import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bg56_512_setup import DEFAULT_RUN, HERE, INITIAL, PROPAINTER, SOURCE, TEACHER_WEIGHTS, batch_layout, cache_path, check_cache_file, load_rgb512, pairs, prepare, read_json, sha256, verify_source, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def status(run, phase, **details):
    previous = read_json(run / "status.json") if (run / "status.json").is_file() else {}
    previous.update(phase=phase, updated_at=now(), supervisor_pid=os.getpid(), **details)
    write_json(run / "status.json", previous)
    print(json.dumps({"phase": phase, **details}), flush=True)


def gpu_memory(gpus):
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free", "--format=csv,noheader,nounits"], text=True)
    records = {int(parts[0]): {"used": int(parts[1]), "free": int(parts[2])} for parts in (line.split(",") for line in output.splitlines())}
    return {gpu: records[gpu] for gpu in gpus}


def environment(gpus):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", NCCL_ASYNC_ERROR_HANDLING="1")
    env["PYTHONPATH"] = str(HERE.parents[1]) + os.pathsep + str(HERE.parents[0]) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def launch(command, log, gpus):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment(gpus), start_new_session=True, cwd=str(HERE.parents[3]))
    return process


def tail(path):
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        return stream.read().decode("utf-8", "replace")


def wait_jobs(run, jobs, phase):
    try:
        while True:
            for process, log in jobs:
                if process.poll() not in (None, 0):
                    raise RuntimeError("Process exited {}: {}\n{}".format(process.returncode, log, tail(log)[-4000:]))
            if all(process.poll() is not None for process, log in jobs):
                return
            details = {"child_pids": [process.pid for process, log in jobs if process.poll() is None]}
            if phase == "teacher_running":
                progress = [read_json(path) for path in (run / "logs").glob("teacher_shard_*.json")]
                details["teacher_completed_pairs"] = sum(item["completed"] for item in progress)
                details["teacher_total_pairs"] = 6002
            if phase == "training":
                steps = re.findall(r'^TRAIN .*"step": (\d+)', tail(jobs[0][1]), re.MULTILINE)
                if steps:
                    details["train_step"] = int(steps[-1])
                latest = run / "student/latest.json"
                if latest.is_file():
                    details["last_checkpoint_step"] = read_json(latest)["global_step"]
            status(run, phase, **details)
            time.sleep(15)
    finally:
        for process, log in jobs:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, log in jobs:
            if process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def ddp_command(script, gpus, extra):
    with socket.socket() as listener:
        listener.bind(("", 0))
        port = listener.getsockname()[1]
    return [sys.executable, "-m", "accelerate.commands.launch", "--multi_gpu", "--num_processes", str(len(gpus)), "--num_machines", "1", "--main_process_port", str(port), "--mixed_precision", "fp16", "--dynamo_backend", "no", str(script), *extra]


def training_args(run, batch, accumulation, resume=False, probe=False):
    args = ["--dataset_root", str(run / "dataset"), "--teacher_flow_root", str(run / "teacher_flows_512x512"), "--output_dir", str(run / ("probes/batch{}".format(batch) if probe else "student")), "--student_backend", "propainter_raft", "--propainter_root", str(PROPAINTER), "--raft_checkpoint", str(TEACHER_WEIGHTS), "--train_split", "train", "--valid_split", "valid", "--resolution", "512", "--spatial_mode", "square", "--max_input_side", "512", "--train_batch_size", str(batch), "--raft_pair_batch_size", str(batch), "--gradient_accumulation_steps", str(accumulation), "--dataloader_num_workers", "0" if probe else "2", "--learning_rate", "1e-5", "--lr_scheduler", "constant", "--lr_warmup_steps", "0", "--max_train_steps", "2" if probe else "4000", "--checkpointing_steps", "250", "--validation_steps", "250", "--checkpoints_total_limit", "20", "--valid_pairs_per_sequence", "8", "--raft_iterations", "20", "--raft_sequence_gamma", "0.8", "--teacher_weight", "1", "--fb_weight", "0.1", "--smoothness_weight", "0.01", "--smoothness_edge_weight", "10", "--flow_charbonnier_eps", "0.001", "--flow_loss_region", "all", "--adam_beta1", "0.9", "--adam_beta2", "0.999", "--adam_weight_decay", "1e-5", "--adam_epsilon", "1e-8", "--max_grad_norm", "1", "--mixed_precision", "fp16", "--freeze_batchnorm", "--seed", "20260903", "--log_steps", "20"]
    args += ["--resume_from_checkpoint", "latest"] if resume else ["--init_from_checkpoint", str(INITIAL)]
    if probe:
        args += ["--memory_probe"]
    return args


def audit(run, manifest):
    import torch
    from raft_teacher_pair_data import RAFTTeacherFlowPairDataset
    count = 0
    for pair in pairs(manifest):
        check_cache_file(cache_path(run, pair))
        count += 1
    samples = 0
    for split in ("train", "valid"):
        dataset = RAFTTeacherFlowPairDataset(run / "dataset", run / "teacher_flows_512x512", split, 512, spatial_mode="square", max_input_side=512)
        seen = set()
        for index, entry in enumerate(dataset.entries):
            branch, first, second, path = entry
            if branch in seen:
                continue
            sample = dataset[index]
            if sample["degraded0"].shape != (3, 512, 512) or sample["bg0"].shape != (1, 512, 512):
                raise ValueError("Unexpected student input shape")
            source = run / "dataset" / split / "GT" / branch / "{:06d}.png".format(first)
            if not torch.equal(dataset._rgb(source), load_rgb512(source)):
                raise ValueError("Teacher/student RGB preprocessing differs")
            seen.add(branch)
            samples += 1
    verify_source(run)
    write_json(run / "preflight_audit.json", {"success": True, "teacher_pairs_checked": count, "student_branch_split_samples_checked": samples, "teacher_student_resize_exactly_equal": True, "source_files_unchanged": True, "resolution": [512, 512], "checked_at": now()})


def choose_batch(run, gpus):
    failures = []
    available_gib = min(value["free"] for value in gpu_memory(gpus).values()) / 1024
    for batch in (4, 2, 1):
        log = run / "logs/batch_probe_{}.log".format(batch)
        status(run, "batch_probe", candidate_batch_per_gpu=batch)
        process = launch([sys.executable, str(HERE / "train_bg56_student_512.py"), *training_args(run, batch, 1, probe=True)], log, [gpus[0]])
        try:
            wait_jobs(run, [(process, log)], "batch_probe")
        except RuntimeError:
            output = tail(log)
            if "out of memory" not in output.lower():
                raise
            failures.append({"batch": batch, "reason": "CUDA out of memory"})
            continue
        records = [json.loads(line.split("MEMORY_PROBE ", 1)[1]) for line in tail(log).splitlines() if line.startswith("MEMORY_PROBE ")]
        if len(records) != 2:
            raise RuntimeError("Memory probe did not finish two optimizer steps")
        peak = max(item["max_reserved_gib"] for item in records)
        if peak + 4 > available_gib:
            failures.append({"batch": batch, "reason": "Insufficient DDP memory headroom", "peak_reserved_gib": peak})
            continue
        config = {"batch_per_gpu": batch, "gradient_accumulation_steps": 1, "global_batch": batch * len(gpus), "initial_gpu_count": len(gpus), "peak_probe_reserved_gib": peak, "probe_log": str(log), "rejected_candidates": failures}
        write_json(run / "batch_config.json", config)
        return config
    raise RuntimeError("No batch fits with four GiB of DDP headroom")


def check_resume_topology(run, gpus, batch):
    output = run / "probes/resume_topology"
    result = output / "resumed_2gpu.json"
    if result.is_file() and read_json(result).get("success"):
        return
    common = ["--output", str(output), "--batch", str(batch), "--global-batch", str(batch * 4)]
    for label, devices, extra in (("save4", gpus, []), ("resume2", gpus[:2], ["--resume"])):
        log = run / "logs/resume_topology_{}.log".format(label)
        process = launch(ddp_command(HERE / "probe_bg56_resume_topology.py", devices, common + extra), log, devices)
        wait_jobs(run, [(process, log)], "resume_topology_check")
    if not read_json(result)["success"]:
        raise RuntimeError("4-to-2 GPU state restoration probe failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--gpus", nargs="+", type=int, default=[1, 3, 4, 5])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request-pause", action="store_true")
    args = parser.parse_args()
    run = args.run_root.resolve()
    if args.request_pause:
        if not (run / "experiment_config.json").is_file():
            raise FileNotFoundError(run)
        write_json(run / "pause.request.json", {"requested_at": now(), "policy": "exit immediately after the next complete checkpoint"})
        print("Pause requested; the next checkpoint will be completed before exit.")
        return
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        manifest = prepare(run)
        (run / "logs").mkdir(exist_ok=True)
        snapshots = run / "scripts"
        snapshots.mkdir(exist_ok=True)
        files = [HERE / name for name in ("bg56_512_setup.py", "precompute_bg56_teacher_512.py", "train_bg56_student_512.py", "run_bg56_student_512.py", "probe_bg56_resume_topology.py", "train_raft_student_flow.py", "raft_teacher_pair_data.py", "raft_student.py")]
        for file in files:
            target = snapshots / file.name
            if target.exists():
                if sha256(target) != sha256(file):
                    raise ValueError("Training code changed since initial preparation: {}".format(file))
            else:
                shutil.copy2(file, target)
        if args.prepare_only:
            status(run, "prepared", teacher_total_pairs=6002, gpus=args.gpus)
            return
        gpus = args.gpus
        if len(gpus) not in (2, 4) or len(set(gpus)) != len(gpus):
            raise ValueError("Use two or four distinct GPUs")
        if not args.resume and len(gpus) != 4:
            raise ValueError("The initial run was requested on four GPUs")
        memory = gpu_memory(gpus)
        if any(value["used"] > 2048 for value in memory.values()):
            raise RuntimeError("Selected GPUs are no longer idle: {}".format(memory))
        if args.resume and (run / "pause.request.json").is_file():
            (run / "pause.request.json").rename(run / "pause.request.consumed.{}.json".format(time.time_ns()))
        if (run / "student/latest.json").is_file() and not args.resume:
            raise ValueError("Existing student checkpoints require --resume")
        missing = sum(not cache_path(run, pair).is_file() for pair in pairs(manifest))
        if missing:
            status(run, "teacher_running", gpus=gpus, missing_teacher_pairs=missing, teacher_total_pairs=6002)
            jobs = []
            for shard, gpu in enumerate(gpus):
                command = [sys.executable, str(HERE / "precompute_bg56_teacher_512.py"), "--run-root", str(run), "--shard-index", str(shard), "--num-shards", str(len(gpus))]
                log = run / "logs/teacher_gpu{}.log".format(gpu)
                jobs.append((launch(command, log, [gpu]), log))
            wait_jobs(run, jobs, "teacher_running")
        status(run, "cache_audit", teacher_completed_pairs=6002)
        audit(run, manifest)
        if (run / "pause.request.json").is_file():
            status(run, "paused_before_training")
            return
        batch_config = read_json(run / "batch_config.json") if (run / "batch_config.json").is_file() else choose_batch(run, gpus)
        if len(gpus) == 4:
            status(run, "resume_topology_check")
            check_resume_topology(run, gpus, batch_config["batch_per_gpu"])
        batch, accumulation = batch_layout(batch_config["global_batch"], len(gpus), batch_config["batch_per_gpu"])
        resume = args.resume and (run / "student/latest.json").is_file()
        command = ddp_command(HERE / "train_bg56_student_512.py", gpus, training_args(run, batch, accumulation, resume=resume))
        runtime = {"gpus": gpus, "gpu_count": len(gpus), "batch_per_gpu": batch, "gradient_accumulation_steps": accumulation, "global_batch": batch_config["global_batch"], "resume": resume, "command": command, "created_at": now()}
        write_json(run / "runtime_config.json", runtime)
        write_json(run / "runtime_history/{}.json".format(time.time_ns()), runtime)
        log = run / "logs/train{}.log".format("_resume_{}".format(time.time_ns()) if resume else "")
        status(run, "training", **{key: runtime[key] for key in ("gpus", "batch_per_gpu", "gradient_accumulation_steps", "global_batch")}, training_log=str(log))
        process = launch(command, log, gpus)
        wait_jobs(run, [(process, log)], "training")
        final = read_json(run / "student/latest.json")
        if final["global_step"] == 4000:
            status(run, "complete", train_step=4000, best=read_json(run / "student/best.json"))
        elif (run / "pause.request.json").is_file() and (run / "paused_checkpoint.json").is_file():
            status(run, "paused", train_step=final["global_step"], checkpoint=final["checkpoint"])
        else:
            raise RuntimeError("Training exited before its target without a saved pause checkpoint")
    except BaseException as error:
        status(run, "failed", error=repr(error))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
