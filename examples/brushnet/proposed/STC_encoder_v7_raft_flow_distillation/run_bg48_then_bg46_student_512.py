#!/usr/bin/env python3
"""Train BG48 on three GPUs, then warm-start BG46 from the best BG48 weights."""
import argparse
import fcntl
import os
import shutil
import sys
import time
from pathlib import Path

from bg56_512_setup import HERE, INITIAL, batch_layout, cache_path, fixed_json, pairs, read_json, sha256, write_json
from run_bg56_student_512 import ddp_command, gpu_memory, launch, now, status, training_args, wait_jobs
from sfu_qp_512_setup import QUEUE, RUN46, RUN48, audit_case, prepare_case, reuse_teacher, verify_case_source


def snapshot(run):
    folder = run / "scripts"
    folder.mkdir(parents=True, exist_ok=True)
    names = ("sfu_qp_512_setup.py", "train_sfu_qp_student_512.py", "run_bg48_then_bg46_student_512.py", "bg56_512_setup.py", "run_bg56_student_512.py", "precompute_bg56_teacher_512.py", "train_raft_student_flow.py", "raft_teacher_pair_data.py", "raft_student.py")
    for name in names:
        source, destination = HERE / name, folder / name
        if destination.exists():
            if sha256(source) != sha256(destination):
                raise ValueError("Stage code changed since preparation: {}".format(source))
        else:
            shutil.copy2(source, destination)


def pause_requested(run):
    return (QUEUE / "pause.request.json").is_file() or (run / "pause.request.json").is_file()


def wait_idle(run, gpus):
    while True:
        memory = gpu_memory(gpus)
        if all(value["used"] < 2048 for value in memory.values()):
            return True
        if pause_requested(run):
            return False
        status(run, "waiting_for_gpus", gpus=gpus, gpu_memory_mib=memory)
        time.sleep(30)


def stage_args(run, initial, gpus, resume):
    batch, accumulation = batch_layout(12, len(gpus), 4)
    args = training_args(run, batch, accumulation, resume=resume)
    if not resume:
        args[args.index("--init_from_checkpoint") + 1] = str(initial)
    return args, batch, accumulation


def choose_best48():
    latest = read_json(RUN48 / "student/latest.json")
    if latest["global_step"] != 4000:
        raise ValueError("BG48 must finish 4000 steps before BG46 starts")
    records = []
    for path in sorted((RUN48 / "student").glob("checkpoint-*/metadata.json")):
        data = read_json(path)
        if (path.parent / "raft_student/pytorch_model.bin").is_file():
            records.append((float(data["validation"]["bg_flow_epe"]), path.parent, data))
    if len(records) != 16:
        raise ValueError("BG48 is missing scheduled checkpoint metadata")
    score, checkpoint, metadata = min(records, key=lambda item: item[0])
    pointer = read_json(RUN48 / "student/best.json")
    if checkpoint.name != pointer["checkpoint"]:
        raise ValueError("BG48 best pointer disagrees with the minimum validation BG EPE")
    selected = {"checkpoint": str(checkpoint), "global_step": metadata["global_step"], "bg_flow_epe": score, "selection_metric": "lowest validation bg_flow_epe across all 16 saved BG48 checkpoints", "weights_sha256": sha256(checkpoint / "raft_student/pytorch_model.bin")}
    write_json(QUEUE / "selected_bg48_checkpoint_for_bg46.json", selected)
    return checkpoint


def run_stage(run, qp, initial, gpus, resume_allowed):
    manifest = prepare_case(run, qp, initial)
    snapshot(run)
    latest_path = run / "student/latest.json"
    if latest_path.is_file() and read_json(latest_path)["global_step"] == 4000:
        status(run, "complete", train_step=4000, skipped_finished_stage=True, best=read_json(run / "student/best.json"))
        return True
    if latest_path.is_file() and not resume_allowed:
        raise ValueError("Existing stage checkpoints require --resume")
    status(run, "teacher_identity_check", bg_qp=qp, initial_checkpoint=str(initial), gpus=gpus)
    reused = reuse_teacher(run, manifest)
    if not reused:
        if not wait_idle(run, gpus):
            status(run, "paused_before_training")
            return False
        jobs = []
        for shard, gpu in enumerate(gpus):
            log = run / "logs/teacher_gpu{}.log".format(gpu)
            command = [sys.executable, str(HERE / "precompute_bg56_teacher_512.py"), "--run-root", str(run), "--shard-index", str(shard), "--num-shards", str(len(gpus))]
            jobs.append((launch(command, log, [gpu]), log))
        wait_jobs(run, jobs, "teacher_running")
    status(run, "cache_audit", teacher_completed_pairs=6002, teacher_cache_reused=reused)
    audit_case(run, manifest, reused)
    if pause_requested(run) or not wait_idle(run, gpus):
        status(run, "paused_before_training")
        return False
    verify_case_source(run)
    resume = latest_path.is_file()
    args, batch, accumulation = stage_args(run, initial, gpus, resume)
    command = ddp_command(HERE / "train_sfu_qp_student_512.py", gpus, args)
    runtime = {"bg_qp": qp, "gpus": gpus, "gpu_count": len(gpus), "batch_per_gpu": batch, "gradient_accumulation_steps": accumulation, "global_batch": 12, "initial_checkpoint": str(initial), "resume": resume, "command": command, "created_at": now()}
    write_json(run / "runtime_config.json", runtime)
    write_json(run / "runtime_history/{}.json".format(time.time_ns()), runtime)
    log = run / "logs/train{}.log".format("_resume_{}".format(time.time_ns()) if resume else "")
    status(run, "training", bg_qp=qp, gpus=gpus, batch_per_gpu=batch, gradient_accumulation_steps=accumulation, global_batch=12, initial_checkpoint=str(initial), training_log=str(log))
    status(QUEUE, "stage_training", active_qp=qp, active_run=str(run), stage_status=str(run / "status.json"), gpus=gpus)
    process = launch(command, log, gpus)
    wait_jobs(run, [(process, log)], "training")
    latest = read_json(latest_path)
    if latest["global_step"] == 4000:
        status(run, "complete", train_step=4000, last_checkpoint_step=4000, child_pids=[], best=read_json(run / "student/best.json"))
        return True
    if pause_requested(run) and (run / "paused_checkpoint.json").is_file():
        status(run, "paused", train_step=latest["global_step"], child_pids=[])
        return False
    raise RuntimeError("Stage stopped before 4000 steps without a completed pause checkpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 2, 5])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request-pause", action="store_true")
    args = parser.parse_args()
    if args.request_pause:
        state = read_json(QUEUE / "status.json")
        write_json(QUEUE / "pause.request.json", {"requested_at": now()})
        if state.get("active_run"):
            write_json(Path(state["active_run"]) / "pause.request.json", {"requested_at": now(), "pause_after_complete_checkpoint": True})
        print("Pause requested for the active stage; the dependent stage will wait.")
        return
    if len(set(args.gpus)) != len(args.gpus) or len(args.gpus) not in (2, 3):
        raise ValueError("Use three distinct GPUs initially, or two for a later resume")
    if len(args.gpus) != 3 and not args.resume:
        raise ValueError("The initial queue was requested on three GPUs")
    QUEUE.mkdir(parents=True, exist_ok=True)
    lock = (QUEUE / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        fixed_json(QUEUE / "queue_config.json", {"order": [48, 46], "bg48_initial_checkpoint": str(INITIAL), "bg46_initial_checkpoint_rule": "best BG48 checkpoint after BG48 completes all 4000 steps", "run48": str(RUN48), "run46": str(RUN46), "steps_per_stage": 4000, "resolution": [512, 512], "learning_rate": 1e-5, "lr_scheduler": "constant", "initial_gpu_count": 3, "initial_batch_per_gpu": 4, "initial_accumulation": 1, "global_batch": 12, "validation": "source test folder; eight unique adjacent pairs per sequence", "no_max960_artifacts": True})
        RUN46.mkdir(parents=True, exist_ok=True)
        fixed_json(RUN46 / "queued_config.json", {"phase": "waiting_for_bg48_best", "source_qp": 46, "initialization_rule": "use queue/selected_bg48_checkpoint_for_bg46.json after BG48 completes", "upstream_best_pointer": str(RUN48 / "student/best.json"), "upstream_required_step": 4000, "steps": 4000, "resolution": [512, 512]})
        if args.resume:
            for root in (QUEUE, RUN48, RUN46):
                request = root / "pause.request.json"
                if request.is_file():
                    request.rename(root / "pause.request.consumed.{}.json".format(time.time_ns()))
        for qp, run in ((48, RUN48), (46, RUN46)):
            if (QUEUE / "pause.request.json").is_file():
                status(QUEUE, "paused", active_qp=qp, active_run=str(run))
                return
            initial = INITIAL if qp == 48 else choose_best48()
            status(QUEUE, "stage_preparing", active_qp=qp, active_run=str(run), initial_checkpoint=str(initial), gpus=args.gpus)
            if not run_stage(run, qp, initial, args.gpus, args.resume):
                status(QUEUE, "paused", active_qp=qp, active_run=str(run))
                return
        status(QUEUE, "complete", active_qp=None, bg48_best=read_json(RUN48 / "student/best.json"), bg46_best=read_json(RUN46 / "student/best.json"))
    except BaseException as error:
        status(QUEUE, "failed", error=repr(error))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
