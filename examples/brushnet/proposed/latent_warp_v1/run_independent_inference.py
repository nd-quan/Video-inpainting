#!/usr/bin/env python
"""CPU coordinator for independent one-GPU inference jobs and video export.

Resume partial shards without rerunning valid PNGs. Each scheduler allocation
ends when its own shard finishes, avoiding idle GPUs in a shared allocation.
An optional second config runs only after the first evaluation/videos complete.
"""
import argparse
from datetime import datetime, timezone, timedelta
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from infer_latent_warp_v1 import (
    HERE,
    read_config,
    discover,
    preflight,
    save_json,
    validate_resume_output,
)
from inference_jobs import partition_sequences, merge_results, query_task
from export_inference_videos import convert_evaluation_videos


def timestamp():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def shard_ready(output, sequences, config):
    output = Path(output)
    if (
        not (output / "summary.json").is_file()
        or not (output / "frame_metrics.json").is_file()
    ):
        return False
    summary = json.loads((output / "summary.json").read_text())
    if summary.get("complete_evaluation") is not True:
        return False
    expected = {(s["name"], stem) for s in sequences for stem in s["stems"]}
    rows = json.loads((output / "frame_metrics.json").read_text())
    actual = {(r["sequence"], r["frame"]) for r in rows}
    if (
        actual != expected
        or len(rows) != len(expected)
        or summary["frames"] != len(expected)
    ):
        return False
    shard_config = dict(config, output_dir=str(output))
    count = validate_resume_output(output, shard_config, sequences)
    return count == len(expected)


def submit_shard(config_path, output, sequences, resume, temporal_enabled):
    command = [
        sys.executable,
        str(HERE / "infer_latent_warp_v1.py"),
        "--config",
        str(config_path),
        "--output-dir",
        str(output),
        "--device",
        "cuda:0",
        "--sequences",
    ] + [s["name"] for s in sequences]
    if resume:
        command += ["--resume"]
    request = [
        "job",
        "run",
        "-g",
        "1",
        "-e",
        str(Path(sys.executable).parents[1]),
        "-n",
        "Latent warp TG {}: independent shard {}".format(
            "ON" if temporal_enabled else "OFF", output.name
        ),
        shlex.join(command),
    ]
    response = subprocess.run(request, check=True, capture_output=True, text=True)
    match = re.search(r"Task #(\d+)", response.stdout)
    if not match:
        raise RuntimeError("No task ID in submission response: " + response.stdout)
    print(response.stdout, flush=True)
    return int(match.group(1))


def run_phase(config, workers, resume, poll_seconds):
    sequences = discover(config)
    preflight(config, sequences)
    output = Path(config["output_dir"]).resolve()
    config = dict(config, output_dir=str(output))
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".independent.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = output / "resolved_inference_config.json"
        if saved.exists():
            if not resume:
                raise FileExistsError("Use --resume for existing output " + str(output))
            previous = read_config(saved)
            different = [k for k in config if previous[k] != config[k]]
            if different:
                raise ValueError("Evaluation config differs: " + str(different))
        save_json(saved, config)
        plan_path = output / "shard_plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            by_name = {s["name"]: s for s in sequences}
            assigned = [name for row in plan for name in row["sequences"]]
            if set(assigned) != set(by_name) or len(assigned) != len(by_name):
                raise ValueError("Existing shard plan has missing/duplicate sequences")
            if [p["rank"] for p in plan] != list(range(len(plan))):
                raise ValueError("Shard ranks must be contiguous")
            shards = [[by_name[name] for name in row["sequences"]] for row in plan]
        else:
            shards = partition_sequences(sequences, workers, config)
            plan = [
                dict(
                    rank=i,
                    sequences=[s["name"] for s in shard],
                    frames=sum(len(s["stems"]) for s in shard),
                )
                for i, shard in enumerate(shards)
            ]
            save_json(plan_path, plan)
        previous_status = output / "run_status.json"
        backup = output / "run_status.before_independent_resume.json"
        if previous_status.exists() and not backup.exists():
            backup.write_bytes(previous_status.read_bytes())
        job_path = output / "independent_jobs.json"
        jobs = (
            json.loads(job_path.read_text()).get("jobs", {})
            if job_path.exists()
            else {}
        )

        def status(state, **extra):
            save_json(
                output / "run_status.json",
                dict(
                    state=state,
                    time=timestamp(),
                    execution_mode="independent_single_gpu_jobs",
                    **extra
                ),
            )

        def persist():
            save_json(
                job_path, dict(jobs=jobs, execution_mode="independent_single_gpu_jobs")
            )

        try:
            status("submitting_independent_jobs")
            finished = set()
            for rank, shard in enumerate(shards):
                folder = output / "shards" / ("shard-%02d" % rank)
                if shard_ready(folder, shard, config):
                    print("Reuse completed shard " + str(rank), flush=True)
                    finished.add(rank)
                    continue
                previous_job = jobs.get(str(rank))
                if previous_job:
                    task = query_task(previous_job)
                    if task["status"] not in {"RUNNING", "PENDING"}:
                        raise RuntimeError(
                            "Previous shard job {} is {}; inspect logs before retry".format(
                                previous_job, task["status"]
                            )
                        )
                    continue
                nonempty = folder.exists() and any(folder.iterdir())
                if nonempty and not resume:
                    raise FileExistsError(str(folder))
                task_id = submit_shard(
                    saved, folder, shard, nonempty, config["temporal_guidance_enabled"]
                )
                jobs[str(rank)] = task_id
                persist()
            status("running_independent_jobs", jobs=jobs)
            last = None
            while True:
                for rank, shard in enumerate(shards):
                    if rank not in finished and shard_ready(
                        output / "shards" / ("shard-%02d" % rank), shard, config
                    ):
                        finished.add(rank)
                if len(finished) == len(shards):
                    break
                statuses = {
                    rank: query_task(task_id)["status"]
                    for rank, task_id in jobs.items()
                    if int(rank) not in finished
                }
                if statuses != last:
                    print(
                        "Independent shard status: " + json.dumps(statuses), flush=True
                    )
                    last = statuses
                failed = [
                    rank
                    for rank, state in statuses.items()
                    if state not in {"RUNNING", "PENDING"}
                ]
                if failed:
                    raise RuntimeError(
                        "Shard jobs ended without complete outputs: "
                        + str({rank: jobs[rank] for rank in failed})
                    )
                time.sleep(poll_seconds)
            status("merging_metrics", jobs=jobs)
            merge_results(output, shards, config)
            summary = json.loads((output / "summary.json").read_text())
            summary["execution_mode"] = "independent_single_gpu_jobs"
            summary["scheduler_jobs"] = jobs
            save_json(output / "summary.json", summary)
            status("encoding_videos", jobs=jobs)
            save_json(
                output / "video_status.json",
                dict(state="encoding_videos", time=timestamp()),
            )
            manifest = convert_evaluation_videos(output, config, workers=2)
            save_json(
                output / "video_status.json",
                dict(
                    state="complete",
                    videos=len(manifest["videos"]),
                    frames=manifest["frames"],
                    time=timestamp(),
                ),
            )
            status(
                "complete",
                jobs=jobs,
                frames=summary["frames"],
                videos=len(manifest["videos"]),
            )
        except BaseException as error:
            status("failed", error=str(error), jobs=jobs)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of independent one-GPU shards for a new evaluation",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--then-config",
        type=Path,
        help="Run this second evaluation after first metrics and videos succeed",
    )
    args = parser.parse_args(argv)
    if args.workers < 1 or args.poll_seconds < 1:
        parser.error("workers/poll-seconds must be positive")
    config = read_config(args.config)
    if args.output_dir:
        config["output_dir"] = str(args.output_dir.resolve())
    run_phase(config, args.workers, args.resume, args.poll_seconds)
    if args.then_config:
        second = read_config(args.then_config)
        differences = [
            key
            for key in config
            if key not in {"output_dir", "temporal_guidance_enabled"}
            and config[key] != second[key]
        ]
        if differences:
            raise ValueError(
                "Follow-up ablation changes other settings: " + str(differences)
            )
        if second["output_dir"] == config["output_dir"]:
            raise ValueError("Follow-up needs a distinct output directory")
        run_phase(
            second, args.workers, Path(second["output_dir"]).exists(), args.poll_seconds
        )


if __name__ == "__main__":
    main()
