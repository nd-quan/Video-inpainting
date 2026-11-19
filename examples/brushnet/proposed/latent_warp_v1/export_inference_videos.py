#!/usr/bin/env python
"""Export completed latent-warp PNGs using the existing Quan_test MP4 encoder.

--wait attaches to an already-running evaluation without restarting inference.
This CPU-only stage never changes PNGs, ROI composition, or image metrics.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from PIL import Image

from infer_latent_warp_v1 import REPO, discover, save_json

sys.path.insert(0, str(REPO / "Quan_test"))
from encode_v8r_original_eval_videos import encode, probe
from imgToVideo import SEQUENCE_FPS


def build_video_jobs(output, config):
    output = Path(output).resolve()
    summary = json.loads((output / "summary.json").read_text())
    if summary.get("complete_evaluation") is not True:
        raise ValueError("Video export requires a completed, full evaluation")
    metrics = json.loads((output / "frame_metrics.json").read_text())
    frame_keys = {(r["sequence"], r["frame"]) for r in metrics}
    if len(frame_keys) != len(metrics) or len(metrics) != summary["frames"]:
        raise ValueError("Duplicate metrics or summary frame-count mismatch")
    names = {r["sequence"] for r in metrics}
    sequences = discover(config, sorted(names))
    prediction_dirs = [
        Path(path).resolve() for path in summary.get("prediction_dirs", [str(output)])
    ]
    jobs = []
    for sequence in sequences:
        name = sequence["name"]
        if {stem for seq, stem in frame_keys if seq == name} != set(sequence["stems"]):
            raise ValueError("Incomplete metric coverage for " + name)
        folders = [root / name for root in prediction_dirs if (root / name).is_dir()]
        if len(folders) != 1:
            raise ValueError("Expected one prediction folder for " + name)
        folder = folders[0]
        frames = {p.stem: p.resolve() for p in folder.glob("*.png")}
        if set(frames) != set(sequence["stems"]):
            raise ValueError("Incomplete/extra prediction PNGs for " + name)
        sizes = set()
        for stem in sequence["stems"]:
            with Image.open(frames[stem]) as image:
                sizes.add(image.size)
        if len(sizes) != 1:
            raise ValueError("Inconsistent frame sizes for " + name)
        width, height = sizes.pop()
        if width % 2 or height % 2:
            raise ValueError("MPEG-4 yuv420p requires even dimensions for " + name)
        short_name = name.split("/")[-1]
        if short_name not in SEQUENCE_FPS:
            raise ValueError("Unknown sequence FPS: " + name)
        jobs.append(
            dict(
                sequence=name,
                kind="prediction",
                width=width,
                height=height,
                fps=SEQUENCE_FPS[short_name],
                frame_count=len(frames),
                first_source_frame=int(sequence["stems"][0]),
                last_source_frame=int(sequence["stems"][-1]),
                output=str(output / "videos" / (name + ".mp4")),
                # Sequential staging also supports nonzero starts or source gaps.
                frames=[(i, frames[stem]) for i, stem in enumerate(sequence["stems"])],
            )
        )
    return jobs


def valid_existing(job):
    from fractions import Fraction

    try:
        info = probe(Path(job["output"]))
        return (
            int(info["nb_read_frames"]) == job["frame_count"]
            and (int(info["width"]), int(info["height"]))
            == (job["width"], job["height"])
            and Fraction(info["avg_frame_rate"]) == job["fps"]
        )
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError):
        return False


def convert_evaluation_videos(output, config, workers=2):
    output = Path(output).resolve()
    for command in ("ffmpeg", "ffprobe"):
        if not shutil.which(command):
            raise RuntimeError(command + " is not available in PATH")
    jobs = build_video_jobs(output, config)

    def encode_one(job):
        if Path(job["output"]).is_file():
            if not valid_existing(job):
                raise ValueError("Existing video failed verification: " + job["output"])
            result = {key: value for key, value in job.items() if key != "frames"}
        else:
            # The reused encoder also puts its sequence label in a temp prefix.
            result = encode(dict(job, sequence=job["sequence"].replace("/", "__")))
            result["sequence"] = job["sequence"]
        print(
            "VIDEO verified: {} ({} frames, {} FPS, {}x{})".format(
                job["sequence"],
                job["frame_count"],
                job["fps"],
                job["width"],
                job["height"],
            ),
            flush=True,
        )
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(encode_one, job) for job in jobs]
        results = [future.result() for future in as_completed(futures)]
    manifest = dict(
        eval_root=str(output),
        codec="mpeg4",
        quality=2,
        pixel_format="yuv420p",
        encoder="Quan_test/encode_v8r_original_eval_videos.py:encode",
        fps_source="Quan_test/imgToVideo.py:SEQUENCE_FPS",
        frames=sum(j["frame_count"] for j in jobs),
        png_metrics_unchanged=True,
        videos=sorted(results, key=lambda row: row["sequence"]),
    )
    save_json(output / "videos/video_manifest.json", manifest)
    return manifest


def wait_for_inference(output, task_id=None, poll_seconds=30):
    output = Path(output)
    previous = None
    while True:
        status_path = output / "run_status.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        state = status.get("state", "waiting_for_inference")
        if state != previous:
            print("Waiting for inference: " + state, flush=True)
            previous = state
        if state == "complete":
            return
        if state == "failed":
            raise RuntimeError("Inference failed: " + status.get("error", "unknown"))
        if task_id is not None:
            query = subprocess.run(
                ["job", "status", str(task_id), "--json"],
                capture_output=True,
                text=True,
            )
            try:
                job = json.loads(query.stdout)
            except ValueError:
                job = {}
            if job.get("status") in {"FAILED", "KILLED", "CANCELLED", "COMPLETED"}:
                # Re-read after querying, in case completion occurred meanwhile.
                if (
                    status_path.is_file()
                    and json.loads(status_path.read_text()).get("state") == "complete"
                ):
                    return
                raise RuntimeError(
                    "Source job {} ended as {} without complete evaluation".format(
                        task_id, job["status"]
                    )
                )
        time.sleep(poll_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.poll_seconds < 1:
        parser.error("workers and poll-seconds must be positive")
    output = args.eval_root.resolve()
    if args.wait:
        wait_for_inference(output, args.task_id, args.poll_seconds)
    config = json.loads((output / "resolved_inference_config.json").read_text())
    if args.dry_run:
        jobs = build_video_jobs(output, config)
        print(
            json.dumps(
                [{k: v for k, v in j.items() if k != "frames"} for j in jobs], indent=2
            )
        )
        return

    def state(value, **extra):
        save_json(
            output / "video_status.json",
            dict(
                state=value,
                time=datetime.now(timezone(timedelta(hours=9))).isoformat(),
                **extra
            ),
        )

    try:
        state("encoding_videos")
        manifest = convert_evaluation_videos(output, config, args.workers)
        state("complete", videos=len(manifest["videos"]), frames=manifest["frames"])
    except BaseException as error:
        state("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
