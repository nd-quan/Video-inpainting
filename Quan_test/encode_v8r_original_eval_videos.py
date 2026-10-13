#!/usr/bin/env python3
"""Stitch native-size V8-R clip PNGs into verified five-second MP4s."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import tempfile

from imgToVideo_rgb_stc_eval import discover_sources, select_unique_frames


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = ROOT / "experiments/eval_v8r_original_size_5s_ckpt2500_latent_tg0p1"
DATASET_MANIFEST = ROOT / "examples/brushnet/dataset/original_size/manifest.json"
KINDS = ("final", "raw", "input", "gt", "mask_roi")


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries",
         "stream=width,height,nb_read_frames,avg_frame_rate,duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream in {path}")
    return streams[0]


def encode(job: dict) -> dict:
    name, kind, frames = job["sequence"], job["kind"], job["frames"]
    width, height, fps = job["width"], job["height"], job["fps"]
    output = Path(job["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.mp4")
    first_id = frames[0][0]
    with tempfile.TemporaryDirectory(prefix=f"stitch_{name}_{kind}_") as folder:
        stage = Path(folder)
        for frame_id, source in frames:
            (stage / f"{frame_id:06d}.png").symlink_to(source)
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", str(fps), "-start_number", str(first_id),
            "-i", str(stage / "%06d.png"), "-frames:v", str(len(frames)),
            "-an", "-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(temporary),
        ]
        try:
            subprocess.run(command, check=True)
            info = probe(temporary)
            actual_fps = info["avg_frame_rate"]
            num, den = (int(x) for x in actual_fps.split("/"))
            if (int(info["nb_read_frames"]) != len(frames)
                    or (int(info["width"]), int(info["height"])) != (width, height)
                    or num / den != fps):
                raise ValueError(f"Video validation failed for {temporary}: {info}")
            temporary.replace(output)
        finally:
            if temporary.exists():
                temporary.unlink()
    return {key: value for key, value in job.items() if key != "frames"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--frame-kinds", nargs="+", choices=KINDS, default=["final"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    eval_root = args.eval_root.resolve()
    reference = json.loads(DATASET_MANIFEST.read_text())["sequences"]
    expected = {record["sequence"]: record for record in reference}
    jobs = []
    for kind in args.frame_kinds:
        sources = discover_sources(eval_root, kind, recursive=True)
        if set(sources) != set(expected):
            raise ValueError(f"Incomplete sequence set for {kind}: {set(sources) ^ set(expected)}")
        for name in sorted(expected):
            record = expected[name]
            frames = [(frame_id, path) for frame_id, path, _ in
                      select_unique_frames(sources[name], "first")]
            if (len(frames) != record["frames"]
                    or frames[0][0] != record["first_frame"]
                    or frames[-1][0] != record["last_frame"]):
                raise ValueError(f"Incomplete five-second sequence: {name}/{kind}")
            jobs.append({
                "sequence": name, "kind": kind, "fps": record["fps"],
                "width": record["width"], "height": record["height"],
                "frame_count": len(frames), "first_frame": frames[0][0],
                "last_frame": frames[-1][0],
                "selection": "first clip occurrence by ascending clip start",
                "output": str(eval_root / "videos" / name / f"{kind}.mp4"),
                "frames": frames,
            })
    for job in jobs:
        print(job["sequence"], job["kind"], job["frame_count"],
              job["fps"], job["output"], flush=True)
    if args.dry_run:
        return
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(encode, job): job for job in jobs}
        for future in as_completed(pending):
            result = future.result()
            completed.append(result)
            print("DONE", result["sequence"], result["kind"], flush=True)
    manifest = {
        "eval_root": str(eval_root), "source_manifest": str(DATASET_MANIFEST),
        "codec": "mpeg4", "quality": 2, "frame_kinds": args.frame_kinds,
        "videos": sorted(completed, key=lambda item: (item["sequence"], item["kind"])),
    }
    path = eval_root / "videos/video_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("VERIFIED", len(completed), "videos", path, flush=True)


if __name__ == "__main__":
    main()
