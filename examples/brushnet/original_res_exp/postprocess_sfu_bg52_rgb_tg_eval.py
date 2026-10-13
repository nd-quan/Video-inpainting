#!/usr/bin/env python3
"""Make eval MP4s, reuse SFU's six-metric pipeline, and summarize sequences."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import subprocess
import sys


BRUSHNET = Path(__file__).resolve().parents[2]
PROJECT = Path("/home/cilab/ndquan/vcm-rs")
DATASET = PROJECT / "output/sfu_qp0_bg_table_window_test/bg52/test"
SOURCE_MANIFEST = PROJECT / "output/postprocess_sfu_full_qp0_bg_multiqp/05_table_window/manifest.json"
OUTPUT = BRUSHNET / "experiments/eval_sfu_bg52_table_window_ckpt1000_rgb_tg"
METRIC_SCRIPT = PROJECT / "Plot/compute_sfu_table_window_metrics.py"
FFMPEG = Path("/home/cilab/ndquan/envs/vcm/bin/ffmpeg")
FFPROBE = FFMPEG.with_name("ffprobe")
PYTHON = Path("/home/cilab/ndquan/envs/guided_diff/bin/python")
METRICS = ("PSNR", "SSIM", "MS-SSIM", "LPIPS", "Frame-level FID", "FVD")
IMAGE_METRICS = METRICS[:4]


def video_info(path: Path) -> dict:
    result = subprocess.run(
        [str(FFPROBE), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
         "-of", "json", str(path)], capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)["streams"][0]


def valid_video(path: Path, row: dict) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        info = video_info(path)
        numerator, denominator = map(int, info["r_frame_rate"].split("/"))
        return (int(info["width"]) == int(row["width"])
                and int(info["height"]) == int(row["height"])
                and int(info["nb_frames"]) == int(row["frames"])
                and abs(numerator / denominator - int(row["fps"])) < 0.001)
    except (OSError, KeyError, ValueError, IndexError, subprocess.CalledProcessError):
        return False


def load_cases() -> list[dict]:
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))["cases"]
    rows = sorted((dict(row) for row in source if row["bg_qp"] == 52),
                  key=lambda row: row["sequence"])
    if len(rows) != 14 or len({r["sequence"] for r in rows}) != 14:
        raise ValueError("Expected 14 BG52 cases")
    for row in rows:
        name, cls = row["sequence"], row["class"]
        first, last = row["first_frame"], row["last_frame"]
        expected = {f"{index:06d}.png" for index in range(first, last + 1)}
        if len(expected) != row["frames"]:
            raise ValueError(f"Invalid frame range: {name}")
        for label, folder in (
            ("prediction", OUTPUT / "test" / cls / name),
            ("GT", DATASET / "GT" / cls / name),
            ("mask", DATASET / "mask" / cls / name),
        ):
            found = {path.name for path in folder.glob("*.png")}
            if found != expected:
                raise ValueError(f"{name} {label}: {len(found)}/{len(expected)} aligned frames")
        reference = Path(row["reference_video"])
        info = video_info(reference)
        row["width"], row["height"] = int(info["width"]), int(info["height"])
        if not valid_video(reference, row):
            raise ValueError(f"Invalid reference video: {row['reference_video']}")
        row["combined_video"] = str(OUTPUT / "videos" / cls /
                                    f"{name}_bg52_ckpt1000_sharednoise_rgb_tg.mp4")
    return rows


def make_video(row: dict) -> None:
    video = Path(row["combined_video"])
    if valid_video(video, row):
        print(f"[VIDEO SKIP] {row['sequence']}", flush=True)
        return
    video.parent.mkdir(parents=True, exist_ok=True)
    image_pattern = OUTPUT / "test" / row["class"] / row["sequence"] / "%06d.png"
    temporary = video.with_suffix(".tmp.mp4")
    command = [str(FFMPEG), "-nostdin", "-hide_banner", "-loglevel", "error",
               "-y", "-framerate", str(row["fps"]),
               "-start_number", str(row["first_frame"]), "-i", str(image_pattern),
               "-frames:v", str(row["frames"]), "-an", "-c:v", "libx264",
               "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2", str(temporary)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        if not valid_video(temporary, row):
            raise RuntimeError(f"Wrong video dimensions, FPS, or frames: {temporary}")
        os.replace(temporary, video)
        print(f"[VIDEO DONE] {row['sequence']}: {row['frames']} frames at {row['fps']} fps",
              flush=True)
    finally:
        temporary.unlink(missing_ok=True)


def write_manifest(rows: list[dict]) -> Path:
    manifest = OUTPUT / "metric_manifest.json"
    data = {"cases": rows, "metric_domain": "8-bit H.264 MP4 CRF 18",
            "prediction_source": str(OUTPUT / "test"),
            "reference_source": str(DATASET / "GT")}
    temporary = manifest.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest)
    return manifest


def summarize(rows: list[dict]) -> None:
    metric_csv = OUTPUT / "metrics/rd_metrics.csv"
    with metric_csv.open(newline="", encoding="utf-8") as stream:
        scores = list(csv.DictReader(stream))
    if len(scores) != len(rows) or {r["sequence"] for r in scores} != {r["sequence"] for r in rows}:
        raise ValueError(f"Expected {len(rows)} metric rows; got {len(scores)}")
    if any(not math.isfinite(float(row[key])) for row in scores for key in METRICS):
        raise ValueError("Non-finite metric value")
    for score in scores:
        score["model"] = "checkpoint-1000 shared noise 0.95 RGB temporal guidance"
    summary = []
    for method in ("sequence_macro_mean", "frame_weighted_mean"):
        weights = [1 if method == "sequence_macro_mean" else int(row["frames"])
                   for row in scores]
        item = {"aggregation": method, "sequences": len(scores),
                "frames": sum(int(row["frames"]) for row in scores)}
        for key in METRICS:
            if method == "frame_weighted_mean" and key not in IMAGE_METRICS:
                item[key] = ""
            else:
                item[key] = sum(float(row[key]) * w for row, w in zip(scores, weights)) / sum(weights)
        summary.append(item)
    summary_path = OUTPUT / "metrics/summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (OUTPUT / "metrics/summary.json").write_text(json.dumps({
        "rows": summary,
        "definitions": {
            "sequence_macro_mean": "Arithmetic mean of 14 per-sequence scores",
            "frame_weighted_mean": "Per-sequence frame means weighted by coded frame count; image metrics only",
            "FID_FVD_note": "Macro mean is descriptive; it is not FID or FVD pooled across all frames/clips",
            "metric_domain": "Prediction and GT videos encoded as 8-bit H.264 CRF 18",
            "overlap_note": "BG52 test set includes 31 frames also present in the earlier training split",
        },
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[SUMMARY] {len(scores)} sequence rows, {summary[0]['frames']} frames: {summary_path}",
          flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    rows = load_cases()
    if args.summary_only:
        summarize(rows)
        return 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(make_video, row) for row in rows]):
            future.result()
    manifest = write_manifest(rows)
    if args.videos_only:
        return 0
    command = [str(PYTHON), "-u", str(METRIC_SCRIPT),
               "--manifest", str(manifest), "--output", str(OUTPUT / "metrics"),
               "--expected-cases", "14", "--gpus", "0", "1", "2", "3", "4", "5"]
    print("[METRICS] Starting six-GPU SFU metric pipeline", flush=True)
    code = subprocess.call(command, cwd=PROJECT)
    if code:
        raise RuntimeError(f"Metric pipeline exited {code}")
    summarize(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
