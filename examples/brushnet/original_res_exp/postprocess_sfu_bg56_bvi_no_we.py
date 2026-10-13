#!/usr/bin/env python3
"""Score native BG56 PNGs without warping error, then encode the sequence videos."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image


BRUSHNET = Path(__file__).resolve().parents[2]
EVAL = BRUSHNET / "experiments/eval_sfu_bg56_table_window_bvi_ckpt2000_rgb_tg"
DATASET = Path("/home/cilab/ndquan/vcm-rs/output/sfu_qp0_bg_table_window_test/bg56/test")
SOURCE_MANIFEST = Path(
    "/home/cilab/ndquan/vcm-rs/output/postprocess_sfu_full_qp0_bg_multiqp/05_table_window/manifest.json"
)
METRIC_SCRIPT = BRUSHNET / "examples/brushnet/proposed/DGAF_VSR_original/postprocess_eval.py"
EVALUATOR_ROOT = BRUSHNET.parent / "video-inpainting-evaluation"
PYTHON = Path("/home/cilab/ndquan/envs/guided_diff/bin/python")
FFMPEG = Path("/home/cilab/ndquan/envs/vcm/bin/ffmpeg")
FFPROBE = FFMPEG.with_name("ffprobe")
OUTPUT = EVAL / "metrics_full_reference_no_we"
METRICS = ("PSNR", "SSIM", "LPIPS", "FS", "VFID_Clips")
FIELDS = ("sequence", "class", "frames", "pairs", *METRICS)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_jobs() -> list[dict]:
    status = json.loads((EVAL / "logs/status.json").read_text(encoding="utf-8"))
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))["cases"]
    source = {row["sequence"]: row for row in source if row["bg_qp"] == 56}
    if len(status) != 14 or len(source) != 14 or any(row["state"] != "complete" for row in status):
        raise ValueError("Expected 14 complete BG56 sequences and 14 source manifest entries")
    jobs = []
    for record in status:
        name, cls = record["sequence"], record["class"]
        metadata = source[name]
        if metadata["class"] != cls or metadata["frames"] != record["frame_count"]:
            raise ValueError(f"Manifest/status mismatch for {name}")
        ids = list(range(record["frame_range"][0], record["frame_range"][1] + 1))
        if (len(ids) != metadata["frames"] or ids[0] != metadata["first_frame"]
                or ids[-1] != metadata["last_frame"]):
            raise ValueError(f"Frame-range mismatch for {name}")
        pred_dir = EVAL / "test" / cls / name
        gt_dir = DATASET / "GT" / cls / name
        expected = {f"{frame_id:06d}.png" for frame_id in ids}
        if ({path.name for path in pred_dir.glob("*.png")} != expected
                or {path.name for path in gt_dir.glob("*.png")} != expected):
            raise ValueError(f"Prediction/GT frame IDs differ for {name}")
        for frame_id in (ids[0], ids[-1]):
            with Image.open(pred_dir / f"{frame_id:06d}.png") as pred, Image.open(
                gt_dir / f"{frame_id:06d}.png"
            ) as gt:
                if pred.size != gt.size or list(pred.size) != record["size_wh"]:
                    raise ValueError(f"Prediction/GT resolution differs for {name}, frame {frame_id}")
        jobs.append({"sequence": name, "class": cls, "ids": ids, "fps": metadata["fps"],
                     "width": record["size_wh"][0], "height": record["size_wh"][1],
                     "pred_dir": pred_dir, "gt_dir": gt_dir})
    if sum(len(job["ids"]) for job in jobs) != 1198:
        raise ValueError("Expected 1198 BG56 frames")
    return jobs


def case_dir(job: dict) -> Path:
    return OUTPUT / "cases" / job["sequence"]


def case_result(job: dict) -> dict | None:
    folder = case_dir(job)
    try:
        with (folder / "metrics.csv").open(newline="", encoding="utf-8") as stream:
            row = next(row for row in csv.DictReader(stream) if row["sequence"] == job["sequence"])
        if (int(row["frames"]) != len(job["ids"])
                or int(row["pairs"]) != len(job["ids"]) - 1
                or "WE" in row
                or any(not math.isfinite(float(row[key])) for key in METRICS)
                or ("WE_L2" in row and not math.isfinite(float(row["WE_L2"])))):
            return None
        if not all((folder / f"{job['sequence']}_{kind}_i3d.npy").is_file()
                   for kind in ("gt", "final")):
            return None
        return row
    except (FileNotFoundError, StopIteration, KeyError, ValueError):
        return None


def score_one(job: dict, gpu: int) -> None:
    if case_result(job) is not None:
        print(f"[METRIC SKIP] {job['sequence']}", flush=True)
        return
    name = job["sequence"]
    with tempfile.TemporaryDirectory(prefix=f"sfu_bg56_metric_{name}_") as dirname:
        stage = Path(dirname)
        sequence = stage / name
        for kind, folder in (("final", job["pred_dir"]), ("gt", job["gt_dir"])):
            target = sequence / kind
            target.mkdir(parents=True)
            for frame_id in job["ids"]:
                (target / f"{frame_id:06d}.png").symlink_to(
                    (folder / f"{frame_id:06d}.png").resolve()
                )
        (sequence / "clip_metrics.json").write_text(
            json.dumps({"video": name, "frame_ids": job["ids"]}), encoding="utf-8"
        )
        (stage / "summary.json").write_text('{"clips": 1}', encoding="utf-8")
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        log = OUTPUT / "logs" / f"{name}_gpu{gpu}.log"
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\n=== {datetime.now().isoformat()} GPU {gpu} {name} ===\n")
            stream.flush()
            result = subprocess.run(
                [str(PYTHON), "-u", str(METRIC_SCRIPT), "--eval_root", str(stage),
                 "--evaluator_root", str(EVALUATOR_ROOT), "--skip_warping"],
                cwd=BRUSHNET, env=env, stdout=stream, stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"{name} metric process exited {result.returncode}; see {log}")
        source = stage / "metrics"
        if not (source / "metrics.csv").is_file():
            raise RuntimeError(f"Missing metrics.csv for {name}; see {log}")
        destination = case_dir(job)
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        save_json(destination / "source.json", {
            "sequence": name, "class": job["class"], "frames": len(job["ids"]),
            "prediction": str(job["pred_dir"]), "ground_truth": str(job["gt_dir"]),
            "resolution": "native PNG", "evaluator": str(METRIC_SCRIPT),
            "warping_errors_computed": False, "gpu": gpu,
        })
    if case_result(job) is None:
        raise RuntimeError(f"Invalid metric result for {name}")
    print(f"[METRIC DONE] {name}: {case_dir(job) / 'metrics.csv'}", flush=True)


def summarize(jobs: list[dict], method_label: str = "BG56 BVI ckpt2000 SN+TG") -> None:
    sys.path.insert(0, str(BRUSHNET / "Quan_test"))
    from compute_vfid_clips_4datasets import frechet_distance_low_rank

    rows = []
    features = {kind: [] for kind in ("gt", "final")}
    l2_presence = []
    for job in sorted(jobs, key=lambda item: item["sequence"]):
        raw = case_result(job)
        if raw is None:
            raise ValueError(f"Missing metric result for {job['sequence']}")
        l2_presence.append("WE_L2" in raw)
        row = {"sequence": job["sequence"], "class": job["class"],
               "frames": int(raw["frames"]), "pairs": int(raw["pairs"])}
        row.update({key: float(raw[key]) for key in METRICS})
        if "WE_L2" in raw:
            row["WE_L2"] = float(raw["WE_L2"])
            row["WE-L2 (×10⁻²)"] = row["WE_L2"] * 100
        rows.append(row)
        for kind in features:
            features[kind].append(np.load(case_dir(job) / f"{job['sequence']}_{kind}_i3d.npy"))
    if any(l2_presence) and not all(l2_presence):
        raise ValueError("Only some cases have WE-L2; finish add_we_l2_sfu_bg56.py first")
    has_l2 = all(l2_presence)
    fields = list(FIELDS)
    if has_l2:
        fields[fields.index("FS"):fields.index("FS")] = ["WE_L2", "WE-L2 (×10⁻²)"]
    with (OUTPUT / "per_sequence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for aggregation in ("sequence_macro_mean", "frame_pair_weighted_mean"):
        row = {"aggregation": aggregation, "sequences": len(rows),
               "frames": sum(item["frames"] for item in rows),
               "pairs": sum(item["pairs"] for item in rows)}
        for key in (*METRICS, *(("WE_L2",) if has_l2 else ())):
            if aggregation == "sequence_macro_mean":
                row[key] = float(np.mean([item[key] for item in rows]))
            elif key == "VFID_Clips":
                row[key] = float(frechet_distance_low_rank(
                    np.concatenate(features["gt"]), np.concatenate(features["final"])
                ))
            else:
                weight = "pairs" if key in ("FS", "WE_L2") else "frames"
                row[key] = float(np.average([item[key] for item in rows],
                                            weights=[item[weight] for item in rows]))
        if has_l2:
            row["WE-L2 (×10⁻²)"] = row["WE_L2"] * 100
        summaries.append(row)
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,
                                fieldnames=["aggregation", "sequences", "frames", "pairs",
                                            *fields[4:]])
        writer.writeheader()
        writer.writerows(summaries)
    macro = summaries[0]
    table_row = {"Method": method_label, "Sequences": len(rows),
                 "PSNR": macro["PSNR"], "SSIM": macro["SSIM"],
                 "LPIPS": macro["LPIPS"], "FS": macro["FS"],
                 "VFID": macro["VFID_Clips"]}
    if has_l2:
        table_row["WE-L2 (×10⁻²)"] = macro["WE-L2 (×10⁻²)"]
        table_row = {key: table_row[key] for key in
                     ("Method", "Sequences", "PSNR", "SSIM", "LPIPS",
                      "WE-L2 (×10⁻²)", "FS", "VFID")}
    with (OUTPUT / "comparison_row.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_row))
        writer.writeheader()
        writer.writerow(table_row)
    header = "| Method | Sequences | PSNR ↑ | SSIM ↑ | LPIPS ↓ |"
    divider = "|---|---:|---:|---:|---:|"
    data = (f"| {table_row['Method']} | {len(rows)} | {macro['PSNR']:.4f} | "
            f"{macro['SSIM']:.5f} | {macro['LPIPS']:.5f} | ")
    if has_l2:
        header += " WE-L2 (×10⁻²) ↓ |"
        divider += "---:|"
        data += f"{macro['WE-L2 (×10⁻²)']:.5f} | "
    header += " FS ↑ | VFID ↓ |"
    divider += "---:|---:|"
    data += f"{macro['FS']:.5f} | {macro['VFID_Clips']:.4f} |"
    (OUTPUT / "comparison_row.md").write_text(
        "\n".join((header, divider, data)) + "\n", encoding="utf-8"
    )
    save_json(OUTPUT / "summary.json", {
        "results": summaries,
        "definitions": {
            "scope": "native prediction/GT PNG, before video compression",
            "PSNR": "Mean over frames of RGB uint8 PSNR",
            "SSIM": "Mean over frames of torchmetrics SSIM",
            "LPIPS": "Mean over frames of AlexNet LPIPS",
            "FS": "Mean SSIM between adjacent prediction frames",
            "VFID_Clips": "I3D rgb_imagenet, 10-frame sliding windows, 224x224 bicubic",
            "sequence_macro_mean": f"Arithmetic mean of {len(rows)} per-sequence scores, matching the comparison table's averaging method",
            "frame_pair_weighted_mean": "Image metrics weighted by frames; FS by frame pairs; pooled VFID features",
            "WE": "not computed",
            "WE_L2": ("Farneback backward flow; valid-pixel squared RGB error on adjacent predictions"
                      if has_l2 else "not computed"),
            "WE_L2_display_multiplier": 100 if has_l2 else None,
            "comparison_scope_note": "The referenced 2026-09-18 table uses a different sequence/frame subset; compare only after matching scope",
        },
    })
    print(f"[SUMMARY] {len(rows)} sequences, {sum(item['frames'] for item in rows)} frames: {OUTPUT / 'summary.csv'}", flush=True)


def video_path(job: dict) -> Path:
    return EVAL / "videos" / job["class"] / f"{job['sequence']}_bg56_ckpt2000_sharednoise_rgb_tg.mp4"


def video_info(path: Path) -> dict:
    result = subprocess.run(
        [str(FFPROBE), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
         "-of", "json", str(path)], capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)["streams"][0]


def valid_video(path: Path, job: dict) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        info = video_info(path)
        num, den = map(int, info["r_frame_rate"].split("/"))
        return (int(info["width"]) == job["width"]
                and int(info["height"]) == job["height"]
                and int(info["nb_frames"]) == len(job["ids"])
                and abs(num / den - job["fps"]) < 0.001)
    except (OSError, KeyError, ValueError, IndexError, subprocess.CalledProcessError):
        return False


def encode_video(job: dict) -> None:
    output = video_path(job)
    if valid_video(output, job):
        print(f"[VIDEO SKIP] {job['sequence']}", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.mp4")
    command = [str(FFMPEG), "-nostdin", "-hide_banner", "-loglevel", "error",
               "-y", "-framerate", str(job["fps"]), "-start_number", str(job["ids"][0]),
               "-i", str(job["pred_dir"] / "%06d.png"),
               "-frames:v", str(len(job["ids"])), "-an", "-c:v", "libx264",
               "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2", str(temporary)]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"FFmpeg failed for {job['sequence']}: {result.stderr[-1000:]}")
        if not valid_video(temporary, job):
            raise RuntimeError(f"Video failed frame/FPS/resolution verification: {temporary}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[VIDEO DONE] {job['sequence']}: {output}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=2,
                        help="Physical GPU for serial metric extraction; default 2")
    parser.add_argument("--videos-only", action="store_true")
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--score-only", nargs="+", metavar="SEQUENCE",
                        help="Score named sequences without writing shared status or summaries")
    args = parser.parse_args()
    if sum(bool(x) for x in (args.videos_only, args.metrics_only, args.score_only)) > 1:
        parser.error("Choose at most one mode")
    jobs = load_jobs()
    for path in (METRIC_SCRIPT, EVALUATOR_ROOT / "pretrained_models/rgb_imagenet.pt",
                 PYTHON, FFMPEG, FFPROBE):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "logs").mkdir(exist_ok=True)
    if args.score_only:
        by_name = {job["sequence"]: job for job in jobs}
        if len(set(args.score_only)) != len(args.score_only) or set(args.score_only) - set(by_name):
            raise ValueError(f"Unknown or repeated sequence in {args.score_only}")
        for name in args.score_only:
            score_one(by_name[name], args.gpu)
        return 0
    if not args.videos_only:
        status_path = OUTPUT / "status.json"
        status = [{"sequence": job["sequence"], "state":
                   "complete" if case_result(job) is not None else "queued"} for job in jobs]
        save_json(status_path, status)
        for record, job in zip(status, jobs):
            if record["state"] == "complete":
                continue
            record.update(state="running", gpu=args.gpu,
                          started_at=datetime.now().isoformat(timespec="seconds"))
            save_json(status_path, status)
            try:
                score_one(job, args.gpu)
            except Exception as error:
                record.update(state="failed", error=repr(error))
                save_json(status_path, status)
                raise
            record.update(state="complete",
                          finished_at=datetime.now().isoformat(timespec="seconds"))
            save_json(status_path, status)
        summarize(jobs)
    if not args.metrics_only:
        manifest = []
        for job in jobs:
            encode_video(job)
            manifest.append({"sequence": job["sequence"], "class": job["class"],
                             "fps": job["fps"], "frames": len(job["ids"]),
                             "width": job["width"], "height": job["height"],
                             "frame_range": [job["ids"][0], job["ids"][-1]],
                             "video": str(video_path(job))})
        save_json(EVAL / "videos/video_manifest.json", {
            "source": str(EVAL / "test"), "codec": "H.264 CRF 18", "videos": manifest,
        })
    return 0


if __name__ == "__main__":
    sys.exit(main())
