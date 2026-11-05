#!/usr/bin/env python3
"""Score BG48 predictions (excluding WE) and encode all four test videos."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

from PIL import Image

import add_we_l2_sfu_bg56 as we_l2
import postprocess_sfu_bg56_bvi_no_we as shared


EVAL = shared.BRUSHNET / "experiments/eval_sfu_bg48_table_window_bvi_ckpt2000_rgb_tg"
DATASET = Path("/home/cilab/ndquan/vcm-rs/output/sfu_qp0_bg_table_window_test/bg48/test")
OUTPUT = EVAL / "metrics_full_reference_no_we"
WORKFLOW_STATUS = OUTPUT / "workflow_status.json"
SEQUENCES = {
    "Traffic", "BQTerrace", "BasketballDrive", "Cactus", "Kimono", "ParkScene",
    "BQMall", "BasketballDrill", "PartyScene", "RaceHorsesC", "BQSquare",
    "BasketballPass", "BlowingBubbles", "RaceHorsesD",
}
EXPECTED_FRAMES = 1198
METHOD_LABEL = "BG48 BVI ckpt2000 SN+TG"


def configure_shared() -> None:
    shared.EVAL = EVAL
    shared.DATASET = DATASET
    shared.OUTPUT = OUTPUT
    shared.video_path = video_path
    we_l2.EVAL = EVAL
    we_l2.OUTPUT = OUTPUT
    we_l2.CASE_CACHE = OUTPUT / "we_l2_cases"


def video_path(job: dict) -> Path:
    return EVAL / "videos" / job["class"] / (
        f"{job['sequence']}_bg48_ckpt2000_sharednoise_rgb_tg.mp4"
    )


def source_metadata() -> dict[str, dict]:
    rows = json.loads(shared.SOURCE_MANIFEST.read_text(encoding="utf-8"))["cases"]
    groups = {}
    for row in rows:
        if row["sequence"] in SEQUENCES:
            groups.setdefault(row["sequence"], []).append(row)
    if set(groups) != SEQUENCES:
        raise ValueError("Source manifest does not contain all BG48 sequence names")
    result = {}
    for name, entries in groups.items():
        signatures = {
            (row["class"], row["frames"], row["first_frame"],
             row["last_frame"], row["fps"])
            for row in entries
        }
        if len(signatures) != 1:
            raise ValueError(f"Source frame range or FPS differs across QPs for {name}")
        result[name] = entries[0]
    return result


def load_jobs() -> tuple[list[dict], dict[str, str]]:
    records = json.loads((EVAL / "logs/status.json").read_text(encoding="utf-8"))
    if len(records) != len(SEQUENCES) or {row["sequence"] for row in records} != SEQUENCES:
        raise ValueError(f"Expected {len(SEQUENCES)} BG48 evaluation records")
    source = source_metadata()
    jobs, states = [], {}
    for record in records:
        name, cls = record["sequence"], record["class"]
        metadata = source[name]
        if (cls != metadata["class"] or record["frame_count"] != metadata["frames"]
                or record["frame_range"] != [metadata["first_frame"], metadata["last_frame"]]):
            raise ValueError(f"Source/status mismatch for {name}")
        ids = list(range(record["frame_range"][0], record["frame_range"][1] + 1))
        expected = {f"{frame_id:06d}.png" for frame_id in ids}
        pred_dir = EVAL / "test" / cls / name
        gt_dir = DATASET / "GT" / cls / name
        if {path.name for path in gt_dir.glob("*.png")} != expected:
            raise ValueError(f"Ground-truth frame IDs differ for {name}")
        if record["state"] == "complete":
            if {path.name for path in pred_dir.glob("*.png")} != expected:
                raise ValueError(f"Prediction frame IDs differ for completed {name}")
            for frame_id in (ids[0], ids[-1]):
                with Image.open(pred_dir / f"{frame_id:06d}.png") as pred, Image.open(
                    gt_dir / f"{frame_id:06d}.png"
                ) as gt:
                    if pred.size != gt.size or list(pred.size) != record["size_wh"]:
                        raise ValueError(f"Prediction/GT resolution differs for {name}")
        elif record["state"] not in {"running", "queued"}:
            raise RuntimeError(f"Evaluation stopped for {name}: {record['state']}")
        jobs.append({
            "sequence": name, "class": cls, "ids": ids,
            "fps": metadata["fps"], "width": record["size_wh"][0],
            "height": record["size_wh"][1], "pred_dir": pred_dir, "gt_dir": gt_dir,
        })
        states[name] = record["state"]
    if sum(len(job["ids"]) for job in jobs) != EXPECTED_FRAMES:
        raise ValueError(f"Expected {EXPECTED_FRAMES} BG48 frames")
    return jobs, states


def save_workflow(state: str, cases: dict, **extra) -> None:
    shared.save_json(WORKFLOW_STATUS, {
        "state": state, "updated_at": datetime.now().isoformat(timespec="seconds"),
        "cases": cases, **extra,
    })


def process_ready_job(job: dict, gpu: int, cases: dict) -> None:
    name = job["sequence"]
    if shared.case_result(job) is None:
        cases[name] = "scoring"
        save_workflow("running", cases)
        shared.score_one(job, gpu)
    record = {"sequence": name, "class": job["class"],
              "frame_count": len(job["ids"]),
              "frame_range": [job["ids"][0], job["ids"][-1]]}
    score = we_l2.cached_score(record)
    if score is None:
        cases[name] = "computing_WE_L2"
        save_workflow("running", cases)
        score = we_l2.case_score(record)
    if "WE_L2" not in shared.case_result(job):
        we_l2.update_case_files(record, score)
    cases[name] = "encoding_video"
    save_workflow("running", cases)
    shared.encode_video(job)
    cases[name] = "complete"
    save_workflow("running", cases)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1,
                        help="GPU for PSNR/SSIM/LPIPS/I3D metrics; default 1")
    parser.add_argument("--ready-only", action="store_true",
                        help="Process complete sequences and exit without waiting")
    args = parser.parse_args()
    configure_shared()
    for path in (shared.SOURCE_MANIFEST, shared.METRIC_SCRIPT,
                 shared.EVALUATOR_ROOT / "pretrained_models/rgb_imagenet.pt",
                 shared.PYTHON, shared.FFMPEG, shared.FFPROBE):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "logs").mkdir(exist_ok=True)
    we_l2.CASE_CACHE.mkdir(exist_ok=True)
    cases = {name: "waiting_for_evaluation" for name in SEQUENCES}
    try:
        while True:
            jobs, states = load_jobs()
            for job in jobs:
                name = job["sequence"]
                if states[name] == "complete" and cases[name] != "complete":
                    process_ready_job(job, args.gpu, cases)
            if all(states[name] == "complete" for name in SEQUENCES):
                break
            if args.ready_only:
                save_workflow("waiting_for_evaluation", cases)
                return 0
            save_workflow("waiting_for_evaluation", cases)
            time.sleep(30)
        shared.summarize(jobs, method_label=METHOD_LABEL)
        manifest = []
        for job in jobs:
            path = video_path(job)
            if not shared.valid_video(path, job):
                raise RuntimeError(f"Video validation failed for {job['sequence']}: {path}")
            manifest.append({
                "sequence": job["sequence"], "class": job["class"],
                "fps": job["fps"], "frames": len(job["ids"]),
                "width": job["width"], "height": job["height"],
                "frame_range": [job["ids"][0], job["ids"][-1]],
                "video": str(path),
            })
        shared.save_json(EVAL / "videos/video_manifest.json", {
            "source": str(EVAL / "test"), "codec": "H.264 CRF 18",
            "fps_source": str(shared.SOURCE_MANIFEST), "videos": manifest,
        })
        save_workflow("complete", cases, frames=EXPECTED_FRAMES,
                      summary=str(OUTPUT / "summary.csv"),
                      video_manifest=str(EVAL / "videos/video_manifest.json"))
        print(f"BG48 metrics and videos complete: {OUTPUT}", flush=True)
        return 0
    except Exception as exc:
        save_workflow("failed", cases, error=repr(exc))
        raise


if __name__ == "__main__":
    sys.exit(main())
