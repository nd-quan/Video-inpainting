#!/usr/bin/env python3
"""Score native BG52 latent-TG frames, encode videos, and retain two PNGs per sequence."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import queue
import subprocess
import sys
from threading import Lock

from PIL import Image


BRUSHNET = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BRUSHNET / "Quan_test"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "original_res_exp"))
import add_we_l2_sfu_bg56 as we_l2  # noqa: E402
import postprocess_sfu_bg56_bvi_no_we as shared  # noqa: E402


EVAL = BRUSHNET / "experiments/eval_sfu_bg52_table_window_bvi_ckpt2000_latent_tg0p1"
DATASET = Path("/home/cilab/ndquan/vcm-rs/output/sfu_qp0_bg_table_window_test/bg52/test")
OUTPUT = EVAL / "metrics_full_reference_no_we"
WORKFLOW_STATUS = OUTPUT / "workflow_status.json"
SOURCE_MANIFEST = Path(
    "/home/cilab/ndquan/vcm-rs/output/postprocess_sfu_full_qp0_bg_multiqp/05_table_window/manifest.json"
)
GPUS = (1, 2)
EXPECTED_SEQUENCES = 14
EXPECTED_FRAMES = 1198


def configure_shared() -> None:
    shared.BRUSHNET = BRUSHNET
    shared.EVAL = EVAL
    shared.DATASET = DATASET
    shared.SOURCE_MANIFEST = SOURCE_MANIFEST
    shared.METRIC_SCRIPT = BRUSHNET / "examples/brushnet/proposed/DGAF_VSR_original/postprocess_eval.py"
    shared.EVALUATOR_ROOT = BRUSHNET.parent / "video-inpainting-evaluation"
    shared.OUTPUT = OUTPUT
    shared.video_path = video_path
    we_l2.EVAL = EVAL
    we_l2.OUTPUT = OUTPUT
    we_l2.CASE_CACHE = OUTPUT / "we_l2_cases"


def video_path(job: dict) -> Path:
    return EVAL / "videos" / job["class"] / (
        f"{job['sequence']}_bg52_ckpt2000_sharednoise_latent_tg.mp4"
    )


def load_jobs() -> list[dict]:
    records = json.loads((EVAL / "logs/status.json").read_text(encoding="utf-8"))
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))["cases"]
    source = {item["sequence"]: item for item in source if item["bg_qp"] == 52}
    if (len(records) != EXPECTED_SEQUENCES or len(source) != EXPECTED_SEQUENCES
            or any(item["state"] != "complete" for item in records)):
        raise ValueError("Expected 14 completed BG52 evaluation sequences")
    jobs = []
    for record in records:
        name, cls = record["sequence"], record["class"]
        meta = source[name]
        ids = list(range(record["frame_range"][0], record["frame_range"][1] + 1))
        if (meta["class"] != cls or meta["frames"] != record["frame_count"]
                or meta["first_frame"] != ids[0] or meta["last_frame"] != ids[-1]
                or len(ids) != meta["frames"]):
            raise ValueError(f"Manifest/status mismatch for {name}")
        pred_dir = EVAL / "test" / cls / name
        gt_dir = DATASET / "GT" / cls / name
        expected = {f"{frame_id:06d}.png" for frame_id in ids}
        for kind, folder in (("prediction", pred_dir), ("GT", gt_dir)):
            if {path.name for path in folder.glob("*.png")} != expected:
                raise ValueError(f"{kind} frame IDs differ for {name}")
        for frame_id in (ids[0], ids[-1]):
            filename = f"{frame_id:06d}.png"
            with Image.open(pred_dir / filename) as pred, Image.open(gt_dir / filename) as gt:
                if pred.size != gt.size or list(pred.size) != record["size_wh"]:
                    raise ValueError(f"Prediction/GT resolution differs for {name}: {filename}")
        jobs.append({
            "sequence": name, "class": cls, "ids": ids, "fps": meta["fps"],
            "width": record["size_wh"][0], "height": record["size_wh"][1],
            "pred_dir": pred_dir, "gt_dir": gt_dir,
        })
    if sum(len(job["ids"]) for job in jobs) != EXPECTED_FRAMES:
        raise ValueError("Expected 1198 BG52 frames")
    return jobs


def save_status(phase: str, cases: dict[str, str], **extra) -> None:
    shared.save_json(WORKFLOW_STATUS, {
        "phase": phase, "updated_at": datetime.now().isoformat(timespec="seconds"),
        "cases": dict(sorted(cases.items())), **extra,
    })


def score_cases(jobs: list[dict], cases: dict[str, str]) -> None:
    pending = queue.Queue()
    for job in sorted(jobs, key=lambda item: len(item["ids"]) * item["width"] * item["height"], reverse=True):
        if shared.case_result(job) is None:
            pending.put(job)
        else:
            cases[job["sequence"]] = "scored"
    save_status("image_metrics", cases)
    lock = Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            name = job["sequence"]
            with lock:
                cases[name] = f"scoring_gpu{gpu}"
                save_status("image_metrics", cases)
            shared.score_one(job, gpu)
            with lock:
                cases[name] = "scored"
                save_status("image_metrics", cases)

    with ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        for future in [pool.submit(worker, gpu) for gpu in GPUS]:
            future.result()
    if any(shared.case_result(job) is None for job in jobs):
        raise ValueError("Image metric results are incomplete")


def score_we_l2(jobs: list[dict], cases: dict[str, str]) -> None:
    records = [{
        "sequence": job["sequence"], "class": job["class"],
        "frame_count": len(job["ids"]),
        "frame_range": [job["ids"][0], job["ids"][-1]],
    } for job in jobs]
    we_l2.CASE_CACHE.mkdir(parents=True, exist_ok=True)
    scores = {record["sequence"]: we_l2.cached_score(record) for record in records}
    pending = [record for record in records if scores[record["sequence"]] is None]
    save_status("WE_L2", cases, pending=len(pending))
    if pending:
        with ProcessPoolExecutor(max_workers=min(4, len(pending))) as pool:
            futures = {pool.submit(we_l2.case_score, record): record["sequence"] for record in pending}
            for future in as_completed(futures):
                score = future.result()
                name = score["sequence"]
                scores[name] = score
                cases[name] = "WE_L2_scored"
                save_status("WE_L2", cases, pending=sum(value is None for value in scores.values()))
                print(f"[WE-L2 DONE] {name}: {score['WE_L2']:.8f}", flush=True)
    for record in records:
        name = record["sequence"]
        if scores[name] is None:
            raise ValueError(f"Missing WE-L2 for {name}")
        if "WE_L2" not in shared.case_result(next(job for job in jobs if job["sequence"] == name)):
            we_l2.update_case_files(record, scores[name])
        cases[name] = "metrics_complete"
    shared.summarize(jobs, method_label="BG52 BVI ckpt2000 shared noise + latent TG")
    save_status("metrics_complete", cases, summary=str(OUTPUT / "summary.csv"))


def encode_and_verify(jobs: list[dict], cases: dict[str, str]) -> None:
    def encode(job: dict) -> None:
        shared.encode_video(job)
        video = video_path(job)
        if not shared.valid_video(video, job):
            raise ValueError(f"Video metadata check failed: {video}")
        result = subprocess.run([
            str(shared.FFMPEG), "-nostdin", "-v", "error", "-xerror", "-i", str(video),
            "-f", "null", "-",
        ], capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f"Full video decode failed: {video}: {result.stderr[-1000:]}")
        print(f"[VIDEO VERIFIED] {job['sequence']}", flush=True)

    save_status("videos", cases)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(encode, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            future.result()
            cases[job["sequence"]] = "video_verified"
            save_status("videos", cases)
    manifest = [{
        "sequence": job["sequence"], "class": job["class"],
        "fps": job["fps"], "frames": len(job["ids"]),
        "width": job["width"], "height": job["height"],
        "frame_range": [job["ids"][0], job["ids"][-1]],
        "video": str(video_path(job)),
    } for job in jobs]
    shared.save_json(EVAL / "videos/video_manifest.json", {
        "source": str(EVAL / "test"), "codec": "H.264 CRF 18",
        "fps_source": str(SOURCE_MANIFEST), "videos": manifest,
    })


def retain_two_pngs(jobs: list[dict], cases: dict[str, str]) -> None:
    # Check every deliverable before removing any original generated frame.
    if not (OUTPUT / "summary.csv").is_file():
        raise ValueError("Metrics summary missing before PNG cleanup")
    for job in jobs:
        if shared.case_result(job) is None or "WE_L2" not in shared.case_result(job):
            raise ValueError(f"Metrics incomplete for {job['sequence']}")
        if not shared.valid_video(video_path(job), job):
            raise ValueError(f"Video invalid before cleanup: {job['sequence']}")
        expected = {f"{frame_id:06d}.png" for frame_id in job["ids"]}
        if {path.name for path in job["pred_dir"].glob("*.png")} != expected:
            raise ValueError(f"Generated PNGs changed before cleanup: {job['sequence']}")
    kept = []
    for job in jobs:
        ids = job["ids"]
        keep = {f"{ids[len(ids) // 3]:06d}.png", f"{ids[2 * len(ids) // 3]:06d}.png"}
        for path in job["pred_dir"].glob("*.png"):
            if path.name not in keep:
                path.unlink()
        if {path.name for path in job["pred_dir"].glob("*.png")} != keep:
            raise ValueError(f"Two-frame retention failed for {job['sequence']}")
        kept.append({
            "sequence": job["sequence"], "class": job["class"],
            "kept_pngs": sorted(keep), "removed_pngs": len(ids) - 2,
            "video": str(video_path(job)),
        })
        cases[job["sequence"]] = "complete"
        save_status("cleanup", cases)
    if len(list(EVAL.rglob("*.png"))) != 2 * len(jobs):
        raise ValueError("Unexpected PNGs remain elsewhere in the evaluation folder")
    shared.save_json(EVAL / "videos/retained_png_manifest.json", {
        "selection": "Frames at one-third and two-thirds of each sequence",
        "original_generated_frames": EXPECTED_FRAMES,
        "removed_pngs": EXPECTED_FRAMES - 2 * len(jobs),
        "retained_pngs": 2 * len(jobs), "sequences": kept,
    })
    save_status("complete", cases, summary=str(OUTPUT / "summary.csv"),
                video_manifest=str(EVAL / "videos/video_manifest.json"),
                retained_png_manifest=str(EVAL / "videos/retained_png_manifest.json"))


def main() -> int:
    configure_shared()
    for path in (SOURCE_MANIFEST, shared.METRIC_SCRIPT,
                 shared.EVALUATOR_ROOT / "pretrained_models/rgb_imagenet.pt",
                 shared.PYTHON, shared.FFMPEG, shared.FFPROBE):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "logs").mkdir(exist_ok=True)
    jobs = load_jobs()
    cases = {job["sequence"]: "queued" for job in jobs}
    try:
        score_cases(jobs, cases)
        score_we_l2(jobs, cases)
        encode_and_verify(jobs, cases)
        retain_two_pngs(jobs, cases)
    except Exception as error:
        save_status("failed", cases, error=repr(error))
        raise
    print(f"[COMPLETE] Metrics, 14 videos, and 28 retained PNGs: {EVAL}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
