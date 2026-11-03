#!/usr/bin/env python3
"""Snapshot completed common frames and compare partial x1/x3/x10 results on CPU."""
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import cv2
import numpy as np
from PIL import Image
import torch
import lpips
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import multiscale_structural_similarity_index_measure as msssim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Quan_test"))
from evaluate_temporal_metrics import warping_error

BASE = ROOT / "experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_cge_then_tg_equivclean_C25-40_T25-40_cge1em4_tg0p1_v7student_512"
FULL = ROOT / "experiments/EVAL_FOR_FINAL/ablation_bg52_cge_x3_x10_tg0p1_C25-40_T25-40_FULL"


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def tensor(image):
    return torch.from_numpy(image.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    moment = datetime.now().astimezone()
    output = FULL / "comparison" / ("progress_" + moment.strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True)
    config = json.loads((FULL / "cge_x3/run_config.json").read_text())
    dataset = Path(config["dataset"])
    runs = {"cge_x1": BASE, "cge_x3": FULL / "cge_x3", "cge_x10": FULL / "cge_x10"}
    progress, shared = [], {}
    for source in sorted((dataset / "input").glob("Class_*/*")):
        relative = source.relative_to(dataset / "input")
        expected = {p.name for p in source.glob("*.png")}
        sets = {label: {p.name for p in (folder / "test" / relative).glob("*.png")}
                for label, folder in runs.items()}
        common = sorted(expected & sets["cge_x3"] & sets["cge_x10"])
        # Ignore a file still being written while taking the progress snapshot.
        valid = []
        for name in common:
            try:
                for label in ("cge_x3", "cge_x10"):
                    with Image.open(runs[label] / "test" / relative / name) as image:
                        image.verify()
                valid.append(name)
            except (OSError, ValueError):
                pass
        if valid:
            shared[relative] = valid
        progress.append({"sequence": source.name, "expected": len(expected),
                         "x3_saved": len(sets["cge_x3"] & expected), "x10_saved": len(sets["cge_x10"] & expected),
                         "common_completed": len(valid)})
    manifest = {str(relative): names for relative, names in shared.items()}
    (output / "matched_frames.json").write_text(json.dumps(manifest, indent=2) + "\n")
    save_csv(output / "progress.csv", progress)
    if not shared:
        raise RuntimeError("No completed common frames")
    common_count = sum(map(len, shared.values()))
    include_x1 = all((BASE / "test" / rel / name).is_file() for rel, names in shared.items() for name in names)
    if not include_x1:
        del runs["cge_x1"]
    perceptual = lpips.LPIPS(net="alex").cpu().eval()
    per_frame, per_pair, direct_rows = [], [], []
    with torch.inference_mode():
        for relative, names in shared.items():
            previous = {}
            previous_id = None
            for name in names:
                with Image.open(dataset / "GT" / relative / name) as im:
                    gt = np.array(im.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR))
                with Image.open(dataset / "mask" / relative / name) as im:
                    background = np.array(im.convert("L").resize((512, 512), Image.Resampling.NEAREST)) <= 127
                gt_tensor = tensor(gt)
                arrays = {}
                for label, folder in runs.items():
                    with Image.open(folder / "test" / relative / name) as im:
                        image = np.array(im.convert("RGB"))
                    arrays[label] = image
                    pred = tensor(image)
                    squared = (image.astype(np.float32) - gt.astype(np.float32))**2
                    mse = float(squared.mean())
                    bg_mse = float(squared[background].mean())
                    per_frame.append({"run": label, "sequence": relative.name, "frame": name,
                                      "PSNR": float(10*np.log10(255**2/max(mse,1e-12))),
                                      "BG_PSNR": float(10*np.log10(255**2/max(bg_mse,1e-12))),
                                      "SSIM": float(ssim(pred, gt_tensor, data_range=1)),
                                      "MS_SSIM": float(msssim(pred, gt_tensor, data_range=1)),
                                      "LPIPS": float(perceptual(pred*2-1, gt_tensor*2-1))})
                    if previous_id is not None and int(Path(name).stem) == previous_id + 1:
                        l1, l2 = warping_error(previous[label].astype(np.float32)/255, image.astype(np.float32)/255)
                        per_pair.append({"run": label, "sequence": relative.name, "next_frame": name,
                                         "WE": l1, "WE_L2": l2})
                    previous[label] = image
                previous_id = int(Path(name).stem)
                difference = np.abs(arrays["cge_x3"].astype(np.float32) - arrays["cge_x10"].astype(np.float32))
                direct_rows.append({"sequence": relative.name, "frame": name, "MAE": float(difference.mean()),
                                    "BG_MAE": float(difference[background].mean()),
                                    "changed_pixels_pct": float(100*np.any(difference>0,axis=2).mean())})
            print("Completed matched sequence",relative.name,len(names),flush=True)
    summaries = []
    for label in runs:
        for sequence in [rel.name for rel in shared] + ["ALL"]:
            frames = [r for r in per_frame if r["run"] == label and (sequence == "ALL" or r["sequence"] == sequence)]
            pairs = [r for r in per_pair if r["run"] == label and (sequence == "ALL" or r["sequence"] == sequence)]
            row = {"run": label, "sequence": sequence, "frames": len(frames), "pairs": len(pairs),
                   **{key: float(np.mean([r[key] for r in frames])) for key in ("PSNR","BG_PSNR","SSIM","MS_SSIM","LPIPS")},
                   **{key: float(np.mean([r[key] for r in pairs])) if pairs else None for key in ("WE","WE_L2")}}
            summaries.append(row)
    timing = {}
    for label in ("cge_x3", "cge_x10"):
        records = []
        for relative, names in shared.items():
            path = runs[label] / "test" / relative / "generation_timing.jsonl"
            if path.is_file():
                latest = {r["clip_index"]:r for r in map(json.loads,path.read_text().splitlines())}
                for record in latest.values():
                    if set(record["frame_ids"]).issubset({int(Path(n).stem) for n in names}):
                        records.append(record)
        seconds = sum(r["generation_seconds"] for r in records)
        frame_count = sum(r["batch_frames"] for r in records)
        timing[label] = {"completed_clips":len(records),"timed_clip_frames":frame_count,
                         "generation_seconds":seconds,"seconds_per_generated_frame":seconds/frame_count if frame_count else None,
                         "includes_reused_pilot_clip":True}
    report = {"snapshot_at":moment.isoformat(),"expected_frames_per_variant":1198,"progress":progress,
              "common_compared_frames":common_count,"matched_frames":manifest,"metrics":summaries,
              "timing":timing,"direct_difference_mean":{key:float(np.mean([r[key] for r in direct_rows])) for key in ("MAE","BG_MAE","changed_pixels_pct")},
              "scope":"Partial results only. Exactly the same completed frames for all variants. Image metrics frame-weighted; temporal metrics pair-weighted. WE uses backward Farneback on generated RGB [0,1], not the earlier fixed-V7-flow diagnostic metric.",
              "gt_preprocessing":"PIL RGB bilinear resize to 512x512"}
    (output / "summary.json").write_text(json.dumps(report,indent=2)+'\n')
    save_csv(output / "metrics.csv",summaries)
    save_csv(output / "per_frame.csv",per_frame)
    save_csv(output / "per_pair.csv",per_pair)
    save_csv(output / "direct_x3_vs_x10.csv",direct_rows)
    (FULL / "comparison/latest_progress.json").write_text(json.dumps({"report":str(output),"snapshot_at":moment.isoformat()},indent=2)+'\n')
    print("REPORT",output)
    print("COMMON",common_count,"TIMING",json.dumps(timing))
    print("AGGREGATE",json.dumps([r for r in summaries if r['sequence']=='ALL'],indent=2))


if __name__ == "__main__":
    main()
