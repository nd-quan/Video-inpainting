#!/usr/bin/env python3
"""Detailed image/temporal comparison of completed four-frame CGE ablations."""
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples/brushnet"))

import numpy as np
from PIL import Image
import torch
import lpips
from torchmetrics.functional import structural_similarity_index_measure, multiscale_structural_similarity_index_measure
from diffusers.schedulers.scheduling_ddim_temporal import bg_temporal_loss

BASE = ROOT / "experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_cge_then_tg_equivclean_C25-40_T25-40_cge1em4_tg0p1_v7student_512"
ABLATION = ROOT / "experiments/EVAL_FOR_FINAL/ablation_bg52_cge_x3_x10_tg0p1_C25-40_T25-40_Traffic4"


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ABLATION / "comparison"
    dataset = Path(json.loads((BASE / "run_config.json").read_text())["dataset"])
    names = ["{:06d}.png".format(i) for i in range(117, 121)]
    paths = {"temporal_only": BASE / "diagnostics/temporal_control",
             "cge_x1": BASE, "cge_x3": ABLATION / "cge_x3", "cge_x10": ABLATION / "cge_x10"}
    arrays = {}
    for label, path in paths.items():
        arrays[label] = np.stack([np.asarray(Image.open(path / "test/Class_A/Traffic" / n).convert("RGB")) for n in names])
    arrays["gt"] = np.stack([np.asarray(Image.open(dataset / "GT/Class_A/Traffic" / n).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)) for n in names])
    masks = np.stack([np.asarray(Image.open(dataset / "mask/Class_A/Traffic" / n).convert("L").resize((512, 512), Image.Resampling.NEAREST)) > 127 for n in names])
    tensors = {label: torch.from_numpy(a.copy()).permute(0, 3, 1, 2).float().to(device) / 255
               for label, a in arrays.items()}
    state = torch.load(BASE / "diagnostics/probe/step25_state.pt", map_location="cpu")
    flow, bg = state["flow"].to(device), state["stable_bg"].to(device)
    perceptual = lpips.LPIPS(net="alex").to(device).eval()
    per_frame, results = [], []
    with torch.inference_mode():
        for label in paths:
            rows = []
            for i, name in enumerate(names):
                pred, gt = tensors[label][i:i+1], tensors["gt"][i:i+1]
                rows.append({"run": label, "frame": name,
                             "ssim_gt": float(structural_similarity_index_measure(pred, gt, data_range=1)),
                             "ms_ssim_gt": float(multiscale_structural_similarity_index_measure(pred, gt, data_range=1)),
                             "lpips_alex_gt": float(perceptual(pred * 2 - 1, gt * 2 - 1))})
            per_frame.extend(rows)
            result = {"run": label, **{key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key not in ("run", "frame")},
                      "fixed_v7_background_warp_mse": float(bg_temporal_loss(tensors[label], flow, bg, detach_previous=False, loss_type="l2"))}
            results.append(result)
            print(json.dumps(result), flush=True)
        direct_rows = []
        for i, name in enumerate(names):
            a, b = arrays["cge_x3"][i].astype(np.float32), arrays["cge_x10"][i].astype(np.float32)
            delta = np.abs(a - b)
            mean_delta = delta.mean(2)
            changed = np.any(delta > 0, axis=2)
            gt = arrays["gt"][i].astype(np.float32)
            mse3, mse10 = np.mean((a-gt)**2), np.mean((b-gt)**2)
            direct_rows.append({"frame": name, "mae": float(delta.mean()), "background_mae": float(delta[~masks[i]].mean()),
                                "roi_mae": float(delta[masks[i]].mean()), "rmse": float(np.sqrt(np.mean((a-b)**2))),
                                "changed_pixels_pct": float(changed.mean()*100),
                                "mean_rgb_delta_above_1_pct": float((mean_delta>1).mean()*100),
                                "mean_rgb_delta_above_3_pct": float((mean_delta>3).mean()*100),
                                "mean_rgb_delta_above_5_pct": float((mean_delta>5).mean()*100),
                                "ssim_between": float(structural_similarity_index_measure(tensors["cge_x3"][i:i+1], tensors["cge_x10"][i:i+1], data_range=1)),
                                "lpips_between": float(perceptual(tensors["cge_x3"][i:i+1]*2-1, tensors["cge_x10"][i:i+1]*2-1)),
                                "mse_reduction_x10_vs_x3_pct": float((1-mse10/mse3)*100)})
        direct = {key: float(np.mean([row[key] for row in direct_rows])) for key in direct_rows[0] if key != "frame"}
        gt_warp = float(bg_temporal_loss(tensors["gt"], flow, bg, detach_previous=False, loss_type="l2"))
    report = {"scope": "Only Traffic 000117..000120, four frames and three temporal pairs, matched settings except CGE scale",
              "definitions": {"ssim_gt": "torchmetrics RGB [0,1] SSIM, data_range=1, frame mean",
                              "ms_ssim_gt": "torchmetrics default five-scale MS-SSIM, data_range=1, frame mean",
                              "lpips_alex_gt": "LPIPS AlexNet, RGB [-1,1], frame mean; lower is better",
                              "fixed_v7_background_warp_mse": "RGB [0,1] adjacent-frame masked MSE with same V7 backward flow and stable BG from source inputs for all variants; three pairs. Not the separate Farneback WE metric."},
              "metrics": results, "direct_x3_vs_x10_mean": direct,
              "direct_x3_vs_x10_per_frame": direct_rows, "per_frame_gt_metrics": per_frame,
              "gt_fixed_v7_background_warp_mse": gt_warp}
    (out / "x3_vs_x10_detailed.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / "perceptual_temporal_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    print("DIRECT", json.dumps(direct, indent=2))
    print("GT fixed-flow background warp MSE", gt_warp)


if __name__ == "__main__":
    main()
