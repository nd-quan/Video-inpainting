#!/usr/bin/env python
"""Visualize coarse, refined, and clean-teacher flow on one cached pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
PROPOSED = HERE.parent
V7 = PROPOSED / "STC_encoder_v7_raft_flow_distillation"
for directory in (str(PROPOSED), str(V7)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from raft_teacher_pair_data import RAFTTeacherFlowPairDataset
from STC_encoder_v8_raft_deformable.raft_flow_provider import FrozenV7RAFTFlowProvider
from flow_refinement.provider import FrozenRefinedFlowProvider


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--teacher_flow_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--branch", required=True, help="Class/sequence, e.g. Class_D/BasketballPass")
    parser.add_argument("--frame0", type=int, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--tile_size", type=int, default=384)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.tile_size < 128 or args.resolution % 8:
        parser.error("tile_size must be >= 128 and resolution divisible by 8")
    return args


def chw_to_hwc(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy().transpose(1, 2, 0)


def input_bgr(tensor: torch.Tensor) -> np.ndarray:
    rgb = ((chw_to_hwc(tensor) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def clean_bgr(path: Path, resolution: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB").resize((resolution, resolution),
                         resample=Image.Resampling.BILINEAR), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def flow_color(flow: np.ndarray, scale: float) -> np.ndarray:
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((angle * 90 / np.pi) % 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude / scale * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def heatmap(value: np.ndarray, scale: float, support: np.ndarray = None) -> np.ndarray:
    result = cv2.applyColorMap(np.clip(value / scale * 255, 0, 255).astype(np.uint8),
                               cv2.COLORMAP_TURBO)
    if support is not None:
        result[~support] = (65, 65, 65)
    return result


def gain_map(coarse_epe: np.ndarray, refined_epe: np.ndarray, scale: float,
             support: np.ndarray) -> np.ndarray:
    """Green means refinement helped; magenta means it hurt."""
    gain = np.clip((coarse_epe - refined_epe) / scale, -1.0, 1.0)
    result = np.full((*gain.shape, 3), 35, dtype=np.uint8)
    positive = gain > 0
    negative = gain < 0
    result[positive, 1] = (35 + 220 * gain[positive]).astype(np.uint8)
    result[negative, 0] = (35 - 220 * gain[negative]).astype(np.uint8)
    result[negative, 2] = (35 - 220 * gain[negative]).astype(np.uint8)
    result[~support] = (65, 65, 65)
    return result


def label(image: np.ndarray, title: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 30), (10, 18, 30), -1)
    cv2.putText(result, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .48,
                (255, 255, 255), 1, cv2.LINE_AA)
    return result


def montage(tiles, tile_size: int) -> np.ndarray:
    rendered = [label(cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA), title)
                for title, image in tiles]
    if len(rendered) % 4:
        raise ValueError("Montage needs a multiple of four tiles")
    return np.concatenate([np.concatenate(rendered[i:i + 4], axis=1)
                           for i in range(0, len(rendered), 4)], axis=0)


def edge_support(teacher: np.ndarray, valid: np.ndarray, threshold: float = 1.0):
    edge_pairs = []
    for axis in (1, 0):
        dt = np.diff(teacher, axis=axis)
        if axis == 1:
            pair_valid = valid[:, 1:] & valid[:, :-1]
        else:
            pair_valid = valid[1:, :] & valid[:-1, :]
        edge_pairs.append(pair_valid & (np.linalg.norm(dt, axis=2) >= threshold))
    return edge_pairs


def direction_metrics(flow: np.ndarray, teacher: np.ndarray, valid: np.ndarray,
                      bg: np.ndarray) -> dict:
    epe = np.linalg.norm(flow - teacher, axis=2)
    support = valid & bg
    edges = edge_support(teacher, valid)
    edge_values = []
    for axis, mask in zip((1, 0), edges):
        pair_epe = (.5 * (epe[:, 1:] + epe[:, :-1]) if axis == 1 else
                    .5 * (epe[1:, :] + epe[:-1, :]))
        edge_values.append(pair_epe[mask])
    selected_edges = np.concatenate(edge_values)
    return {
        "epe_px": float(epe[valid].mean()) if valid.any() else None,
        "bg_epe_px": float(epe[support].mean()) if support.any() else None,
        "boundary_epe_px": float(selected_edges.mean()) if selected_edges.size else None,
        "valid_pixels": int(valid.sum()),
        "bg_valid_pixels": int(support.sum()),
        "boundary_pairs": int(selected_edges.size),
    }


def warp(source: np.ndarray, backward: np.ndarray):
    h, w = backward.shape[:2]
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = xx + backward[..., 0], yy + backward[..., 1]
    valid = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    warped = cv2.remap(source, mx, my, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    return warped, valid, mx, my


def save_image(directory: Path, filename: str, image: np.ndarray):
    if not cv2.imwrite(str(directory / filename), image):
        raise OSError(directory / filename)


def main():
    args = parse_args()
    dataset = RAFTTeacherFlowPairDataset(args.dataset_root, args.teacher_flow_root,
                                         args.split, resolution=args.resolution,
                                         include_branches=[args.branch],
                                         validate_files=False)
    matches = [i for i, entry in enumerate(dataset.entries) if entry[1] == args.frame0]
    if len(matches) != 1:
        raise ValueError(f"Pair {args.branch}:{args.frame0}->{args.frame0+1} absent from {args.split}")
    sample = dataset[matches[0]]
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("The frozen RAFT provider requires CUDA")
    payload = torch.load(args.checkpoint, map_location="cpu")
    base = FrozenV7RAFTFlowProvider(payload["base_raft_component"], device=device,
                                    pair_batch_size=1, mixed_precision=True)
    checkpoint_name = base.raft_student_path.parent.name
    student_label = (f"Student {int(checkpoint_name.rsplit('-', 1)[-1])}"
                     if checkpoint_name.startswith("checkpoint-") else "Student")
    refiner = FrozenRefinedFlowProvider(base, args.checkpoint, pair_batch_size=1)
    frames = torch.stack((sample["degraded0"], sample["degraded1"]), dim=0)[None].to(device)
    with torch.inference_mode():
        coarse = base.predict_sequence(frames)
        coarse_f, coarse_b = coarse.forward[:, 0], coarse.backward[:, 0]
        refined_f, refined_b = refiner.refiner(frames[:, 0], frames[:, 1], coarse_f, coarse_b)
    flows = {
        "teacher_forward": chw_to_hwc(sample["teacher_forward"]),
        "teacher_backward": chw_to_hwc(sample["teacher_backward"]),
        "coarse_forward": chw_to_hwc(coarse_f[0]),
        "coarse_backward": chw_to_hwc(coarse_b[0]),
        "refined_forward": chw_to_hwc(refined_f[0]),
        "refined_backward": chw_to_hwc(refined_b[0]),
    }
    valid_f = sample["valid_forward"][0].numpy() >= .5
    valid_b = sample["valid_backward"][0].numpy() >= .5
    bg0 = sample["bg0"][0].numpy() >= .5
    bg1 = sample["bg1"][0].numpy() >= .5
    supports = {"forward": valid_f & bg0, "backward": valid_b & bg1}
    epe = {f"{kind}_{direction}": np.linalg.norm(
        flows[f"{kind}_{direction}"] - flows[f"teacher_{direction}"], axis=2)
        for kind in ("coarse", "refined") for direction in ("forward", "backward")}
    metrics = {
        "split": args.split, "branch": args.branch, "frame0": args.frame0,
        "frame1": args.frame0 + 1, "refiner_checkpoint": str(args.checkpoint.resolve()),
        "refiner_step": refiner.step, "base_raft_component": str(base.raft_student_path),
        "coarse": {}, "refined": {},
    }
    for kind in ("coarse", "refined"):
        for direction, valid, bg in (("forward", valid_f, bg0), ("backward", valid_b, bg1)):
            metrics[kind][direction] = direction_metrics(
                flows[f"{kind}_{direction}"], flows[f"teacher_{direction}"], valid, bg)
    for key in ("epe_px", "bg_epe_px", "boundary_epe_px"):
        metrics[f"bidirectional_{key}"] = {
            kind: float(np.mean([metrics[kind][d][key] for d in ("forward", "backward")]))
            for kind in ("coarse", "refined")
        }
    root = Path(args.dataset_root)
    clean_root = root / args.split / "GT" / Path(args.branch)
    input0 = input_bgr(sample["degraded0"])
    input1 = input_bgr(sample["degraded1"])
    clean0 = clean_bgr(clean_root / f"{args.frame0:06d}.png", args.resolution)
    clean1 = clean_bgr(clean_root / f"{args.frame0+1:06d}.png", args.resolution)
    all_magnitudes = [np.linalg.norm(value, axis=2).ravel() for value in flows.values()]
    flow_scale = max(float(np.percentile(np.concatenate(all_magnitudes), 99)), 1e-6)
    selected_epe = np.concatenate([epe[f"{k}_{d}"][supports[d]]
        for k in ("coarse", "refined") for d in ("forward", "backward")])
    epe_scale = max(float(np.percentile(selected_epe, 99)), 1e-6)
    selected_gain = np.concatenate([np.abs(epe[f"coarse_{d}"] - epe[f"refined_{d}"])[supports[d]]
                                    for d in ("forward", "backward")])
    gain_scale = max(float(np.percentile(selected_gain, 99)), 1e-6)
    delta_f = np.linalg.norm(flows["refined_forward"] - flows["coarse_forward"], axis=2)
    delta_b = np.linalg.norm(flows["refined_backward"] - flows["coarse_backward"], axis=2)
    residual_scale = max(float(np.percentile(np.concatenate((delta_f.ravel(), delta_b.ravel())), 99)), 1e-6)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    images = {}
    for direction in ("forward", "backward"):
        for kind in ("teacher", "coarse", "refined"):
            images[f"{kind}_{direction}_flow.png"] = flow_color(flows[f"{kind}_{direction}"], flow_scale)
        for kind in ("coarse", "refined"):
            images[f"{kind}_{direction}_epe.png"] = heatmap(epe[f"{kind}_{direction}"],
                                                              epe_scale, supports[direction])
        images[f"{direction}_gain.png"] = gain_map(epe[f"coarse_{direction}"],
                epe[f"refined_{direction}"], gain_scale, supports[direction])
        images[f"{direction}_bg_support.png"] = cv2.cvtColor(
            supports[direction].astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    images["forward_correction_magnitude.png"] = heatmap(delta_f, residual_scale)
    images["backward_correction_magnitude.png"] = heatmap(delta_b, residual_scale)
    warped = {}
    warp_errors = {}
    warp_supports = {}
    for kind in ("coarse", "refined"):
        warped[kind], in_bounds, mx, my = warp(input0, flows[f"{kind}_backward"])
        source_bg = cv2.remap(bg0.astype(np.uint8), mx, my, cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT).astype(bool)
        warp_supports[kind] = valid_b & bg1 & source_bg & in_bounds
        warp_errors[kind] = np.abs(warped[kind].astype(np.float32) - input1).mean(axis=2)
    common_warp_support = warp_supports["coarse"] & warp_supports["refined"]
    for kind in ("coarse", "refined"):
        metrics[kind]["warp_mae_bg_0_255"] = (float(warp_errors[kind][common_warp_support].mean())
                                            if common_warp_support.any() else None)
        images[f"{kind}_warped_t_to_t1.png"] = warped[kind]
    warp_scale = max(float(np.percentile(np.concatenate([
        warp_errors[k][common_warp_support] for k in ("coarse", "refined")]), 99))
        if common_warp_support.any() else 1.0, 1e-6)
    for kind in ("coarse", "refined"):
        images[f"{kind}_warp_error.png"] = heatmap(warp_errors[kind], warp_scale,
                                                    common_warp_support)
    metrics["display_scales_px"] = {"flow_magnitude_p99": flow_scale,
                                     "teacher_epe_p99": epe_scale,
                                     "absolute_gain_p99": gain_scale,
                                     "correction_p99": residual_scale}
    metrics["warp_support_pixels"] = int(common_warp_support.sum())
    tiles = [
        ("Degraded t", input0), ("Degraded t+1", input1),
        ("Clean t (reference)", clean0), ("Clean t+1 (reference)", clean1),
        ("Teacher forward", images["teacher_forward_flow.png"]),
        (f"{student_label} forward", images["coarse_forward_flow.png"]),
        ("Refined forward", images["refined_forward_flow.png"]),
        ("Forward correction |delta|", images["forward_correction_magnitude.png"]),
        ("Teacher backward", images["teacher_backward_flow.png"]),
        (f"{student_label} backward", images["coarse_backward_flow.png"]),
        ("Refined backward", images["refined_backward_flow.png"]),
        ("Backward correction |delta|", images["backward_correction_magnitude.png"]),
        (f"{student_label} forward EPE", images["coarse_forward_epe.png"]),
        ("Refined forward EPE", images["refined_forward_epe.png"]),
        (f"{student_label} backward EPE", images["coarse_backward_epe.png"]),
        ("Refined backward EPE", images["refined_backward_epe.png"]),
        ("Forward gain (green=better)", images["forward_gain.png"]),
        ("Backward gain (green=better)", images["backward_gain.png"]),
        ("Forward BG support", images["forward_bg_support.png"]),
        ("Backward BG support", images["backward_bg_support.png"]),
        ("Warped t, student backward", images["coarse_warped_t_to_t1.png"]),
        ("Warped t, refined backward", images["refined_warped_t_to_t1.png"]),
        ("Student warp error", images["coarse_warp_error.png"]),
        ("Refined warp error", images["refined_warp_error.png"]),
    ]
    images["montage.png"] = montage(tiles, args.tile_size)
    for filename, image in images.items():
        save_image(save_dir, filename, image)
    np.savez_compressed(save_dir / "flows.npz", **flows,
                        valid_forward=valid_f, valid_backward=valid_b,
                        bg0=bg0, bg1=bg1)
    (save_dir / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(json.dumps({"montage": str(save_dir / "montage.png"),
                      "metrics": metrics["bidirectional_bg_epe_px"]}, indent=2))


if __name__ == "__main__":
    main()
