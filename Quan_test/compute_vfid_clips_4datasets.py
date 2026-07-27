#!/usr/bin/env python3

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch


DEFAULT_EVALUATOR_ROOT = Path(
    "/media/ssd1/ndquan/videoInpainting/code/video-inpainting-evaluation"
)
DEFAULT_STAGE_ROOT = Path(
    "/media/ssd1/ndquan/videoInpainting/code/BrushNet/Quan_test/results/"
    "vfid_eval_4datasets_224"
)
DEFAULT_CSV_PATH = Path(
    "/media/ssd1/ndquan/videoInpainting/code/BrushNet/Quan_test/results/metric/"
    "vfid_clips_4datasets_3cases.csv"
)

CASES = (
    "fixedBG_nulltext",
    "nulltext_modelBase",
    "sharedNoise_fixedBG_095_temporal_v0",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute 10-frame I3D VFID-Clips for the staged VCM datasets."
    )
    parser.add_argument("--evaluator_root", type=Path, default=DEFAULT_EVALUATOR_ROOT)
    parser.add_argument("--stage_root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_CSV_PATH)
    return parser.parse_args()


def frechet_distance_low_rank(features_a, features_b):
    """Exact empirical Frechet distance without a feature_dim x feature_dim sqrtm."""
    a = np.asarray(features_a, dtype=np.float64)
    b = np.asarray(features_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"Incompatible feature shapes: {a.shape} and {b.shape}")
    if a.shape[0] < 2 or b.shape[0] < 2:
        raise ValueError("At least two clips per distribution are required.")

    mean_diff = np.square(a.mean(axis=0) - b.mean(axis=0)).sum()
    a_centered = a - a.mean(axis=0, keepdims=True)
    b_centered = b - b.mean(axis=0, keepdims=True)
    trace_a = np.square(a_centered).sum() / (a.shape[0] - 1)
    trace_b = np.square(b_centered).sum() / (b.shape[0] - 1)

    cross = a_centered @ b_centered.T
    cross /= np.sqrt((a.shape[0] - 1) * (b.shape[0] - 1))
    trace_covmean = np.linalg.svd(cross, compute_uv=False).sum()
    distance = mean_diff + trace_a + trace_b - 2.0 * trace_covmean
    return float(max(distance, 0.0))


def main():
    args = parse_args()
    sys.path.insert(0, str(args.evaluator_root))

    from src.common_util.image import get_comp_frame
    from src.common_util.misc import get_video_names_and_frame_counts
    from src.fid.util import extract_video_clip_features
    from src.models.i3d.pytorch_i3d import InceptionI3d

    gt_root = args.stage_root / "gt"
    pred_root = args.stage_root / "pred"
    feature_root = args.stage_root / "raw_results"
    feature_root.mkdir(parents=True, exist_ok=True)

    video_names, frame_counts = get_video_names_and_frame_counts(str(gt_root), None)
    expected_names = ["BasketballPass", "ParkScene", "PartyScene", "Traffic"]
    if video_names != expected_names:
        raise ValueError(f"Unexpected staged videos: {video_names}")

    clip_counts = [max(1, count - 10 + 1) for count in frame_counts]
    gt_feature_path = args.stage_root / "eval_features" / "vfid_clips.npy"
    gt_features = np.load(gt_feature_path)
    if gt_features.shape[0] != sum(clip_counts):
        raise ValueError(
            f"GT feature count mismatch: {gt_features.shape[0]} vs {sum(clip_counts)}"
        )

    checkpoint = args.evaluator_root / "pretrained_models" / "rgb_imagenet.pt"
    model = InceptionI3d(400, in_channels=3)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.cuda().eval()
    torch.set_grad_enabled(False)

    rows = []
    for case_name in CASES:
        case_root = pred_root / case_name

        def get_comp_frame_wrapper(video_name, frame_index):
            return get_comp_frame(
                str(gt_root), str(case_root), video_name, frame_index
            )

        print(f"Extracting prediction features: {case_name}", flush=True)
        pred_features = extract_video_clip_features(
            model,
            get_comp_frame_wrapper,
            video_names,
            frame_counts,
        )
        np.save(feature_root / f"{case_name}_vfid_clips.npy", pred_features)
        if pred_features.shape != gt_features.shape:
            raise ValueError(
                f"Feature shape mismatch for {case_name}: "
                f"{pred_features.shape} vs {gt_features.shape}"
            )

        start = 0
        for video_name, frame_count, clip_count in zip(
            video_names, frame_counts, clip_counts
        ):
            end = start + clip_count
            score = frechet_distance_low_rank(
                gt_features[start:end], pred_features[start:end]
            )
            rows.append(
                {
                    "dataset": video_name,
                    "case": case_name,
                    "frames": frame_count,
                    "clips": clip_count,
                    "clip_length": 10,
                    "resize": "224x224",
                    "vfid_clips": score,
                }
            )
            start = end

        rows.append(
            {
                "dataset": "ALL",
                "case": case_name,
                "frames": sum(frame_counts),
                "clips": sum(clip_counts),
                "clip_length": 10,
                "resize": "224x224",
                "vfid_clips": frechet_distance_low_rank(
                    gt_features, pred_features
                ),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved VFID-Clips results to: {args.output_csv}")
    for row in rows:
        if row["dataset"] == "ALL":
            print(f"{row['case']}: {row['vfid_clips']:.6f}")


if __name__ == "__main__":
    main()
