#!/usr/bin/env python3
"""Evaluate V7 RAFT student on flat test_1 against its matched clean 512 teacher."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


BRUSHNET = Path(__file__).resolve().parents[4]
ROOT = BRUSHNET / "examples/brushnet"
PROPAINTER = BRUSHNET.parents[1] / "pretrained/ProPainter"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROPAINTER))
import visualize_raft_student_flow as vis  # noqa: E402
from core.vcm_flow_utils import directional_fb_confidence  # noqa: E402


DEFAULT_CHECKPOINT = BRUSHNET / "experiments/train_v7_raft_student_flow/checkpoint-0004750"
DEFAULT_CACHE = BRUSHNET / "experiments/eval_temporal_guidance_student_vs_teacher_latent_s0p1_20260929/teacher_cache"
DEFAULT_DATASET = ROOT / "dataset/test_1"
REFERENCE = BRUSHNET / "experiments/eval_temporal_guidance_rgb_vs_latent_student_s0p1_20260918"
LABELS = ("BasketballPass", "ParkScene", "PartyScene", "RaceHorses", "Traffic")
DIRECTIONS = ("forward", "backward")


def rgb(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (512, 512):
            raise ValueError(f"Expected 512x512 source RGB: {path}: {image.size}")
        array = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).div_(127.5).sub_(1.0)


def bg_mask(path):
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (512, 512):
            raise ValueError(f"Expected 512x512 source mask: {path}: {image.size}")
        roi = np.asarray(image, dtype=np.uint8) >= 128
    return torch.from_numpy(np.ascontiguousarray((~roi).astype(np.float32)[None]))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def average(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model_mode", choices=("student", "coarse"), default="student")
    parser.add_argument("--teacher_cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cache = args.teacher_cache.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    meta = json.loads((cache / "metadata.json").read_text())
    if (meta["height"], meta["width"]) != (512, 512):
        raise ValueError("Teacher cache is not 512x512")
    if Path(meta["dataset_root"]).resolve() != dataset.parent:
        raise ValueError("Teacher cache comes from a different flat dataset root")
    specs = []
    for label in LABELS:
        source = REFERENCE / label / "latent/test_1" / label / "source_metadata.json"
        spec = json.loads(source.read_text())
        if spec["split"] != "test_1" or spec["label"] != label:
            raise ValueError(f"Reference test_1 mapping differs: {label}")
        start, stop = (int(value) for value in spec["frame_range"])
        branch = f"{spec['class']}/{spec['source_sequence']}"
        if (start, stop) != (0, 149):
            raise ValueError(f"Unexpected 5-second cached window: {label}")
        for frame in range(start, stop + 1):
            for folder in ("inputs", "masks"):
                path = dataset / label / folder / f"{frame:06d}.png"
                if not path.is_file():
                    raise FileNotFoundError(path)
        for frame in range(start, stop):
            path = cache / "matched" / branch / f"{frame:06d}_{frame + 1:06d}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
        specs.append((label, branch, start, stop))
    checkpoint_root, model_dir = vis.resolve_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = json.loads((model_dir / "config.json").read_text())
    if args.model_mode == "coarse":
        model = vis.RAFTStudentFlowPredictor(
            propainter_root=config["propainter_root"],
            raft_checkpoint=config["source_checkpoint"],
            iterations=int(config["iterations"]), mixed_precision=False,
            freeze_batchnorm=bool(config["freeze_batchnorm"]),
        )
    else:
        model = vis.RAFTStudentFlowPredictor.from_pretrained(
            model_dir, mixed_precision=False,
        )
    model = model.to(device).eval().requires_grad_(False)
    output.mkdir(parents=True)
    rows = []
    for label, branch, start, stop in specs:
        for frame in range(start, stop):
            source = dataset / label
            a = rgb(source / "inputs" / f"{frame:06d}.png")[None].to(device)
            b = rgb(source / "inputs" / f"{frame + 1:06d}.png")[None].to(device)
            backgrounds = [bg_mask(source / "masks" / f"{f:06d}.png").to(device)
                           for f in (frame, frame + 1)]
            path = cache / "matched" / branch / f"{frame:06d}_{frame + 1:06d}.npz"
            with np.load(path, allow_pickle=False) as file:
                teachers = [torch.from_numpy(file[key].copy()).to(device)
                            for key in ("teacher_f", "teacher_b")]
            if any(flow.shape != (2, 512, 512) for flow in teachers):
                raise ValueError(f"Malformed teacher flow: {path}")
            with torch.inference_mode():
                predictions = [flow[0].float() for flow in
                               model.predict_bidirectional(a, b, pair_batch_size=1)]
                valid_forward, _, _ = directional_fb_confidence(
                    teachers[0][None], teachers[1][None])
                valid_backward, _, _ = directional_fb_confidence(
                    teachers[1][None], teachers[0][None])
            valids = [valid_forward[0], valid_backward[0]]
            row = {"sequence": label, "teacher_branch": branch,
                   "frame_t": frame, "frame_t1": frame + 1}
            for index, direction in enumerate(DIRECTIONS):
                stats = vis.direction_metrics(
                    predictions[index], teachers[index], valids[index], backgrounds[index])
                row.update({f"{direction}_{key}": value for key, value in stats.items()})
                row[f"{direction}_valid_bg_pixels"] = float(
                    (valids[index] * backgrounds[index]).sum())
            rows.append(row)
            if frame == start or (frame - start + 1) % 25 == 0 or frame == stop - 1:
                print(f"{label}: {frame - start + 1}/{stop - start} pairs", flush=True)
    write_csv(output / "per_pair_metrics.csv", rows)
    sequence_rows = []
    for label in LABELS:
        group = [row for row in rows if row["sequence"] == label]
        result = {"sequence": label, "pair_count": len(group), "frame_count": len(group) + 1}
        for key in ("epe", "zero_epe", "bg_epe", "bg_zero_epe", "valid_ratio", "valid_bg_ratio"):
            result[key] = float(np.mean([average(group, f"{direction}_{key}")
                                         for direction in DIRECTIONS]))
        result["zero_gain"] = 1 - result["epe"] / max(result["zero_epe"], 1e-8)
        result["bg_zero_gain"] = 1 - result["bg_epe"] / max(result["bg_zero_epe"], 1e-8)
        result["bg_epe_pixel_weighted"] = float(
            sum(float(row[f"{direction}_bg_epe"]) * float(row[f"{direction}_valid_bg_pixels"])
                for row in group for direction in DIRECTIONS)
            / max(sum(float(row[f"{direction}_valid_bg_pixels"])
                      for row in group for direction in DIRECTIONS), 1e-8))
        sequence_rows.append(result)
    write_csv(output / "per_sequence_metrics.csv", sequence_rows)
    numeric = [key for key in sequence_rows[0] if key not in
               {"sequence", "pair_count", "frame_count"}]
    summary = {
        "checkpoint": str(checkpoint_root) if args.model_mode == "student" else config["source_checkpoint"],
        "model_mode": args.model_mode,
        "student_reference_checkpoint": str(checkpoint_root),
        "teacher_cache": str(cache),
        "dataset": str(dataset), "split": "test_1",
        "teacher_model_sha256": meta["raft_sha256"],
        "flow_resolution": [512, 512], "student_input_resolution": [512, 512],
        "pair_count": len(rows), "sequence_count": len(sequence_rows),
        "coverage": "Frames 0..149 of each sequence; BasketballPass and PartyScene have additional test frames without a matching teacher in this cache",
        "validity": "Directional teacher forward-backward consistency, alpha=0.01 beta=0.5, and in-bounds",
        "mask": "0=BG, 255=ROI; BG pixels selected with grayscale mask <128",
        "aggregation": "Mean per adjacent pair and direction, then equal-weight mean of 5 sequences",
        "macro_average": {key: average(sequence_rows, key) for key in numeric},
        "global_bg_epe_pixel_weighted": float(
            sum(float(row[f"{direction}_bg_epe"]) * float(row[f"{direction}_valid_bg_pixels"])
                for row in rows for direction in DIRECTIONS)
            / max(sum(float(row[f"{direction}_valid_bg_pixels"])
                      for row in rows for direction in DIRECTIONS), 1e-8)),
        "per_sequence": sequence_rows,
    }
    vis.json_dump(output / "summary.json", summary)
    print(json.dumps({"pair_count": len(rows), "macro_average": summary["macro_average"]}), flush=True)


if __name__ == "__main__":
    main()
