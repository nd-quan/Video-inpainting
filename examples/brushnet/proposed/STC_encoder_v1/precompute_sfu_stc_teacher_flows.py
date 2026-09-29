#!/usr/bin/env python3
"""Precompute clean-frame RAFT teachers for the hierarchical SFU STC dataset.

Pairs are created only inside contiguous manifest segments, so no flow crosses
video or train/valid/test boundaries.  The cache is consumed by
``STCFlowDataset`` and the Stage-1 full-flow trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_PROPAINTER = Path(
    "/home/cilab/ndquan/videoInpainting/pretrained/ProPainter"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("/home/cilab/ndquan/videoInpainting/SFU_STC_flow"),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--propainter-root", type=Path, default=DEFAULT_PROPAINTER)
    parser.add_argument("--raft-checkpoint", type=Path)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "valid", "test"),
        default=("train", "valid", "test"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sfu_stc_teacher_flow")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def iter_manifest_pairs(manifest: dict, splits: set) -> Iterable[Tuple[dict, str, int, int]]:
    for sequence in manifest["sequences"]:
        for split in ("train", "valid", "test"):
            if split not in splits:
                continue
            for segment in sequence["splits"][split]["segments"]:
                for index in range(int(segment["start"]), int(segment["end"])):
                    yield sequence, split, index, index + 1


def load_frame(path: Path, height: int, width: int, device: torch.device) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = F.interpolate(
        tensor, size=(height, width), mode="bilinear", align_corners=False
    )
    return (tensor * 2.0 - 1.0).to(device)


@torch.inference_mode()
def estimate_pair(model, frame0: torch.Tensor, frame1: torch.Tensor, iters: int):
    _, forward = model(frame0, frame1, iters=iters, test_mode=True)
    _, backward = model(frame1, frame0, iters=iters, test_mode=True)
    return forward, backward


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    if args.height % 8 or args.width % 8:
        raise SystemExit("RAFT height and width must be divisible by 8")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    dataset_root = args.dataset_root.resolve()
    manifest_path = (args.manifest or dataset_root / "manifest.json").resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_root = (
        args.output_root
        or dataset_root / f"teacher_flows_{args.width}x{args.height}"
    ).resolve()
    propainter_root = args.propainter_root.resolve()
    checkpoint = (
        args.raft_checkpoint or propainter_root / "weights" / "raft-things.pth"
    ).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    sys.path.insert(0, str(propainter_root))
    from core.vcm_flow_utils import directional_fb_confidence
    from model.modules.flow_comp_raft import initialize_RAFT

    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_root / "logs" / "precompute.log")
    metadata = {
        "version": 1,
        "dataset_root": str(dataset_root),
        "manifest": str(manifest_path),
        "raft_checkpoint": str(checkpoint),
        "raft_sha256": sha256(checkpoint),
        "height": args.height,
        "width": args.width,
        "iters": args.iters,
        "normalization": "RGB [0,1] -> [-1,1]",
        "format": (
            "npz teacher_f/teacher_b [2,H,W], valid_f/valid_b [1,H,W]"
        ),
        "layout": "<split>/<class>/<sequence>/<frame0>_<frame1>.npz",
    }
    metadata_path = output_root / "metadata.json"
    if metadata_path.is_file() and not args.overwrite:
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("raft_sha256", "height", "width", "iters"):
            if old.get(key) != metadata[key]:
                raise RuntimeError(
                    f"Existing metadata differs at {key}; use another output root "
                    "or pass --overwrite"
                )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    pairs = list(iter_manifest_pairs(manifest, set(args.splits)))
    device = torch.device(args.device)
    logger.info("Loading ProPainter RAFT teacher from %s", checkpoint)
    model = initialize_RAFT(str(checkpoint), device=device)
    model.eval()
    written = skipped = 0
    started = time.time()
    for sequence, split, index0, index1 in pairs:
        class_name = sequence["class"]
        name = sequence["name"]
        destination = (
            output_root / split / class_name / name
            / f"{index0:06d}_{index1:06d}.npz"
        )
        if destination.is_file() and args.resume:
            skipped += 1
            continue
        if destination.is_file() and not args.overwrite:
            raise FileExistsError(
                f"{destination} exists; pass --resume or --overwrite"
            )
        frame_root = dataset_root / split / "GT" / class_name / name
        frame0 = load_frame(
            frame_root / f"{index0:06d}.png", args.height, args.width, device
        )
        frame1 = load_frame(
            frame_root / f"{index1:06d}.png", args.height, args.width, device
        )
        teacher_f, teacher_b = estimate_pair(model, frame0, frame1, args.iters)
        valid_f, _, _ = directional_fb_confidence(teacher_f, teacher_b)
        valid_b, _, _ = directional_fb_confidence(teacher_b, teacher_f)
        arrays = {
            "teacher_f": teacher_f[0].cpu().numpy().astype(np.float32),
            "teacher_b": teacher_b[0].cpu().numpy().astype(np.float32),
            "valid_f": valid_f[0].cpu().numpy().astype(np.float32),
            "valid_b": valid_b[0].cpu().numpy().astype(np.float32),
        }
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise FloatingPointError(f"Non-finite teacher flow: {name} {index0}->{index1}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(destination)
        written += 1
        processed = written + skipped
        if written == 1 or processed % 25 == 0 or processed == len(pairs):
            elapsed = time.time() - started
            logger.info(
                "%d/%d pairs | written=%d skipped=%d | %.2f pairs/s",
                processed, len(pairs), written, skipped,
                processed / max(elapsed, 1e-6),
            )
    logger.info(
        "Finished total=%d written=%d skipped=%d", len(pairs), written, skipped
    )


if __name__ == "__main__":
    main()
