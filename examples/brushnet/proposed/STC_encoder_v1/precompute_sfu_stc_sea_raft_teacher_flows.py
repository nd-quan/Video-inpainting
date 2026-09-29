#!/usr/bin/env python3
"""Cache clean VCM flows from a frozen SEA-RAFT teacher.

The VCM/STC dataset contains clean RGB frames, not optical-flow ground truth.
This script therefore makes a *pseudo-flow* teacher cache.  It follows the
same manifest-segment and NPZ contract as ``precompute_sfu_stc_teacher_flows``
so it can be consumed directly by the V7 distillation dataset.

The resulting cache is intentionally model-agnostic at its boundary:

    teacher_f, teacher_b: [2, H, W] pixel flow, channel order [dx, dy]
    valid_f, valid_b:     [1, H, W] forward/backward consistency masks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_DATASET = Path("/home/cilab/ndquan/videoInpainting/SFU_STC_flow")
DEFAULT_SEA_RAFT = Path("/home/cilab/ndquan/videoInpainting/pretrained/SEA-RAFT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--sea-raft-root", type=Path, default=DEFAULT_SEA_RAFT)
    parser.add_argument(
        "--cfg",
        type=Path,
        required=True,
        help="SEA-RAFT JSON config matching --checkpoint architecture.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Inference refinements; defaults to the value in --cfg.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "valid", "test"),
        default=("train", "valid"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sfu_stc_sea_raft_teacher_flow")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def iter_manifest_pairs(manifest: Dict, splits: set) -> Iterable[Tuple[Dict, str, int, int]]:
    """Yield only adjacent frames inside authoritative manifest segments."""
    for sequence in manifest["sequences"]:
        for split in ("train", "valid", "test"):
            if split not in splits:
                continue
            for segment in sequence["splits"][split]["segments"]:
                start, end = int(segment["start"]), int(segment["end"])
                for index in range(start, end):
                    yield sequence, split, index, index + 1


def load_frame(path: Path, height: int, width: int, device: torch.device) -> torch.Tensor:
    """Load SEA-RAFT input as float RGB in [0, 255]."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return tensor.to(device)


def flow_warp(tensor: torch.Tensor, flow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Warp ``tensor`` by pixel-space [dx,dy] flow and return in-bounds mask."""
    batch, _, height, width = tensor.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=tensor.device, dtype=tensor.dtype),
        torch.arange(width, device=tensor.device, dtype=tensor.dtype),
        indexing="ij",
    )
    coordinates = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
    sample = coordinates + flow
    in_bounds = (
        (sample[:, 0] >= 0)
        & (sample[:, 0] <= width - 1)
        & (sample[:, 1] >= 0)
        & (sample[:, 1] <= height - 1)
    ).unsqueeze(1)
    sample_x = 2.0 * sample[:, 0] / max(width - 1, 1) - 1.0
    sample_y = 2.0 * sample[:, 1] / max(height - 1, 1) - 1.0
    sample_grid = torch.stack((sample_x, sample_y), dim=-1)
    warped = F.grid_sample(
        tensor, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return warped, in_bounds.to(tensor.dtype)


def directional_fb_confidence(
    flow: torch.Tensor,
    opposite_flow: torch.Tensor,
    alpha: float = 0.01,
    beta: float = 0.5,
) -> torch.Tensor:
    """Return a standard forward/backward consistency validity mask."""
    warped_opposite, in_bounds = flow_warp(opposite_flow, flow)
    residual = flow + warped_opposite
    error_sq = residual.square().sum(dim=1, keepdim=True)
    magnitude_sq = flow.square().sum(dim=1, keepdim=True)
    magnitude_sq = magnitude_sq + warped_opposite.square().sum(dim=1, keepdim=True)
    return (error_sq <= alpha * magnitude_sq + beta).to(flow.dtype) * in_bounds


def load_sea_model(sea_raft_root: Path, cfg_path: Path, checkpoint: Path, device: torch.device):
    """Import SEA-RAFT without depending on the launch directory."""
    core_root = sea_raft_root / "core"
    if not (core_root / "raft.py").is_file():
        raise FileNotFoundError(core_root / "raft.py")
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    model_args = SimpleNamespace(**config)
    # The flow checkpoint below supplies every model parameter.  Avoid a
    # needless torchvision/ImageNet download while constructing the model.
    model_args.init_weight = False
    # SEA-RAFT's core modules use absolute intra-core imports (``from update``).
    core_string = str(core_root)
    if core_string not in sys.path:
        sys.path.insert(0, core_string)
    from raft import RAFT  # pylint: disable=import-outside-toplevel,import-error

    if checkpoint.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError(
                "Loading an official Hugging Face SEA-RAFT checkpoint requires "
                "`pip install safetensors`."
            ) from exc
        state = load_file(str(checkpoint), device="cpu")
    else:
        try:
            state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.0
            state = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint must contain a state dict: {checkpoint}")
    state = {
        str(key)[len("module.") :] if str(key).startswith("module.") else str(key): value
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    # ``BasicBlock`` registers its shortcut BatchNorm twice: once as ``bn3``
    # and once under ``downsample.1``.  Hugging Face safetensors retains just
    # the canonical ``bn3`` name to avoid serializing shared tensors twice,
    # whereas PyTorch strict loading expects both state-dict aliases.
    for key, value in tuple(state.items()):
        if ".bn3." in key:
            state.setdefault(key.replace(".bn3.", ".downsample.1."), value)
    if not state:
        raise ValueError(f"No tensor parameters found in {checkpoint}")
    model = RAFT(model_args)
    report = model.load_state_dict(state, strict=True)
    if report.missing_keys or report.unexpected_keys:
        raise RuntimeError(
            f"SEA-RAFT checkpoint/config mismatch: missing={report.missing_keys}, "
            f"unexpected={report.unexpected_keys}"
        )
    model.to(device).eval()
    return model, model_args


@torch.inference_mode()
def estimate_pair(model, frame0: torch.Tensor, frame1: torch.Tensor, iters: int):
    forward = model(frame0, frame1, iters=iters, test_mode=True)["final"]
    backward = model(frame1, frame0, iters=iters, test_mode=True)["final"]
    return forward, backward


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    if args.height <= 0 or args.width <= 0 or args.height % 8 or args.width % 8:
        raise SystemExit("--height and --width must be positive multiples of 8")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    dataset_root = args.dataset_root.expanduser().resolve()
    manifest_path = (args.manifest or dataset_root / "manifest.json").expanduser().resolve()
    sea_raft_root = args.sea_raft_root.expanduser().resolve()
    cfg_path = args.cfg.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    for path in (manifest_path, cfg_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_root = (
        args.output_root
        or dataset_root / f"teacher_flows_sea_raft_{args.width}x{args.height}"
    ).expanduser().resolve()

    device = torch.device(args.device)
    model, model_args = load_sea_model(sea_raft_root, cfg_path, checkpoint, device)
    iters = int(args.iters if args.iters is not None else model_args.iters)
    if iters < 1:
        raise ValueError("SEA-RAFT inference iterations must be positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs = list(iter_manifest_pairs(manifest, set(args.splits)))
    if not pairs:
        raise ValueError("No manifest pairs selected")

    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_root / "logs" / "precompute.log")
    metadata = {
        "version": 2,
        "dataset_root": str(dataset_root),
        "manifest": str(manifest_path),
        "teacher_backend": "sea_raft",
        "teacher_checkpoint": str(checkpoint),
        "teacher_sha256": sha256(checkpoint),
        "teacher_config": str(cfg_path),
        "teacher_config_sha256": sha256(cfg_path),
        "height": int(args.height),
        "width": int(args.width),
        "iters": iters,
        "normalization": "RGB [0,255] passed to SEA-RAFT; SEA-RAFT normalizes internally",
        "format": "npz teacher_f/teacher_b [2,H,W], valid_f/valid_b [1,H,W]",
        "layout": "<split>/<class>/<sequence>/<frame0>_<frame1>.npz",
    }
    metadata_path = output_root / "metadata.json"
    if metadata_path.is_file() and not args.overwrite:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("teacher_backend", "teacher_sha256", "height", "width", "iters"):
            if existing.get(key) != metadata[key]:
                raise RuntimeError(
                    f"Existing teacher metadata differs at {key}; choose another output root "
                    "or pass --overwrite"
                )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    logger.info("Loaded SEA-RAFT teacher from %s", checkpoint)
    logger.info("Caching %d pairs at %dx%d with %d refinement iterations", len(pairs), args.width, args.height, iters)
    written = skipped = 0
    started = time.time()
    for sequence, split, index0, index1 in pairs:
        class_name, sequence_name = sequence["class"], sequence["name"]
        destination = output_root / split / class_name / sequence_name / f"{index0:06d}_{index1:06d}.npz"
        if destination.is_file() and args.resume:
            skipped += 1
            continue
        if destination.is_file() and not args.overwrite:
            raise FileExistsError(f"{destination} exists; pass --resume or --overwrite")

        frame_root = dataset_root / split / "GT" / class_name / sequence_name
        frame0 = load_frame(frame_root / f"{index0:06d}.png", args.height, args.width, device)
        frame1 = load_frame(frame_root / f"{index1:06d}.png", args.height, args.width, device)
        teacher_f, teacher_b = estimate_pair(model, frame0, frame1, iters)
        arrays = {
            "teacher_f": teacher_f[0].cpu().numpy().astype(np.float32),
            "teacher_b": teacher_b[0].cpu().numpy().astype(np.float32),
            "valid_f": directional_fb_confidence(teacher_f, teacher_b)[0].cpu().numpy().astype(np.float32),
            "valid_b": directional_fb_confidence(teacher_b, teacher_f)[0].cpu().numpy().astype(np.float32),
        }
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise FloatingPointError(f"Non-finite teacher flow: {split}/{class_name}/{sequence_name} {index0}->{index1}")
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
                processed, len(pairs), written, skipped, processed / max(elapsed, 1e-6),
            )
    logger.info("Finished total=%d written=%d skipped=%d", len(pairs), written, skipped)


if __name__ == "__main__":
    main()
