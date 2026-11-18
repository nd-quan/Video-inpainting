#!/usr/bin/env python3
"""Phase 0 (run 1): clean-GT RAFT teacher flows for the BVI-DVC bg52 train set.

Frames are resized exactly like ``HierarchicalV8ClipDataset`` does for
``--resolution 512`` (PIL ``BILINEAR`` to a 512x512 square), so cached flow
vectors are already in the training pixel grid and need no rescaling.

Output layout (consumed unchanged by ``TeacherFlowV8ClipDataset``)::

    <output_root>/metadata.json
    <output_root>/manifest.json
    <output_root>/<split>/<Class>/<sequence>/<i:06d>_<i+1:06d>.npz
        teacher_f [2,H,W]  frame i   -> frame i+1   (pixels, float16 by default)
        teacher_b [2,H,W]  frame i+1 -> frame i     (pixels; used for backward warp)
        valid_f   [1,H,W]  FB-consistent & in-bounds for teacher_f (uint8 0/1)
        valid_b   [1,H,W]  FB-consistent & in-bounds for teacher_b (uint8 0/1)

GPU safety: by default the script refuses to start on a GPU that already runs
another process (see ``--allow-shared-gpu`` / ``--wait-for-free-gpu``).  The
check runs before this process creates any CUDA context.
"""

from __future__ import annotations

import os

# Make CUDA device indices match nvidia-smi indices before torch touches CUDA.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import datetime as _dt
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

REPO_DIR = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_ROOT = REPO_DIR / "examples" / "brushnet" / "dataset" / "bg52"
DEFAULT_OUTPUT_ROOT = REPO_DIR.parent / "flow-training" / "flows" / "bvi_dvc_bg52_512"
DEFAULT_PROPAINTER = REPO_DIR.parent / "pretrained" / "ProPainter"
FB_ALPHA = 0.01
FB_BETA = 0.5


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--propainter-root", type=Path, default=DEFAULT_PROPAINTER)
    parser.add_argument("--raft-checkpoint", type=Path, default=None)
    parser.add_argument("--resolution", type=int, default=512,
                        help="Square size; must equal the trainer's --resolution.")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--flow-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default="cuda:0",
                        help="cuda:N, cpu, or auto (first GPU with no other process).")
    parser.add_argument("--cpu-threads", type=int, default=8,
                        help="torch threads when --device cpu (keep small on a shared server).")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--include-branches", nargs="*", default=None,
                        help="Optional Class/sequence subset (tests). Manifest still lists all.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    gpu = parser.add_argument_group("GPU safety")
    gpu.add_argument("--allow-shared-gpu", action="store_true",
                     help="Run even if another process already uses the GPU (NOT default).")
    gpu.add_argument("--min-free-mib", type=int, default=6000,
                     help="Required free memory on the target GPU.")
    gpu.add_argument("--wait-for-free-gpu", type=int, default=0, metavar="SECONDS",
                     help="If >0, poll every SECONDS until the GPU is free instead of exiting.")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# GPU guard (nvidia-smi only; never initialises CUDA in this process)
# --------------------------------------------------------------------------

def _nvidia_smi(query: str, kind: str) -> List[List[str]]:
    output = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={query}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout
    return [[field.strip() for field in line.split(",")] for line in output.splitlines() if line.strip()]


def _process_owner(pid: str) -> str:
    try:
        return subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True, text=True).stdout.strip() or "?"
    except Exception:  # pragma: no cover - informational only
        return "?"


def gpu_occupancy() -> List[Dict]:
    gpus = [
        {"index": row[0], "uuid": row[1], "used_mib": int(row[2]), "total_mib": int(row[3]), "procs": []}
        for row in _nvidia_smi("index,uuid,memory.used,memory.total", "gpu")
    ]
    by_uuid = {gpu["uuid"]: gpu for gpu in gpus}
    for gpu_uuid, pid, used in _nvidia_smi("gpu_uuid,pid,used_memory", "compute-apps"):
        if int(pid) == os.getpid() or gpu_uuid not in by_uuid:
            continue
        by_uuid[gpu_uuid]["procs"].append({"pid": pid, "user": _process_owner(pid), "mib": used})
    return gpus


def _physical_id(cuda_index: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return str(cuda_index)
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if cuda_index >= len(entries):
        raise SystemExit(f"cuda:{cuda_index} is not visible (CUDA_VISIBLE_DEVICES={visible!r})")
    return entries[cuda_index]


def _gpu_is_free(gpu: Dict, min_free_mib: int) -> Tuple[bool, str]:
    free = gpu["total_mib"] - gpu["used_mib"]
    if gpu["procs"]:
        owners = ", ".join(f"pid {p['pid']} ({p['user']}, {p['mib']} MiB)" for p in gpu["procs"])
        return False, f"GPU {gpu['index']} is used by: {owners}"
    if free < min_free_mib:
        return False, f"GPU {gpu['index']} has only {free} MiB free (< {min_free_mib})"
    return True, f"GPU {gpu['index']} is free ({free} MiB)"


def resolve_device(args: argparse.Namespace, logger: logging.Logger) -> str:
    """Return a torch device string, enforcing the no-sharing policy."""
    if args.device == "cpu":
        return "cpu"
    while True:
        gpus = gpu_occupancy()
        if args.device == "auto":
            if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
                raise SystemExit("--device auto requires CUDA_VISIBLE_DEVICES to be unset")
            for gpu in gpus:
                ok, reason = _gpu_is_free(gpu, args.min_free_mib)
                if ok:
                    logger.info("Auto-selected %s", reason)
                    return f"cuda:{gpu['index']}"
            reason = "no GPU is free: " + "; ".join(_gpu_is_free(g, args.min_free_mib)[1] for g in gpus)
        else:
            if not args.device.startswith("cuda"):
                raise SystemExit(f"Unsupported --device {args.device!r}")
            cuda_index = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
            physical = _physical_id(cuda_index)
            matches = [g for g in gpus if physical in (g["index"], g["uuid"])]
            if not matches:
                raise SystemExit(f"Cannot map {args.device} (physical {physical}) to nvidia-smi output")
            ok, reason = _gpu_is_free(matches[0], args.min_free_mib)
            if ok or args.allow_shared_gpu:
                level = logging.INFO if ok else logging.WARNING
                logger.log(level, "%s%s", reason, "" if ok else " -> continuing because --allow-shared-gpu")
                return f"cuda:{cuda_index}"
        if args.wait_for_free_gpu <= 0:
            raise SystemExit(
                f"Refusing to start: {reason}. Other users' training must not be "
                "disturbed. Re-run when the GPU is free, use --wait-for-free-gpu SECONDS, "
                "or --device cpu."
            )
        logger.info("Waiting %ds: %s", args.wait_for_free_gpu, reason)
        time.sleep(args.wait_for_free_gpu)


# --------------------------------------------------------------------------
# Dataset scan / manifest
# --------------------------------------------------------------------------

def scan_sequences(dataset_root: Path, split: str) -> List[Dict]:
    gt_root = dataset_root / split / "GT"
    if not gt_root.is_dir():
        raise FileNotFoundError(gt_root)
    sequences = []
    for class_dir in sorted(path for path in gt_root.iterdir() if path.is_dir()):
        for sequence_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            indices = sorted(int(path.stem) for path in sequence_dir.glob("*.png"))
            if len(indices) < 2:
                continue
            segments, start = [], indices[0]
            for previous, current in zip(indices[:-1], indices[1:]):
                if current != previous + 1:
                    segments.append({"start": start, "end": previous})
                    start = current
            segments.append({"start": start, "end": indices[-1]})
            with Image.open(sequence_dir / f"{indices[0]:06d}.png") as image:
                native_width, native_height = image.size
            sequences.append({
                "name": sequence_dir.name,
                "class": class_dir.name,
                "native_width": native_width,
                "native_height": native_height,
                "mask_semantics": {"background": 0, "roi": 255},
                "splits": {
                    split: {
                        "segments": segments,
                        "frame_count": len(indices),
                        "relative_root": f"{split}/{{kind}}/{class_dir.name}/{sequence_dir.name}",
                    }
                },
            })
    if not sequences:
        raise ValueError(f"No sequences below {gt_root}")
    return sequences


def iter_pairs(sequences: Iterable[Dict], split: str) -> Iterable[Tuple[str, str, int, int]]:
    for sequence in sequences:
        for segment in sequence["splits"][split]["segments"]:
            for index in range(int(segment["start"]), int(segment["end"])):
                yield sequence["class"], sequence["name"], index, index + 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bg52_teacher_flow")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_frame(path: Path, resolution: int) -> np.ndarray:
    """RGB uint8 HxWx3, resized exactly like HierarchicalV8ClipDataset._rgb_tensor."""
    resampling = getattr(Image, "Resampling", Image)
    with Image.open(path) as image:
        image = image.convert("RGB")
        resized = image.resize((resolution, resolution), resample=resampling.BILINEAR)
        return np.asarray(resized, dtype=np.uint8).copy()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    if args.resolution % 8:
        raise SystemExit("--resolution must be divisible by 8 (RAFT)")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("Invalid --num-shards/--shard-index")

    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if str(output_root).startswith(str(dataset_root)) or "/vcm-rs/" in str(output_root) + "/":
        raise SystemExit("Output must not be written inside the dataset / vcm-rs tree")
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_root / "logs" / f"precompute_{args.split}_shard{args.shard_index}.log")
    logger.info("Args: %s", vars(args))

    device_name = resolve_device(args, logger)

    import torch  # imported after the GPU guard decided on the device

    if device_name == "cpu":
        torch.set_num_threads(max(1, args.cpu_threads))
    elif not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    propainter_root = args.propainter_root.resolve()
    checkpoint = (args.raft_checkpoint or propainter_root / "weights" / "raft-things.pth").resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(propainter_root))
    from core.vcm_flow_utils import directional_fb_confidence
    from model.modules.flow_comp_raft import initialize_RAFT

    # ---- manifest + metadata (shard 0 writes; every shard validates) ----
    sequences = scan_sequences(dataset_root, args.split)
    manifest = {
        "version": 1,
        "description": "Auto-generated pair manifest for BVI-DVC bg52 teacher flows (latent_warp_v1)",
        "dataset_root": str(dataset_root),
        "layout": "<split>/<kind>/<class>/<sequence>/<frame>.png",
        "kinds": ["GT", "input", "mask"],
        "splits": [args.split],
        "mask_semantics": {"background": 0, "roi": 255},
        "sequences": sequences,
    }
    manifest_path = output_root / "manifest.json"
    metadata = {
        "version": 1,
        "dataset_root": str(dataset_root),
        "manifest": str(manifest_path),
        "split": args.split,
        "source_kind": "GT",
        "raft_checkpoint": str(checkpoint),
        "raft_sha256": sha256(checkpoint),
        "height": args.resolution,
        "width": args.resolution,
        "resize": "PIL.Image.resize((W,H), BILINEAR) from native size, identical to HierarchicalV8ClipDataset",
        "iters": args.iters,
        "normalization": "RGB [0,1] -> [-1,1]",
        "flow_dtype": args.flow_dtype,
        "valid_definition": (
            f"|b(p)+f(p+b(p))|^2 <= {FB_ALPHA}*(|b|^2+|f_w|^2)+{FB_BETA}, times in-bounds "
            "(ProPainter core.vcm_flow_utils.directional_fb_confidence)"
        ),
        "format": "npz teacher_f/teacher_b [2,H,W] pixels; valid_f/valid_b [1,H,W] uint8",
        "layout": "<split>/<class>/<sequence>/<frame0>_<frame1>.npz",
    }
    metadata_path = output_root / "metadata.json"
    if metadata_path.is_file() and not args.overwrite:
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("dataset_root", "raft_sha256", "height", "width", "iters", "flow_dtype", "split"):
            if old.get(key) != metadata[key]:
                raise RuntimeError(f"Existing metadata differs at {key!r}; use another --output-root or --overwrite")
    if args.shard_index == 0:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        metadata["created"] = _dt.datetime.now().isoformat(timespec="seconds")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    selected = sequences
    if args.include_branches:
        wanted = {branch.strip("/") for branch in args.include_branches}
        selected = [s for s in sequences if f"{s['class']}/{s['name']}" in wanted]
        missing = wanted - {f"{s['class']}/{s['name']}" for s in selected}
        if missing:
            raise SystemExit(f"Unknown --include-branches: {sorted(missing)}")
    selected = [s for i, s in enumerate(selected) if i % args.num_shards == args.shard_index]
    pairs = list(iter_pairs(selected, args.split))
    logger.info("Sequences in manifest=%d, this shard=%d, pairs=%d", len(sequences), len(selected), len(pairs))

    device = torch.device(device_name)
    logger.info("Loading RAFT %s on %s", checkpoint, device)
    model = initialize_RAFT(str(checkpoint), device=device)
    model.eval()
    flow_dtype = np.float16 if args.flow_dtype == "float16" else np.float32

    frame_cache: Dict[Path, "torch.Tensor"] = {}

    def frame_tensor(path: Path) -> "torch.Tensor":
        if path not in frame_cache:
            while len(frame_cache) >= 2:  # keep only the two most recent frames
                frame_cache.pop(next(iter(frame_cache)))
            array = load_frame(path, args.resolution)
            tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0).mul_(2.0).sub_(1.0)
            frame_cache[path] = tensor.unsqueeze(0).to(device)
        return frame_cache[path]

    stats_path = output_root / "logs" / f"stats_{args.split}_shard{args.shard_index}.jsonl"
    stats_handle = stats_path.open("a", encoding="utf-8")
    written = skipped = 0
    started = time.time()
    with torch.inference_mode():
        for class_name, name, index0, index1 in pairs:
            destination = output_root / args.split / class_name / name / f"{index0:06d}_{index1:06d}.npz"
            if destination.is_file() and args.resume:
                skipped += 1
                continue
            if destination.is_file() and not args.overwrite:
                raise FileExistsError(f"{destination} exists; pass --resume or --overwrite")
            frame_root = dataset_root / args.split / "GT" / class_name / name
            frame0 = frame_tensor(frame_root / f"{index0:06d}.png")
            frame1 = frame_tensor(frame_root / f"{index1:06d}.png")
            # One batched call: [f0->f1, f1->f0].
            _, flows = model(torch.cat([frame0, frame1]), torch.cat([frame1, frame0]), iters=args.iters, test_mode=True)
            teacher_f, teacher_b = flows[0:1].float(), flows[1:2].float()
            valid_f, _, _ = directional_fb_confidence(teacher_f, teacher_b, alpha=FB_ALPHA, beta=FB_BETA)
            valid_b, _, _ = directional_fb_confidence(teacher_b, teacher_f, alpha=FB_ALPHA, beta=FB_BETA)
            arrays = {
                "teacher_f": teacher_f[0].cpu().numpy(),
                "teacher_b": teacher_b[0].cpu().numpy(),
                "valid_f": valid_f[0].cpu().numpy(),
                "valid_b": valid_b[0].cpu().numpy(),
            }
            if not all(np.isfinite(value).all() for value in arrays.values()):
                raise FloatingPointError(f"Non-finite flow: {class_name}/{name} {index0}->{index1}")
            if flow_dtype == np.float16 and max(np.abs(arrays["teacher_f"]).max(), np.abs(arrays["teacher_b"]).max()) > 6.0e4:
                raise OverflowError(f"Flow exceeds float16 range: {class_name}/{name} {index0}->{index1}")
            magnitude_b = float(np.sqrt((arrays["teacher_b"] ** 2).sum(0)).mean())
            to_save = {
                "teacher_f": arrays["teacher_f"].astype(flow_dtype),
                "teacher_b": arrays["teacher_b"].astype(flow_dtype),
                "valid_f": (arrays["valid_f"] > 0.5).astype(np.uint8),
                "valid_b": (arrays["valid_b"] > 0.5).astype(np.uint8),
            }
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.stem + ".tmp.npz")
            np.savez_compressed(temporary, **to_save)
            temporary.replace(destination)
            stats_handle.write(json.dumps({
                "branch": f"{class_name}/{name}", "pair": [index0, index1],
                "valid_f": float(to_save["valid_f"].mean()), "valid_b": float(to_save["valid_b"].mean()),
                "mean_abs_flow_b_px": magnitude_b,
            }) + "\n")
            written += 1
            processed = written + skipped
            if written == 1 or processed % 50 == 0 or processed == len(pairs):
                stats_handle.flush()
                elapsed = time.time() - started
                rate = written / max(elapsed, 1e-6)
                remaining = (len(pairs) - processed) / max(rate, 1e-6)
                logger.info("%d/%d pairs | written=%d skipped=%d | %.2f pairs/s | ETA %.1f min",
                            processed, len(pairs), written, skipped, rate, remaining / 60.0)
    stats_handle.close()
    logger.info("Finished: total=%d written=%d skipped=%d in %.1f min",
                len(pairs), written, skipped, (time.time() - started) / 60.0)


if __name__ == "__main__":
    main()
