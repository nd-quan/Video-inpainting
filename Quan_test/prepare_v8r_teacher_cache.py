#!/usr/bin/env python3
"""Compute clean ProPainter RAFT flow on frozen V8-R student references."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROPAINTER = Path("/home/cilab/ndquan/videoInpainting/pretrained/ProPainter")
CHECKPOINT = PROPAINTER / "weights/raft-things.pth"


def worker(output, records):
    import torch
    sys.path.insert(0, str(PROPAINTER))
    sys.path.insert(0, str(ROOT / "examples/brushnet"))
    from model.modules.flow_comp_raft import initialize_RAFT
    from proposed.STC_encoder_v1.precompute_sfu_stc_teacher_flows import load_frame, estimate_pair
    device = torch.device("cuda")
    model = initialize_RAFT(str(CHECKPOINT), device=device).eval()
    for record in records:
        branch = output / "matched" / record["sequence"]
        branch.mkdir(parents=True, exist_ok=True)
        gt = Path(record["frozen_dataset_root"]) / record["sequence"] / "gt"
        first, last = record["frame_range"]
        done = 0
        for a in range(first, last):
            dst = branch / f"{a:06d}_{a+1:06d}.npz"
            if dst.is_file():
                done += 1
                continue
            frame_a = load_frame(gt / f"{a:06d}.png", 512, 512, device)
            frame_b = load_frame(gt / f"{a+1:06d}.png", 512, 512, device)
            forward, backward = estimate_pair(model, frame_a, frame_b, 20)
            arrays = {"teacher_f": forward[0].cpu().numpy().astype(np.float32),
                      "teacher_b": backward[0].cpu().numpy().astype(np.float32)}
            if not all(np.isfinite(x).all() for x in arrays.values()):
                raise FloatingPointError(dst)
            tmp = dst.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **arrays)
            tmp.replace(dst)
            done += 1
            if done % 25 == 0:
                print(f"{record['sequence']}: {done}/{last-first} pairs", flush=True)
        print(f"{record['sequence']}: complete {done}/{last-first} pairs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--sequences", nargs="+")
    args = parser.parse_args()
    root = args.output_root.resolve()
    manifest = json.loads((root / "reference_manifest.json").read_text())
    records = manifest["records"]
    cache = root / "teacher_cache"
    cache.mkdir(exist_ok=True)
    if args.worker:
        selected = set(args.sequences or ())
        worker(cache, [r for r in records if r["sequence"] in selected])
        return
    metadata = {"version": 1, "height": 512, "width": 512, "iters": 20,
                "raft_checkpoint": str(CHECKPOINT),
                "source": "frozen student-reference GT", "format": "npz teacher_f/teacher_b [2,H,W]"}
    (cache / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    buckets = [[] for _ in args.gpus]
    loads = [0 for _ in args.gpus]
    for record in sorted(records, key=lambda x: x["unique_frames"], reverse=True):
        index = min(range(len(buckets)), key=lambda i: loads[i])
        buckets[index].append(record["sequence"])
        loads[index] += record["unique_frames"] - 1
    def launch(item):
        gpu, sequences = item
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        log_path = cache / f"build_gpu{gpu}.log"
        with log_path.open("w") as log:
            return subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                "--output-root", str(root), "--worker", "--sequences", *sequences],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    assignments = list(zip(args.gpus, buckets))
    print("Assignments:", assignments, flush=True)
    with ThreadPoolExecutor(max_workers=len(assignments)) as pool:
        statuses = list(pool.map(launch, assignments))
    if any(statuses):
        raise RuntimeError(f"Teacher cache build failed: {list(zip(args.gpus, statuses))}")
    expected = sum(r["unique_frames"] - 1 for r in records)
    actual = len(list((cache / "matched").glob("*/*.npz")))
    if actual != expected:
        raise RuntimeError(f"Teacher cache incomplete: {actual}/{expected}")
    (cache / "build_report.json").write_text(json.dumps({"pairs": actual,
        "sequences": len(records), "gpus": args.gpus, "completed_at_unix": time.time()}, indent=2) + "\n")
    print(f"Teacher cache complete: {actual} pairs", flush=True)


if __name__ == "__main__":
    main()
