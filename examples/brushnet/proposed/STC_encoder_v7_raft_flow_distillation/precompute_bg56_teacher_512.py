#!/usr/bin/env python3
"""Generate new BG56 clean-GT labels using the student's exact square loader."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bg56_512_setup import PROPAINTER, TEACHER_WEIGHTS, cache_path, check_cache_file, load_rgb512, pairs, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard")
    run = args.run_root.resolve()
    metadata = read_json(run / "teacher_flows_512x512/metadata.json")
    if (metadata["height"], metadata["width"], metadata["iters"]) != (512, 512, 20):
        raise ValueError("Teacher generation is restricted to 512x512, 20 iterations")
    manifest = read_json(run / "dataset/manifest.json")
    entries = list(pairs(manifest))[args.shard_index::args.num_shards]
    sys.path.insert(0, str(PROPAINTER))
    from core.vcm_flow_utils import directional_fb_confidence
    from model.modules.flow_comp_raft import initialize_RAFT
    model = initialize_RAFT(str(TEACHER_WEIGHTS), device=torch.device("cuda")).eval()
    model.requires_grad_(False)
    report = run / "logs/teacher_shard_{}.json".format(args.shard_index)
    started = time.time()
    for index, pair in enumerate(entries, 1):
        destination = cache_path(run, pair)
        if destination.is_file():
            check_cache_file(destination)
        else:
            split, branch, first, second = pair
            frames = run / "dataset" / split / "GT" / branch
            frame0 = load_rgb512(frames / "{:06d}.png".format(first))[None].cuda()
            frame1 = load_rgb512(frames / "{:06d}.png".format(second))[None].cuda()
            if frame0.shape != (1, 3, 512, 512) or frame1.shape != frame0.shape:
                raise ValueError("Invalid teacher input shape")
            with torch.inference_mode():
                _, forward = model(frame0, frame1, iters=20, test_mode=True)
                _, backward = model(frame1, frame0, iters=20, test_mode=True)
                valid_f, _, _ = directional_fb_confidence(forward, backward)
                valid_b, _, _ = directional_fb_confidence(backward, forward)
            arrays = {key: value[0].cpu().numpy().astype(np.float32) for key, value in (("teacher_f", forward), ("teacher_b", backward), ("valid_f", valid_f), ("valid_b", valid_b))}
            if not all(np.isfinite(value).all() for value in arrays.values()):
                raise FloatingPointError(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **arrays)
            temporary.replace(destination)
        if index == 1 or index % 25 == 0 or index == len(entries):
            progress = {"completed": index, "total": len(entries), "elapsed_seconds": time.time() - started, "shard": args.shard_index, "complete": index == len(entries)}
            write_json(report, progress)
            print(progress, flush=True)


if __name__ == "__main__":
    main()
