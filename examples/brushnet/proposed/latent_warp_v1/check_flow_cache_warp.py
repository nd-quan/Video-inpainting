#!/usr/bin/env python3
"""CPU sanity check of a teacher-flow cache on real frames (no GPU, no VAE).

For sampled pairs (k-1, k) it loads GT frames and masks exactly like the
trainer (PIL 512x512), builds the stable-BG mask from teacher_b / valid_b and
reports, inside the stable region:

* pixel level : PSNR(warp(frame_{k-1}), frame_k)  vs  PSNR(frame_{k-1}, frame_k)
* latent level: the trainer's x8 path (build_latent_warp_condition) applied to
  8x8 average-pooled frames ("pseudo latents") vs the unwarped pseudo latent.

Warped must beat unwarped whenever there is motion; otherwise the flow
direction / units are wrong.  A PNG [prev | current | warped prev | stable | error]
is saved for the first pair of every branch.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (str(HERE.parents[1]), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffusers.schedulers.scheduling_ddim_temporal import backward_warp, build_stable_bg_mask
from latent_warp_utils import build_latent_warp_condition


def load_rgb(path: Path, size: int) -> torch.Tensor:
    resampling = getattr(Image, "Resampling", Image)
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((size, size), resampling.BILINEAR), dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1)[None] / 255.0


def load_bg(path: Path, size: int) -> torch.Tensor:
    resampling = getattr(Image, "Resampling", Image)
    with Image.open(path) as image:
        array = np.asarray(image.convert("L").resize((size, size), resampling.NEAREST), dtype=np.uint8)
    return torch.from_numpy((array < 128).astype(np.float32))[None, None]


def psnr(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor) -> float:
    weight = weight.expand_as(a)
    mse = float(((a - b) ** 2 * weight).sum() / weight.sum().clamp_min(1.0))
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flow-root", type=Path, required=True)
    parser.add_argument("--branches", nargs="*", default=None, help="Class/sequence; default: random cached ones")
    parser.add_argument("--num-branches", type=int, default=6)
    parser.add_argument("--pairs-per-branch", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)

    metadata = json.loads((args.flow_root / "metadata.json").read_text())
    dataset_root, split, size = Path(metadata["dataset_root"]), metadata["split"], int(metadata["height"])
    split_root = args.flow_root / split
    cached = sorted(p.relative_to(split_root).as_posix() for p in split_root.glob("*/*") if any(p.glob("*.npz")))
    rng = random.Random(args.seed)
    branches = args.branches or rng.sample(cached, min(args.num_branches, len(cached)))
    out_dir = args.out_dir or args.flow_root / "logs" / "warp_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for branch in branches:
        files = sorted((split_root / branch).glob("*.npz"))
        for index, path in enumerate(rng.sample(files, min(args.pairs_per_branch, len(files)))):
            i0, i1 = (int(v) for v in path.stem.split("_"))
            with np.load(path) as data:
                flow_b = torch.from_numpy(data["teacher_b"].astype(np.float32))[None]
                valid_b = torch.from_numpy(data["valid_b"].astype(np.float32))[None]
            prev = load_rgb(dataset_root / split / "GT" / branch / f"{i0:06d}.png", size)
            cur = load_rgb(dataset_root / split / "GT" / branch / f"{i1:06d}.png", size)
            bg = torch.cat([load_bg(dataset_root / split / "mask" / branch / f"{i:06d}.png", size) for i in (i0, i1)])
            stable = build_stable_bg_mask(bg, flow_b, visibility=valid_b)
            warped, _ = backward_warp(prev, flow_b)

            # latent path on pseudo latents (8x8 mean RGB): exactly the trainer's warp code
            pseudo = F.avg_pool2d(torch.cat([prev, cur]), 8)
            lat_warp, lat_valid, _ = build_latent_warp_condition(
                pseudo, bg, flow_b[:, None].reshape(1, 1, 2, size, size), valid_b.reshape(1, 1, 1, size, size), 1, 2
            )
            lat_weight = (lat_valid[1:] > 0.99).float()
            row = {
                "branch": branch, "pair": [i0, i1],
                "mean_abs_flow_px": float(flow_b.norm(dim=1).mean()),
                "stable_frac": float(stable.mean()),
                "pix_psnr_unwarped": psnr(prev, cur, stable),
                "pix_psnr_warped": psnr(warped, cur, stable),
                "lat_psnr_unwarped": psnr(pseudo[:1], pseudo[1:], lat_weight),
                "lat_psnr_warped": psnr(lat_warp[1:], pseudo[1:], lat_weight),
            }
            rows.append(row)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}))
            if index == 0:
                error = (warped - cur).abs().mean(1, keepdim=True).clamp(0, 0.25) * 4 * stable
                panel = torch.cat([prev, cur, warped, stable.expand(-1, 3, -1, -1), error.expand(-1, 3, -1, -1)], dim=3)
                image = (panel[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(image).save(out_dir / f"{branch.replace('/', '__')}_{i0:06d}.png")

    def mean(key):
        return float(np.mean([row[key] for row in rows]))

    summary = {key: round(mean(key), 3) for key in rows[0] if key not in ("branch", "pair")}
    summary["pairs"] = len(rows)
    summary["pix_warped_better_frac"] = float(np.mean([r["pix_psnr_warped"] >= r["pix_psnr_unwarped"] for r in rows]))
    summary["lat_warped_better_frac"] = float(np.mean([r["lat_psnr_warped"] >= r["lat_psnr_unwarped"] for r in rows]))
    print("SUMMARY", json.dumps(summary))
    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"panels + summary -> {out_dir}")


if __name__ == "__main__":
    main()
