#!/usr/bin/env python3
"""Phase 1 (run 1): build the 14-channel BrushNet initialisation from T8 checkpoint-2250.

* BrushNet: ``conv_in_condition`` widened 9 -> 14 inputs (old slices + bias
  copied, 5 new slices zero); config ``conditioning_channels`` 5 -> 10.
* IP-Adapter and FG/BG fusion module: copied byte-for-byte.
* Only weights are reused; optimizer / scheduler / RNG states are NOT (the new
  run starts a fresh optimizer).

Verification (CPU only, never touches a GPU):
  1. every copied tensor is bit-identical to the source,
  2. new input slices are exactly zero,
  3. forward equivalence: old model with 5-channel cond vs widened model with
     (a) zero warp channels and (b) random warp channels -> identical residuals.
A ``report.json`` with all numbers is written next to the weights.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only: never touch shared GPUs

import argparse
import datetime as _dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRUSHNET_DIR = HERE.parents[1]  # examples/brushnet
for path in (str(BRUSHNET_DIR), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
from safetensors.torch import load_file

from diffusers import BrushNetModel
from latent_warp_utils import WARP_CONDITIONING_CHANNELS, expand_brushnet_conditioning

DEFAULT_SOURCE = Path(
    "/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/"
    "train_sharedNoise_sameBG_0.95_T8/checkpoint-2250"
)
DEFAULT_OUTPUT = Path("/user/Cilab/Quan/experiment/latent_warp_run1/init_T8_ckpt2250_14ch")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--extra-channels", type=int, default=WARP_CONDITIONING_CHANNELS)
    parser.add_argument("--latent-size", type=int, default=64, help="Latent H=W for the forward check.")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def forward_residuals(model, sample, timestep, hidden, cond):
    down, mid, up = model(sample, timestep, encoder_hidden_states=hidden, brushnet_cond=cond, return_dict=False)
    return list(down) + [mid] + list(up)


def max_abs_diff(first, second) -> float:
    return max(float((a - b).abs().max()) for a, b in zip(first, second))


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.cpu_threads))
    source = args.source_checkpoint.resolve()
    output = args.output_dir.resolve()
    source_files = {
        "brushnet_weights": source / "brushnet" / "diffusion_pytorch_model.safetensors",
        "brushnet_config": source / "brushnet" / "config.json",
        "ipadapter": source / "ipadapter" / "model.safetensors",
        "fusion_module": source / "ipadapter" / "fusion_module.safetensors",
    }
    for path in source_files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"{output} is not empty; pass --overwrite")
    (output / "ipadapter").mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading {source / 'brushnet'}")
    old = BrushNetModel.from_pretrained(str(source / "brushnet")).eval()
    old_cc = int(old.config.conditioning_channels)
    print(f"      conditioning_channels={old_cc}, conv_in_condition.in_channels={old.conv_in_condition.in_channels}")

    print(f"[2/4] widening conv_in_condition by {args.extra_channels} zero-initialised inputs")
    new = expand_brushnet_conditioning(old, args.extra_channels).eval()
    new.save_pretrained(str(output / "brushnet"), safe_serialization=True)
    shutil.copy2(source_files["ipadapter"], output / "ipadapter" / "model.safetensors")
    shutil.copy2(source_files["fusion_module"], output / "ipadapter" / "fusion_module.safetensors")

    print("[3/4] verifying saved weights")
    reloaded = BrushNetModel.from_pretrained(str(output / "brushnet")).eval()
    old_state, new_state = old.state_dict(), reloaded.state_dict()
    if set(old_state) != set(new_state):
        raise RuntimeError("Parameter names differ after widening")
    old_in = old.conv_in_condition.in_channels
    copied_max = 0.0
    for key, value in old_state.items():
        if key == "conv_in_condition.weight":
            copied_max = max(copied_max, float((new_state[key][:, :old_in] - value).abs().max()))
            new_slice_max = float(new_state[key][:, old_in:].abs().max())
        else:
            copied_max = max(copied_max, float((new_state[key] - value).abs().max()))
    if copied_max != 0.0 or new_slice_max != 0.0:
        raise RuntimeError(f"Weight copy failed: copied_max={copied_max}, new_slice_max={new_slice_max}")
    for name in ("ipadapter", "fusion_module"):
        target = output / "ipadapter" / ("model.safetensors" if name == "ipadapter" else "fusion_module.safetensors")
        if sha256(target) != sha256(source_files[name]):
            raise RuntimeError(f"{name} copy is not byte-identical")
        load_file(str(target))  # readable

    print("[4/4] forward equivalence on CPU")
    generator = torch.Generator().manual_seed(0)
    size = args.latent_size
    sample = torch.randn(1, 4, size, size, generator=generator)
    hidden = torch.randn(1, 77, old.config.cross_attention_dim, generator=generator)
    base_cond = torch.randn(1, old_cc, size, size, generator=generator)
    base_cond[:, -1:] = (base_cond[:, -1:] > 0).float()  # mask-like channel
    warp_random = torch.randn(1, args.extra_channels, size, size, generator=generator)
    checks = {}
    for timestep in (999, 500, 50):
        t = torch.tensor([timestep])
        reference = forward_residuals(old, sample, t, hidden, base_cond)
        zero_warp = forward_residuals(reloaded, sample, t, hidden, torch.cat([base_cond, torch.zeros_like(warp_random)], 1))
        rand_warp = forward_residuals(reloaded, sample, t, hidden, torch.cat([base_cond, warp_random], 1))
        scale = max(float(r.abs().max()) for r in reference)
        checks[str(timestep)] = {
            "max_abs_residual": scale,
            "zero_warp_max_abs_diff": max_abs_diff(reference, zero_warp),
            "random_warp_max_abs_diff": max_abs_diff(reference, rand_warp),
        }
        print(f"      t={timestep}: {checks[str(timestep)]}")
        if checks[str(timestep)]["zero_warp_max_abs_diff"] > 1e-4 * max(scale, 1.0):
            raise RuntimeError("Widened model is not equivalent with zero warp channels")
        if checks[str(timestep)]["random_warp_max_abs_diff"] > 1e-4 * max(scale, 1.0):
            raise RuntimeError("New input slices are not inert at initialisation")

    report = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "source_checkpoint": str(source),
        "source_sha256": {key: sha256(path) for key, path in source_files.items()},
        "output_dir": str(output),
        "old_conditioning_channels": old_cc,
        "new_conditioning_channels": int(reloaded.config.conditioning_channels),
        "conv_in_condition_in_channels": [old_in, reloaded.conv_in_condition.in_channels],
        "channel_layout": [
            "0-3 z_t (sample)", "4-7 decoded-input latent", "8 BG mask (BG=1)",
            "9-12 warped previous x0 (stable BG only)", "13 validity (stable fraction)",
        ],
        "copied_weights_max_abs_diff": copied_max,
        "new_slices_max_abs": new_slice_max,
        "forward_checks": checks,
        "reused": ["brushnet weights", "ipadapter/model.safetensors", "ipadapter/fusion_module.safetensors"],
        "not_reused": ["optimizer.bin", "scheduler.bin", "scaler.pt", "random_states_*.pkl"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"OK -> {output}")


if __name__ == "__main__":
    main()
