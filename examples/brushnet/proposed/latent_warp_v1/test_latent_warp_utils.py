#!/usr/bin/env python3
"""CPU unit tests for latent_warp_utils (no GPU, no checkpoints needed).

Run:  CUDA_VISIBLE_DEVICES= python test_latent_warp_utils.py
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (str(HERE.parents[1]), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch

from diffusers import DDPMScheduler
from latent_warp_utils import build_latent_warp_condition, latent_warp_drop_mask, predict_x0

torch.set_num_threads(4)
B, T, C, h, w = 1, 3, 4, 8, 10
H, W = 8 * h, 8 * w


def _inputs(flow_x=0.0, flow_y=0.0, masks=None, valid=None):
    generator = torch.Generator().manual_seed(0)
    x0 = torch.randn(B * T, C, h, w, generator=generator)
    if masks is None:
        masks = torch.ones(B * T, 1, H, W)
    flow = torch.zeros(B, T - 1, 2, H, W)
    flow[:, :, 0] = flow_x
    flow[:, :, 1] = flow_y
    if valid is None:
        valid = torch.ones(B, T - 1, 1, H, W)
    return x0, masks, flow, valid


def test_zero_flow_is_identity():
    x0, masks, flow, valid = _inputs()
    warped, validity, stats = build_latent_warp_condition(x0, masks, flow, valid, B, T)
    assert torch.allclose(warped[1:], x0[:-1], atol=1e-5), "zero flow must copy the previous x0"
    assert torch.all(validity[1:] == 1) and torch.all(validity[0] == 0) and torch.all(warped[0] == 0)
    assert abs(stats["stable_frac_of_bg"] - 1.0) < 1e-6


def test_one_latent_pixel_shift():
    # current pixel p samples previous frame at p + 8px -> latent column j takes j+1
    x0, masks, flow, valid = _inputs(flow_x=8.0)
    warped, validity, _ = build_latent_warp_condition(x0, masks, flow, valid, B, T)
    assert torch.allclose(warped[1:, :, :, : w - 1], x0[:-1, :, :, 1:], atol=1e-5)
    assert torch.all(validity[1:, :, :, : w - 1] == 1)
    assert torch.all(validity[1:, :, :, w - 1] == 0), "out-of-bounds block must be invalid"


def test_half_block_shift_is_unbiased():
    # +4 px = half a latent pixel: avg-pool gives exactly the mean of columns j and j+1
    # (DGAF's nearest x0.125 would return column j, i.e. ignore the motion).
    x0, masks, flow, valid = _inputs(flow_x=4.0)
    warped, _, _ = build_latent_warp_condition(x0, masks, flow, valid, B, T)
    expected = 0.5 * (x0[:-1, :, :, : w - 1] + x0[:-1, :, :, 1:])
    assert torch.allclose(warped[1:, :, :, : w - 2], expected[:, :, :, : w - 2], atol=1e-5)


def test_roi_and_fb_invalid_are_excluded():
    x0, masks, flow, valid = _inputs()
    masks = masks.clone()
    masks[0, :, :16, :16] = 0          # ROI in frame 0 -> frame 1 block (0..1, 0..1) unstable
    valid = valid.clone()
    valid[0, 1, :, 16:24, 40:48] = 0   # FB-invalid pixels in frame 2 -> block (2, 5) unstable
    warped, validity, stats = build_latent_warp_condition(x0, masks, flow, valid, B, T)
    assert torch.all(validity[1, :, :2, :2] == 0) and torch.all(warped[1, :, :2, :2] == 0)
    assert torch.all(validity[2, :, 2, 5] == 0) and torch.all(warped[2, :, 2, 5] == 0)
    assert torch.all(validity[1, :, 2:, 2:] == 1)
    assert stats["stable_frac_of_bg"] < 1.0


def test_partial_block_uses_stable_pixels_only():
    x0, masks, flow, valid = _inputs()
    masks = masks.clone()
    masks[1, :, 0:8, 0:4] = 0          # left half of block (0,0) is ROI in frame 1
    warped, validity, _ = build_latent_warp_condition(x0, masks, flow, valid, B, T)
    assert torch.allclose(validity[1, :, 0, 0], torch.tensor([0.5]))
    assert torch.allclose(warped[1, :, 0, 0], x0[0, :, 0, 0], atol=1e-5), "normalised by stable count"


def test_drop_and_first_frame():
    x0, masks, flow, valid = _inputs()
    drop = torch.tensor([False, True, False])
    warped, validity, stats = build_latent_warp_condition(x0, masks, flow, valid, B, T, drop=drop)
    assert torch.all(warped[0] == 0) and torch.all(warped[1] == 0) and torch.all(validity[1] == 0)
    assert torch.allclose(warped[2], x0[1], atol=1e-5)
    assert stats["warp_frames"] == 1
    assert not latent_warp_drop_mask(1, 8, 0.0, "cpu").any()
    assert latent_warp_drop_mask(1, 8, 1.0, "cpu").all()


def test_flow_is_rescaled_when_cached_at_other_resolution():
    x0, masks, _, valid = _inputs()
    flow = torch.zeros(B, T - 1, 2, 2 * H, 2 * W)
    flow[:, :, 0] = 16.0               # 16 px at 2x resolution == 8 px at training resolution
    warped, _, _ = build_latent_warp_condition(x0, masks, flow, None, B, T)
    assert torch.allclose(warped[1:, :, :, : w - 1], x0[:-1, :, :, 1:], atol=1e-4)


def test_predict_x0_inverts_add_noise():
    scheduler = DDPMScheduler(beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012)
    generator = torch.Generator().manual_seed(1)
    x0 = torch.randn(4, 4, 8, 8, generator=generator)
    noise = torch.randn(4, 4, 8, 8, generator=generator)
    timesteps = torch.tensor([0, 250, 700, 999])
    noisy = scheduler.add_noise(x0, noise, timesteps)
    assert torch.allclose(predict_x0(noisy, noise, timesteps, scheduler), x0, atol=1e-3)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
