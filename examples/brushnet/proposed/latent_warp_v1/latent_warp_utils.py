"""Stable-BG latent warp helpers for latent_warp_v1.

BrushNet input layout after expansion (14 channels into ``conv_in_condition``)::

    [ z_t (4) | decoded-input latent (4) | BG mask (1) | warped prev x0 (4) | validity (1) ]
      sample    ------------------- brushnet_cond (10 = 5 base + 5 warp) -------------------

Conventions
-----------
* Masks: ``1 = background`` (the regenerated region), ``0 = ROI``.
* ``flow_backward[k]`` maps pixels of frame ``k+1`` to frame ``k`` (pixel units),
  i.e. cached ``teacher_b`` of pair ``k_k+1``.  ``valid_backward[k]`` lives on
  frame ``k+1`` pixels.
* Warp happens in pixel space so sub-latent-pixel motion is preserved:
  nearest x8 upsample of x0 -> bilinear backward warp -> stable-weighted
  average pool x8 (each latent pixel = mean over the stable pixels of its 8x8
  block).  Validity = stable fraction of the 8x8 block.  Frame 0 of each clip
  (no previous frame) and dropped frames get zeros in both warp channels.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddim_temporal import (
    backward_warp,
    build_stable_bg_mask,
    resize_flow,
)

BASE_CONDITIONING_CHANNELS = 5  # decoded-input latent (4) + BG mask (1)
WARP_CONDITIONING_CHANNELS = 5  # warped previous x0 (4) + validity (1)
LATENT_CHANNELS = 4
VAE_SCALE = 8


# ---------------------------------------------------------------------------
# Phase 1: widen BrushNet's first conv
# ---------------------------------------------------------------------------

def expand_brushnet_conditioning(brushnet, extra_channels: int = WARP_CONDITIONING_CHANNELS):
    """Return a new BrushNetModel whose ``conv_in_condition`` takes ``extra_channels`` more inputs.

    Every existing weight (including the conv bias and the old input slices) is
    copied unchanged; the new input slices are zero, so the widened model is
    numerically identical to the source model at initialisation.
    """
    from diffusers import BrushNetModel

    config = {key: value for key, value in dict(brushnet.config).items() if not key.startswith("_")}
    old_conditioning = int(config["conditioning_channels"])
    config["conditioning_channels"] = old_conditioning + int(extra_channels)
    widened = BrushNetModel.from_config(config)

    state = {key: value.detach().clone() for key, value in brushnet.state_dict().items()}
    old_weight = state["conv_in_condition.weight"]
    out_channels, old_in, kernel_h, kernel_w = old_weight.shape
    expected_old_in = int(config["in_channels"]) + old_conditioning
    if old_in != expected_old_in:
        raise ValueError(f"conv_in_condition has {old_in} inputs, config implies {expected_old_in}")
    new_weight = torch.zeros(
        (out_channels, old_in + int(extra_channels), kernel_h, kernel_w), dtype=old_weight.dtype
    )
    new_weight[:, :old_in] = old_weight
    state["conv_in_condition.weight"] = new_weight
    widened.load_state_dict(state, strict=True)
    return widened.to(dtype=next(brushnet.parameters()).dtype)


# ---------------------------------------------------------------------------
# Phase 3: x0 estimate and warped condition
# ---------------------------------------------------------------------------

def predict_x0(
    noisy_latents: torch.Tensor,
    model_output: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler,
) -> torch.Tensor:
    """x0 = (z_t - sqrt(1-a_t) * eps) / sqrt(a_t), computed in float32.

    Same quantity as DDIM/DDPM ``pred_original_sample`` (no clipping), so the
    training-time condition matches what the sampler can provide at inference.
    """
    alphas = noise_scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=torch.float32)
    alpha = alphas[timesteps.long()].view(-1, 1, 1, 1)
    sample = noisy_latents.float()
    output = model_output.float()
    prediction_type = noise_scheduler.config.prediction_type
    if prediction_type == "epsilon":
        return (sample - (1.0 - alpha).sqrt() * output) / alpha.sqrt()
    if prediction_type == "v_prediction":
        return alpha.sqrt() * sample - (1.0 - alpha).sqrt() * output
    if prediction_type == "sample":
        return output
    raise ValueError(f"Unknown prediction type {prediction_type}")


def latent_warp_drop_mask(
    num_clips: int,
    num_frames: int,
    drop_prob: float,
    device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Bool ``[B*T]``: True = this frame's warp condition is zeroed.

    p=0 and p=1 consume no random numbers, so a p=1 baseline and a p=0 run see
    the same diffusion noise / timesteps for the same seed.
    """
    total = int(num_clips) * int(num_frames)
    if drop_prob <= 0.0:
        return torch.zeros(total, dtype=torch.bool, device=device)
    if drop_prob >= 1.0:
        return torch.ones(total, dtype=torch.bool, device=device)
    return torch.rand(total, device=device, generator=generator) < float(drop_prob)


def _resize_valid(valid: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if tuple(valid.shape[-2:]) == tuple(size):
        return valid
    return F.interpolate(valid, size=size, mode="nearest")


@torch.no_grad()
def build_latent_warp_condition(
    x0: torch.Tensor,
    bg_masks: torch.Tensor,
    flow_backward: torch.Tensor,
    valid_backward: Optional[torch.Tensor],
    num_clips: int,
    num_frames: int,
    drop: Optional[torch.Tensor] = None,
    x0_clamp: Optional[float] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Warp each frame's previous-frame x0 into its own grid, restricted to stable BG.

    Args:
        x0: ``[B*T, 4, h, w]`` predicted clean latents (any dtype; detached).
        bg_masks: ``[B*T, 1, H, W]`` pixel-resolution masks, BG=1, with ``H = 8h``.
        flow_backward: ``[B, T-1, 2, Hf, Wf]``; resized to ``(H, W)`` if needed.
        valid_backward: ``[B, T-1, 1, Hf, Wf]`` FB-consistency mask or ``None``.
        drop: optional bool ``[B*T]``; True zeroes that frame's warp channels.
        x0_clamp: optional symmetric clamp applied to x0 before warping.

    Returns:
        warped ``[B*T, 4, h, w]`` (float32), validity ``[B*T, 1, h, w]`` (float32)
        and scalar statistics for logging.
    """
    total, channels, height, width = x0.shape
    if total != num_clips * num_frames:
        raise ValueError(f"x0 batch {total} != B*T = {num_clips}*{num_frames}")
    pixel_size = (height * VAE_SCALE, width * VAE_SCALE)
    if tuple(bg_masks.shape) != (total, 1, *pixel_size):
        raise ValueError(f"bg_masks must be {(total, 1, *pixel_size)}, got {tuple(bg_masks.shape)}")
    if flow_backward.shape[:3] != (num_clips, num_frames - 1, 2):
        raise ValueError(f"flow_backward must be [B,T-1,2,H,W], got {tuple(flow_backward.shape)}")

    device = x0.device
    x0 = x0.detach().float()
    if x0_clamp is not None and x0_clamp > 0:
        x0 = x0.clamp(-float(x0_clamp), float(x0_clamp))
    warped_out = torch.zeros((total, channels, height, width), device=device, dtype=torch.float32)
    valid_out = torch.zeros((total, 1, height, width), device=device, dtype=torch.float32)

    stable_pixels = 0.0
    bg_pixels = 0.0
    for clip in range(num_clips):
        first, last = clip * num_frames, (clip + 1) * num_frames
        masks = (bg_masks[first:last].float() > 0.5).float().to(device)
        flow = resize_flow(flow_backward[clip].float().to(device), pixel_size)
        visibility = None
        if valid_backward is not None:
            visibility = _resize_valid(valid_backward[clip].float().to(device), pixel_size)
        # stable_k = BG_k * [warp(BG_{k-1}) > 0.5] * in_bounds * fb_valid, for k = 1..T-1
        stable = build_stable_bg_mask(masks, flow, visibility=visibility, threshold=0.5)
        previous_up = F.interpolate(x0[first : last - 1], scale_factor=VAE_SCALE, mode="nearest")
        warped_up, _ = backward_warp(previous_up, flow)
        numerator = F.avg_pool2d(warped_up * stable, kernel_size=VAE_SCALE)
        fraction = F.avg_pool2d(stable, kernel_size=VAE_SCALE)
        warped = torch.where(fraction > 0, numerator / fraction.clamp_min(eps), torch.zeros_like(numerator))
        warped_out[first + 1 : last] = warped
        valid_out[first + 1 : last] = fraction
        stable_pixels += float(stable.sum())
        bg_pixels += float(masks[1:].sum())

    if drop is not None:
        keep = (~drop.to(device=device, dtype=torch.bool)).float().view(-1, 1, 1, 1)
        warped_out = warped_out * keep
        valid_out = valid_out * keep

    has_previous = torch.ones(total, dtype=torch.bool, device=device)
    has_previous[0::num_frames] = False
    if drop is not None:
        has_previous &= ~drop.to(device=device, dtype=torch.bool)
    active = valid_out[has_previous]
    stats = {
        "stable_frac_of_bg": stable_pixels / max(bg_pixels, 1.0),
        "valid_mean": float(active.mean()) if active.numel() else 0.0,
        "x0_abs_mean": float(x0.abs().mean()),
        "x0_abs_max": float(x0.abs().max()),
        "warp_frames": int(has_previous.sum()),
    }
    return warped_out, valid_out, stats
