"""Training utilities for BrushNet temporal adapters and STC noise shaping.

BrushNet and the diffusion U-Net stay parameter-frozen. Autograd remains
enabled through the required input paths so losses can reach either a motion
adapter or an STC-conditioned initial-noise module.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .brushnet_motion_adapter import backward_warp_feature


@torch.no_grad()
def build_flow_confidence(
    flow_forward: torch.Tensor,
    flow_backward: torch.Tensor,
    alpha: float = 0.01,
    beta: float = 0.5,
) -> torch.Tensor:
    """Forward/backward reliability for each adjacent backward flow."""
    if flow_forward.shape != flow_backward.shape or flow_forward.ndim != 5:
        raise ValueError("forward/backward flow must match [B,T-1,2,H,W]")
    batch, pairs, _, height, width = flow_backward.shape
    backward = flow_backward.reshape(-1, 2, height, width)
    forward = flow_forward.reshape(-1, 2, height, width)
    warped_forward = backward_warp_feature(forward, backward)
    fb_error = (backward + warped_forward).square().sum(dim=1, keepdim=True)
    magnitude = backward.square().sum(dim=1, keepdim=True)
    magnitude = magnitude + warped_forward.square().sum(dim=1, keepdim=True)
    in_bounds = backward_warp_feature(
        torch.ones_like(backward[:, :1]), backward
    )
    confidence = (fb_error <= alpha * magnitude + beta).to(backward.dtype)
    return (confidence * in_bounds).clamp_(0.0, 1.0).reshape(
        batch, pairs, 1, height, width
    )


@torch.no_grad()
def build_stable_bg_confidence(
    flow_forward: torch.Tensor,
    flow_backward: torch.Tensor,
    bg_masks: torch.Tensor,
    alpha: float = 0.01,
    beta: float = 0.5,
    flow_confidence: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build confidence for warping frame i-1 features into frame i.

    Args:
        flow_forward: [B,T-1,2,H,W], frame i-1 -> frame i.
        flow_backward: [B,T-1,2,H,W], frame i -> frame i-1.
        bg_masks: [B,T,1,H,W], one on background.
    Returns:
        [B,T-1,1,H,W] stable, visible background confidence at frame i.
    """
    if flow_forward.shape != flow_backward.shape or flow_forward.ndim != 5:
        raise ValueError("forward/backward flow must match [B,T-1,2,H,W]")
    batch, pairs, _, height, width = flow_backward.shape
    if bg_masks.shape != (batch, pairs + 1, 1, height, width):
        raise ValueError("bg_masks must have shape [B,T,1,H,W] at flow resolution")
    backward = flow_backward.reshape(-1, 2, height, width)
    if flow_confidence is None:
        flow_confidence = build_flow_confidence(
            flow_forward,
            flow_backward,
            alpha=alpha,
            beta=beta,
        )
    if flow_confidence.shape != (batch, pairs, 1, height, width):
        raise ValueError("flow_confidence must have shape [B,T-1,1,H,W]")
    fb_confidence = flow_confidence.reshape(-1, 1, height, width)

    previous_bg = bg_masks[:, :-1].reshape(-1, 1, height, width)
    current_bg = bg_masks[:, 1:].reshape(-1, 1, height, width)
    warped_previous_bg = backward_warp_feature(previous_bg, backward)
    stable = current_bg * warped_previous_bg * fb_confidence
    return stable.clamp_(0.0, 1.0).reshape(batch, pairs, 1, height, width)


def temporal_warp_loss(
    decoded_x0: torch.Tensor,
    flow_backward: torch.Tensor,
    confidence: torch.Tensor,
    charbonnier_eps: float = 1e-3,
) -> torch.Tensor:
    """Robust temporal loss on decoded predicted-clean frames."""
    if decoded_x0.ndim != 5:
        raise ValueError("decoded_x0 must have shape [B,T,C,H,W]")
    batch, frames, channels, height, width = decoded_x0.shape
    if flow_backward.shape[:2] != (batch, frames - 1):
        raise ValueError("flow_backward must contain T-1 pairs")
    previous = decoded_x0[:, :-1].reshape(-1, channels, height, width)
    current = decoded_x0[:, 1:].reshape(-1, channels, height, width)
    flow = flow_backward.reshape(-1, 2, *flow_backward.shape[-2:])
    confidence = confidence.reshape(-1, 1, *confidence.shape[-2:])
    warped_previous = backward_warp_feature(previous, flow)
    confidence = F.interpolate(
        confidence.to(decoded_x0.dtype),
        size=(height, width),
        mode="bilinear",
        align_corners=True,
    )
    error = ((current - warped_previous).square() + charbonnier_eps**2).sqrt()
    weight = confidence.expand_as(error)
    return (error * weight).sum() / weight.sum().clamp_min(1e-6)


class FrozenBrushNetMotionModel(nn.Module):
    """Frozen BrushNet/UNet with an optional trainable motion adapter."""

    def __init__(self, brushnet, unet, motion_adapter=None):
        super().__init__()
        self.brushnet = brushnet
        self.unet = unet
        self.motion_adapter = motion_adapter
        self.brushnet.requires_grad_(False).eval()
        self.unet.requires_grad_(False).eval()

    @staticmethod
    def _flatten_frames(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(-1, *tensor.shape[2:])

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        conditioning_latents: torch.Tensor,
        flow_backward: torch.Tensor,
        motion_confidence: torch.Tensor,
        flow_forward: Optional[torch.Tensor] = None,
        unet_added_kwargs: Optional[Dict] = None,
        motion_scale: float = 1.0,
    ):
        """Predict diffusion noise for a temporally ordered clip.

        Shapes:
            noisy_latents: [B,T,4,h,w]
            timesteps: [B], one shared diffusion timestep per clip
            encoder_hidden_states: [B,L,D] or [B,T,L,D]
            conditioning_latents: [B,T,Cc,h,w]
        """
        if noisy_latents.ndim != 5:
            raise ValueError("noisy_latents must have shape [B,T,C,H,W]")
        batch, frames = noisy_latents.shape[:2]
        if timesteps.shape != (batch,):
            raise ValueError("Use exactly one shared diffusion timestep per clip")
        noisy_flat = self._flatten_frames(noisy_latents)
        cond_flat = self._flatten_frames(conditioning_latents)
        timestep_flat = timesteps.repeat_interleave(frames)
        if encoder_hidden_states.ndim == 3:
            text_flat = encoder_hidden_states.repeat_interleave(frames, dim=0)
        elif encoder_hidden_states.ndim == 4:
            text_flat = self._flatten_frames(encoder_hidden_states)
        else:
            raise ValueError("encoder_hidden_states must be [B,L,D] or [B,T,L,D]")

        # Adapter training intentionally detaches frozen BrushNet features.
        # Noise-only training must retain gradients with respect to noisy_flat,
        # even though all BrushNet parameters themselves remain frozen.
        brushnet_context = (
            torch.no_grad()
            if self.motion_adapter is not None
            else nullcontext()
        )
        with brushnet_context:
            down, mid, up = self.brushnet(
                noisy_flat,
                timestep_flat,
                encoder_hidden_states=text_flat,
                brushnet_cond=cond_flat,
                return_dict=False,
            )
        original_down = down
        original_mid = mid
        original_up = up
        if self.motion_adapter is not None:
            down, mid, up = self.motion_adapter(
                down,
                mid,
                up,
                backward_flow=flow_backward,
                confidence=motion_confidence,
                cfg_branches=1,
                scale=motion_scale,
            )
            regularization = (
                sum(
                    (new - old).square().mean()
                    for new, old in zip(down, original_down)
                )
                + (mid - original_mid).square().mean()
                + sum(
                    (new - old).square().mean()
                    for new, old in zip(up, original_up)
                )
            )
        else:
            regularization = noisy_latents.new_zeros(())

        kwargs = dict(unet_added_kwargs or {})
        prediction = self.unet(
            noisy_flat,
            timestep_flat,
            encoder_hidden_states=text_flat,
            down_block_add_samples=list(down),
            mid_block_add_sample=mid,
            up_block_add_samples=list(up),
            return_dict=False,
            **kwargs,
        )
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        return prediction.reshape(batch, frames, *prediction.shape[1:]), regularization
