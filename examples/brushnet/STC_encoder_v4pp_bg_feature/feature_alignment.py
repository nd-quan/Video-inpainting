"""Robust adjacent feature-alignment loss for V4++.

Predicted flow performs every warp.  Clean-teacher flow contributes only its
finite/in-bounds validity map, while the current-frame feature target is
stop-gradient.  This makes the loss train the flow head and the source spatial
feature without introducing a moving target on both sides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from diffusers.models.stc_flow_training import prepare_teacher_flow

try:
    from STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (
        backward_warp_feature,
    )
except ModuleNotFoundError:  # Imported as examples.brushnet.STC_encoder_v4pp_bg_feature.
    from ..STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (
        backward_warp_feature,
    )


@dataclass
class FeatureAlignmentLossOutput:
    loss: torch.Tensor
    loss_forward: torch.Tensor
    loss_backward: torch.Tensor
    valid_forward_ratio: torch.Tensor
    valid_backward_ratio: torch.Tensor
    confidence_mean: torch.Tensor


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1e-6)


def _feature_error(
    warped_source: torch.Tensor,
    detached_target: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return (
        (warped_source - detached_target)
        .float()
        .square()
        .mean(dim=1, keepdim=True)
        .add(float(eps) ** 2)
        .sqrt()
    )


def compute_feature_alignment_loss(
    spatial_features: torch.Tensor,
    predicted_forward: torch.Tensor,
    predicted_backward: torch.Tensor,
    teacher_forward: torch.Tensor,
    teacher_backward: torch.Tensor,
    bg_mask_sequence: torch.Tensor,
    valid_forward: Optional[torch.Tensor] = None,
    valid_backward: Optional[torch.Tensor] = None,
    alignment_confidence: Optional[torch.Tensor] = None,
    region: str = "bg",
    charbonnier_eps: float = 1e-3,
    confidence_floor: float = 0.1,
) -> FeatureAlignmentLossOutput:
    """Match raw adjacent STC features under predicted bidirectional flow."""
    if spatial_features.ndim != 5:
        raise ValueError("spatial_features must have shape [B,T,C,H,W]")
    batch, frames, channels, height, width = spatial_features.shape
    expected_flow = (batch, frames - 1, 2, height, width)
    if tuple(predicted_forward.shape) != expected_flow:
        raise ValueError(
            f"predicted_forward must have shape {expected_flow}, got "
            f"{tuple(predicted_forward.shape)}"
        )
    if predicted_backward.shape != predicted_forward.shape:
        raise ValueError("predicted forward/backward flow shapes must match")
    if bg_mask_sequence.ndim != 5 or bg_mask_sequence.shape[:3] != (
        batch,
        frames,
        1,
    ):
        raise ValueError("bg_mask_sequence must have shape [B,T,1,H,W]")
    if region not in {"bg", "all"}:
        raise ValueError("region must be 'bg' or 'all'")
    if charbonnier_eps <= 0.0:
        raise ValueError("charbonnier_eps must be positive")
    if not 0.0 <= confidence_floor <= 1.0:
        raise ValueError("confidence_floor must be in [0,1]")

    device = spatial_features.device
    features = F.normalize(spatial_features.float(), dim=2, eps=1e-6)
    forward = predicted_forward.float()
    backward = predicted_backward.float()
    pairs = frames - 1

    _, weight_forward = prepare_teacher_flow(
        teacher_forward.to(device=device, dtype=torch.float32),
        (height, width),
        None
        if valid_forward is None
        else valid_forward.to(device=device, dtype=torch.float32),
    )
    _, weight_backward = prepare_teacher_flow(
        teacher_backward.to(device=device, dtype=torch.float32),
        (height, width),
        None
        if valid_backward is None
        else valid_backward.to(device=device, dtype=torch.float32),
    )

    if region == "bg":
        bg = F.interpolate(
            bg_mask_sequence.flatten(0, 1).to(
                device=device, dtype=torch.float32
            ),
            size=(height, width),
            mode="nearest",
        ).reshape(batch, frames, 1, height, width)
        bg = (bg >= 0.5).float()
        weight_forward = weight_forward * bg[:, :-1]
        weight_backward = weight_backward * bg[:, 1:]

    previous = features[:, :-1].reshape(-1, channels, height, width)
    current = features[:, 1:].reshape(-1, channels, height, width)
    forward_flat = forward.reshape(-1, 2, height, width)
    backward_flat = backward.reshape(-1, 2, height, width)

    # No fallback=current: that would make predicted OOB samples have zero
    # feature error and create an escape route for the flow head.
    warped_previous, _ = backward_warp_feature(
        previous, backward_flat, fallback=None
    )
    warped_current, _ = backward_warp_feature(
        current, forward_flat, fallback=None
    )

    if alignment_confidence is None:
        confidence_backward = weight_backward.new_ones(weight_backward.shape)
    else:
        if tuple(alignment_confidence.shape) != (
            batch,
            pairs,
            1,
            height,
            width,
        ):
            raise ValueError("alignment_confidence has an unexpected shape")
        confidence_backward = alignment_confidence.detach().float().clamp(0.0, 1.0)
    confidence_forward_flat, _ = backward_warp_feature(
        confidence_backward.reshape(-1, 1, height, width),
        forward_flat.detach(),
        fallback=None,
    )
    confidence_forward = confidence_forward_flat.reshape(
        batch, pairs, 1, height, width
    ).detach().clamp(0.0, 1.0)

    reliability_backward = confidence_floor + (
        1.0 - confidence_floor
    ) * confidence_backward
    reliability_forward = confidence_floor + (
        1.0 - confidence_floor
    ) * confidence_forward
    weight_backward = weight_backward * reliability_backward
    weight_forward = weight_forward * reliability_forward

    error_backward = _feature_error(
        warped_previous,
        current.detach(),
        charbonnier_eps,
    ).reshape(batch, pairs, 1, height, width)
    error_forward = _feature_error(
        warped_current,
        previous.detach(),
        charbonnier_eps,
    ).reshape(batch, pairs, 1, height, width)
    loss_backward = _weighted_mean(error_backward, weight_backward)
    loss_forward = _weighted_mean(error_forward, weight_forward)
    confidence_mean = 0.5 * (
        confidence_backward.mean() + confidence_forward.mean()
    )
    return FeatureAlignmentLossOutput(
        loss=0.5 * (loss_forward + loss_backward),
        loss_forward=loss_forward,
        loss_backward=loss_backward,
        valid_forward_ratio=(weight_forward > 0).float().mean(),
        valid_backward_ratio=(weight_backward > 0).float().mean(),
        confidence_mean=confidence_mean,
    )
