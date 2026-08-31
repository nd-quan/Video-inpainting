"""Training losses for V6 flow-guided deformable feature alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from diffusers.models.stc_flow_training import prepare_teacher_flow


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1e-6)


def _feature_error(
    candidate: torch.Tensor,
    detached_target: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return (
        (candidate - detached_target)
        .float()
        .square()
        .mean(dim=2, keepdim=True)
        .add(float(eps) ** 2)
        .sqrt()
    )


@dataclass
class DeformableAlignmentLossOutput:
    loss: torch.Tensor
    loss_forward: torch.Tensor
    loss_backward: torch.Tensor
    loss_offset: torch.Tensor
    loss_offset_forward: torch.Tensor
    loss_offset_backward: torch.Tensor
    valid_forward_ratio: torch.Tensor
    valid_backward_ratio: torch.Tensor
    reliability_forward_mean: torch.Tensor
    reliability_backward_mean: torch.Tensor


@dataclass
class WeightedV6ExtraLossOutput:
    loss: torch.Tensor
    metrics: Dict[str, torch.Tensor]


def compute_deformable_alignment_loss(
    spatial_features: torch.Tensor,
    deformed_previous: torch.Tensor,
    deformed_next: torch.Tensor,
    residual_offset_backward: torch.Tensor,
    residual_offset_forward: torch.Tensor,
    reliability_backward: torch.Tensor,
    reliability_forward: torch.Tensor,
    teacher_forward: torch.Tensor,
    teacher_backward: torch.Tensor,
    valid_forward: Optional[torch.Tensor] = None,
    valid_backward: Optional[torch.Tensor] = None,
    charbonnier_eps: float = 1e-3,
) -> DeformableAlignmentLossOutput:
    """Supervise raw DCN candidates before V6's zero-init residual fusion.

    ``deformed_previous`` maps frame ``t-1 -> t`` and is compared with
    ``spatial[:, 1:]``.  ``deformed_next`` maps ``t+1 -> t`` (the forward
    field of each adjacent pair) and is compared with ``spatial[:, :-1]``.
    Teacher flow contributes only its finite/in-bounds validity here.
    """
    if spatial_features.ndim != 5:
        raise ValueError("spatial_features must have shape [B,T,C,H,W]")
    batch, frames, channels, height, width = spatial_features.shape
    pairs = frames - 1
    expected_feature = (batch, pairs, channels, height, width)
    expected_weight = (batch, pairs, 1, height, width)
    for name, value in (
        ("deformed_previous", deformed_previous),
        ("deformed_next", deformed_next),
    ):
        if tuple(value.shape) != expected_feature:
            raise ValueError(f"{name} must have shape {expected_feature}")
    for name, value in (
        ("reliability_backward", reliability_backward),
        ("reliability_forward", reliability_forward),
    ):
        if tuple(value.shape) != expected_weight:
            raise ValueError(f"{name} must have shape {expected_weight}")
    if residual_offset_backward.ndim != 5:
        raise ValueError("residual offsets must have shape [B,T-1,O,H,W]")
    if residual_offset_backward.shape != residual_offset_forward.shape:
        raise ValueError("forward/backward residual offset shapes must match")
    if residual_offset_backward.shape[:2] != (batch, pairs) or tuple(
        residual_offset_backward.shape[-2:]
    ) != (height, width):
        raise ValueError("residual offsets do not match spatial feature layout")
    if charbonnier_eps <= 0.0:
        raise ValueError("charbonnier_eps must be positive")

    if pairs == 0:
        zero = spatial_features.sum() * 0.0
        return DeformableAlignmentLossOutput(
            loss=zero,
            loss_forward=zero,
            loss_backward=zero,
            loss_offset=zero,
            loss_offset_forward=zero,
            loss_offset_backward=zero,
            valid_forward_ratio=zero.detach(),
            valid_backward_ratio=zero.detach(),
            reliability_forward_mean=zero.detach(),
            reliability_backward_mean=zero.detach(),
        )

    device = spatial_features.device
    _, teacher_weight_forward = prepare_teacher_flow(
        teacher_forward.to(device=device, dtype=torch.float32),
        (height, width),
        None
        if valid_forward is None
        else valid_forward.to(device=device, dtype=torch.float32),
    )
    _, teacher_weight_backward = prepare_teacher_flow(
        teacher_backward.to(device=device, dtype=torch.float32),
        (height, width),
        None
        if valid_backward is None
        else valid_backward.to(device=device, dtype=torch.float32),
    )
    reliability_backward = reliability_backward.detach().float().clamp(0.0, 1.0)
    reliability_forward = reliability_forward.detach().float().clamp(0.0, 1.0)
    weight_backward = teacher_weight_backward * reliability_backward
    weight_forward = teacher_weight_forward * reliability_forward

    targets = F.normalize(spatial_features.float(), dim=2, eps=1e-6).detach()
    previous_candidate = F.normalize(deformed_previous.float(), dim=2, eps=1e-6)
    next_candidate = F.normalize(deformed_next.float(), dim=2, eps=1e-6)
    error_backward = _feature_error(
        previous_candidate, targets[:, 1:], charbonnier_eps
    )
    error_forward = _feature_error(
        next_candidate, targets[:, :-1], charbonnier_eps
    )
    loss_backward = _weighted_mean(error_backward, weight_backward)
    loss_forward = _weighted_mean(error_forward, weight_forward)

    # Regularize the actual bounded DCN residual in feature pixels, not logits.
    offset_error_backward = residual_offset_backward.float().abs().mean(
        dim=2, keepdim=True
    )
    offset_error_forward = residual_offset_forward.float().abs().mean(
        dim=2, keepdim=True
    )
    offset_backward = _weighted_mean(offset_error_backward, weight_backward)
    offset_forward = _weighted_mean(offset_error_forward, weight_forward)
    return DeformableAlignmentLossOutput(
        loss=0.5 * (loss_forward + loss_backward),
        loss_forward=loss_forward,
        loss_backward=loss_backward,
        loss_offset=0.5 * (offset_forward + offset_backward),
        loss_offset_forward=offset_forward,
        loss_offset_backward=offset_backward,
        valid_forward_ratio=(weight_forward > 0).float().mean(),
        valid_backward_ratio=(weight_backward > 0).float().mean(),
        reliability_forward_mean=reliability_forward.mean(),
        reliability_backward_mean=reliability_backward.mean(),
    )


def build_v6_extra_train_loss(
    *,
    stc_output,
    batch,
    bg_mask_sequence,
    args,
    global_step: int,
) -> WeightedV6ExtraLossOutput:
    """Trainer-hook adapter returning the already weighted V6 loss."""
    del bg_mask_sequence  # Reliability already contains target/source BG support.
    output = compute_deformable_alignment_loss(
        spatial_features=stc_output.spatial_features,
        deformed_previous=stc_output.deformed_previous_features,
        deformed_next=stc_output.deformed_next_features,
        residual_offset_backward=stc_output.residual_offset_backward,
        residual_offset_forward=stc_output.residual_offset_forward,
        reliability_backward=stc_output.deform_reliability_backward,
        reliability_forward=stc_output.deform_reliability_forward,
        teacher_forward=batch["teacher_flow_forward"],
        teacher_backward=batch["teacher_flow_backward"],
        valid_forward=batch["teacher_valid_forward"],
        valid_backward=batch["teacher_valid_backward"],
        charbonnier_eps=args.deform_alignment_charbonnier_eps,
    )
    if args.deform_alignment_warmup_steps > 0:
        ramp = min(
            1.0,
            float(global_step + 1) / float(args.deform_alignment_warmup_steps),
        )
    else:
        ramp = 1.0
    deform_weight = float(args.deform_alignment_loss_weight) * ramp
    offset_weight = float(args.deform_offset_loss_weight)
    weighted_deform = deform_weight * output.loss
    weighted_offset = offset_weight * output.loss_offset
    total = weighted_deform + weighted_offset
    return WeightedV6ExtraLossOutput(
        loss=total,
        metrics={
            "train/deform_alignment_effective_weight": total.new_tensor(
                deform_weight
            ),
            "train/deform_offset_effective_weight": total.new_tensor(offset_weight),
            "train/loss_deform_alignment": output.loss,
            "train/loss_deform_alignment_weighted": weighted_deform,
            "train/loss_deform_backward": output.loss_backward,
            "train/loss_deform_forward": output.loss_forward,
            "train/loss_deform_offset": output.loss_offset,
            "train/loss_deform_offset_weighted": weighted_offset,
            "train/loss_deform_offset_backward": output.loss_offset_backward,
            "train/loss_deform_offset_forward": output.loss_offset_forward,
            "train/deform_valid_backward": output.valid_backward_ratio,
            "train/deform_valid_forward": output.valid_forward_ratio,
            "train/deform_reliability_backward": output.reliability_backward_mean,
            "train/deform_reliability_forward": output.reliability_forward_mean,
        },
    )
