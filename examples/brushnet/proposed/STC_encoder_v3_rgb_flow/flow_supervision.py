"""Clean-teacher flow supervision for RGB-STC features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from diffusers.models.stc_flow_training import (
    charbonnier_flow_loss,
    endpoint_error,
    prepare_teacher_flow,
)


@dataclass
class FlowLossOutput:
    loss: torch.Tensor
    loss_forward: torch.Tensor
    loss_backward: torch.Tensor
    epe: torch.Tensor
    epe_forward: torch.Tensor
    epe_backward: torch.Tensor
    valid_forward_ratio: torch.Tensor
    valid_backward_ratio: torch.Tensor
    predicted_magnitude: torch.Tensor


def compute_teacher_flow_loss(
    predicted_forward: torch.Tensor,
    predicted_backward: torch.Tensor,
    teacher_forward: torch.Tensor,
    teacher_backward: torch.Tensor,
    bg_mask_sequence: torch.Tensor,
    valid_forward: Optional[torch.Tensor] = None,
    valid_backward: Optional[torch.Tensor] = None,
    region: str = "bg",
    charbonnier_eps: float = 1e-3,
) -> FlowLossOutput:
    """Compute bidirectional robust teacher loss at prediction resolution.

    ``forward[:,t]`` lives on frame-t coordinates, whereas ``backward[:,t]``
    lives on frame-(t+1) coordinates. For BG-only supervision, each direction
    is therefore gated with the degraded mask on its own coordinate grid.
    """
    if predicted_forward.ndim != 5 or predicted_forward.shape[2] != 2:
        raise ValueError("predicted flow must have shape [B,T-1,2,H,W]")
    if predicted_forward.shape != predicted_backward.shape:
        raise ValueError("predicted forward/backward shapes must match")
    batch, pairs, _, height, width = predicted_forward.shape
    if bg_mask_sequence.ndim != 5 or bg_mask_sequence.shape[:3] != (
        batch,
        pairs + 1,
        1,
    ):
        raise ValueError("bg_mask_sequence must have shape [B,T,1,H,W]")
    if region not in {"bg", "all"}:
        raise ValueError("flow region must be 'bg' or 'all'")

    teacher_forward, weight_forward = prepare_teacher_flow(
        teacher_forward.to(
            device=predicted_forward.device, dtype=torch.float32
        ),
        (height, width),
        None if valid_forward is None else valid_forward.to(
            device=predicted_forward.device, dtype=torch.float32
        ),
    )
    teacher_backward, weight_backward = prepare_teacher_flow(
        teacher_backward.to(
            device=predicted_backward.device, dtype=torch.float32
        ),
        (height, width),
        None if valid_backward is None else valid_backward.to(
            device=predicted_backward.device, dtype=torch.float32
        ),
    )
    predicted_forward = predicted_forward.float()
    predicted_backward = predicted_backward.float()

    if region == "bg":
        bg = F.interpolate(
            bg_mask_sequence.flatten(0, 1).to(
                device=predicted_forward.device, dtype=torch.float32
            ),
            size=(height, width),
            mode="nearest",
        ).reshape(batch, pairs + 1, 1, height, width)
        weight_forward = weight_forward * bg[:, :-1]
        weight_backward = weight_backward * bg[:, 1:]

    loss_forward = charbonnier_flow_loss(
        predicted_forward,
        teacher_forward,
        weight_forward,
        eps=float(charbonnier_eps),
    )
    loss_backward = charbonnier_flow_loss(
        predicted_backward,
        teacher_backward,
        weight_backward,
        eps=float(charbonnier_eps),
    )
    epe_forward = endpoint_error(
        predicted_forward, teacher_forward, weight_forward
    )
    epe_backward = endpoint_error(
        predicted_backward, teacher_backward, weight_backward
    )
    magnitude = 0.5 * (
        predicted_forward.float().square().sum(2).sqrt().mean()
        + predicted_backward.float().square().sum(2).sqrt().mean()
    )
    return FlowLossOutput(
        loss=0.5 * (loss_forward + loss_backward),
        loss_forward=loss_forward,
        loss_backward=loss_backward,
        epe=0.5 * (epe_forward + epe_backward),
        epe_forward=epe_forward,
        epe_backward=epe_backward,
        valid_forward_ratio=weight_forward.mean(),
        valid_backward_ratio=weight_backward.mean(),
        predicted_magnitude=magnitude,
    )
