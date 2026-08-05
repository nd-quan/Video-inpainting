"""Losses and checkpoint helpers for Stage-1 STC flow prediction.

Flow convention
---------------
``flow_forward[:, i]`` is defined on frame ``i`` and points to frame ``i+1``.
``flow_backward[:, i]`` is defined on frame ``i+1`` and points to frame ``i``.
All flow values are pixel displacements at the tensor's own spatial resolution.

The utilities deliberately do not depend on the diffusion loss.  Stage 1 first
teaches the STC encoder a geometrically meaningful motion field from a clean
teacher-flow cache; noise shaping is trained/evaluated only after this stage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .brushnet_motion_adapter import backward_warp_feature, resize_flow


def _require_flow(name: str, flow: torch.Tensor) -> Tuple[int, int, int, int, int]:
    if flow.ndim != 5 or flow.shape[2] != 2:
        raise ValueError(f"{name} must have shape [B,T-1,2,H,W], got {tuple(flow.shape)}")
    return tuple(flow.shape)  # type: ignore[return-value]


def resize_flow_sequence(flow: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Resize ``[B,P,2,H,W]`` flow and scale its pixel displacements."""
    batch, pairs, _, _, _ = _require_flow("flow", flow)
    if flow.shape[-2:] == tuple(size):
        return flow
    resized = resize_flow(flow.flatten(0, 1), size)
    return resized.reshape(batch, pairs, 2, *size)


def _canonical_valid(
    valid: Optional[torch.Tensor],
    batch: int,
    pairs: int,
    size: Tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if valid is None:
        return torch.ones(batch, pairs, 1, *size, device=device, dtype=dtype)
    if valid.ndim == 4:
        valid = valid.unsqueeze(2)
    if valid.ndim != 5 or valid.shape[:3] != (batch, pairs, 1):
        raise ValueError(
            "valid mask must have shape [B,T-1,1,H,W] or [B,T-1,H,W], "
            f"got {tuple(valid.shape)}"
        )
    valid = valid.to(device=device, dtype=dtype)
    if valid.shape[-2:] != size:
        valid = F.interpolate(
            valid.flatten(0, 1), size=size, mode="nearest"
        ).reshape(batch, pairs, 1, *size)
    return valid.clamp(0.0, 1.0)


def flow_in_bounds(flow: torch.Tensor) -> torch.Tensor:
    """Return a finite/in-bounds validity mask for a pixel-space flow."""
    batch, pairs, _, height, width = _require_flow("flow", flow)
    finite = torch.isfinite(flow).all(dim=2, keepdim=True)
    safe = torch.nan_to_num(flow)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    sample_x = xx[None, None] + safe[:, :, 0]
    sample_y = yy[None, None] + safe[:, :, 1]
    inside = (
        (sample_x >= 0.0)
        & (sample_x <= max(width - 1, 0))
        & (sample_y >= 0.0)
        & (sample_y <= max(height - 1, 0))
    )[:, :, None]
    return (finite & inside).to(flow.dtype).reshape(batch, pairs, 1, height, width)


def prepare_teacher_flow(
    teacher: torch.Tensor,
    size: Tuple[int, int],
    valid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resize teacher flow and combine supplied, finite, and in-bounds masks."""
    batch, pairs, _, _, _ = _require_flow("teacher", teacher)
    source_size = teacher.shape[-2:]
    finite = torch.isfinite(teacher).all(dim=2, keepdim=True).to(teacher.dtype)
    supplied = _canonical_valid(
        valid,
        batch,
        pairs,
        source_size,
        device=teacher.device,
        dtype=teacher.dtype,
    ) * finite
    teacher = torch.nan_to_num(teacher, nan=0.0, posinf=0.0, neginf=0.0)
    teacher = resize_flow_sequence(teacher, size)
    supplied = _canonical_valid(
        supplied,
        batch,
        pairs,
        size,
        device=teacher.device,
        dtype=teacher.dtype,
    )
    geometric = flow_in_bounds(teacher)
    return teacher, supplied * geometric


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim == value.ndim - 1:
        weight = weight.unsqueeze(2)
    return (value * weight).sum() / weight.sum().clamp_min(1e-6)


def endpoint_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean endpoint error in pixels."""
    if predicted.shape != target.shape:
        raise ValueError("predicted and target flow must have identical shapes")
    batch, pairs, _, height, width = _require_flow("predicted", predicted)
    valid = _canonical_valid(
        valid,
        batch,
        pairs,
        (height, width),
        device=predicted.device,
        dtype=predicted.dtype,
    )
    error = (predicted - target).float().square().sum(dim=2, keepdim=True).sqrt()
    return _weighted_mean(error, valid.float())


def charbonnier_flow_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Robust vector-flow loss, averaged over valid pixels."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    if predicted.shape != target.shape:
        raise ValueError("predicted and target flow must have identical shapes")
    batch, pairs, _, height, width = _require_flow("predicted", predicted)
    valid = _canonical_valid(
        valid,
        batch,
        pairs,
        (height, width),
        device=predicted.device,
        dtype=predicted.dtype,
    )
    error = ((predicted - target).float().square().sum(dim=2, keepdim=True) + eps**2).sqrt()
    return _weighted_mean(error, valid.float())


def _warp_sequence(feature: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp ``[B,P,C,H,W]`` with matching ``[B,P,2,H,W]`` flow."""
    if feature.ndim != 5:
        raise ValueError("feature must have shape [B,T-1,C,H,W]")
    batch, pairs, _, height, width = _require_flow("flow", flow)
    if feature.shape[:2] != (batch, pairs):
        raise ValueError("feature and flow must have the same B,T-1 dimensions")
    channels = feature.shape[2]
    warped = backward_warp_feature(
        feature.reshape(-1, channels, *feature.shape[-2:]),
        flow.reshape(-1, 2, height, width),
    )
    return warped.reshape(batch, pairs, channels, height, width)


def forward_backward_consistency_loss(
    flow_forward: torch.Tensor,
    flow_backward: torch.Tensor,
    valid_forward: Optional[torch.Tensor] = None,
    valid_backward: Optional[torch.Tensor] = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Symmetric adjacent-pair forward/backward cycle consistency.

    At current-frame coordinates the residual is
    ``B + warp(F, B)``; at previous-frame coordinates it is
    ``F + warp(B, F)``.
    """
    if flow_forward.shape != flow_backward.shape:
        raise ValueError("forward and backward flow must have identical shapes")
    batch, pairs, _, height, width = _require_flow("flow_forward", flow_forward)
    size = (height, width)
    vf = _canonical_valid(
        valid_forward,
        batch,
        pairs,
        size,
        device=flow_forward.device,
        dtype=flow_forward.dtype,
    ) * flow_in_bounds(flow_forward)
    vb = _canonical_valid(
        valid_backward,
        batch,
        pairs,
        size,
        device=flow_backward.device,
        dtype=flow_backward.dtype,
    ) * flow_in_bounds(flow_backward)

    warped_forward = _warp_sequence(flow_forward, flow_backward)
    warped_vf = _warp_sequence(vf, flow_backward).clamp(0.0, 1.0)
    residual_backward = (
        (flow_backward + warped_forward).float().square().sum(dim=2, keepdim=True)
        + eps**2
    ).sqrt()
    loss_backward = _weighted_mean(residual_backward, vb * warped_vf)

    warped_backward = _warp_sequence(flow_backward, flow_forward)
    warped_vb = _warp_sequence(vb, flow_forward).clamp(0.0, 1.0)
    residual_forward = (
        (flow_forward + warped_backward).float().square().sum(dim=2, keepdim=True)
        + eps**2
    ).sqrt()
    loss_forward = _weighted_mean(residual_forward, vf * warped_vb)
    return 0.5 * (loss_forward + loss_backward)


def edge_aware_smoothness_loss(
    flow: torch.Tensor,
    reference_frames: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    edge_weight: float = 10.0,
) -> torch.Tensor:
    """First-order flow smoothness, relaxed across image edges.

    ``reference_frames`` is the frame on whose coordinates the flow is defined:
    previous frames for forward flow and current frames for backward flow.
    """
    batch, pairs, _, height, width = _require_flow("flow", flow)
    if reference_frames.ndim != 5 or reference_frames.shape[:2] != (batch, pairs):
        raise ValueError("reference_frames must have shape [B,T-1,C,H,W]")
    images = F.interpolate(
        reference_frames.flatten(0, 1).float(),
        size=(height, width),
        mode="bilinear",
        align_corners=True,
    ).reshape(batch, pairs, reference_frames.shape[2], height, width)
    valid = _canonical_valid(
        valid,
        batch,
        pairs,
        (height, width),
        device=flow.device,
        dtype=flow.dtype,
    )

    flow_x = flow[..., :, 1:] - flow[..., :, :-1]
    flow_y = flow[..., 1:, :] - flow[..., :-1, :]
    image_x = (images[..., :, 1:] - images[..., :, :-1]).abs().mean(2, keepdim=True)
    image_y = (images[..., 1:, :] - images[..., :-1, :]).abs().mean(2, keepdim=True)
    weight_x = torch.exp(-float(edge_weight) * image_x)
    weight_y = torch.exp(-float(edge_weight) * image_y)
    valid_x = valid[..., :, 1:] * valid[..., :, :-1]
    valid_y = valid[..., 1:, :] * valid[..., :-1, :]
    loss_x = _weighted_mean(flow_x.abs().mean(2, keepdim=True), weight_x * valid_x)
    loss_y = _weighted_mean(flow_y.abs().mean(2, keepdim=True), weight_y * valid_y)
    return 0.5 * (loss_x + loss_y)


def compute_flow_training_losses(
    predicted_forward: torch.Tensor,
    predicted_backward: torch.Tensor,
    teacher_forward: torch.Tensor,
    teacher_backward: torch.Tensor,
    decoded_frames: torch.Tensor,
    valid_forward: Optional[torch.Tensor] = None,
    valid_backward: Optional[torch.Tensor] = None,
    loss_config: Optional[Mapping[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute Stage-1 supervision and metrics at prediction resolution."""
    config = dict(loss_config or {})
    if predicted_forward.shape != predicted_backward.shape:
        raise ValueError("predicted forward/backward flow shapes must match")
    batch, pairs, _, height, width = _require_flow(
        "predicted_forward", predicted_forward
    )
    if decoded_frames.ndim != 5 or decoded_frames.shape[:2] != (batch, pairs + 1):
        raise ValueError("decoded_frames must have shape [B,T,C,H,W]")
    teacher_forward, valid_forward = prepare_teacher_flow(
        teacher_forward.to(predicted_forward),
        (height, width),
        valid_forward,
    )
    teacher_backward, valid_backward = prepare_teacher_flow(
        teacher_backward.to(predicted_backward),
        (height, width),
        valid_backward,
    )
    if not bool((valid_forward.sum() + valid_backward.sum()).detach() > 0):
        raise ValueError("Teacher cache has no finite in-bounds flow pixels")
    eps = float(config.get("charbonnier_eps", 1e-3))
    forward_teacher = charbonnier_flow_loss(
        predicted_forward, teacher_forward, valid_forward, eps
    )
    backward_teacher = charbonnier_flow_loss(
        predicted_backward, teacher_backward, valid_backward, eps
    )
    teacher_loss = 0.5 * (forward_teacher + backward_teacher)
    forward_epe = endpoint_error(predicted_forward, teacher_forward, valid_forward)
    backward_epe = endpoint_error(predicted_backward, teacher_backward, valid_backward)
    epe = 0.5 * (forward_epe + backward_epe)
    fb = forward_backward_consistency_loss(
        predicted_forward,
        predicted_backward,
        valid_forward,
        valid_backward,
        eps=eps,
    )
    smooth_forward = edge_aware_smoothness_loss(
        predicted_forward,
        decoded_frames[:, :-1],
        valid_forward,
        edge_weight=float(config.get("smoothness_edge_weight", 10.0)),
    )
    smooth_backward = edge_aware_smoothness_loss(
        predicted_backward,
        decoded_frames[:, 1:],
        valid_backward,
        edge_weight=float(config.get("smoothness_edge_weight", 10.0)),
    )
    smooth = 0.5 * (smooth_forward + smooth_backward)
    total = (
        float(config.get("teacher_weight", 1.0)) * teacher_loss
        + float(config.get("fb_weight", 0.1)) * fb
        + float(config.get("smoothness_weight", 0.01)) * smooth
    )
    return {
        "total": total,
        "teacher": teacher_loss,
        "teacher_forward": forward_teacher,
        "teacher_backward": backward_teacher,
        "epe": epe,
        "epe_forward": forward_epe,
        "epe_backward": backward_epe,
        "fb": fb,
        "smoothness": smooth,
        "valid_forward": valid_forward.mean(),
        "valid_backward": valid_backward.mean(),
    }


def unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def save_stage1_checkpoint(
    checkpoint_root: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    step: int,
    best_valid_epe: float,
    config: Mapping,
    *,
    is_best: bool = False,
    extra_state: Optional[Mapping] = None,
) -> Path:
    """Save model plus resumable trainer state and update JSON pointers."""
    checkpoint_root = Path(checkpoint_root)
    destination = checkpoint_root / f"checkpoint-{int(step):07d}"
    destination.mkdir(parents=True, exist_ok=True)
    bare_model = unwrap_module(model)
    model_dir = destination / "flow_predictor"
    if hasattr(bare_model, "save_pretrained"):
        bare_model.save_pretrained(model_dir)
        model_format = "pretrained"
    else:
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(bare_model.state_dict(), model_dir / "pytorch_model.bin")
        model_format = "state_dict"
    trainer_state = {
        "step": int(step),
        "best_valid_epe": float(best_valid_epe),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "model_format": model_format,
    }
    trainer_state.update(dict(extra_state or {}))
    torch.save(trainer_state, destination / "trainer_state.pt")
    (destination / "config.json").write_text(
        json.dumps(dict(config), indent=2), encoding="utf-8"
    )
    pointer = {"checkpoint": str(destination), "step": int(step)}
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (checkpoint_root / "latest.json").write_text(
        json.dumps(pointer, indent=2), encoding="utf-8"
    )
    if is_best:
        pointer["valid_epe"] = float(best_valid_epe)
        (checkpoint_root / "best.json").write_text(
            json.dumps(pointer, indent=2), encoding="utf-8"
        )
    return destination
