"""Predicted-clean latent temporal loss for V8-R training.

Clean teacher flow supplies training-only correspondence.  The loss keeps the
current predicted-clean latent differentiable while, by default, treating the
previous latent and all geometric support tensors as stop-gradient targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from diffusers.models.stc_flow_training import resize_flow_sequence

try:
    from STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (
        backward_warp_feature,
    )
except ModuleNotFoundError:  # Imported as examples.brushnet.STC_encoder_v8_rescaled_deformable.
    from ..STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (
        backward_warp_feature,
    )


@dataclass
class TemporalTrainingLossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


def _canonical_teacher_valid(
    valid: torch.Tensor,
    *,
    batch: int,
    pairs: int,
    size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    if valid.ndim == 4:
        valid = valid.unsqueeze(2)
    if valid.ndim != 5 or tuple(valid.shape[:3]) != (batch, pairs, 1):
        raise ValueError(
            "teacher_valid_backward must have shape [B,T-1,1,H,W] or "
            f"[B,T-1,H,W], got {tuple(valid.shape)}"
        )
    valid = valid.detach().to(device=device, dtype=torch.float32)
    if valid.shape[-2:] != size:
        valid = F.interpolate(valid.flatten(0, 1), size=size, mode="nearest").reshape(
            batch, pairs, 1, *size
        )
    return (valid >= 0.5).float().detach()


def _masked_ratio(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    expanded = support.expand_as(numerator)
    numerator_sum = (numerator * expanded).sum()
    denominator_sum = (denominator * expanded).sum()
    # The numerator remains graph-connected when support is empty; metrics are
    # detached by the shared logging path after backward.
    return numerator_sum / denominator_sum.clamp_min(1e-6)


def compute_temporal_training_loss(
    *,
    model_prediction: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler,
    batch,
    stc_output,
    bg_mask_sequence: torch.Tensor,
    num_clips: int,
    num_frames: int,
    global_step: int,
    weight: float,
    warmup_steps: int,
    charbonnier_eps: float,
    detach_previous: bool,
    snr_gamma: float | None,
) -> TemporalTrainingLossOutput:
    """Compute the already-weighted previous-to-current temporal contribution."""
    del stc_output  # Kept in the hook contract for future correspondence ablations.

    if not math.isfinite(float(weight)) or weight < 0.0:
        raise ValueError("temporal training loss weight must be finite and non-negative")
    if warmup_steps < 0:
        raise ValueError("temporal training loss warmup steps must be non-negative")
    if not math.isfinite(float(charbonnier_eps)) or charbonnier_eps <= 0.0:
        raise ValueError("temporal training Charbonnier epsilon must be finite and positive")
    if snr_gamma is not None and (
        not math.isfinite(float(snr_gamma)) or snr_gamma <= 0.0
    ):
        raise ValueError("temporal training SNR gamma must be finite and positive")
    if num_clips < 1 or num_frames < 2:
        raise ValueError("temporal training loss requires B>=1 and T>=2")
    if model_prediction.shape != noisy_latents.shape:
        raise ValueError("model_prediction and noisy_latents must have identical shapes")
    expected_latents = num_clips * num_frames
    if (
        model_prediction.ndim != 4
        or model_prediction.shape[0] != expected_latents
        or model_prediction.shape[1] != 4
    ):
        raise ValueError(
            "model_prediction must have shape [B*T,4,H_lat,W_lat], got "
            f"{tuple(model_prediction.shape)}"
        )
    if not model_prediction.requires_grad:
        raise RuntimeError("model_prediction must retain gradients for temporal training")
    if timesteps.numel() != expected_latents:
        raise ValueError("timesteps must contain exactly B*T entries")

    device = model_prediction.device
    height, width = model_prediction.shape[-2:]
    clip_timesteps = timesteps.reshape(num_clips, num_frames)
    if not torch.all(clip_timesteps == clip_timesteps[:, :1]):
        raise ValueError(
            "Temporal training loss requires one shared diffusion timestep per clip"
        )

    alphas_cumprod = noise_scheduler.alphas_cumprod.to(
        device=device, dtype=torch.float32
    )
    alpha_bar = alphas_cumprod[timesteps.reshape(-1).long()].reshape(
        expected_latents, 1, 1, 1
    )
    sqrt_alpha = alpha_bar.clamp_min(1e-8).sqrt()
    sqrt_one_minus_alpha = (1.0 - alpha_bar).clamp_min(0.0).sqrt()
    prediction_type = noise_scheduler.config.prediction_type
    if prediction_type == "epsilon":
        predicted_clean = (
            noisy_latents.float() - sqrt_one_minus_alpha * model_prediction.float()
        ) / sqrt_alpha
    elif prediction_type == "v_prediction":
        predicted_clean = (
            sqrt_alpha * noisy_latents.float()
            - sqrt_one_minus_alpha * model_prediction.float()
        )
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")
    if predicted_clean.shape != noisy_latents.shape or not predicted_clean.requires_grad:
        raise RuntimeError("predicted-clean latent lost its expected shape or autograd graph")
    predicted_clean = predicted_clean.reshape(
        num_clips, num_frames, 4, height, width
    )

    pairs = num_frames - 1
    teacher_flow_rgb = batch["teacher_flow_backward"]
    if (
        teacher_flow_rgb.ndim != 5
        or tuple(teacher_flow_rgb.shape[:3]) != (num_clips, pairs, 2)
    ):
        raise ValueError(
            "teacher_flow_backward must have shape [B,T-1,2,H_rgb,W_rgb], got "
            f"{tuple(teacher_flow_rgb.shape)}"
        )
    teacher_flow_rgb = teacher_flow_rgb.detach().to(device=device, dtype=torch.float32)
    finite_rgb = torch.isfinite(teacher_flow_rgb).all(dim=2, keepdim=True)
    teacher_flow_rgb_safe = torch.nan_to_num(
        teacher_flow_rgb, nan=0.0, posinf=0.0, neginf=0.0
    ).detach()
    flow_backward = resize_flow_sequence(
        teacher_flow_rgb_safe, (height, width)
    ).detach()
    if tuple(flow_backward.shape) != (num_clips, pairs, 2, height, width):
        raise RuntimeError("resized teacher flow does not match latent resolution")

    teacher_valid_rgb = batch["teacher_valid_backward"]
    if teacher_valid_rgb.ndim == 4:
        teacher_valid_rgb = teacher_valid_rgb.unsqueeze(2)
    if (
        teacher_valid_rgb.ndim != 5
        or tuple(teacher_valid_rgb.shape[:3]) != (num_clips, pairs, 1)
        or teacher_valid_rgb.shape[-2:] != teacher_flow_rgb.shape[-2:]
    ):
        raise ValueError("teacher_valid_backward does not match teacher backward flow")
    teacher_valid_rgb = (
        teacher_valid_rgb.detach().to(device=device, dtype=torch.float32) >= 0.5
    ).float() * finite_rgb.float()
    teacher_valid = _canonical_teacher_valid(
        teacher_valid_rgb,
        batch=num_clips,
        pairs=pairs,
        size=(height, width),
        device=device,
    )

    previous = predicted_clean[:, :-1]
    current = predicted_clean[:, 1:]
    if detach_previous:
        previous = previous.detach()
    previous_flat = previous.reshape(-1, 4, height, width)
    current_flat = current.reshape(-1, 4, height, width)
    flow_flat = flow_backward.reshape(-1, 2, height, width)
    warped_previous, valid_bounds = backward_warp_feature(
        previous_flat,
        flow_flat,
        fallback=current_flat.detach(),
    )
    valid_bounds = valid_bounds.detach()

    if (
        bg_mask_sequence.ndim != 5
        or tuple(bg_mask_sequence.shape[:3]) != (num_clips, num_frames, 1)
    ):
        raise ValueError("bg_mask_sequence must have shape [B,T,1,H_rgb,W_rgb]")
    bg_latent = F.interpolate(
        bg_mask_sequence.detach().flatten(0, 1).to(
            device=device, dtype=torch.float32
        ),
        size=(height, width),
        mode="nearest",
    ).reshape(num_clips, num_frames, 1, height, width)
    bg_latent = (bg_latent >= 0.5).float().detach()
    previous_bg = bg_latent[:, :-1].reshape(-1, 1, height, width)
    current_bg = bg_latent[:, 1:].reshape(-1, 1, height, width)
    warped_previous_bg, bg_valid = backward_warp_feature(
        previous_bg, flow_flat, fallback=None
    )
    warped_previous_bg = (warped_previous_bg >= 0.5).float().detach()
    geometric_support = (
        current_bg
        * warped_previous_bg
        * valid_bounds.float()
        * bg_valid.detach().float()
    ).detach()
    teacher_valid_flat = teacher_valid.reshape(-1, 1, height, width)
    support = (geometric_support * teacher_valid_flat).detach()

    residual = current_flat - warped_previous
    charbonnier = (residual.float().square() + float(charbonnier_eps) ** 2).sqrt()
    support_count = support.sum()
    normalizer = (4.0 * support_count).clamp_min(1e-6)
    loss_raw = (charbonnier * support).sum() / normalizer

    clip_alpha = alphas_cumprod[clip_timesteps[:, 0].long()]
    snr = clip_alpha / (1.0 - clip_alpha).clamp_min(1e-8)
    if snr_gamma is None:
        clip_snr_weight = torch.ones_like(snr)
    else:
        clip_snr_weight = (snr / float(snr_gamma)).clamp(max=1.0)
    pair_snr_weight = clip_snr_weight[:, None, None, None, None].expand(
        num_clips, pairs, 1, height, width
    ).reshape(-1, 1, height, width)
    loss_snr_weighted = (charbonnier * support * pair_snr_weight).sum() / normalizer
    ramp = (
        1.0
        if warmup_steps <= 0
        else min(1.0, float(global_step + 1) / float(warmup_steps))
    )
    loss_weighted = float(weight) * ramp * loss_snr_weighted

    residual_squared = residual.float().square()
    current_squared = current_flat.float().square()
    relative_mse = _masked_ratio(residual_squared, current_squared, support)
    no_warp_residual = current_flat - previous_flat
    no_warp_relative_mse = _masked_ratio(
        no_warp_residual.float().square(), current_squared, support
    )
    warp_gain = no_warp_relative_mse - relative_mse

    support_per_clip = support.reshape(
        num_clips, pairs, 1, height, width
    ).sum(dim=(1, 2, 3, 4))
    snr_weight_mean = (
        (clip_snr_weight * support_per_clip).sum()
        / support_per_clip.sum().clamp_min(1e-6)
    )
    # Report the RGB-unit teacher magnitude only where the cached teacher says
    # correspondence is valid and the flow itself is finite.
    rgb_magnitude = teacher_flow_rgb_safe.square().sum(dim=2, keepdim=True).sqrt()
    rgb_valid_count = teacher_valid_rgb.sum().clamp_min(1e-6)
    rgb_magnitude_mean = (rgb_magnitude * teacher_valid_rgb).sum() / rgb_valid_count

    scalar_weight = loss_weighted.new_tensor(float(weight) * ramp)
    return TemporalTrainingLossOutput(
        loss=loss_weighted,
        metrics={
            "train/loss_temporal": loss_raw,
            "train/loss_temporal_weighted": loss_weighted,
            "train/temporal_effective_weight": scalar_weight * snr_weight_mean,
            "train/temporal_charbonnier": loss_raw,
            "train/temporal_relative_mse": relative_mse,
            "train/temporal_no_warp_relative_mse": no_warp_relative_mse,
            "train/temporal_warp_gain": warp_gain,
            "train/temporal_geometric_support_ratio": geometric_support.float().mean(),
            "train/temporal_teacher_valid_support_ratio": support.float().mean(),
            "train/temporal_flow_magnitude_teacher_rgb_px": rgb_magnitude_mean,
            "train/temporal_snr_weight": snr_weight_mean,
        },
    )
