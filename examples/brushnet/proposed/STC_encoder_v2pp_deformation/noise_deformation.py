"""Core STC-v2++ NoiseDeformationHead and full-frame noise transport.

This module deliberately contains no dataset, diffusion pipeline, or mutable
sequence-level inference state.  It implements the pure Phase-D1 operations so
their coordinate, gradient, and Gaussian-statistics contracts can be tested in
isolation before the STC-v2 trainer is forked.

Offsets are backward sampling displacements in latent-pixel units.  At target
pixel ``q``, ``offset(q)`` selects reference location ``q + offset(q)``.  Every
warp uses a pixel-center grid with ``align_corners=False``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(int(maximum), int(channels)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class NoiseDeformationHead(ModelMixin, ConfigMixin):
    """Predict a bounded dense backward offset from an ordered feature pair.

    Input construction is fixed to ``[h_reference, h_target,
    h_target-h_reference]``.  The zero-initialized final layer makes a new head
    an exact identity deformation while still allowing its final layer to learn
    on the first optimizer step.
    """

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        feature_channels: int = 64,
        hidden_channels: int = 128,
        num_hidden_layers: int = 3,
        max_displacement: float = 8.0,
    ):
        super().__init__()
        feature_channels = int(feature_channels)
        hidden_channels = int(hidden_channels)
        num_hidden_layers = int(num_hidden_layers)
        max_displacement = float(max_displacement)
        if feature_channels < 1 or hidden_channels < 2:
            raise ValueError("feature_channels and hidden_channels must be positive")
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least one")
        if not math.isfinite(max_displacement) or max_displacement <= 0.0:
            raise ValueError("max_displacement must be finite and positive")

        layers = []
        in_channels = 3 * feature_channels
        for _ in range(num_hidden_layers):
            layers.extend(
                (
                    nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
                    nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
                    nn.SiLU(),
                )
            )
            in_channels = hidden_channels
        self.hidden = nn.Sequential(*layers)
        self.to_offset = nn.Conv2d(hidden_channels, 2, 3, padding=1)
        nn.init.zeros_(self.to_offset.weight)
        nn.init.zeros_(self.to_offset.bias)

    def forward(
        self,
        reference_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        if reference_features.ndim != 4:
            raise ValueError("features must have shape [B,C,H,W]")
        if reference_features.shape != target_features.shape:
            raise ValueError("reference and target features must have equal shapes")
        if reference_features.shape[1] != int(self.config.feature_channels):
            raise ValueError(
                "feature channel mismatch: expected "
                f"{self.config.feature_channels}, got {reference_features.shape[1]}"
            )
        pair = torch.cat(
            (
                reference_features,
                target_features,
                target_features - reference_features,
            ),
            dim=1,
        )
        raw_offset = self.to_offset(self.hidden(pair))
        return torch.tanh(raw_offset) * float(self.config.max_displacement)


def make_backward_sampling_grid(offset: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert latent-pixel offsets to an ``align_corners=False`` grid.

    Returns ``(grid, valid_mask)`` with shapes ``[B,H,W,2]`` and
    ``[B,1,H,W]``.  A location is valid only when its sampled pixel center lies
    in ``[0, W-1] x [0, H-1]``; partially padded bilinear samples are rejected.
    """
    if offset.ndim != 4 or offset.shape[1] != 2:
        raise ValueError("offset must have shape [B,2,H,W]")
    if not offset.is_floating_point():
        raise TypeError("offset must be floating point")
    batch, _, height, width = offset.shape
    if height < 1 or width < 1:
        raise ValueError("offset spatial dimensions must be positive")

    y = torch.arange(height, device=offset.device, dtype=offset.dtype)
    x = torch.arange(width, device=offset.device, dtype=offset.dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    source_x = grid_x.unsqueeze(0) + offset[:, 0]
    source_y = grid_y.unsqueeze(0) + offset[:, 1]
    valid = (
        (source_x >= 0.0)
        & (source_x <= float(width - 1))
        & (source_y >= 0.0)
        & (source_y <= float(height - 1))
    ).unsqueeze(1)

    normalized_x = 2.0 * (source_x + 0.5) / float(width) - 1.0
    normalized_y = 2.0 * (source_y + 0.5) / float(height) - 1.0
    grid = torch.stack((normalized_x, normalized_y), dim=-1)
    if grid.shape != (batch, height, width, 2):
        raise RuntimeError("internal sampling-grid shape error")
    return grid, valid


def backward_warp(
    reference: torch.Tensor,
    offset: torch.Tensor,
    fallback: Optional[torch.Tensor] = None,
    mode: str = "bilinear",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward-warp ``reference`` and optionally fill invalid coordinates."""
    if reference.ndim != 4:
        raise ValueError("reference must have shape [B,C,H,W]")
    if reference.shape[0] != offset.shape[0] or reference.shape[-2:] != offset.shape[-2:]:
        raise ValueError("reference and offset batch/spatial shapes must match")
    if fallback is not None and fallback.shape != reference.shape:
        raise ValueError("fallback must have the same shape as reference")
    if mode not in {"bilinear", "nearest"}:
        raise ValueError("mode must be 'bilinear' or 'nearest'")

    grid, valid = make_backward_sampling_grid(offset)
    warped = F.grid_sample(
        reference,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=False,
    )
    if fallback is not None:
        warped = torch.where(valid, warped, fallback)
    return warped, valid


def normalize_lineage_channels(
    lineage: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize each ``[B,C]`` lineage channel over its spatial dimensions."""
    if lineage.ndim != 4:
        raise ValueError("lineage must have shape [B,C,H,W]")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    if lineage.shape[-2] * lineage.shape[-1] < 2:
        raise ValueError("lineage normalization requires at least two pixels")
    mean = lineage.mean(dim=(-2, -1), keepdim=True)
    variance = (lineage - mean).square().mean(dim=(-2, -1), keepdim=True)
    return (lineage - mean) / (variance.sqrt() + eps)


def variance_preserving_fusion(
    lineage: torch.Tensor,
    independent: torch.Tensor,
    alpha: float,
    warp_scope: str = "full",
    bg_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fuse lineage and independent noise using the existing V8 law.

    ``alpha`` is the correlated-variance fraction, not the direct amplitude.
    For ``warp_scope='full'`` the mask is deliberately ignored.  In the
    ``'bg'`` ablation, ROI remains exactly equal to ``independent``.
    """
    if lineage.shape != independent.shape or lineage.ndim != 4:
        raise ValueError("lineage and independent must share shape [B,C,H,W]")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    if warp_scope not in {"full", "bg"}:
        raise ValueError("warp_scope must be 'full' or 'bg'")

    if alpha == 0.0:
        fused = independent
    elif alpha == 1.0:
        fused = lineage
    else:
        fused = math.sqrt(alpha) * lineage + math.sqrt(1.0 - alpha) * independent

    if warp_scope == "full":
        return fused
    if bg_mask is None:
        raise ValueError("bg_mask is required for warp_scope='bg'")
    expected = (lineage.shape[0], 1, lineage.shape[2], lineage.shape[3])
    if bg_mask.shape != expected:
        raise ValueError(f"bg_mask must have shape {expected}")
    mask = bg_mask.to(device=lineage.device, dtype=lineage.dtype)
    return fused * mask + independent * (1.0 - mask)


def cosine_feature_matching_loss(
    warped_reference: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean ``1-cosine`` over valid target locations."""
    if warped_reference.shape != target.shape or target.ndim != 4:
        raise ValueError("feature tensors must share shape [B,C,H,W]")
    expected = (target.shape[0], 1, target.shape[2], target.shape[3])
    if valid_mask.shape != expected:
        raise ValueError(f"valid_mask must have shape {expected}")
    mismatch = 1.0 - F.cosine_similarity(
        warped_reference, target, dim=1, eps=float(eps)
    )
    weights = valid_mask[:, 0].to(dtype=mismatch.dtype)
    denominator = weights.sum().clamp_min(1.0)
    return (mismatch * weights).sum() / denominator


def edge_aware_smoothness_loss(
    offset: torch.Tensor,
    target_features: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """First-order offset smoothness weighted by frozen target-feature edges."""
    if offset.ndim != 4 or offset.shape[1] != 2:
        raise ValueError("offset must have shape [B,2,H,W]")
    if target_features.ndim != 4:
        raise ValueError("target_features must have shape [B,C,H,W]")
    if offset.shape[0] != target_features.shape[0] or offset.shape[-2:] != target_features.shape[-2:]:
        raise ValueError("offset and target feature batch/spatial shapes must match")
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and non-negative")

    terms = []
    if offset.shape[-1] > 1:
        offset_dx = offset[..., 1:] - offset[..., :-1]
        feature_dx = target_features[..., 1:] - target_features[..., :-1]
        weight_x = torch.exp(-gamma * feature_dx.square().sum(dim=1).sqrt())
        terms.append((offset_dx.abs() * weight_x.unsqueeze(1)).mean())
    if offset.shape[-2] > 1:
        offset_dy = offset[..., 1:, :] - offset[..., :-1, :]
        feature_dy = target_features[..., 1:, :] - target_features[..., :-1, :]
        weight_y = torch.exp(-gamma * feature_dy.square().sum(dim=1).sqrt())
        terms.append((offset_dy.abs() * weight_y.unsqueeze(1)).mean())
    if not terms:
        return offset.sum() * 0.0
    return torch.stack(terms).sum()


@dataclass
class DeformationNoiseOutput:
    final_noise: torch.Tensor
    lineage_noise: torch.Tensor
    offsets: torch.Tensor
    valid_masks: torch.Tensor
    match_loss: torch.Tensor
    smoothness_loss: torch.Tensor
    pre_normalization_mean: torch.Tensor
    pre_normalization_std: torch.Tensor
    post_normalization_mean: torch.Tensor
    post_normalization_std: torch.Tensor


def build_deformed_clip_noise(
    deformation_head: NoiseDeformationHead,
    stc_features: torch.Tensor,
    anchor_lineage: torch.Tensor,
    independent_noise: torch.Tensor,
    fallback_noise: torch.Tensor,
    alpha: float,
    warp_scope: str = "full",
    bg_mask: Optional[torch.Tensor] = None,
    normalization_eps: float = 1e-6,
    smoothness_gamma: float = 1.0,
    detach_stc_features: bool = True,
    offset_mode: str = "learned",
) -> DeformationNoiseOutput:
    """Recurrently transport one lineage through a training clip.

    The final fused noise is never fed back as lineage.  ``fallback_noise`` and
    ``independent_noise`` are separate tensors by contract, preventing invalid
    pixels from accidentally sharing both terms of the variance-preserving mix.
    """
    if stc_features.ndim != 5:
        raise ValueError("stc_features must have shape [B,T,C,H,W]")
    batch, frames, _, height, width = stc_features.shape
    if frames < 1:
        raise ValueError("a clip must contain at least one frame")
    if anchor_lineage.ndim != 4:
        raise ValueError("anchor_lineage must have shape [B,C_noise,H,W]")
    expected_noise = (
        batch,
        frames,
        anchor_lineage.shape[1],
        height,
        width,
    )
    if anchor_lineage.shape[0] != batch or anchor_lineage.shape[-2:] != (height, width):
        raise ValueError("anchor_lineage must match feature batch/spatial shape")
    if independent_noise.shape != expected_noise or fallback_noise.shape != expected_noise:
        raise ValueError(f"independent/fallback noise must have shape {expected_noise}")
    if warp_scope == "bg":
        expected_mask = (batch, frames, 1, height, width)
        if bg_mask is None or bg_mask.shape != expected_mask:
            raise ValueError(f"bg_mask must have shape {expected_mask}")

    if offset_mode not in {"learned", "zero_grid"}:
        raise ValueError("offset_mode must be 'learned' or 'zero_grid'")

    features = stc_features.detach() if detach_stc_features else stc_features
    lineage_frames = [anchor_lineage]
    final_frames = [
        variance_preserving_fusion(
            anchor_lineage,
            independent_noise[:, 0],
            alpha,
            warp_scope=warp_scope,
            bg_mask=None if bg_mask is None else bg_mask[:, 0],
        )
    ]
    offsets = []
    valid_masks = []
    match_losses = []
    smoothness_losses = []
    pre_normalization_means = []
    pre_normalization_stds = []
    post_normalization_means = []
    post_normalization_stds = []

    for frame_index in range(1, frames):
        reference_feature = features[:, frame_index - 1]
        target_feature = features[:, frame_index]
        if offset_mode == "learned":
            offset = deformation_head(reference_feature, target_feature)
        else:
            # This ablation deliberately keeps the complete sampling path.
            # A zero offset is passed to backward_warp/grid_sample instead of
            # bypassing the warp with an identity tensor assignment.
            offset = reference_feature.new_zeros(
                (batch, 2, height, width)
            )
        warped_lineage, valid = backward_warp(
            lineage_frames[-1],
            offset,
            fallback=fallback_noise[:, frame_index],
            mode="bilinear",
        )
        pre_normalization_means.append(
            warped_lineage.mean(dim=(-2, -1))
        )
        pre_normalization_stds.append(
            warped_lineage.std(dim=(-2, -1), unbiased=False)
        )
        lineage = normalize_lineage_channels(warped_lineage, eps=normalization_eps)
        post_normalization_means.append(lineage.mean(dim=(-2, -1)))
        post_normalization_stds.append(
            lineage.std(dim=(-2, -1), unbiased=False)
        )
        final = variance_preserving_fusion(
            lineage,
            independent_noise[:, frame_index],
            alpha,
            warp_scope=warp_scope,
            bg_mask=None if bg_mask is None else bg_mask[:, frame_index],
        )

        warped_reference_feature, _ = backward_warp(
            reference_feature, offset, mode="bilinear"
        )
        match = cosine_feature_matching_loss(
            warped_reference_feature, target_feature, valid
        )
        smooth = edge_aware_smoothness_loss(
            offset, target_feature, gamma=smoothness_gamma
        )

        lineage_frames.append(lineage)
        final_frames.append(final)
        offsets.append(offset)
        valid_masks.append(valid)
        match_losses.append(match)
        smoothness_losses.append(smooth)

    if offsets:
        offset_tensor = torch.stack(offsets, dim=1)
        valid_tensor = torch.stack(valid_masks, dim=1)
        match_loss = torch.stack(match_losses).mean()
        smoothness_loss = torch.stack(smoothness_losses).mean()
        pre_mean_tensor = torch.stack(pre_normalization_means, dim=1)
        pre_std_tensor = torch.stack(pre_normalization_stds, dim=1)
        post_mean_tensor = torch.stack(post_normalization_means, dim=1)
        post_std_tensor = torch.stack(post_normalization_stds, dim=1)
    else:
        offset_tensor = stc_features.new_empty((batch, 0, 2, height, width))
        valid_tensor = torch.empty(
            (batch, 0, 1, height, width),
            device=stc_features.device,
            dtype=torch.bool,
        )
        match_loss = anchor_lineage.sum() * 0.0
        smoothness_loss = anchor_lineage.sum() * 0.0
        stats_shape = (batch, 0, anchor_lineage.shape[1])
        pre_mean_tensor = anchor_lineage.new_empty(stats_shape)
        pre_std_tensor = anchor_lineage.new_empty(stats_shape)
        post_mean_tensor = anchor_lineage.new_empty(stats_shape)
        post_std_tensor = anchor_lineage.new_empty(stats_shape)

    return DeformationNoiseOutput(
        final_noise=torch.stack(final_frames, dim=1),
        lineage_noise=torch.stack(lineage_frames, dim=1),
        offsets=offset_tensor,
        valid_masks=valid_tensor,
        match_loss=match_loss,
        smoothness_loss=smoothness_loss,
        pre_normalization_mean=pre_mean_tensor,
        pre_normalization_std=pre_std_tensor,
        post_normalization_mean=post_mean_tensor,
        post_normalization_std=post_std_tensor,
    )
