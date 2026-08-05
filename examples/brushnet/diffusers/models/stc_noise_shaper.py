"""STC-conditioned latent-noise shaping for temporally consistent diffusion.

Two explicitly configured prediction paths are supported. ``legacy`` predicts
a residual on an external flow prior, while ``full`` predicts adjacent
bidirectional flow directly from degraded-frame latents and an ROI mask. The
predicted backward flow transports anchor noise through the whole frame; fresh
Gaussian innovation preserves the required marginal noise variance and fills
out-of-bounds samples.

Mask convention used by the VCM/BrushNet pipeline:
    background mask == 1, ROI mask == 0.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from .brushnet_motion_adapter import backward_warp_feature, resize_flow
from .modeling_utils import ModelMixin


def _sinusoidal_temporal_embedding(
    frames: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a variable-length [1,T,C] temporal positional embedding."""
    positions = torch.arange(frames, device=device, dtype=torch.float32)[:, None]
    half = channels // 2
    if half == 0:
        return torch.zeros(1, frames, channels, device=device, dtype=dtype)
    denominator = max(half - 1, 1)
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / denominator
    )
    angles = positions * frequencies[None]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if embedding.shape[-1] < channels:
        embedding = F.pad(embedding, (0, channels - embedding.shape[-1]))
    return embedding[None].to(dtype=dtype)


class _TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        hidden = max(int(channels * mlp_ratio), channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        attended = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )[0]
        tokens = tokens + attended
        return tokens + self.feed_forward(self.norm2(tokens))


class STCConditionEncoder(nn.Module):
    """Lightweight spatial Conv2D + per-location temporal Transformer."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_channels % num_heads:
            raise ValueError("hidden_channels must be divisible by num_heads")
        self.spatial = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
        )
        self.temporal = nn.ModuleList(
            [
                _TemporalTransformerBlock(
                    hidden_channels,
                    num_heads,
                    mlp_ratio,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.GroupNorm(1, hidden_channels)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 5:
            raise ValueError("condition must have shape [B,T,C,H,W]")
        batch, frames, _, height, width = condition.shape
        spatial = self.spatial(condition.flatten(0, 1))
        _, channels, pooled_h, pooled_w = spatial.shape
        tokens = spatial.reshape(
            batch, frames, channels, pooled_h, pooled_w
        ).permute(0, 3, 4, 1, 2)
        tokens = tokens.reshape(batch * pooled_h * pooled_w, frames, channels)
        tokens = tokens + _sinusoidal_temporal_embedding(
            frames,
            channels,
            tokens.device,
            tokens.dtype,
        )
        for block in self.temporal:
            tokens = block(tokens)
        features = tokens.reshape(
            batch, pooled_h, pooled_w, frames, channels
        ).permute(0, 3, 4, 1, 2)
        features = self.output_norm(features.flatten(0, 1))
        features = F.interpolate(
            features,
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        )
        return features.reshape(batch, frames, channels, height, width)


class _VideoComposerAttention(nn.Module):
    """PyTorch-only port of VideoComposer's condition attention.

    VideoComposer uses ``dim_head == dim`` for every head, rather than splitting
    ``dim`` across heads as ``nn.MultiheadAttention`` does. Keeping that detail
    makes this block structurally compatible with the released implementation
    without importing its monolithic U-Net and xFormers dependencies.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_channels = channels
        inner_channels = num_heads * channels
        self.scale = channels**-0.5
        self.to_qkv = nn.Linear(channels, inner_channels * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_channels, channels),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tokens.shape
        query, key, value = self.to_qkv(tokens).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch,
                length,
                self.num_heads,
                self.head_channels,
            ).permute(0, 2, 1, 3)

        query, key, value = map(split_heads, (query, key, value))
        similarity = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention = similarity.float().softmax(dim=-1).to(similarity.dtype)
        attended = torch.matmul(attention, value)
        attended = attended.permute(0, 2, 1, 3).reshape(batch, length, -1)
        return self.to_out(attended)


class _VideoComposerTemporalBlock(nn.Module):
    """Pre-norm attention and residual feed-forward used by VideoComposer."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.attention_norm = nn.LayerNorm(channels)
        self.attention = _VideoComposerAttention(
            channels,
            num_heads,
            dropout,
        )
        hidden_channels = max(int(channels * mlp_ratio), channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, channels),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.attention_norm(tokens))
        return tokens + self.feed_forward(tokens)


class _VideoComposerConditionBranch(nn.Module):
    """One spatial-CNN + per-location temporal-attention condition branch."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float,
        dropout: float,
        pool_size: int,
    ):
        super().__init__()
        expanded_channels = hidden_channels * 4
        self.pool_size = int(pool_size)
        self.spatial_in = nn.Sequential(
            nn.Conv2d(input_channels, expanded_channels, 3, padding=1),
            nn.SiLU(),
        )
        self.spatial_down = nn.Sequential(
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                3,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv2d(
                expanded_channels,
                hidden_channels,
                3,
                stride=2,
                padding=1,
            ),
        )
        self.temporal = nn.ModuleList(
            [
                _VideoComposerTemporalBlock(
                    hidden_channels,
                    num_heads,
                    mlp_ratio,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 5:
            raise ValueError("condition branch must have shape [B,T,C,H,W]")
        batch, frames, _, height, width = condition.shape
        spatial = self.spatial_in(condition.flatten(0, 1))
        pooled_height = min(self.pool_size, spatial.shape[-2])
        pooled_width = min(self.pool_size, spatial.shape[-1])
        if spatial.shape[-2:] != (pooled_height, pooled_width):
            spatial = F.adaptive_avg_pool2d(
                spatial,
                (pooled_height, pooled_width),
            )
        spatial = self.spatial_down(spatial)
        _, channels, encoded_height, encoded_width = spatial.shape
        tokens = spatial.reshape(
            batch,
            frames,
            channels,
            encoded_height,
            encoded_width,
        ).permute(0, 3, 4, 1, 2)
        tokens = tokens.reshape(
            batch * encoded_height * encoded_width,
            frames,
            channels,
        )
        # The official condition encoder has no explicit temporal positional
        # embedding, so none is added in this architecture variant.
        for block in self.temporal:
            tokens = block(tokens)
        features = tokens.reshape(
            batch,
            encoded_height,
            encoded_width,
            frames,
            channels,
        ).permute(0, 3, 4, 1, 2)
        features = F.interpolate(
            features.flatten(0, 1),
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        )
        return features.reshape(batch, frames, channels, height, width)


class VideoComposerSTCConditionEncoder(nn.Module):
    """Resolution-adapted port of VideoComposer's multi-condition STC path.

    The released VideoComposer U-Net builds one spatial/temporal branch per
    condition and adds their outputs. Legacy VCM noise shaping uses three
    groups (structure, motion, reliability). Full-flow prediction deliberately
    uses one five-channel structural group (whole degraded latent + ROI mask),
    because no external motion prior is available at inference.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.05,
        condition_group_channels: Sequence[int] = (5, 2, 2),
        pool_size: int = 128,
    ):
        super().__init__()
        group_channels = tuple(int(channels) for channels in condition_group_channels)
        if any(channels <= 0 for channels in group_channels):
            raise ValueError("condition-group channels must be positive")
        if sum(group_channels) != input_channels:
            raise ValueError(
                "condition-group channels must sum to input_channels, got "
                f"{group_channels} for {input_channels}"
            )
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")
        self.group_channels = group_channels
        self.branches = nn.ModuleList(
            [
                _VideoComposerConditionBranch(
                    group_input_channels,
                    hidden_channels,
                    num_heads,
                    num_layers,
                    mlp_ratio,
                    dropout,
                    pool_size,
                )
                for group_input_channels in group_channels
            ]
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 5:
            raise ValueError("condition must have shape [B,T,C,H,W]")
        if condition.shape[2] != sum(self.group_channels):
            raise ValueError(
                f"Expected {sum(self.group_channels)} condition channels, "
                f"got {condition.shape[2]}"
            )
        groups = condition.split(self.group_channels, dim=2)
        encoded = [
            branch(group)
            for branch, group in zip(self.branches, groups)
        ]
        return torch.stack(encoded, dim=0).sum(dim=0)


class _SharedDirectionalFlowDecoder(nn.Module):
    """Predict query-to-reference flow from one ordered adjacent feature pair.

    The same weights are used for both temporal directions. ``query`` is the
    frame on whose pixel grid the backward sampling field is defined and
    ``reference`` is the frame that will be sampled. The explicit ordered
    representation avoids asking a per-frame head to infer adjacency and flow
    direction implicitly from temporally mixed features.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3 * channels, 2 * channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(2 * channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, 2, 3, padding=1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        reference: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        if reference.shape != query.shape or reference.ndim != 4:
            raise ValueError("reference and query must match [N,C,H,W]")
        pair = torch.cat((reference, query, query - reference), dim=1)
        return self.network(pair)


def _standardize_channels(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    mean = tensor.mean(dim=(-2, -1), keepdim=True)
    variance = (tensor - mean).square().mean(dim=(-2, -1), keepdim=True)
    return (tensor - mean) * torch.rsqrt(variance + eps)


def _standardize_globally(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    dimensions = tuple(range(1, tensor.ndim))
    mean = tensor.mean(dim=dimensions, keepdim=True)
    variance = (tensor - mean).square().mean(dim=dimensions, keepdim=True)
    return (tensor - mean) * torch.rsqrt(variance + eps)


def gaussian_backward_warp(
    feature: torch.Tensor,
    backward_flow: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bilinearly warp white noise and correct its per-pixel marginal variance.

    Standard bilinear interpolation has variance equal to the sum of squared
    interpolation weights, which is usually below one. Dividing by its square
    root restores unit marginal variance. This does not remove the intentional
    temporal correlation or all spatial correlation introduced by resampling.
    """
    if feature.ndim != 4:
        raise ValueError("feature must have shape [N,C,H,W]")
    if backward_flow.ndim != 4 or backward_flow.shape[:2] != (feature.shape[0], 2):
        raise ValueError("backward_flow must have shape [N,2,H,W]")
    if backward_flow.shape[-2:] != feature.shape[-2:]:
        backward_flow = resize_flow(backward_flow, feature.shape[-2:])
    batch, _, height, width = feature.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=feature.device, dtype=feature.dtype),
        torch.arange(width, device=feature.device, dtype=feature.dtype),
        indexing="ij",
    )
    sample_x = xx[None] + backward_flow[:, 0].to(feature.dtype)
    sample_y = yy[None] + backward_flow[:, 1].to(feature.dtype)
    normalized_x = 2.0 * sample_x / max(width - 1, 1) - 1.0
    normalized_y = 2.0 * sample_y / max(height - 1, 1) - 1.0
    grid = torch.stack((normalized_x, normalized_y), dim=-1)
    warped = F.grid_sample(
        feature,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    x0 = sample_x.floor()
    y0 = sample_y.floor()
    x1 = x0 + 1.0
    y1 = y0 + 1.0
    wx = sample_x - x0
    wy = sample_y - y0

    def valid(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (
            (x >= 0)
            & (x <= width - 1)
            & (y >= 0)
            & (y <= height - 1)
        ).to(feature.dtype)

    variance = (
        ((1.0 - wx) * (1.0 - wy)).square() * valid(x0, y0)
        + (wx * (1.0 - wy)).square() * valid(x1, y0)
        + ((1.0 - wx) * wy).square() * valid(x0, y1)
        + (wx * wy).square() * valid(x1, y1)
    )[:, None]
    in_bounds = (variance > eps).to(feature.dtype)
    normalized = warped * torch.rsqrt(variance.clamp_min(eps))
    return normalized * in_bounds, in_bounds


def _resize_scalar_sequence(
    sequence: torch.Tensor,
    size: Tuple[int, int],
    batch: int,
    pairs: int,
    name: str,
) -> torch.Tensor:
    if sequence.ndim == 4:
        sequence = sequence.unsqueeze(0)
    if sequence.shape[:3] != (batch, pairs, 1):
        raise ValueError(
            f"{name} must have shape [B,T-1,1,H,W], got {tuple(sequence.shape)}"
        )
    if pairs == 0:
        return sequence.new_zeros(batch, 0, 1, *size)
    resized = F.interpolate(
        sequence.flatten(0, 1),
        size=size,
        mode="bilinear",
        align_corners=True,
    )
    return resized.reshape(batch, pairs, 1, *size).clamp(0.0, 1.0)


class STCConditionedNoiseShaper(ModelMixin, ConfigMixin):
    """Generate a condition-aware temporally correlated diffusion noise clip.

    The output has shape [B,T,C,H,W]. Spatial mean/variance normalization only
    matches first and second moments; it is not a proof of i.i.d. Gaussianity.
    A configurable warp region supports either stable-background gating or
    all-frame warping. Geometric in-bounds validity and fresh-noise innovation
    are always retained because an out-of-frame sample has no source noise.
    """

    @register_to_config
    def __init__(
        self,
        latent_channels: int = 4,
        condition_channels: int = 9,
        hidden_channels: int = 64,
        num_attention_heads: int = 4,
        num_transformer_layers: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        encoder_architecture: str = "native",
        condition_group_channels: Tuple[int, ...] = (5, 2, 2),
        videocomposer_pool_size: int = 128,
        flow_residual_scale: float = 2.0,
        use_refined_flow_prior: bool = True,
        use_input_flow_prior: Optional[bool] = None,
        predict_flow_residual: bool = True,
        flow_prediction_mode: str = "legacy",
        full_flow_max_displacement: Tuple[float, float] = (32.0, 32.0),
        beta_min: float = 0.0,
        beta_max: float = 0.95,
        initial_beta: float = 0.10,
        beta_mode: str = "learned",
        fixed_beta: float = 0.5,
        warp_region: str = "stable_bg",
        channel_normalize: bool = False,
        global_normalize: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        flow_prediction_mode = str(flow_prediction_mode).lower()
        if flow_prediction_mode not in {"legacy", "full"}:
            raise ValueError("flow_prediction_mode must be 'legacy' or 'full'")
        expected_condition_channels = (
            latent_channels + 1
            if flow_prediction_mode == "full"
            else latent_channels + 5
        )
        if condition_channels != expected_condition_channels:
            condition_description = (
                "full decoded latent and mask"
                if flow_prediction_mode == "full"
                else "decoded BG latent, BG mask, flow xy, confidence, stable BG"
            )
            raise ValueError(
                "condition_channels must equal "
                f"{expected_condition_channels} for {condition_description}, "
                f"got {condition_channels}"
            )
        if isinstance(full_flow_max_displacement, (int, float)):
            full_flow_bounds = (
                float(full_flow_max_displacement),
                float(full_flow_max_displacement),
            )
        else:
            full_flow_bounds = tuple(float(value) for value in full_flow_max_displacement)
        if len(full_flow_bounds) != 2 or any(value <= 0 for value in full_flow_bounds):
            raise ValueError(
                "full_flow_max_displacement must be a positive scalar or (x,y) pair"
            )
        if not 0.0 <= beta_min < beta_max <= 1.0:
            raise ValueError("Require 0 <= beta_min < beta_max <= 1")
        if not beta_min < initial_beta < beta_max:
            raise ValueError("initial_beta must be in (beta_min,beta_max)")
        beta_mode = str(beta_mode).lower()
        if beta_mode not in {"learned", "fixed"}:
            raise ValueError("beta_mode must be 'learned' or 'fixed'")
        if not 0.0 <= fixed_beta <= 1.0:
            raise ValueError("fixed_beta must be in [0,1]")
        warp_region = str(warp_region).lower()
        if warp_region not in {"stable_bg", "all"}:
            raise ValueError("warp_region must be 'stable_bg' or 'all'")
        if flow_prediction_mode == "full" and warp_region != "all":
            raise ValueError("Full-flow prediction requires warp_region='all'")
        encoder_architecture = str(encoder_architecture).lower()
        if encoder_architecture == "native":
            self.stc_encoder = STCConditionEncoder(
                condition_channels,
                hidden_channels,
                num_attention_heads,
                num_transformer_layers,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
        elif encoder_architecture in {"videocomposer", "videocomposer_stc"}:
            self.stc_encoder = VideoComposerSTCConditionEncoder(
                condition_channels,
                hidden_channels,
                num_attention_heads,
                num_transformer_layers,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                condition_group_channels=condition_group_channels,
                pool_size=videocomposer_pool_size,
            )
        else:
            raise ValueError(
                "encoder_architecture must be 'native' or 'videocomposer', "
                f"got {encoder_architecture!r}"
            )
        if flow_prediction_mode == "full":
            self.flow_head = None
            self.full_flow_head = _SharedDirectionalFlowDecoder(hidden_channels)
        else:
            self.full_flow_head = None
            self.flow_head = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, 2, 3, padding=1),
            )
            nn.init.zeros_(self.flow_head[-1].weight)
            nn.init.zeros_(self.flow_head[-1].bias)
            if not predict_flow_residual:
                self.flow_head.requires_grad_(False)
        if beta_mode == "learned":
            self.beta_head = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, 1, 3, padding=1),
            )
            nn.init.zeros_(self.beta_head[-1].weight)
            initial_probability = (initial_beta - beta_min) / (
                beta_max - beta_min
            )
            initial_logit = math.log(initial_probability / (1.0 - initial_probability))
            nn.init.constant_(self.beta_head[-1].bias, initial_logit)
        else:
            self.beta_head = None

    def _validate_structural_inputs(
        self,
        decoded_latents: torch.Tensor,
        bg_mask: torch.Tensor,
    ) -> Tuple[int, int, int, int]:
        if decoded_latents.ndim != 5:
            raise ValueError("decoded_latents must have shape [B,T,C,H,W]")
        batch, frames, channels, height, width = decoded_latents.shape
        if channels != self.config.latent_channels:
            raise ValueError(
                f"Expected {self.config.latent_channels} latent channels, got {channels}"
            )
        if bg_mask.shape != (batch, frames, 1, height, width):
            raise ValueError("bg_mask must have shape [B,T,1,H,W]")
        return batch, frames, height, width

    def _full_flow_bounds(self, tensor: torch.Tensor) -> torch.Tensor:
        configured = self.config.full_flow_max_displacement
        if isinstance(configured, (int, float)):
            bounds = (float(configured), float(configured))
        else:
            bounds = tuple(float(value) for value in configured)
        return tensor.new_tensor(bounds).reshape(1, 2, 1, 1)

    def _encode_full_flow_condition(
        self,
        decoded_latents: torch.Tensor,
        bg_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode_stc_features(decoded_latents, bg_mask)

    def encode_stc_features(
        self,
        decoded_latents: torch.Tensor,
        bg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the degraded clip into reusable spatio-temporal features.

        Stage 1 consumes these features through the bidirectional flow decoder,
        Stage 2 additionally consumes them through the beta head, and Stage 3
        projects the same representation into the BrushNet residual ports.  A
        public entry point keeps Stage 3 independent of private implementation
        details while preserving the exact Stage-1 conditioning convention.
        """

        if str(self.config.flow_prediction_mode).lower() != "full":
            raise ValueError("Reusable STC features require flow_prediction_mode='full'")
        self._validate_structural_inputs(decoded_latents, bg_mask)
        # Full-flow prediction deliberately sees the whole degraded frame. The
        # mask is semantic context only and never zeros the ROI feature values.
        condition = torch.cat((decoded_latents, bg_mask), dim=2)
        return self.stc_encoder(condition)

    def _decode_full_bidirectional_flow(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, frames, channels, height, width = features.shape
        pairs = frames - 1
        if pairs == 0:
            empty = features.new_zeros(batch, 0, 2, height, width)
            return empty, empty
        previous = features[:, :-1].reshape(-1, channels, height, width)
        current = features[:, 1:].reshape(-1, channels, height, width)
        bounds = self._full_flow_bounds(features)
        # Backward: the current-frame grid samples the previous frame.
        backward = self.full_flow_head(previous, current).tanh() * bounds
        # Forward: the previous-frame grid samples the current frame. Reusing
        # the same ordered decoder enforces a shared directional representation.
        forward = self.full_flow_head(current, previous).tanh() * bounds
        backward = backward.reshape(batch, pairs, 2, height, width)
        forward = forward.reshape(batch, pairs, 2, height, width)
        return backward, forward

    def predict_flow(
        self,
        decoded_latents: torch.Tensor,
        bg_mask: torch.Tensor,
        return_dict: bool = True,
    ):
        """Predict adjacent bidirectional latent-pixel flow without an input prior.

        This entry point is intended for teacher-flow pretraining and inference.
        ``predicted_flow_backward[:, i]`` is defined on frame ``i+1`` and samples
        frame ``i``; the forward tensor uses the opposite direction.
        """
        if str(self.config.flow_prediction_mode).lower() != "full":
            raise ValueError("predict_flow requires flow_prediction_mode='full'")
        features = self._encode_full_flow_condition(decoded_latents, bg_mask)
        backward, forward = self._decode_full_bidirectional_flow(features)
        result = {
            "predicted_flow": backward,
            "predicted_flow_backward": backward,
            "predicted_flow_forward": forward,
        }
        if return_dict:
            return result
        return backward, forward

    def flow_parameters(self):
        """Iterate parameters used by the configured flow-prediction path."""
        yield from self.stc_encoder.parameters()
        head = (
            self.full_flow_head
            if str(self.config.flow_prediction_mode).lower() == "full"
            else self.flow_head
        )
        if head is not None:
            yield from head.parameters()

    def _prepare_inputs(
        self,
        independent_noise: torch.Tensor,
        decoded_latents: torch.Tensor,
        bg_mask: torch.Tensor,
        backward_flow: Optional[torch.Tensor],
        motion_confidence: Optional[torch.Tensor],
        stable_bg: Optional[torch.Tensor],
    ):
        if independent_noise.ndim != 5:
            raise ValueError("independent_noise must have shape [B,T,C,H,W]")
        if decoded_latents.shape != independent_noise.shape:
            raise ValueError("decoded_latents must match independent_noise")
        batch, frames, height, width = self._validate_structural_inputs(
            decoded_latents,
            bg_mask,
        )
        pairs = frames - 1
        if str(self.config.flow_prediction_mode).lower() == "full":
            # Placeholders preserve downstream diagnostic shapes. They are not
            # encoder conditions and are not used as flow priors in full mode.
            flow = independent_noise.new_zeros(batch, pairs, 2, height, width)
            confidence = independent_noise.new_ones(
                batch, pairs, 1, height, width
            )
            stable = independent_noise.new_ones(batch, pairs, 1, height, width)
            return flow, confidence, stable
        if backward_flow is None:
            raise ValueError("Legacy flow mode requires backward_flow")
        if backward_flow.ndim == 4:
            backward_flow = backward_flow.unsqueeze(0)
        if backward_flow.shape[:3] != (batch, pairs, 2):
            raise ValueError(
                "backward_flow must have shape [B,T-1,2,Hf,Wf], got "
                f"{tuple(backward_flow.shape)}"
            )
        if pairs:
            flow = resize_flow(
                backward_flow.flatten(0, 1).to(independent_noise.dtype),
                (height, width),
            ).reshape(batch, pairs, 2, height, width)
        else:
            flow = independent_noise.new_zeros(batch, 0, 2, height, width)
        if motion_confidence is None:
            confidence = torch.ones(
                batch,
                pairs,
                1,
                height,
                width,
                device=flow.device,
                dtype=flow.dtype,
            )
        else:
            confidence = _resize_scalar_sequence(
                motion_confidence.to(flow.dtype),
                (height, width),
                batch,
                pairs,
                "motion_confidence",
            )
        if stable_bg is None:
            stable = confidence * bg_mask[:, 1:]
        else:
            stable = _resize_scalar_sequence(
                stable_bg.to(flow.dtype),
                (height, width),
                batch,
                pairs,
                "stable_bg",
            )
            stable = stable * bg_mask[:, 1:]
        return flow, confidence, stable

    def forward(
        self,
        independent_noise: Optional[torch.Tensor] = None,
        decoded_latents: Optional[torch.Tensor] = None,
        bg_mask: Optional[torch.Tensor] = None,
        backward_flow: Optional[torch.Tensor] = None,
        motion_confidence: Optional[torch.Tensor] = None,
        stable_bg: Optional[torch.Tensor] = None,
        strength: float = 1.0,
        return_dict: bool = True,
        operation: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        if operation is not None:
            if operation != "predict_flow":
                raise ValueError("operation must be None or 'predict_flow'")
            if decoded_latents is None or bg_mask is None:
                raise ValueError(
                    "operation='predict_flow' requires decoded_latents and bg_mask"
                )
            return self.predict_flow(
                decoded_latents,
                bg_mask,
                return_dict=return_dict,
            )
        if independent_noise is None or decoded_latents is None or bg_mask is None:
            raise ValueError(
                "Noise shaping requires independent_noise, decoded_latents, and bg_mask"
            )
        strength = float(strength)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0,1]")
        flow, confidence, stable = self._prepare_inputs(
            independent_noise,
            decoded_latents,
            bg_mask,
            backward_flow,
            motion_confidence,
            stable_bg,
        )
        batch, frames, _, height, width = independent_noise.shape
        full_flow_mode = str(self.config.flow_prediction_mode).lower() == "full"
        if full_flow_mode:
            features = self._encode_full_flow_condition(decoded_latents, bg_mask)
        else:
            zero_flow = torch.zeros(
                batch, 1, 2, height, width, device=flow.device, dtype=flow.dtype
            )
            zero_scalar = torch.zeros(
                batch, 1, 1, height, width, device=flow.device, dtype=flow.dtype
            )
            padded_flow = torch.cat((zero_flow, flow), dim=1)
            padded_confidence = torch.cat((zero_scalar, confidence), dim=1)
            padded_stable = torch.cat((zero_scalar, stable), dim=1)
            normalized_flow = padded_flow.clone()
            normalized_flow[:, :, 0].div_(max(width - 1, 1))
            normalized_flow[:, :, 1].div_(max(height - 1, 1))
            condition = torch.cat(
                (
                    decoded_latents * bg_mask,
                    bg_mask,
                    normalized_flow,
                    padded_confidence,
                    padded_stable,
                ),
                dim=2,
            )
            features = self.stc_encoder(condition)
        flat_features = features.flatten(0, 1)
        if full_flow_mode:
            predicted_flow, predicted_flow_forward = (
                self._decode_full_bidirectional_flow(features)
            )
            flow_residual = torch.zeros_like(predicted_flow)
        elif self.config.predict_flow_residual:
            flow_residual = self.flow_head(flat_features).reshape(
                batch, frames, 2, height, width
            )
            flow_residual = (
                float(self.config.flow_residual_scale) * flow_residual.tanh()
            )[:, 1:]
            predicted_flow = flow_residual
            use_input_flow_prior = self.config.use_input_flow_prior
            if use_input_flow_prior is None:
                use_input_flow_prior = self.config.use_refined_flow_prior
            if use_input_flow_prior:
                predicted_flow = predicted_flow + flow
            predicted_flow_forward = None
        else:
            flow_residual = torch.zeros_like(flow)
            predicted_flow = flow_residual
            use_input_flow_prior = self.config.use_input_flow_prior
            if use_input_flow_prior is None:
                use_input_flow_prior = self.config.use_refined_flow_prior
            if use_input_flow_prior:
                predicted_flow = predicted_flow + flow
            predicted_flow_forward = None

        if str(self.config.beta_mode).lower() == "fixed":
            beta = independent_noise.new_full(
                (batch, frames - 1, 1, height, width),
                float(self.config.fixed_beta),
            )
            beta = (beta * strength).clamp(0.0, 1.0)
        else:
            beta = self.config.beta_min + (
                self.config.beta_max - self.config.beta_min
            ) * torch.sigmoid(
                self.beta_head(flat_features).reshape(
                    batch, frames, 1, height, width
                )[:, 1:]
            )
            beta = (beta * strength).clamp(0.0, self.config.beta_max)
        shaped_frames = [independent_noise[:, 0]]
        cumulative_flows = []
        cumulative_stable = []
        warp_region = str(getattr(self.config, "warp_region", "stable_bg")).lower()
        coherence_numerator = independent_noise.new_zeros(())
        coherence_denominator = independent_noise.new_zeros(())
        for frame_index in range(1, frames):
            pair_flow = predicted_flow[:, frame_index - 1]
            pair_stable = stable[:, frame_index - 1]
            if frame_index == 1:
                cumulative_flow = pair_flow
            else:
                cumulative_flow = pair_flow + backward_warp_feature(
                    cumulative_flows[-1], pair_flow
                )
            if warp_region == "stable_bg":
                if frame_index == 1:
                    stable_to_anchor = pair_stable
                else:
                    stable_to_anchor = pair_stable * backward_warp_feature(
                        cumulative_stable[-1], pair_flow
                    ).clamp(0.0, 1.0)
            warped, warp_valid = gaussian_backward_warp(
                independent_noise[:, 0],
                cumulative_flow,
                eps=self.config.norm_eps,
            )
            if warp_region == "all":
                stable_to_anchor = warp_valid
            else:
                stable_to_anchor = stable_to_anchor * warp_valid
            if self.config.channel_normalize:
                warped = _standardize_channels(warped, self.config.norm_eps)
            coefficient = beta[:, frame_index - 1] * stable_to_anchor
            innovation = torch.sqrt((1.0 - coefficient.square()).clamp_min(0.0))
            current = coefficient * warped + innovation * independent_noise[:, frame_index]
            shaped_frames.append(current)
            cumulative_flows.append(cumulative_flow)
            cumulative_stable.append(stable_to_anchor)
            stable_weight = stable_to_anchor
            coherence_numerator = coherence_numerator + (
                (current - warped).square() * stable_weight
            ).sum()
            coherence_denominator = coherence_denominator + (
                stable_weight.sum() * current.shape[1]
            )
        shaped_noise = torch.stack(shaped_frames, dim=1)
        if self.config.global_normalize:
            shaped_noise = _standardize_globally(
                shaped_noise,
                self.config.norm_eps,
            )

        if cumulative_stable:
            cumulative_stable_tensor = torch.stack(cumulative_stable, dim=1)
            effective_beta = beta * cumulative_stable_tensor
            beta_reliable_mean = effective_beta.sum() / (
                cumulative_stable_tensor.sum().clamp_min(1e-6)
            )
        else:
            cumulative_stable_tensor = stable
            effective_beta = beta
            beta_reliable_mean = independent_noise.new_zeros(())
        flow_weight = (
            torch.ones_like(stable)
            if warp_region == "all"
            else stable
        )
        stable_weight = flow_weight.expand_as(flow_residual[:, :, :1])
        flow_residual_energy = (
            flow_residual.square().sum(dim=2, keepdim=True) * stable_weight
        ).sum() / (stable_weight.sum() * 2.0).clamp_min(1e-6)
        flow_prediction_energy = (
            predicted_flow.square().sum(dim=2, keepdim=True) * flow_weight
        ).sum() / (flow_weight.sum() * 2.0).clamp_min(1e-6)
        result = {
            "noise": shaped_noise,
            # Reuse the exact feature tensor in Stage 3 instead of evaluating
            # the relatively expensive STC encoder a second time.
            "stc_features": features,
            "predicted_flow": predicted_flow,
            "predicted_flow_backward": predicted_flow,
            "predicted_flow_forward": predicted_flow_forward,
            "flow_residual": flow_residual,
            "flow_residual_energy": flow_residual_energy,
            "flow_prediction_energy": flow_prediction_energy,
            "beta": beta,
            "effective_beta": effective_beta,
            "cumulative_flow": (
                torch.stack(cumulative_flows, dim=1)
                if cumulative_flows
                else predicted_flow
            ),
            "cumulative_stable": cumulative_stable_tensor,
            "beta_mean": beta_reliable_mean,
            "effective_beta_mean": (
                effective_beta.mean()
                if frames > 1
                else independent_noise.new_zeros(())
            ),
            "beta_raw_mean": (
                beta.mean()
                if frames > 1
                else independent_noise.new_zeros(())
            ),
            "noise_mean": shaped_noise.mean(),
            "noise_std": shaped_noise.std(unbiased=False),
            "noise_warp_error": coherence_numerator
            / coherence_denominator.clamp_min(1e-6),
        }
        if return_dict:
            return result
        return (
            result["noise"],
            result["predicted_flow"],
            result["beta"],
        )
