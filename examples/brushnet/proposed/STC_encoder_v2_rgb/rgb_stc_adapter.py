"""A small RGB spatio-temporal condition adapter for V8 BrushNet.

The adapter deliberately has one job.  It reads a sequence of fully degraded
RGB frames together with the per-frame degraded-background mask, predicts a
four-channel latent correction, and gates that correction to the degraded
background.  It does not predict flow and it does not alter the shared-noise
law.

Mask convention used by this experiment:

* ``M_BG == 1``: strongly degraded background that must be restored.
* ``M_BG == 0``: high-quality ROI that should remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _VideoComposerAttention(nn.Module):
    """Temporal attention matching VideoComposer's per-head channel layout."""

    def __init__(self, channels: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_channels = int(channels)
        inner_channels = self.num_heads * self.head_channels
        self.scale = self.head_channels**-0.5
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
                batch, length, self.num_heads, self.head_channels
            ).permute(0, 2, 1, 3)

        query, key, value = map(split_heads, (query, key, value))
        similarity = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention = similarity.float().softmax(dim=-1).to(similarity.dtype)
        attended = torch.matmul(attention, value)
        attended = attended.permute(0, 2, 1, 3).reshape(batch, length, -1)
        return self.to_out(attended)


class _VideoComposerTemporalBlock(nn.Module):
    """Pre-norm temporal attention and residual feed-forward block."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.attention_norm = nn.LayerNorm(channels)
        self.attention = _VideoComposerAttention(channels, num_heads, dropout)
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


class _SpatialDownBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        groups = _group_count(output_channels)
        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.block(tensor)


@dataclass
class RGBSTCConditionOutput:
    """Intermediate tensors useful for training diagnostics and inference."""

    delta_bg: torch.Tensor
    features: torch.Tensor
    latent_bg_mask: torch.Tensor


class RGBSTCConditionAdapter(ModelMixin, ConfigMixin):
    """Encode pixel-space artifacts and emit a BG-gated latent correction.

    ``full_rgb_bg_mask`` is the primary experiment: all degraded RGB pixels and
    ``M_BG`` are visible to the encoder.  ``videocomposer_roi_masked`` is kept
    only as a faithful ablation: it supplies RGB from the high-quality ROI and
    its complementary ROI mask.
    """

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        hidden_channels: int = 64,
        num_heads: int = 2,
        num_layers: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        downsample_factor: int = 8,
        output_channels: int = 4,
        condition_mode: str = "full_rgb_bg_mask",
    ):
        super().__init__()
        hidden_channels = int(hidden_channels)
        if hidden_channels < 8:
            raise ValueError("hidden_channels must be at least 8")
        if int(num_heads) < 1 or int(num_layers) < 1:
            raise ValueError("num_heads and num_layers must be positive")
        if int(downsample_factor) != 8:
            raise ValueError("This phase-1 RGB encoder supports downsample_factor=8")
        if int(output_channels) != 4:
            raise ValueError("BrushNet condition latents require output_channels=4")
        if condition_mode not in {
            "full_rgb_bg_mask",
            "videocomposer_roi_masked",
        }:
            raise ValueError(f"Unknown condition_mode: {condition_mode}")

        half_channels = max(hidden_channels // 2, 8)
        self.spatial_encoder = nn.Sequential(
            _SpatialDownBlock(4, half_channels),
            _SpatialDownBlock(half_channels, hidden_channels),
            _SpatialDownBlock(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
        )
        self.temporal_blocks = nn.ModuleList(
            [
                _VideoComposerTemporalBlock(
                    hidden_channels,
                    int(num_heads),
                    float(mlp_ratio),
                    float(dropout),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.output_norm = nn.GroupNorm(
            _group_count(hidden_channels), hidden_channels
        )
        self.output_act = nn.SiLU()
        self.zero_conv = nn.Conv2d(hidden_channels, output_channels, 1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    @staticmethod
    def _validate_inputs(
        rgb_sequence: torch.Tensor,
        bg_mask_sequence: torch.Tensor,
    ) -> Tuple[int, int, int, int]:
        if rgb_sequence.ndim != 5 or rgb_sequence.shape[2] != 3:
            raise ValueError("rgb_sequence must have shape [B,T,3,H,W]")
        batch, frames, _, height, width = rgb_sequence.shape
        expected_mask = (batch, frames, 1, height, width)
        if tuple(bg_mask_sequence.shape) != expected_mask:
            raise ValueError(
                "bg_mask_sequence must have shape [B,T,1,H,W] matching RGB; "
                f"expected {expected_mask}, got {tuple(bg_mask_sequence.shape)}"
            )
        if height % 8 or width % 8:
            raise ValueError("RGB spatial dimensions must be divisible by 8")
        if not rgb_sequence.is_floating_point():
            raise TypeError("rgb_sequence must be a floating-point tensor")
        if not bg_mask_sequence.is_floating_point():
            raise TypeError("bg_mask_sequence must be a floating-point tensor")
        return batch, frames, height, width

    def build_pixel_condition(
        self,
        rgb_sequence: torch.Tensor,
        bg_mask_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """Build the explicit four-channel pixel-space STC input."""
        self._validate_inputs(rgb_sequence, bg_mask_sequence)
        mask = bg_mask_sequence.to(
            device=rgb_sequence.device, dtype=rgb_sequence.dtype
        )
        if self.config.condition_mode == "full_rgb_bg_mask":
            return torch.cat((rgb_sequence, mask), dim=2)
        roi_mask = 1.0 - mask
        return torch.cat((rgb_sequence * roi_mask, roi_mask), dim=2)

    def encode(
        self,
        rgb_sequence: torch.Tensor,
        bg_mask_sequence: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, frames, height, width = self._validate_inputs(
            rgb_sequence, bg_mask_sequence
        )
        if output_size is None:
            output_size = (height // 8, width // 8)
        output_size = (int(output_size[0]), int(output_size[1]))
        if output_size[0] < 1 or output_size[1] < 1:
            raise ValueError("output_size must contain positive dimensions")

        condition = self.build_pixel_condition(
            rgb_sequence, bg_mask_sequence
        ).flatten(0, 1)
        features = self.spatial_encoder(condition)
        if features.shape[-2:] != output_size:
            features = F.interpolate(
                features,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        _, channels, encoded_height, encoded_width = features.shape
        tokens = features.reshape(
            batch, frames, channels, encoded_height, encoded_width
        ).permute(0, 3, 4, 1, 2)
        tokens = tokens.reshape(
            batch * encoded_height * encoded_width, frames, channels
        )
        # VideoComposer's condition encoder has no explicit temporal position
        # embedding; this phase keeps that behavior.
        for temporal_block in self.temporal_blocks:
            tokens = temporal_block(tokens)
        features = tokens.reshape(
            batch, encoded_height, encoded_width, frames, channels
        ).permute(0, 3, 4, 1, 2)
        features_flat = features.flatten(0, 1)
        features_flat = self.output_act(self.output_norm(features_flat))
        delta = self.zero_conv(features_flat).reshape(
            batch, frames, 4, encoded_height, encoded_width
        )
        return delta, features

    def forward(
        self,
        rgb_sequence: torch.Tensor,
        bg_mask_sequence: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        return_dict: bool = False,
    ):
        delta, features = self.encode(
            rgb_sequence,
            bg_mask_sequence,
            output_size=output_size,
        )
        latent_bg_mask = F.interpolate(
            bg_mask_sequence.flatten(0, 1).to(
                device=delta.device, dtype=delta.dtype
            ),
            size=delta.shape[-2:],
            mode="nearest",
        ).reshape(
            bg_mask_sequence.shape[0],
            bg_mask_sequence.shape[1],
            1,
            *delta.shape[-2:],
        )
        delta_bg = delta * latent_bg_mask
        if return_dict:
            return RGBSTCConditionOutput(
                delta_bg=delta_bg,
                features=features,
                latent_bg_mask=latent_bg_mask,
            )
        return delta_bg


def augment_brushnet_condition(
    adapter: RGBSTCConditionAdapter,
    base_condition_latents: torch.Tensor,
    rgb_sequence: torch.Tensor,
    bg_mask_sequence: torch.Tensor,
    injection_scale: float = 1.0,
) -> Tuple[torch.Tensor, RGBSTCConditionOutput, torch.Tensor]:
    """Inject RGB-STC into only the four latent channels of BrushNet input.

    Returns ``(brushnet_condition, stc_output, augmented_latents)``.  The
    fifth BrushNet channel remains the original nearest-neighbour ``M_BG``.
    """
    if base_condition_latents.ndim != 4 or base_condition_latents.shape[1] != 4:
        raise ValueError(
            "base_condition_latents must have shape [B*T,4,h,w]"
        )
    batch, frames = rgb_sequence.shape[:2]
    if base_condition_latents.shape[0] != batch * frames:
        raise ValueError(
            "Flattened condition batch does not match B*T: "
            f"{base_condition_latents.shape[0]} != {batch}*{frames}"
        )
    scale = float(injection_scale)
    if not torch.isfinite(torch.tensor(scale)):
        raise ValueError("injection_scale must be finite")

    output = adapter(
        rgb_sequence,
        bg_mask_sequence,
        output_size=base_condition_latents.shape[-2:],
        return_dict=True,
    )
    base_sequence = base_condition_latents.reshape(
        batch, frames, 4, *base_condition_latents.shape[-2:]
    )
    # Accelerator converts prepared-model outputs back to FP32. Cast the new
    # correction and mask to the original VAE-condition dtype so zero-init is
    # also numerically identical to V8's five-channel BrushNet input under AMP.
    delta_for_condition = output.delta_bg.to(dtype=base_sequence.dtype)
    mask_for_condition = output.latent_bg_mask.to(dtype=base_sequence.dtype)
    augmented = base_sequence + scale * delta_for_condition
    brushnet_condition = torch.cat(
        (augmented, mask_for_condition), dim=2
    ).flatten(0, 1)
    return brushnet_condition, output, augmented
