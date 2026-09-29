"""Flow-align adjacent RGB-STC features before temporal attention.

This variant deliberately aligns features, not noise.  For frame ``t`` the
flow head predicts a backward field defined on frame ``t`` that samples the
raw spatial feature of frame ``t-1``.  A confidence-aware, zero-initialized
residual fuses that warped feature into the raw feature of frame ``t`` before
VideoComposer-style temporal attention.

Alignment is local to one input clip.  The implementation never recursively
warps an already-warped feature, so it does not reproduce the long-horizon
bilinear-noise collapse observed in the V2++ sequence-state experiment.

Mask convention after dataset loading:

* ``M_BG == 1``: strongly degraded background that must be restored.
* ``M_BG == 0``: high-quality ROI that must be preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

try:
    from STC_encoder_v2_rgb.rgb_stc_adapter import RGBSTCConditionAdapter
    from STC_encoder_v3_rgb_flow.rgb_stc_flow_adapter import (
        SharedDirectionalFlowHead,
    )
except ModuleNotFoundError:  # Imported as examples.brushnet.STC_encoder_v4_flow_aligned.
    from ..STC_encoder_v2_rgb.rgb_stc_adapter import RGBSTCConditionAdapter
    from ..STC_encoder_v3_rgb_flow.rgb_stc_flow_adapter import (
        SharedDirectionalFlowHead,
    )

 
def _flow_bounds(value: Sequence[float]) -> Tuple[float, float]:
    if isinstance(value, (int, float)):
        result = (float(value), float(value))
    else:
        result = tuple(float(item) for item in value)
    if len(result) != 2 or any(item <= 0.0 for item in result):
        raise ValueError("flow_max_displacement must be positive (x,y)")
    return result


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def make_backward_sampling_grid(
    backward_flow: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert latent-pixel backward flow to an align_corners=False grid."""
    if backward_flow.ndim != 4 or backward_flow.shape[1] != 2:
        raise ValueError("backward_flow must have shape [N,2,H,W]")
    if not backward_flow.is_floating_point():
        raise TypeError("backward_flow must be floating point")
    batch, _, height, width = backward_flow.shape
    if height < 1 or width < 1:
        raise ValueError("flow spatial dimensions must be positive")

    y = torch.arange(height, device=backward_flow.device, dtype=backward_flow.dtype)
    x = torch.arange(width, device=backward_flow.device, dtype=backward_flow.dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    source_x = grid_x.unsqueeze(0) + backward_flow[:, 0]
    source_y = grid_y.unsqueeze(0) + backward_flow[:, 1]
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


def backward_warp_feature(
    reference: torch.Tensor,
    backward_flow: torch.Tensor,
    fallback: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample ``reference`` on query coordinates using a backward field."""
    if reference.ndim != 4:
        raise ValueError("reference must have shape [N,C,H,W]")
    if (
        reference.shape[0] != backward_flow.shape[0]
        or reference.shape[-2:] != backward_flow.shape[-2:]
    ):
        raise ValueError("reference and flow batch/spatial shapes must match")
    if fallback is not None and fallback.shape != reference.shape:
        raise ValueError("fallback must have the same shape as reference")
    grid, valid = make_backward_sampling_grid(backward_flow)
    warped = F.grid_sample(
        reference,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    if fallback is not None:
        warped = torch.where(valid, warped, fallback)
    return warped, valid


class ConfidenceAwareAlignmentFusion(nn.Module):
    """Zero-init residual fusion of current and motion-aligned features."""

    def __init__(self, channels: int):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Conv2d(3 * channels + 1, 2 * channels, 3, padding=1),
            nn.GroupNorm(_group_count(2 * channels), 2 * channels),
            nn.SiLU(),
            nn.Conv2d(2 * channels, channels, 3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
        )
        self.to_residual_and_gate = nn.Conv2d(channels, channels + 1, 1)
        nn.init.zeros_(self.to_residual_and_gate.weight)
        nn.init.zeros_(self.to_residual_and_gate.bias)

    def forward(
        self,
        current: torch.Tensor,
        warped_previous: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        if current.shape != warped_previous.shape:
            raise ValueError("current and warped_previous must have matching shapes")
        if confidence.shape != (current.shape[0], 1, *current.shape[-2:]):
            raise ValueError("confidence must have shape [N,1,H,W]")
        fused = torch.cat(
            (current, warped_previous, current - warped_previous, confidence),
            dim=1,
        )
        residual, gate = self.to_residual_and_gate(self.hidden(fused)).split(
            (current.shape[1], 1), dim=1
        )
        return current + confidence * torch.sigmoid(gate) * residual


@dataclass
class FlowAlignedRGBSTCOutput:
    delta_bg: torch.Tensor
    features: torch.Tensor
    spatial_features: torch.Tensor
    aligned_spatial_features: torch.Tensor
    latent_bg_mask: torch.Tensor
    predicted_flow_forward: Optional[torch.Tensor]
    predicted_flow_backward: Optional[torch.Tensor]
    alignment_confidence: torch.Tensor


class FlowAlignedRGBSTCAdapter(ModelMixin, ConfigMixin):
    """Checkpointable T-variable STC with clip-local adjacent alignment."""

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
        flow_max_displacement: Tuple[float, float] = (8.0, 8.0),
        flow_confidence_scale: float = 1.0,
        detach_flow_confidence: bool = True,
    ):
        super().__init__()
        bounds = _flow_bounds(flow_max_displacement)
        if float(flow_confidence_scale) <= 0.0:
            raise ValueError("flow_confidence_scale must be positive")
        self.stc_adapter = RGBSTCConditionAdapter(
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            downsample_factor=downsample_factor,
            output_channels=output_channels,
            condition_mode=condition_mode,
        )
        self.flow_head = SharedDirectionalFlowHead(int(hidden_channels))
        self.alignment_fusion = ConfidenceAwareAlignmentFusion(int(hidden_channels))
        self._flow_max_displacement = bounds

    @property
    def zero_conv(self):
        return self.stc_adapter.zero_conv

    def build_pixel_condition(
        self,
        rgb_sequence: torch.Tensor,
        bg_mask_sequence: torch.Tensor,
    ) -> torch.Tensor:
        return self.stc_adapter.build_pixel_condition(
            rgb_sequence, bg_mask_sequence
        )

    def _decode_bidirectional_flow(
        self,
        spatial_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if spatial_features.ndim != 5:
            raise ValueError("spatial_features must have shape [B,T,C,H,W]")
        batch, frames, channels, height, width = spatial_features.shape
        if frames <= 1:
            empty = spatial_features.new_zeros(batch, 0, 2, height, width)
            return empty, empty
        previous = spatial_features[:, :-1].reshape(-1, channels, height, width)
        current = spatial_features[:, 1:].reshape(-1, channels, height, width)
        bounds = spatial_features.new_tensor(self._flow_max_displacement).reshape(
            1, 2, 1, 1
        )
        backward = self.flow_head(previous, current).tanh() * bounds
        forward = self.flow_head(current, previous).tanh() * bounds
        pairs = frames - 1
        return (
            forward.reshape(batch, pairs, 2, height, width),
            backward.reshape(batch, pairs, 2, height, width),
        )

    def _align_spatial_features(
        self,
        spatial_features: torch.Tensor,
        flow_forward: torch.Tensor,
        flow_backward: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, frames, channels, height, width = spatial_features.shape
        if frames <= 1:
            confidence = spatial_features.new_zeros(batch, 0, 1, height, width)
            return spatial_features, confidence

        previous = spatial_features[:, :-1].reshape(-1, channels, height, width)
        current = spatial_features[:, 1:].reshape(-1, channels, height, width)
        backward = flow_backward.reshape(-1, 2, height, width)
        forward = flow_forward.reshape(-1, 2, height, width)

        warped_previous, feature_valid = backward_warp_feature(
            previous, backward, fallback=current
        )
        warped_forward, flow_valid = backward_warp_feature(forward, backward)
        consistency_error = backward + warped_forward
        scale = float(self.config.flow_confidence_scale)
        confidence = torch.exp(
            -consistency_error.square().sum(dim=1, keepdim=True)
            / (2.0 * scale * scale)
        )
        confidence = confidence * (feature_valid & flow_valid).to(confidence.dtype)
        if bool(self.config.detach_flow_confidence):
            confidence = confidence.detach()

        aligned_current = self.alignment_fusion(
            current, warped_previous, confidence
        ).reshape(batch, frames - 1, channels, height, width)
        aligned = torch.cat((spatial_features[:, :1], aligned_current), dim=1)
        return aligned, confidence.reshape(batch, frames - 1, 1, height, width)

    def encode(
        self,
        rgb_sequence: torch.Tensor,
        bg_mask_sequence: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch, frames, height, width = self.stc_adapter._validate_inputs(
            rgb_sequence, bg_mask_sequence
        )
        if output_size is None:
            output_size = (height // 8, width // 8)
        output_size = (int(output_size[0]), int(output_size[1]))
        if output_size[0] < 1 or output_size[1] < 1:
            raise ValueError("output_size must contain positive dimensions")

        pixel_condition = self.build_pixel_condition(
            rgb_sequence, bg_mask_sequence
        ).flatten(0, 1)
        spatial_flat = self.stc_adapter.spatial_encoder(pixel_condition)
        if spatial_flat.shape[-2:] != output_size:
            spatial_flat = F.interpolate(
                spatial_flat,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        _, channels, encoded_height, encoded_width = spatial_flat.shape
        spatial_features = spatial_flat.reshape(
            batch, frames, channels, encoded_height, encoded_width
        )
        flow_forward, flow_backward = self._decode_bidirectional_flow(
            spatial_features
        )
        aligned_spatial, confidence = self._align_spatial_features(
            spatial_features, flow_forward, flow_backward
        )

        tokens = aligned_spatial.permute(0, 3, 4, 1, 2).reshape(
            batch * encoded_height * encoded_width, frames, channels
        )
        # Preserve V2/VideoComposer behavior: no explicit temporal position
        # embedding. T=16 is therefore shape-compatible but still requires
        # fine-tuning because the attention-token distribution changed from T=8.
        for temporal_block in self.stc_adapter.temporal_blocks:
            tokens = temporal_block(tokens)
        features = tokens.reshape(
            batch, encoded_height, encoded_width, frames, channels
        ).permute(0, 3, 4, 1, 2)
        features_flat = features.flatten(0, 1)
        features_flat = self.stc_adapter.output_act(
            self.stc_adapter.output_norm(features_flat)
        )
        delta = self.stc_adapter.zero_conv(features_flat).reshape(
            batch, frames, 4, encoded_height, encoded_width
        )
        return (
            delta,
            features,
            spatial_features,
            aligned_spatial,
            flow_forward,
            flow_backward,
            confidence,
        )

    def forward(
        self,
        rgb_sequence: torch.Tensor,
        bg_mask_sequence: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        predict_flow: bool = True,
        return_dict: bool = True,
    ):
        (
            delta,
            features,
            spatial_features,
            aligned_spatial,
            flow_forward,
            flow_backward,
            confidence,
        ) = self.encode(rgb_sequence, bg_mask_sequence, output_size=output_size)
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
        result = FlowAlignedRGBSTCOutput(
            delta_bg=delta * latent_bg_mask,
            features=features,
            spatial_features=spatial_features,
            aligned_spatial_features=aligned_spatial,
            latent_bg_mask=latent_bg_mask,
            predicted_flow_forward=flow_forward if predict_flow else None,
            predicted_flow_backward=flow_backward if predict_flow else None,
            alignment_confidence=confidence,
        )
        if return_dict:
            return result
        return (
            result.delta_bg,
            result.features,
            result.latent_bg_mask,
            result.predicted_flow_forward,
            result.predicted_flow_backward,
            result.alignment_confidence,
        )


def augment_brushnet_condition(
    model: FlowAlignedRGBSTCAdapter,
    base_condition_latents: torch.Tensor,
    rgb_sequence: torch.Tensor,
    bg_mask_sequence: torch.Tensor,
    injection_scale: float = 1.0,
    predict_flow: bool = True,
):
    """Inject the BG-gated aligned STC delta; keep mask channel unchanged."""
    if base_condition_latents.ndim != 4 or base_condition_latents.shape[1] != 4:
        raise ValueError("base_condition_latents must have shape [B*T,4,h,w]")
    batch, frames = rgb_sequence.shape[:2]
    if base_condition_latents.shape[0] != batch * frames:
        raise ValueError("base_condition_latents batch must equal B*T")
    output = model(
        rgb_sequence,
        bg_mask_sequence,
        output_size=base_condition_latents.shape[-2:],
        predict_flow=predict_flow,
        return_dict=True,
    )
    base_sequence = base_condition_latents.reshape(
        batch, frames, 4, *base_condition_latents.shape[-2:]
    )
    delta = output.delta_bg.to(dtype=base_sequence.dtype)
    latent_bg_mask = output.latent_bg_mask.to(dtype=base_sequence.dtype)
    augmented = base_sequence + float(injection_scale) * delta
    brushnet_condition = torch.cat(
        (augmented, latent_bg_mask), dim=2
    ).flatten(0, 1)
    return brushnet_condition, output, augmented
