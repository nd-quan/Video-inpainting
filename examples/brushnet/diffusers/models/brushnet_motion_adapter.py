"""Flow-guided motion adapters for BrushNet multi-scale residual features.

The adapter consumes backward optical flow, i.e. a flow defined at frame i that
samples frame i-1. BrushNet residuals remain flattened as [B*T, C, H, W] (or
[2*B*T, C, H, W] under classifier-free guidance), matching the existing
BrushNet/UNet interface.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from .modeling_utils import ModelMixin


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def resize_flow(flow: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Resize pixel-space flow while scaling horizontal/vertical displacement."""
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"flow must have shape [N,2,H,W], got {tuple(flow.shape)}")
    source_h, source_w = flow.shape[-2:]
    target_h, target_w = size
    resized = F.interpolate(flow, size=size, mode="bilinear", align_corners=True)
    resized = resized.clone()
    resized[:, 0].mul_(target_w / max(source_w, 1))
    resized[:, 1].mul_(target_h / max(source_h, 1))
    return resized


def backward_warp_feature(feature: torch.Tensor, backward_flow: torch.Tensor) -> torch.Tensor:
    """Warp previous-frame features into current-frame coordinates."""
    if feature.ndim != 4:
        raise ValueError(f"feature must be NCHW, got {tuple(feature.shape)}")
    if backward_flow.shape[0] != feature.shape[0] or backward_flow.shape[1] != 2:
        raise ValueError(
            "backward_flow must have shape [N,2,H,W] with the same batch as feature"
        )
    if backward_flow.shape[-2:] != feature.shape[-2:]:
        backward_flow = resize_flow(backward_flow, feature.shape[-2:])
    n, _, height, width = feature.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=feature.device, dtype=feature.dtype),
        torch.arange(width, device=feature.device, dtype=feature.dtype),
        indexing="ij",
    )
    base_grid = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(n, -1, -1, -1)
    sample = base_grid + backward_flow.to(feature.dtype)
    sample_x = 2.0 * sample[:, 0] / max(width - 1, 1) - 1.0
    sample_y = 2.0 * sample[:, 1] / max(height - 1, 1) - 1.0
    grid = torch.stack((sample_x, sample_y), dim=-1)
    return F.grid_sample(
        feature,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


class MotionResidualBlock(nn.Module):
    """Convert flow-aligned temporal features into a zero-initialized correction."""

    def __init__(self, channels: int, bottleneck_channels: int, flow_channels: int):
        super().__init__()
        groups = _group_count(bottleneck_channels)
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(2, flow_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(flow_channels, flow_channels, 3, padding=1),
            nn.SiLU(),
        )
        input_channels = 3 * channels + flow_channels + 1
        self.input_projection = nn.Conv2d(input_channels, bottleneck_channels, 1)
        self.residual = nn.Sequential(
            nn.GroupNorm(groups, bottleneck_channels),
            nn.SiLU(),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1),
            nn.GroupNorm(groups, bottleneck_channels),
            nn.SiLU(),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1),
        )
        self.output_projection = nn.Conv2d(bottleneck_channels, channels, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.gain = nn.Parameter(torch.ones(()))

    def forward(
        self,
        current: torch.Tensor,
        aligned_previous: torch.Tensor,
        flow: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        height, width = current.shape[-2:]
        normalized_flow = flow.clone()
        normalized_flow[:, 0].div_(max(width - 1, 1))
        normalized_flow[:, 1].div_(max(height - 1, 1))
        flow_feature = self.flow_encoder(normalized_flow)
        inputs = torch.cat(
            (
                current,
                aligned_previous,
                current - aligned_previous,
                flow_feature,
                confidence,
            ),
            dim=1,
        )
        hidden = self.input_projection(inputs)
        hidden = hidden + self.residual(hidden)
        return self.gain * confidence * self.output_projection(hidden)


class BrushNetFlowMotionAdapter(ModelMixin, ConfigMixin):
    """Apply trainable flow-guided corrections to BrushNet residual ports."""

    @register_to_config
    def __init__(
        self,
        down_channels: Sequence[int],
        mid_channel: int,
        up_channels: Sequence[int],
        bottleneck_channels: int = 64,
        flow_channels: int = 16,
        use_down: bool = True,
        use_mid: bool = True,
        use_up: bool = False,
    ):
        super().__init__()
        self.down_adapters = nn.ModuleList(
            [
                MotionResidualBlock(channel, bottleneck_channels, flow_channels)
                for channel in down_channels
            ]
            if use_down
            else []
        )
        self.mid_adapter = (
            MotionResidualBlock(mid_channel, bottleneck_channels, flow_channels)
            if use_mid
            else None
        )
        self.up_adapters = nn.ModuleList(
            [
                MotionResidualBlock(channel, bottleneck_channels, flow_channels)
                for channel in up_channels
            ]
            if use_up
            else []
        )

    @classmethod
    def from_brushnet(
        cls,
        brushnet,
        bottleneck_channels: int = 64,
        flow_channels: int = 16,
        use_down: bool = True,
        use_mid: bool = True,
        use_up: bool = False,
    ):
        return cls(
            down_channels=[module.out_channels for module in brushnet.brushnet_down_blocks],
            mid_channel=brushnet.brushnet_mid_block.out_channels,
            up_channels=[module.out_channels for module in brushnet.brushnet_up_blocks],
            bottleneck_channels=bottleneck_channels,
            flow_channels=flow_channels,
            use_down=use_down,
            use_mid=use_mid,
            use_up=use_up,
        )

    @staticmethod
    def _normalize_motion_inputs(
        backward_flow: torch.Tensor,
        confidence: Optional[torch.Tensor],
    ):
        if backward_flow.ndim == 4:
            backward_flow = backward_flow.unsqueeze(0)
        if backward_flow.ndim != 5 or backward_flow.shape[2] != 2:
            raise ValueError(
                "backward_flow must have shape [B,T-1,2,H,W] or [T-1,2,H,W]"
            )
        if confidence is None:
            confidence = torch.ones(
                backward_flow.shape[0],
                backward_flow.shape[1],
                1,
                backward_flow.shape[-2],
                backward_flow.shape[-1],
                device=backward_flow.device,
                dtype=backward_flow.dtype,
            )
        elif confidence.ndim == 4:
            confidence = confidence.unsqueeze(0)
        if confidence.ndim != 5 or confidence.shape[2] != 1:
            raise ValueError(
                "confidence must have shape [B,T-1,1,H,W] or [T-1,1,H,W]"
            )
        if confidence.shape[:2] != backward_flow.shape[:2]:
            raise ValueError("confidence and backward_flow must have matching B,T-1")
        return backward_flow, confidence

    @staticmethod
    def _apply_one_scale(
        residual: torch.Tensor,
        adapter: MotionResidualBlock,
        backward_flow: torch.Tensor,
        confidence: torch.Tensor,
        cfg_branches: int,
        scale: float,
    ) -> torch.Tensor:
        batch, pairs = backward_flow.shape[:2]
        frames = pairs + 1
        expected = cfg_branches * batch * frames
        if residual.shape[0] != expected:
            raise ValueError(
                f"Residual batch {residual.shape[0]} does not match "
                f"cfg_branches*B*T={cfg_branches}*{batch}*{frames}={expected}. "
                "Expected CFG layout [all unconditional frames, all conditional frames]."
            )
        _, channels, height, width = residual.shape
        sequence = residual.reshape(cfg_branches, batch, frames, channels, height, width)
        previous = sequence[:, :, :-1].reshape(-1, channels, height, width)
        current = sequence[:, :, 1:].reshape(-1, channels, height, width)

        flow = backward_flow[:, None].expand(batch, cfg_branches, pairs, 2, *backward_flow.shape[-2:])
        flow = flow.permute(1, 0, 2, 3, 4, 5).reshape(-1, 2, *backward_flow.shape[-2:])
        flow = resize_flow(flow.to(residual.dtype), (height, width))
        conf = confidence[:, None].expand(batch, cfg_branches, pairs, 1, *confidence.shape[-2:])
        conf = conf.permute(1, 0, 2, 3, 4, 5).reshape(-1, 1, *confidence.shape[-2:])
        conf = F.interpolate(conf.to(residual.dtype), (height, width), mode="bilinear", align_corners=True)
        conf = conf.clamp_(0.0, 1.0)

        aligned_previous = backward_warp_feature(previous, flow)
        delta = adapter(current, aligned_previous, flow, conf)
        correction = torch.zeros_like(sequence)
        correction[:, :, 1:] = delta.reshape(
            cfg_branches, batch, pairs, channels, height, width
        )
        return residual + float(scale) * correction.reshape_as(residual)

    def forward(
        self,
        down_residuals: Sequence[torch.Tensor],
        mid_residual: torch.Tensor,
        up_residuals: Sequence[torch.Tensor],
        backward_flow: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        cfg_branches: int = 1,
        scale: float = 1.0,
    ):
        backward_flow, confidence = self._normalize_motion_inputs(
            backward_flow, confidence
        )
        if cfg_branches not in (1, 2):
            raise ValueError("cfg_branches must be 1 or 2")
        down = list(down_residuals)
        up = list(up_residuals)
        if self.down_adapters:
            if len(down) != len(self.down_adapters):
                raise ValueError(
                    f"Expected {len(self.down_adapters)} down residuals, got {len(down)}"
                )
            down = [
                self._apply_one_scale(
                    residual, adapter, backward_flow, confidence, cfg_branches, scale
                )
                for residual, adapter in zip(down, self.down_adapters)
            ]
        if self.mid_adapter is not None:
            mid_residual = self._apply_one_scale(
                mid_residual,
                self.mid_adapter,
                backward_flow,
                confidence,
                cfg_branches,
                scale,
            )
        if self.up_adapters:
            if len(up) != len(self.up_adapters):
                raise ValueError(
                    f"Expected {len(self.up_adapters)} up residuals, got {len(up)}"
                )
            up = [
                self._apply_one_scale(
                    residual, adapter, backward_flow, confidence, cfg_branches, scale
                )
                for residual, adapter in zip(up, self.up_adapters)
            ]
        return down, mid_residual, up
