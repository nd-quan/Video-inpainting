"""Zero-initialized Stage-3 injection of STC features into BrushNet ports.

The adapter deliberately does not estimate or warp optical flow.  Motion has
already entered the pipeline through the Stage-1/2 STC noise shaper.  Stage 3
reuses the same spatio-temporal feature tensor and converts it into corrections
whose shapes exactly match BrushNet's down, mid, and optional up residuals.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

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


class STCResidualProjection(nn.Module):
    """Project one STC scale to a zero-initialized residual correction."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        bottleneck_channels: int,
    ):
        super().__init__()
        groups = _group_count(bottleneck_channels)
        self.input_projection = nn.Conv2d(
            input_channels, bottleneck_channels, kernel_size=1
        )
        self.residual = nn.Sequential(
            nn.GroupNorm(groups, bottleneck_channels),
            nn.SiLU(),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1),
            nn.GroupNorm(groups, bottleneck_channels),
            nn.SiLU(),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1),
        )
        self.output_projection = nn.Conv2d(
            bottleneck_channels, output_channels, kernel_size=1
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(feature)
        hidden = hidden + self.residual(hidden)
        return self.output_projection(hidden)


class STCBrushNetConditionAdapter(ModelMixin, ConfigMixin):
    """Create and fuse multi-scale STC corrections at BrushNet residual ports."""

    @register_to_config
    def __init__(
        self,
        input_channels: int,
        down_channels: Sequence[int],
        mid_channel: int,
        up_channels: Sequence[int],
        bottleneck_channels: int = 64,
        use_down: bool = True,
        use_mid: bool = True,
        use_up: bool = False,
        injection_scale: float = 1.0,
    ):
        super().__init__()
        if input_channels < 1 or bottleneck_channels < 1:
            raise ValueError("input_channels and bottleneck_channels must be positive")
        if not down_channels or mid_channel < 1:
            raise ValueError("BrushNet down and mid channel descriptions are required")
        if float(injection_scale) < 0.0:
            raise ValueError("injection_scale must be non-negative")

        self.down_projections = nn.ModuleList(
            [
                STCResidualProjection(
                    input_channels, int(channel), bottleneck_channels
                )
                for channel in down_channels
            ]
            if use_down
            else []
        )
        self.mid_projection = (
            STCResidualProjection(
                input_channels, int(mid_channel), bottleneck_channels
            )
            if use_mid
            else None
        )
        self.up_projections = nn.ModuleList(
            [
                STCResidualProjection(
                    input_channels, int(channel), bottleneck_channels
                )
                for channel in up_channels
            ]
            if use_up
            else []
        )

    @classmethod
    def from_brushnet(
        cls,
        brushnet,
        input_channels: int,
        bottleneck_channels: int = 64,
        use_down: bool = True,
        use_mid: bool = True,
        use_up: bool = False,
        injection_scale: float = 1.0,
    ):
        """Derive every target channel count from the deployed BrushNet."""

        return cls(
            input_channels=int(input_channels),
            down_channels=tuple(
                int(module.out_channels)
                for module in brushnet.brushnet_down_blocks
            ),
            mid_channel=int(brushnet.brushnet_mid_block.out_channels),
            up_channels=tuple(
                int(module.out_channels)
                for module in brushnet.brushnet_up_blocks
            ),
            bottleneck_channels=int(bottleneck_channels),
            use_down=bool(use_down),
            use_mid=bool(use_mid),
            use_up=bool(use_up),
            injection_scale=float(injection_scale),
        )

    @staticmethod
    def _validate_features(features: torch.Tensor) -> Tuple[int, int]:
        if features.ndim != 5:
            raise ValueError("stc_features must have shape [B,T,C,H,W]")
        batch, frames = features.shape[:2]
        if batch < 1 or frames < 1:
            raise ValueError("stc_features must contain at least one frame")
        return batch, frames

    @staticmethod
    def _project_one(
        flat_features: torch.Tensor,
        reference: torch.Tensor,
        projection: STCResidualProjection,
    ) -> torch.Tensor:
        if reference.ndim != 4:
            raise ValueError("BrushNet residual references must be NCHW tensors")
        if reference.shape[0] != flat_features.shape[0]:
            raise ValueError(
                "STC and BrushNet residual batches disagree: "
                f"{flat_features.shape[0]} vs {reference.shape[0]}"
            )
        feature = flat_features
        if feature.shape[-2:] != reference.shape[-2:]:
            feature = F.interpolate(
                feature,
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        correction = projection(feature)
        if correction.shape != reference.shape:
            raise ValueError(
                "Projected STC residual does not match BrushNet port: "
                f"{tuple(correction.shape)} vs {tuple(reference.shape)}"
            )
        # Residual ports are strict dtype/device interfaces. This is normally
        # handled by autocast during mixed-precision training, but the explicit
        # conversion also protects validation and external inference callers.
        return correction.to(device=reference.device, dtype=reference.dtype)

    def forward(
        self,
        stc_features: torch.Tensor,
        down_residuals: Sequence[torch.Tensor],
        mid_residual: torch.Tensor,
        up_residuals: Sequence[torch.Tensor],
        injection_scale: Optional[float] = None,
        return_dict: bool = True,
    ):
        batch, frames = self._validate_features(stc_features)
        if stc_features.shape[2] != int(self.config.input_channels):
            raise ValueError(
                f"Expected {self.config.input_channels} STC channels, "
                f"got {stc_features.shape[2]}"
            )
        flat_features = stc_features.reshape(
            batch * frames, *stc_features.shape[2:]
        )
        down_residuals = tuple(down_residuals)
        up_residuals = tuple(up_residuals)
        if self.down_projections and len(self.down_projections) != len(down_residuals):
            raise ValueError("Configured down projections do not match BrushNet outputs")
        if self.up_projections and len(self.up_projections) != len(up_residuals):
            raise ValueError("Configured up projections do not match BrushNet outputs")

        down_corrections = tuple(
            self._project_one(flat_features, reference, projection)
            for reference, projection in zip(down_residuals, self.down_projections)
        )
        mid_correction = (
            self._project_one(flat_features, mid_residual, self.mid_projection)
            if self.mid_projection is not None
            else None
        )
        up_corrections = tuple(
            self._project_one(flat_features, reference, projection)
            for reference, projection in zip(up_residuals, self.up_projections)
        )
        scale = (
            float(self.config.injection_scale)
            if injection_scale is None
            else float(injection_scale)
        )
        if scale < 0.0:
            raise ValueError("injection_scale must be non-negative")
        fused_down = tuple(
            reference + scale * correction
            for reference, correction in zip(down_residuals, down_corrections)
        )
        if not down_corrections:
            fused_down = down_residuals
        fused_mid = (
            mid_residual + scale * mid_correction
            if mid_correction is not None
            else mid_residual
        )
        fused_up = tuple(
            reference + scale * correction
            for reference, correction in zip(up_residuals, up_corrections)
        )
        if not up_corrections:
            fused_up = up_residuals

        mid_corrections = () if mid_correction is None else (mid_correction,)
        corrections = down_corrections + mid_corrections + up_corrections
        if corrections:
            element_count = sum(tensor.numel() for tensor in corrections)
            squared = sum(tensor.float().square().sum() for tensor in corrections)
            absolute = sum(tensor.float().abs().sum() for tensor in corrections)
            # Optimize the mean square directly. Backpropagating through
            # square(sqrt(mean(square(correction)))) at the exact zero
            # initialization produces the undefined product 0 * inf and NaN
            # gradients. RMS/absolute values are reporting metrics only.
            correction_energy = squared / max(element_count, 1)
            correction_rms = correction_energy.detach().sqrt()
            correction_abs = (absolute / max(element_count, 1)).detach()
        else:
            correction_energy = stc_features.new_zeros(())
            correction_rms = stc_features.new_zeros(())
            correction_abs = stc_features.new_zeros(())

        result: Dict[str, object] = {
            "down": fused_down,
            "mid": fused_mid,
            "up": fused_up,
            "down_corrections": down_corrections,
            "mid_correction": mid_correction,
            "up_corrections": up_corrections,
            "correction_energy": correction_energy,
            "correction_rms": correction_rms,
            "correction_abs": correction_abs,
        }
        if return_dict:
            return result
        return fused_down, fused_mid, fused_up
