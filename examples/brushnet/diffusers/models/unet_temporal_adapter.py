"""Zero-initialized temporal residual adapters for a frozen 2D diffusion U-Net.

The adapter keeps the deployed image U-Net and BrushNet checkpoints unchanged.
It receives flattened video features ``[B*T,C,H,W]``, performs only temporal
mixing with a factorized ``(3,1,1)`` convolution, and returns a residual update.
The final projection of every block is initialized to zero, so enabling a new
adapter is an exact identity operation before training.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from .modeling_utils import ModelMixin


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class TemporalConvResidualBlock(nn.Module):
    """Bottlenecked local temporal mixer with an identity initialization."""

    def __init__(
        self,
        channels: int,
        bottleneck_channels: int = 64,
        temporal_kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if channels <= 0 or bottleneck_channels <= 0:
            raise ValueError("channels and bottleneck_channels must be positive")
        if temporal_kernel_size <= 0 or temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd integer")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")

        groups = _group_count(channels)
        temporal_padding = temporal_kernel_size // 2
        self.input_norm = nn.GroupNorm(groups, channels)
        self.input_projection = nn.Conv2d(channels, bottleneck_channels, 1)
        self.input_activation = nn.SiLU()
        # Depth-wise temporal filtering preserves spatial locations. A cheap
        # point-wise projection then allows communication between bottleneck
        # channels without introducing spatial convolution in this module.
        self.temporal_depthwise = nn.Conv3d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=(temporal_kernel_size, 1, 1),
            padding=(temporal_padding, 0, 0),
            groups=bottleneck_channels,
        )
        self.temporal_pointwise = nn.Conv3d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=1,
        )
        self.temporal_activation = nn.SiLU()
        self.dropout = nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity()
        self.output_projection = nn.Conv2d(bottleneck_channels, channels, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        batch_size: int,
        num_frames: int,
        scale: float = 1.0,
    ) -> torch.Tensor:
        if hidden_states.ndim != 4:
            raise ValueError("hidden_states must have shape [B*T,C,H,W]")
        if batch_size <= 0 or num_frames <= 0:
            raise ValueError("batch_size and num_frames must be positive")
        if hidden_states.shape[0] != batch_size * num_frames:
            raise ValueError(
                f"feature batch {hidden_states.shape[0]} does not match "
                f"batch_size*num_frames={batch_size}*{num_frames}"
            )
        # Preserve the exact image-model path for single-frame inference even
        # after the adapter has been trained.
        if num_frames == 1 or float(scale) == 0.0:
            return hidden_states

        frames = self.input_activation(
            self.input_projection(self.input_norm(hidden_states))
        )
        _, channels, height, width = frames.shape
        video = frames.reshape(
            batch_size, num_frames, channels, height, width
        ).permute(0, 2, 1, 3, 4)
        mixed = self.temporal_depthwise(video)
        mixed = self.temporal_activation(self.temporal_pointwise(mixed))
        mixed = self.dropout(mixed)
        mixed = mixed.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
        residual = self.output_projection(mixed)
        return hidden_states + float(scale) * residual


class DiffusionUNetTemporalAdapter(ModelMixin, ConfigMixin):
    """Multi-scale temporal adapters called from selected 2D U-Net stages.

    The default placement targets the outputs of down blocks 0, 1, and 2 plus
    the middle block. For a 512-pixel SD 1.5 input these correspond to latent
    feature resolutions 32, 16, 8, and 8. Up-block adapters are supported for
    ablations but intentionally disabled by default.
    """

    @register_to_config
    def __init__(
        self,
        block_out_channels: Sequence[int] = (320, 640, 1280, 1280),
        down_block_indices: Sequence[int] = (0, 1, 2),
        use_mid: bool = True,
        up_block_indices: Sequence[int] = (),
        bottleneck_channels: int = 64,
        temporal_kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        channels = tuple(int(value) for value in block_out_channels)
        if not channels or any(value <= 0 for value in channels):
            raise ValueError("block_out_channels must contain positive integers")
        down_indices = tuple(int(value) for value in down_block_indices)
        up_indices = tuple(int(value) for value in up_block_indices)
        valid = set(range(len(channels)))
        if not set(down_indices).issubset(valid):
            raise ValueError("down_block_indices contains an invalid block index")
        if not set(up_indices).issubset(valid):
            raise ValueError("up_block_indices contains an invalid block index")
        if len(set(down_indices)) != len(down_indices):
            raise ValueError("down_block_indices must not contain duplicates")
        if len(set(up_indices)) != len(up_indices):
            raise ValueError("up_block_indices must not contain duplicates")

        block_kwargs = {
            "bottleneck_channels": int(bottleneck_channels),
            "temporal_kernel_size": int(temporal_kernel_size),
            "dropout": float(dropout),
        }
        self.down_adapters = nn.ModuleDict(
            {
                str(index): TemporalConvResidualBlock(channels[index], **block_kwargs)
                for index in down_indices
            }
        )
        self.mid_adapter = (
            TemporalConvResidualBlock(channels[-1], **block_kwargs)
            if use_mid
            else None
        )
        up_channels = tuple(reversed(channels))
        self.up_adapters = nn.ModuleDict(
            {
                str(index): TemporalConvResidualBlock(
                    up_channels[index], **block_kwargs
                )
                for index in up_indices
            }
        )

    @classmethod
    def from_unet(cls, unet, **kwargs):
        """Build an adapter whose channels match a loaded 2D U-Net."""
        return cls(
            block_out_channels=tuple(unet.config.block_out_channels),
            **kwargs,
        )

    def has_adapter(self, stage: str, block_index: Optional[int] = None) -> bool:
        stage = str(stage).lower()
        if stage == "down":
            return str(block_index) in self.down_adapters
        if stage == "mid":
            return self.mid_adapter is not None
        if stage == "up":
            return str(block_index) in self.up_adapters
        raise ValueError("stage must be 'down', 'mid', or 'up'")

    def forward(
        self,
        hidden_states: torch.Tensor,
        batch_size: int,
        num_frames: int,
        stage: str,
        block_index: Optional[int] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        stage = str(stage).lower()
        if stage == "down":
            key = str(block_index)
            adapter = self.down_adapters[key] if key in self.down_adapters else None
        elif stage == "mid":
            adapter = self.mid_adapter
        elif stage == "up":
            key = str(block_index)
            adapter = self.up_adapters[key] if key in self.up_adapters else None
        else:
            raise ValueError("stage must be 'down', 'mid', or 'up'")
        if adapter is None:
            return hidden_states
        return adapter(hidden_states, batch_size, num_frames, scale=scale)
