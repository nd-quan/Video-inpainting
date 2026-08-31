"""V6: V5 plus first-order flow-guided modulated deformable alignment.

The module aligns *raw* adjacent spatial STC features once in each direction.
Predicted flow is the base DCN offset and a bounded learned residual refines
the sampling locations.  The resulting candidates are fused into the V5
feature only on reliable BG-to-BG correspondences, before V5 temporal
self-attention and exact-overlap cross-clip memory.

Flow convention
---------------
``flow_forward[:, i]`` is defined on frame ``i`` and samples frame ``i+1``.
``flow_backward[:, i]`` is defined on frame ``i+1`` and samples frame ``i``.
Both store ``[dx, dy]`` in feature-grid pixels.  Torchvision DCN offsets are
interleaved ``[dy, dx]`` for every deformable group and kernel point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from diffusers.configuration_utils import register_to_config

try:
    from STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (
        backward_warp_feature,
    )
    from STC_encoder_v5_relative_crossclip.relative_crossclip_stc_adapter import (
        RelativeCrossClipBGSTCAdapter,
        RelativeCrossClipRGBSTCOutput,
        TemporalMemoryState,
    )
except ModuleNotFoundError:  # Imported through examples.brushnet.
    from ..STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (
        backward_warp_feature,
    )
    from ..STC_encoder_v5_relative_crossclip.relative_crossclip_stc_adapter import (
        RelativeCrossClipBGSTCAdapter,
        RelativeCrossClipRGBSTCOutput,
        TemporalMemoryState,
    )


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(int(maximum), int(channels)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def flow_xy_to_dcn_offset(
    flow_xy: torch.Tensor,
    deform_groups: int,
    kernel_size: int,
) -> torch.Tensor:
    """Repeat ``[dx,dy]`` flow as DCN's interleaved ``[dy,dx]`` offsets."""
    if flow_xy.ndim != 4 or flow_xy.shape[1] != 2:
        raise ValueError("flow_xy must have shape [N,2,H,W]")
    deform_groups = int(deform_groups)
    kernel_size = int(kernel_size)
    if deform_groups < 1 or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("deform_groups must be positive and kernel_size odd")
    repeats = deform_groups * kernel_size * kernel_size
    flow_yx = flow_xy[:, [1, 0]]
    return (
        flow_yx[:, None]
        .expand(-1, repeats, -1, -1, -1)
        .reshape(flow_xy.shape[0], 2 * repeats, *flow_xy.shape[-2:])
        .contiguous()
    )


@dataclass
class DeformablePairOutput:
    aligned_source: torch.Tensor
    residual_offset: torch.Tensor
    modulation_mask: torch.Tensor
    reliability: torch.Tensor
    base_difference_abs_mean: torch.Tensor


class FlowGuidedModulatedDeformableAlignment(nn.Module):
    """One-shot raw-source DCNv2 alignment around a predicted-flow prior."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        deform_groups: int,
        residual_max_displacement: float,
        flow_max_displacement: Tuple[float, float],
        detach_reliability: bool = True,
    ):
        super().__init__()
        channels = int(channels)
        hidden_channels = int(hidden_channels)
        kernel_size = int(kernel_size)
        deform_groups = int(deform_groups)
        residual_max_displacement = float(residual_max_displacement)
        if channels < 1 or hidden_channels < 1:
            raise ValueError("channels and hidden_channels must be positive")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if deform_groups < 1 or channels % deform_groups:
            raise ValueError("deform_groups must divide channels")
        if not math.isfinite(residual_max_displacement) or residual_max_displacement <= 0:
            raise ValueError("residual_max_displacement must be finite and positive")
        flow_bounds = tuple(float(value) for value in flow_max_displacement)
        if len(flow_bounds) != 2 or any(value <= 0 for value in flow_bounds):
            raise ValueError("flow_max_displacement must contain positive (x,y)")

        self.channels = channels
        self.kernel_size = kernel_size
        self.deform_groups = deform_groups
        self.residual_max_displacement = residual_max_displacement
        self.flow_max_displacement = flow_bounds
        self.detach_reliability = bool(detach_reliability)

        # [target, base-warped source, difference, normalized flow,
        #  detached FB confidence, target BG, base-warped source BG].
        head_in_channels = 3 * channels + 5
        output_channels = 3 * deform_groups * kernel_size * kernel_size
        self.offset_mask_head = nn.Sequential(
            nn.Conv2d(head_in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, output_channels, 3, padding=1),
        )
        nn.init.zeros_(self.offset_mask_head[-1].weight)
        nn.init.zeros_(self.offset_mask_head[-1].bias)

        # Grouped DCN keeps T=16 memory/compute tractable.  At initialization,
        # center weight 2 times sigmoid(0)=0.5 reproduces one base-flow warp.
        self.weight = nn.Parameter(
            torch.zeros(
                channels,
                channels // deform_groups,
                kernel_size,
                kernel_size,
            )
        )
        self.bias = nn.Parameter(torch.zeros(channels))
        center = kernel_size // 2
        with torch.no_grad():
            for channel in range(channels):
                self.weight[
                    channel, channel % (channels // deform_groups), center, center
                ] = 2.0

    @property
    def offset_channels(self) -> int:
        return 2 * self.deform_groups * self.kernel_size * self.kernel_size

    @property
    def mask_channels(self) -> int:
        return self.deform_groups * self.kernel_size * self.kernel_size

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        flow_xy: torch.Tensor,
        confidence: torch.Tensor,
        source_bg: torch.Tensor,
        target_bg: torch.Tensor,
    ) -> DeformablePairOutput:
        if source.ndim != 4 or source.shape != target.shape:
            raise ValueError("source and target must share shape [N,C,H,W]")
        expected_scalar = (source.shape[0], 1, *source.shape[-2:])
        if confidence.shape != expected_scalar:
            raise ValueError("confidence must have shape [N,1,H,W]")
        if source_bg.shape != expected_scalar or target_bg.shape != expected_scalar:
            raise ValueError("source_bg and target_bg must have shape [N,1,H,W]")
        if flow_xy.shape != (source.shape[0], 2, *source.shape[-2:]):
            raise ValueError("flow_xy must match source batch/spatial dimensions")

        base_warp, valid = backward_warp_feature(source, flow_xy, fallback=target)
        warped_source_bg, _ = backward_warp_feature(source_bg, flow_xy)
        warped_source_bg = warped_source_bg.clamp(0.0, 1.0)
        confidence_input = confidence.detach() if self.detach_reliability else confidence
        bounds = flow_xy.new_tensor(self.flow_max_displacement).reshape(1, 2, 1, 1)
        normalized_flow = flow_xy / bounds
        head_input = torch.cat(
            (
                target,
                base_warp,
                target - base_warp,
                normalized_flow,
                confidence_input,
                target_bg,
                warped_source_bg,
            ),
            dim=1,
        )
        raw = self.offset_mask_head(head_input)
        raw_offset, mask_logits = raw.split(
            (self.offset_channels, self.mask_channels), dim=1
        )
        residual_offset = (
            torch.tanh(raw_offset) * self.residual_max_displacement
        )
        modulation_mask = torch.sigmoid(mask_logits)
        offset = flow_xy_to_dcn_offset(
            flow_xy, self.deform_groups, self.kernel_size
        ) + residual_offset
        deformed = deform_conv2d(
            source,
            offset,
            self.weight,
            self.bias,
            stride=(1, 1),
            padding=(self.kernel_size // 2, self.kernel_size // 2),
            dilation=(1, 1),
            mask=modulation_mask,
        )
        # A fully invalid base coordinate is never allowed to inject padded
        # zeros.  The reliability below also disables its final contribution.
        deformed = torch.where(valid, deformed, target)
        reliability = (
            target_bg
            * warped_source_bg
            * valid.to(target.dtype)
            * confidence_input.clamp(0.0, 1.0)
        )
        if self.detach_reliability:
            reliability = reliability.detach()
        with torch.no_grad():
            base_difference = (deformed - base_warp).float().abs().mean()
        return DeformablePairOutput(
            aligned_source=deformed,
            residual_offset=residual_offset,
            modulation_mask=modulation_mask,
            reliability=reliability,
            base_difference_abs_mean=base_difference,
        )


class BidirectionalDeformableResidualFusion(nn.Module):
    """Zero-initialized residual fusion of previous/current/next features."""

    def __init__(self, channels: int):
        super().__init__()
        channels = int(channels)
        self.hidden = nn.Sequential(
            nn.Conv2d(5 * channels + 2, 2 * channels, 3, padding=1),
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
        base: torch.Tensor,
        previous: torch.Tensor,
        following: torch.Tensor,
        previous_reliability: torch.Tensor,
        following_reliability: torch.Tensor,
        target_bg: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        if base.shape != previous.shape or base.shape != following.shape:
            raise ValueError("base/previous/following feature shapes must match")
        expected = (base.shape[0], 1, *base.shape[-2:])
        if any(
            value.shape != expected
            for value in (previous_reliability, following_reliability, target_bg)
        ):
            raise ValueError("reliability and target_bg must be [N,1,H,W]")
        scale = float(scale)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("deformable fusion scale must be finite and non-negative")
        fused = torch.cat(
            (
                base,
                previous,
                following,
                base - previous,
                base - following,
                previous_reliability,
                following_reliability,
            ),
            dim=1,
        )
        residual, gate = self.to_residual_and_gate(self.hidden(fused)).split(
            (base.shape[1], 1), dim=1
        )
        union_reliability = 1.0 - (
            1.0 - previous_reliability.clamp(0.0, 1.0)
        ) * (1.0 - following_reliability.clamp(0.0, 1.0))
        return base + (
            scale
            * target_bg
            * union_reliability
            * torch.sigmoid(gate)
            * residual
        )


@dataclass
class FlowGuidedDeformableRGBSTCOutput(RelativeCrossClipRGBSTCOutput):
    base_aligned_spatial_features: torch.Tensor
    deformed_previous_features: torch.Tensor
    deformed_next_features: torch.Tensor
    deform_reliability_backward: torch.Tensor
    deform_reliability_forward: torch.Tensor
    residual_offset_backward: torch.Tensor
    residual_offset_forward: torch.Tensor
    modulation_mask_backward: torch.Tensor
    modulation_mask_forward: torch.Tensor
    deformation_minus_base_abs_mean: torch.Tensor


class FlowGuidedDeformableBGSTCAdapter(RelativeCrossClipBGSTCAdapter):
    """V5 plus bidirectional first-order flow-guided deformable alignment."""

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
        relative_position_max_distance: int = 32,
        cross_clip_memory_frames: int = 4,
        detach_cross_clip_memory: bool = True,
        require_memory_overlap: bool = True,
        deform_hidden_channels: int = 128,
        deform_kernel_size: int = 3,
        deform_groups: int = 4,
        deform_residual_max_displacement: float = 2.0,
        detach_deform_reliability: bool = True,
    ):
        super().__init__(
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            downsample_factor=downsample_factor,
            output_channels=output_channels,
            condition_mode=condition_mode,
            flow_max_displacement=flow_max_displacement,
            flow_confidence_scale=flow_confidence_scale,
            detach_flow_confidence=detach_flow_confidence,
            relative_position_max_distance=relative_position_max_distance,
            cross_clip_memory_frames=cross_clip_memory_frames,
            detach_cross_clip_memory=detach_cross_clip_memory,
            require_memory_overlap=require_memory_overlap,
        )
        self.deformable_alignment = FlowGuidedModulatedDeformableAlignment(
            channels=int(hidden_channels),
            hidden_channels=int(deform_hidden_channels),
            kernel_size=int(deform_kernel_size),
            deform_groups=int(deform_groups),
            residual_max_displacement=float(deform_residual_max_displacement),
            flow_max_displacement=tuple(flow_max_displacement),
            detach_reliability=bool(detach_deform_reliability),
        )
        self.deformable_fusion = BidirectionalDeformableResidualFusion(
            int(hidden_channels)
        )

    @classmethod
    def from_v5_pretrained(
        cls,
        pretrained_model_path,
        deform_hidden_channels: int = 128,
        deform_kernel_size: int = 3,
        deform_groups: int = 4,
        deform_residual_max_displacement: float = 2.0,
        detach_deform_reliability: bool = True,
    ) -> "FlowGuidedDeformableBGSTCAdapter":
        """Upgrade one complete V5 component and audit every transferred key."""
        source = RelativeCrossClipBGSTCAdapter.from_pretrained(
            str(Path(pretrained_model_path).expanduser().resolve())
        )
        config = source.config
        model = cls(
            hidden_channels=int(config.hidden_channels),
            num_heads=int(config.num_heads),
            num_layers=int(config.num_layers),
            mlp_ratio=float(config.mlp_ratio),
            dropout=float(config.dropout),
            downsample_factor=int(config.downsample_factor),
            output_channels=int(config.output_channels),
            condition_mode=str(config.condition_mode),
            flow_max_displacement=tuple(config.flow_max_displacement),
            flow_confidence_scale=float(config.flow_confidence_scale),
            detach_flow_confidence=bool(config.detach_flow_confidence),
            relative_position_max_distance=int(config.relative_position_max_distance),
            cross_clip_memory_frames=int(config.cross_clip_memory_frames),
            detach_cross_clip_memory=bool(config.detach_cross_clip_memory),
            require_memory_overlap=bool(config.require_memory_overlap),
            deform_hidden_channels=int(deform_hidden_channels),
            deform_kernel_size=int(deform_kernel_size),
            deform_groups=int(deform_groups),
            deform_residual_max_displacement=float(
                deform_residual_max_displacement
            ),
            detach_deform_reliability=bool(detach_deform_reliability),
        )
        source_state = source.state_dict()
        transfer = model.load_state_dict(source_state, strict=False)
        expected_missing = set(model.state_dict()) - set(source_state)
        if (
            set(transfer.missing_keys) != expected_missing
            or transfer.unexpected_keys
            or any(
                not key.startswith(("deformable_alignment.", "deformable_fusion."))
                for key in expected_missing
            )
        ):
            raise RuntimeError(
                "V5 -> V6 transfer mismatch: "
                f"missing={transfer.missing_keys}, "
                f"unexpected={transfer.unexpected_keys}"
            )
        return model

    def deformable_parameters(self):
        yield from self.deformable_alignment.parameters()
        yield from self.deformable_fusion.parameters()

    def _forward_confidence(
        self,
        spatial: torch.Tensor,
        flow_forward: torch.Tensor,
        flow_backward: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames, channels, height, width = spatial.shape
        if frames <= 1:
            return spatial.new_zeros(batch, 0, 1, height, width)
        previous = spatial[:, :-1].reshape(-1, channels, height, width)
        current = spatial[:, 1:].reshape(-1, channels, height, width)
        forward = flow_forward.reshape(-1, 2, height, width)
        backward = flow_backward.reshape(-1, 2, height, width)
        _, feature_valid = backward_warp_feature(current, forward, fallback=previous)
        warped_backward, flow_valid = backward_warp_feature(backward, forward)
        error = forward + warped_backward
        scale = float(self.config.flow_confidence_scale)
        confidence = torch.exp(
            -error.square().sum(dim=1, keepdim=True) / (2.0 * scale * scale)
        )
        confidence = confidence * (feature_valid & flow_valid).to(confidence.dtype)
        if bool(self.config.detach_flow_confidence):
            confidence = confidence.detach()
        return confidence.reshape(batch, frames - 1, 1, height, width)

    def _deformable_spatial_alignment(
        self,
        spatial: torch.Tensor,
        base_aligned: torch.Tensor,
        flow_forward: torch.Tensor,
        flow_backward: torch.Tensor,
        confidence_backward: torch.Tensor,
        bg_mask_sequence: torch.Tensor,
        fusion_scale: float,
    ):
        batch, frames, channels, height, width = spatial.shape
        pairs = max(frames - 1, 0)
        offset_channels = self.deformable_alignment.offset_channels
        mask_channels = self.deformable_alignment.mask_channels
        if pairs == 0:
            empty_feature = spatial.new_empty(batch, 0, channels, height, width)
            empty_weight = spatial.new_empty(batch, 0, 1, height, width)
            empty_offset = spatial.new_empty(
                batch, 0, offset_channels, height, width
            )
            empty_mask = spatial.new_empty(batch, 0, mask_channels, height, width)
            return (
                base_aligned,
                empty_feature,
                empty_feature,
                empty_weight,
                empty_weight,
                empty_offset,
                empty_offset,
                empty_mask,
                empty_mask,
                spatial.new_zeros(()),
            )

        bg = F.interpolate(
            bg_mask_sequence.flatten(0, 1).to(
                device=spatial.device, dtype=spatial.dtype
            ),
            size=(height, width),
            mode="nearest",
        ).reshape(batch, frames, 1, height, width)
        bg = (bg >= 0.5).to(spatial.dtype)
        confidence_forward = self._forward_confidence(
            spatial, flow_forward, flow_backward
        )

        previous = spatial[:, :-1].reshape(-1, channels, height, width)
        current = spatial[:, 1:].reshape(-1, channels, height, width)
        previous_bg = bg[:, :-1].reshape(-1, 1, height, width)
        current_bg = bg[:, 1:].reshape(-1, 1, height, width)
        backward = flow_backward.reshape(-1, 2, height, width)
        forward = flow_forward.reshape(-1, 2, height, width)
        backward_pair = self.deformable_alignment(
            source=previous,
            target=current,
            flow_xy=backward,
            confidence=confidence_backward.reshape(-1, 1, height, width),
            source_bg=previous_bg,
            target_bg=current_bg,
        )
        forward_pair = self.deformable_alignment(
            source=current,
            target=previous,
            flow_xy=forward,
            confidence=confidence_forward.reshape(-1, 1, height, width),
            source_bg=current_bg,
            target_bg=previous_bg,
        )

        deformed_previous = backward_pair.aligned_source.reshape(
            batch, pairs, channels, height, width
        )
        deformed_next = forward_pair.aligned_source.reshape(
            batch, pairs, channels, height, width
        )
        reliability_backward = backward_pair.reliability.reshape(
            batch, pairs, 1, height, width
        )
        reliability_forward = forward_pair.reliability.reshape(
            batch, pairs, 1, height, width
        )

        previous_full = base_aligned.clone()
        following_full = base_aligned.clone()
        previous_reliability = spatial.new_zeros(batch, frames, 1, height, width)
        following_reliability = spatial.new_zeros(batch, frames, 1, height, width)
        previous_full[:, 1:] = deformed_previous
        following_full[:, :-1] = deformed_next
        previous_reliability[:, 1:] = reliability_backward
        following_reliability[:, :-1] = reliability_forward
        fused = self.deformable_fusion(
            base_aligned.flatten(0, 1),
            previous_full.flatten(0, 1),
            following_full.flatten(0, 1),
            previous_reliability.flatten(0, 1),
            following_reliability.flatten(0, 1),
            bg.flatten(0, 1),
            scale=fusion_scale,
        ).reshape_as(base_aligned)
        with torch.no_grad():
            difference = 0.5 * (
                backward_pair.base_difference_abs_mean
                + forward_pair.base_difference_abs_mean
            )
        return (
            fused,
            deformed_previous,
            deformed_next,
            reliability_backward,
            reliability_forward,
            backward_pair.residual_offset.reshape(
                batch, pairs, offset_channels, height, width
            ),
            forward_pair.residual_offset.reshape(
                batch, pairs, offset_channels, height, width
            ),
            backward_pair.modulation_mask.detach().reshape(
                batch, pairs, mask_channels, height, width
            ),
            forward_pair.modulation_mask.detach().reshape(
                batch, pairs, mask_channels, height, width
            ),
            difference,
        )

    def encode(
        self,
        rgb_sequence: torch.Tensor,
        bg_mask_sequence: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        frame_ids: Optional[torch.Tensor] = None,
        temporal_memory: Optional[TemporalMemoryState] = None,
        frame_valid_mask: Optional[torch.Tensor] = None,
        deformable_alignment_scale: float = 1.0,
    ):
        batch, frames, height, width = self.stc_adapter._validate_inputs(
            rgb_sequence, bg_mask_sequence
        )
        if output_size is None:
            output_size = (height // 8, width // 8)
        output_size = (int(output_size[0]), int(output_size[1]))
        if min(output_size) < 1:
            raise ValueError("output_size must contain positive dimensions")
        frame_ids = self._frame_ids(frame_ids, batch, frames, rgb_sequence.device)
        if frame_valid_mask is None:
            frame_valid_mask = torch.ones_like(frame_ids, dtype=torch.bool)
        elif tuple(frame_valid_mask.shape) != (batch, frames):
            raise ValueError("frame_valid_mask must match frame_ids")
        else:
            frame_valid_mask = frame_valid_mask.to(
                device=frame_ids.device, dtype=torch.bool
            )

        (
            spatial,
            base_aligned,
            flow_forward,
            flow_backward,
            confidence,
        ) = super()._spatial_and_bg_alignment(
            rgb_sequence, bg_mask_sequence, output_size
        )
        (
            aligned,
            deformed_previous,
            deformed_next,
            reliability_backward,
            reliability_forward,
            residual_backward,
            residual_forward,
            mask_backward,
            mask_forward,
            deformation_minus_base,
        ) = self._deformable_spatial_alignment(
            spatial,
            base_aligned,
            flow_forward,
            flow_backward,
            confidence,
            bg_mask_sequence,
            deformable_alignment_scale,
        )

        _, _, channels, encoded_height, encoded_width = aligned.shape
        memory, overlap_count = self._validated_memory(
            temporal_memory, frame_ids, channels, encoded_height, encoded_width
        )
        tokens = aligned.permute(0, 3, 4, 1, 2).reshape(
            batch * encoded_height * encoded_width, frames, channels
        )
        token_frame_ids = self._expand_ids(frame_ids, encoded_height, encoded_width)
        memory_frame_ids = memory_valid = None
        if memory is not None:
            memory_frame_ids = self._expand_ids(
                memory.frame_ids, encoded_height, encoded_width
            )
            memory_valid = self._expand_valid(
                memory.valid_mask, encoded_height, encoded_width
            )

        memory_frames = min(int(self.config.cross_clip_memory_frames), frames)
        next_layer_features = []
        for layer_index, temporal_block in enumerate(
            self.stc_adapter.temporal_blocks
        ):
            memory_tokens = None
            if memory is not None:
                memory_feature = memory.layer_features[layer_index].to(
                    device=tokens.device, dtype=tokens.dtype
                )
                memory_tokens = memory_feature.permute(0, 3, 4, 1, 2).reshape(
                    batch * encoded_height * encoded_width,
                    memory_feature.shape[1],
                    channels,
                )
            tokens = temporal_block(
                tokens,
                token_frame_ids,
                memory_tokens=memory_tokens,
                memory_frame_ids=memory_frame_ids,
                memory_valid_mask=memory_valid,
            )
            layer_sequence = tokens.reshape(
                batch, encoded_height, encoded_width, frames, channels
            ).permute(0, 3, 4, 1, 2)
            cached = layer_sequence[:, -memory_frames:]
            if bool(self.config.detach_cross_clip_memory):
                cached = cached.detach()
            next_layer_features.append(cached)

        features = tokens.reshape(
            batch, encoded_height, encoded_width, frames, channels
        ).permute(0, 3, 4, 1, 2)
        features_flat = self.stc_adapter.output_act(
            self.stc_adapter.output_norm(features.flatten(0, 1))
        )
        delta = self.stc_adapter.zero_conv(features_flat).reshape(
            batch, frames, 4, encoded_height, encoded_width
        )
        next_memory = TemporalMemoryState(
            frame_ids=frame_ids[:, -memory_frames:].detach(),
            valid_mask=frame_valid_mask[:, -memory_frames:].detach(),
            layer_features=tuple(next_layer_features),
        )
        return (
            delta,
            features,
            spatial,
            aligned,
            base_aligned,
            flow_forward,
            flow_backward,
            confidence,
            next_memory,
            overlap_count,
            deformed_previous,
            deformed_next,
            reliability_backward,
            reliability_forward,
            residual_backward,
            residual_forward,
            mask_backward,
            mask_forward,
            deformation_minus_base,
        )

    def forward(
        self,
        rgb_sequence: torch.Tensor,
        bg_mask_sequence: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        predict_flow: bool = True,
        return_dict: bool = True,
        frame_ids: Optional[torch.Tensor] = None,
        temporal_memory: Optional[TemporalMemoryState] = None,
        frame_valid_mask: Optional[torch.Tensor] = None,
        deformable_alignment_scale: float = 1.0,
    ):
        values = self.encode(
            rgb_sequence,
            bg_mask_sequence,
            output_size=output_size,
            frame_ids=frame_ids,
            temporal_memory=temporal_memory,
            frame_valid_mask=frame_valid_mask,
            deformable_alignment_scale=deformable_alignment_scale,
        )
        (
            delta,
            features,
            spatial,
            aligned,
            base_aligned,
            flow_forward,
            flow_backward,
            confidence,
            next_memory,
            overlap_count,
            deformed_previous,
            deformed_next,
            reliability_backward,
            reliability_forward,
            residual_backward,
            residual_forward,
            mask_backward,
            mask_forward,
            deformation_minus_base,
        ) = values
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
        with torch.no_grad():
            bias_values = torch.stack(
                [
                    block.attention.relative_position_bias.abs().mean()
                    for block in self.stc_adapter.temporal_blocks
                ]
            ).mean()
            gate_values = torch.stack(
                [
                    torch.tanh(block.cross_clip_gate).abs()
                    for block in self.stc_adapter.temporal_blocks
                ]
            ).mean()
        result = FlowGuidedDeformableRGBSTCOutput(
            delta_bg=delta * latent_bg_mask,
            features=features,
            spatial_features=spatial,
            aligned_spatial_features=aligned,
            latent_bg_mask=latent_bg_mask,
            predicted_flow_forward=flow_forward if predict_flow else None,
            predicted_flow_backward=flow_backward if predict_flow else None,
            alignment_confidence=confidence,
            temporal_memory=next_memory,
            memory_overlap_count=overlap_count,
            relative_bias_abs_mean=bias_values,
            cross_clip_gate_abs_mean=gate_values,
            base_aligned_spatial_features=base_aligned,
            deformed_previous_features=deformed_previous,
            deformed_next_features=deformed_next,
            deform_reliability_backward=reliability_backward,
            deform_reliability_forward=reliability_forward,
            residual_offset_backward=residual_backward,
            residual_offset_forward=residual_forward,
            modulation_mask_backward=mask_backward,
            modulation_mask_forward=mask_forward,
            deformation_minus_base_abs_mean=deformation_minus_base,
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
            result.temporal_memory,
        )


def augment_brushnet_condition_v6(
    model: FlowGuidedDeformableBGSTCAdapter,
    base_condition_latents: torch.Tensor,
    rgb_sequence: torch.Tensor,
    bg_mask_sequence: torch.Tensor,
    injection_scale: float = 1.0,
    predict_flow: bool = True,
    frame_ids: Optional[torch.Tensor] = None,
    previous_rgb_sequence: Optional[torch.Tensor] = None,
    previous_bg_mask_sequence: Optional[torch.Tensor] = None,
    previous_frame_ids: Optional[torch.Tensor] = None,
    previous_valid_mask: Optional[torch.Tensor] = None,
    deformable_alignment_scale: float = 1.0,
):
    """V5's stateless predecessor/current contract with V6 deformation."""
    if base_condition_latents.ndim != 4 or base_condition_latents.shape[1] != 4:
        raise ValueError("base_condition_latents must have shape [B*T,4,h,w]")
    batch, frames = rgb_sequence.shape[:2]
    if base_condition_latents.shape[0] != batch * frames:
        raise ValueError("base_condition_latents batch must equal B*T")

    temporal_memory = None
    if previous_rgb_sequence is not None:
        if previous_bg_mask_sequence is None or previous_frame_ids is None:
            raise ValueError("Previous RGB requires previous masks and frame IDs")
        if previous_valid_mask is None:
            previous_valid_mask = torch.ones_like(
                previous_frame_ids, dtype=torch.bool
            )
        if bool(previous_valid_mask.any()):
            bare_model = model.module if hasattr(model, "module") else model
            with torch.no_grad():
                previous_output = bare_model(
                    previous_rgb_sequence,
                    previous_bg_mask_sequence,
                    output_size=base_condition_latents.shape[-2:],
                    predict_flow=False,
                    return_dict=True,
                    frame_ids=previous_frame_ids,
                    frame_valid_mask=previous_valid_mask,
                    deformable_alignment_scale=deformable_alignment_scale,
                )
            temporal_memory = previous_output.temporal_memory.detach()

    output = model(
        rgb_sequence,
        bg_mask_sequence,
        output_size=base_condition_latents.shape[-2:],
        predict_flow=predict_flow,
        return_dict=True,
        frame_ids=frame_ids,
        temporal_memory=temporal_memory,
        deformable_alignment_scale=deformable_alignment_scale,
    )
    base_sequence = base_condition_latents.reshape(
        batch, frames, 4, *base_condition_latents.shape[-2:]
    )
    delta = output.delta_bg.to(dtype=base_sequence.dtype)
    latent_bg_mask = output.latent_bg_mask.to(dtype=base_sequence.dtype)
    augmented = base_sequence + float(injection_scale) * delta
    brushnet_condition = torch.cat((augmented, latent_bg_mask), dim=2).flatten(0, 1)
    return brushnet_condition, output, augmented
