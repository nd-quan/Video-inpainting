"""V8-R: high-resolution RAFT pre-warp and native-grid DCNv2 refinement.

The frozen RAFT student still predicts RGB-pixel forward/backward flow.  For
each adjacent pair, the raw spatial STC source feature is nearest-upsampled,
warped with flow resized *directly from RGB* to that intermediate grid, and
nearest-downsampled.  DCNv2 then learns only a bounded residual offset around
the resulting motion-compensated feature.  The RAFT flow is therefore never
applied twice.

Forward/backward consistency is retained as a detached diagnostic in the V8
output, but it is intentionally absent from the active DCN head and fusion
reliability.  Active support is target BG x warped-source BG x in-bounds.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from diffusers.configuration_utils import register_to_config
from diffusers.models.brushnet_motion_adapter import resize_flow

try:
    from STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (
        backward_warp_feature,
    )
    from STC_encoder_v6_flow_deformable.flow_guided_deformable_stc_adapter import (
        BidirectionalDeformableResidualFusion,
        DeformablePairOutput,
        FlowGuidedModulatedDeformableAlignment,
    )
    from STC_encoder_v8_raft_deformable.raft_guided_deformable_stc_adapter import (
        RAFTGuidedDeformableBGSTCAdapter,
        augment_brushnet_condition_v8,
    )
except ModuleNotFoundError:  # Imported through examples.brushnet.
    from ..STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (
        backward_warp_feature,
    )
    from ..STC_encoder_v6_flow_deformable.flow_guided_deformable_stc_adapter import (
        BidirectionalDeformableResidualFusion,
        DeformablePairOutput,
        FlowGuidedModulatedDeformableAlignment,
    )
    from ..STC_encoder_v8_raft_deformable.raft_guided_deformable_stc_adapter import (
        RAFTGuidedDeformableBGSTCAdapter,
        augment_brushnet_condition_v8,
    )


class RescaledPrewarpModulatedDeformableAlignment(
    FlowGuidedModulatedDeformableAlignment
):
    """DCNv2 residual refinement of a source that was already flow-warped."""

    def forward_prewarped(
        self,
        prewarped_source: torch.Tensor,
        target: torch.Tensor,
        flow_xy_native: torch.Tensor,
        warped_source_bg: torch.Tensor,
        target_bg: torch.Tensor,
        valid: torch.Tensor,
    ) -> DeformablePairOutput:
        if prewarped_source.ndim != 4 or prewarped_source.shape != target.shape:
            raise ValueError("prewarped_source and target must share [N,C,H,W]")
        expected_scalar = (target.shape[0], 1, *target.shape[-2:])
        if warped_source_bg.shape != expected_scalar or target_bg.shape != expected_scalar:
            raise ValueError("BG masks must have shape [N,1,H,W]")
        if valid.shape != expected_scalar or valid.dtype != torch.bool:
            raise ValueError("valid must be a bool tensor with shape [N,1,H,W]")
        if flow_xy_native.shape != (target.shape[0], 2, *target.shape[-2:]):
            raise ValueError("native flow must match target batch/spatial dimensions")

        # Preserve the old head schema for checkpoint compatibility, but make
        # it explicit that FB confidence is not an active model input.
        confidence_input = torch.ones_like(target_bg)
        bounds = flow_xy_native.new_tensor(self.flow_max_displacement).reshape(
            1, 2, 1, 1
        )
        normalized_flow = flow_xy_native / bounds
        head_input = torch.cat(
            (
                target,
                prewarped_source,
                target - prewarped_source,
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
        residual_offset = torch.tanh(raw_offset) * self.residual_max_displacement
        modulation_mask = torch.sigmoid(mask_logits)

        # The source is already motion compensated.  Adding RAFT flow here
        # would apply the same motion twice, so DCN receives residual offsets
        # only.
        deformed = deform_conv2d(
            prewarped_source,
            residual_offset,
            self.weight,
            self.bias,
            stride=(1, 1),
            padding=(self.kernel_size // 2, self.kernel_size // 2),
            dilation=(1, 1),
            mask=modulation_mask,
        )
        deformed = torch.where(valid, deformed, target)
        reliability = (
            target_bg
            * warped_source_bg.clamp(0.0, 1.0)
            * valid.to(target.dtype)
        )
        if self.detach_reliability:
            reliability = reliability.detach()
        with torch.no_grad():
            base_difference = (
                deformed - prewarped_source
            ).float().abs().mean()
        return DeformablePairOutput(
            aligned_source=deformed,
            residual_offset=residual_offset,
            modulation_mask=modulation_mask,
            reliability=reliability,
            base_difference_abs_mean=base_difference,
        )


class RescaledRAFTGuidedDeformableBGSTCAdapter(
    RAFTGuidedDeformableBGSTCAdapter
):
    """V8 with xS flow pre-warp, geometric-only support, and native DCNv2."""

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
        deformable_alignment_direction: str = "bidirectional",
        rescaled_warp_scale: int = 4,
        rescaled_warp_up_mode: str = "nearest",
        rescaled_warp_down_mode: str = "nearest",
        deform_reliability_mode: str = "geometric_only",
    ):
        scale = int(rescaled_warp_scale)
        if scale < 1:
            raise ValueError("rescaled_warp_scale must be >= 1")
        if rescaled_warp_up_mode != "nearest" or rescaled_warp_down_mode != "nearest":
            raise ValueError("V8-R currently requires nearest up/down sampling")
        if deform_reliability_mode != "geometric_only":
            raise ValueError("V8-R supports only geometric_only reliability")
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
            deform_hidden_channels=deform_hidden_channels,
            deform_kernel_size=deform_kernel_size,
            deform_groups=deform_groups,
            deform_residual_max_displacement=deform_residual_max_displacement,
            detach_deform_reliability=detach_deform_reliability,
            deformable_alignment_direction=deformable_alignment_direction,
        )
        self._replace_deformable_alignment()

    def _new_deformable_alignment(self):
        return RescaledPrewarpModulatedDeformableAlignment(
            channels=int(self.config.hidden_channels),
            hidden_channels=int(self.config.deform_hidden_channels),
            kernel_size=int(self.config.deform_kernel_size),
            deform_groups=int(self.config.deform_groups),
            residual_max_displacement=float(
                self.config.deform_residual_max_displacement
            ),
            flow_max_displacement=tuple(self.config.flow_max_displacement),
            detach_reliability=bool(self.config.detach_deform_reliability),
        )

    def _replace_deformable_alignment(self) -> None:
        reference = next(self.parameters(), None)
        module = self._new_deformable_alignment()
        if reference is not None:
            module.to(device=reference.device, dtype=reference.dtype)
        self.deformable_alignment = module

    def reset_rescaled_deformable_branch(self) -> None:
        """Reset geometry-dependent DCN/fusion while retaining V8 temporal weights."""
        reference = next(self.parameters())
        alignment = self._new_deformable_alignment().to(
            device=reference.device, dtype=reference.dtype
        )
        fusion = BidirectionalDeformableResidualFusion(
            int(self.config.hidden_channels)
        ).to(device=reference.device, dtype=reference.dtype)
        self.deformable_alignment = alignment
        self.deformable_fusion = fusion

    @classmethod
    def from_v8_pretrained(
        cls,
        pretrained_model_path,
        *,
        rescaled_warp_scale: int = 4,
        reset_deformable_branch: bool = True,
    ) -> "RescaledRAFTGuidedDeformableBGSTCAdapter":
        source = RAFTGuidedDeformableBGSTCAdapter.from_pretrained(
            str(Path(pretrained_model_path).expanduser().resolve())
        )
        config = source.config
        keys = (
            "hidden_channels", "num_heads", "num_layers", "mlp_ratio", "dropout",
            "downsample_factor", "output_channels", "condition_mode",
            "flow_max_displacement", "flow_confidence_scale", "detach_flow_confidence",
            "relative_position_max_distance", "cross_clip_memory_frames",
            "detach_cross_clip_memory", "require_memory_overlap",
            "deform_hidden_channels", "deform_kernel_size", "deform_groups",
            "deform_residual_max_displacement", "detach_deform_reliability",
        )
        kwargs = {key: getattr(config, key) for key in keys}
        kwargs["deformable_alignment_direction"] = getattr(
            config, "deformable_alignment_direction", "bidirectional"
        )
        model = cls(
            **kwargs,
            rescaled_warp_scale=int(rescaled_warp_scale),
        )
        transfer = model.load_state_dict(source.state_dict(), strict=True)
        if transfer.missing_keys or transfer.unexpected_keys:
            raise RuntimeError("V8 -> V8-R strict transfer unexpectedly failed")
        if reset_deformable_branch:
            model.reset_rescaled_deformable_branch()
        return model

    def _rescaled_prewarp(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_bg: torch.Tensor,
        flow_rgb: torch.Tensor,
    ):
        native_size = source.shape[-2:]
        scale = int(self.config.rescaled_warp_scale)
        high_size = (native_size[0] * scale, native_size[1] * scale)
        source_up = F.interpolate(source, size=high_size, mode="nearest")
        target_up = F.interpolate(target, size=high_size, mode="nearest")
        source_bg_up = F.interpolate(source_bg, size=high_size, mode="nearest")
        # Resize directly from the original RGB flow.  resize_flow performs
        # both spatial interpolation and dx/dy unit conversion.
        flow_high = resize_flow(flow_rgb.detach().float(), high_size).to(
            device=source.device, dtype=source.dtype
        )
        warped_up, valid_up = backward_warp_feature(
            source_up, flow_high, fallback=target_up
        )
        warped_bg_up, _ = backward_warp_feature(source_bg_up, flow_high)
        prewarped = F.interpolate(warped_up, size=native_size, mode="nearest")
        warped_source_bg = F.interpolate(
            warped_bg_up, size=native_size, mode="nearest"
        ).clamp(0.0, 1.0)
        valid = F.interpolate(
            valid_up.to(source.dtype), size=native_size, mode="nearest"
        ) >= 0.5
        return prewarped, warped_source_bg, valid

    def _raft_deformable_spatial_alignment(
        self,
        spatial: torch.Tensor,
        base_aligned: torch.Tensor,
        flow_forward: torch.Tensor,
        flow_backward: torch.Tensor,
        confidence_backward: torch.Tensor,
        bg_mask_sequence: torch.Tensor,
        fusion_scale: float,
        raft_flow_forward_rgb: torch.Tensor,
        raft_flow_backward_rgb: torch.Tensor,
    ):
        # confidence_backward remains part of the returned V8 diagnostics but
        # is deliberately excluded from the active V8-R alignment path.
        del confidence_backward
        batch, frames, channels, height, width = spatial.shape
        pairs = max(frames - 1, 0)
        offset_channels = self.deformable_alignment.offset_channels
        mask_channels = self.deformable_alignment.mask_channels
        if pairs == 0:
            empty_feature = spatial.new_empty(batch, 0, channels, height, width)
            empty_weight = spatial.new_empty(batch, 0, 1, height, width)
            empty_offset = spatial.new_empty(batch, 0, offset_channels, height, width)
            empty_mask = spatial.new_empty(batch, 0, mask_channels, height, width)
            return (
                base_aligned, empty_feature, empty_feature, empty_weight,
                empty_weight, empty_offset, empty_offset, empty_mask,
                empty_mask, spatial.new_zeros(()),
            )

        bg = F.interpolate(
            bg_mask_sequence.flatten(0, 1).to(
                device=spatial.device, dtype=spatial.dtype
            ),
            size=(height, width),
            mode="nearest",
        ).reshape(batch, frames, 1, height, width)
        bg = (bg >= 0.5).to(spatial.dtype)

        previous = spatial[:, :-1].reshape(-1, channels, height, width)
        current = spatial[:, 1:].reshape(-1, channels, height, width)
        previous_bg = bg[:, :-1].reshape(-1, 1, height, width)
        current_bg = bg[:, 1:].reshape(-1, 1, height, width)
        backward_native = flow_backward.reshape(-1, 2, height, width)
        forward_native = flow_forward.reshape(-1, 2, height, width)
        backward_rgb = raft_flow_backward_rgb.reshape(
            -1, 2, *raft_flow_backward_rgb.shape[-2:]
        )
        forward_rgb = raft_flow_forward_rgb.reshape(
            -1, 2, *raft_flow_forward_rgb.shape[-2:]
        )

        previous_prewarped, previous_bg_warped, previous_valid = (
            self._rescaled_prewarp(
                previous, current, previous_bg, backward_rgb
            )
        )
        next_prewarped, next_bg_warped, next_valid = self._rescaled_prewarp(
            current, previous, current_bg, forward_rgb
        )
        backward_pair = self.deformable_alignment.forward_prewarped(
            prewarped_source=previous_prewarped,
            target=current,
            flow_xy_native=backward_native,
            warped_source_bg=previous_bg_warped,
            target_bg=current_bg,
            valid=previous_valid,
        )
        forward_pair = self.deformable_alignment.forward_prewarped(
            prewarped_source=next_prewarped,
            target=previous,
            flow_xy_native=forward_native,
            warped_source_bg=next_bg_warped,
            target_bg=previous_bg,
            valid=next_valid,
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

        direction = getattr(
            self.config, "deformable_alignment_direction", "bidirectional"
        )
        if direction == "previous_only":
            following_full = base_aligned.clone()
            following_reliability.zero_()
            reliability_forward = torch.zeros_like(reliability_forward)
        elif direction == "next_only":
            previous_full = base_aligned.clone()
            previous_reliability.zero_()
            reliability_backward = torch.zeros_like(reliability_backward)
        elif direction != "bidirectional":
            raise ValueError(f"Invalid deformable_alignment_direction: {direction}")

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


augment_brushnet_condition_v8_rescaled = augment_brushnet_condition_v8

