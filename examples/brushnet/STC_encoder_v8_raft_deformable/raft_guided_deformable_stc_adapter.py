"""V8: V5 base path plus V7-RAFT-guided feature-space DCN.

V8 intentionally retains V5's inherited lightweight flow alignment as the
``base_aligned`` stream.  The frozen V7 RAFT flow is used only as the *base
offset* of the new V6-style DCN branch.  Since the DCN fusion is zero
initialized, upgrading V5 to V8 is exact at step zero even though V7 flow can
be much different from the legacy light flow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from diffusers.configuration_utils import register_to_config
from diffusers.models.stc_flow_training import resize_flow_sequence

try:
    from STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import backward_warp_feature
    from STC_encoder_v5_relative_crossclip.relative_crossclip_stc_adapter import (
        RelativeCrossClipBGSTCAdapter,
        TemporalMemoryState,
    )
    from STC_encoder_v6_flow_deformable.flow_guided_deformable_stc_adapter import (
        FlowGuidedDeformableBGSTCAdapter,
        FlowGuidedDeformableRGBSTCOutput,
    )
except ModuleNotFoundError:  # Imported through examples.brushnet.
    from ..STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import backward_warp_feature
    from ..STC_encoder_v5_relative_crossclip.relative_crossclip_stc_adapter import (
        RelativeCrossClipBGSTCAdapter,
        TemporalMemoryState,
    )
    from ..STC_encoder_v6_flow_deformable.flow_guided_deformable_stc_adapter import (
        FlowGuidedDeformableBGSTCAdapter,
        FlowGuidedDeformableRGBSTCOutput,
    )


@dataclass
class RAFTGuidedDeformableRGBSTCOutput(FlowGuidedDeformableRGBSTCOutput):
    """V6 output with explicit V7 RGB/feature-grid flow diagnostics."""

    raft_flow_forward_rgb: torch.Tensor
    raft_flow_backward_rgb: torch.Tensor
    legacy_flow_forward: torch.Tensor
    legacy_flow_backward: torch.Tensor
    legacy_alignment_confidence: torch.Tensor


class RAFTGuidedDeformableBGSTCAdapter(FlowGuidedDeformableBGSTCAdapter):
    """V5-preserving DCN whose base offsets come from frozen external V7 RAFT."""

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
            deform_hidden_channels=deform_hidden_channels,
            deform_kernel_size=deform_kernel_size,
            deform_groups=deform_groups,
            deform_residual_max_displacement=deform_residual_max_displacement,
            detach_deform_reliability=detach_deform_reliability,
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
    ) -> "RAFTGuidedDeformableBGSTCAdapter":
        """Upgrade a full V5 component while adding only DCN/fusion weights."""
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
            deform_residual_max_displacement=float(deform_residual_max_displacement),
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
                "V5 -> V8 transfer mismatch: "
                f"missing={transfer.missing_keys}, unexpected={transfer.unexpected_keys}"
            )
        return model

    @classmethod
    def from_v6_pretrained(cls, pretrained_model_path) -> "RAFTGuidedDeformableBGSTCAdapter":
        """Import a complete V6 component without changing any trainable tensors."""
        source = FlowGuidedDeformableBGSTCAdapter.from_pretrained(
            str(Path(pretrained_model_path).expanduser().resolve())
        )
        config = source.config
        model = cls(**{
            key: getattr(config, key)
            for key in (
                "hidden_channels", "num_heads", "num_layers", "mlp_ratio", "dropout",
                "downsample_factor", "output_channels", "condition_mode",
                "flow_max_displacement", "flow_confidence_scale", "detach_flow_confidence",
                "relative_position_max_distance", "cross_clip_memory_frames",
                "detach_cross_clip_memory", "require_memory_overlap",
                "deform_hidden_channels", "deform_kernel_size", "deform_groups",
                "deform_residual_max_displacement", "detach_deform_reliability",
            )
        })
        transfer = model.load_state_dict(source.state_dict(), strict=True)
        if transfer.missing_keys or transfer.unexpected_keys:
            raise RuntimeError("V6 -> V8 strict transfer unexpectedly failed")
        return model

    @staticmethod
    def _validate_rgb_flow(
        flow: torch.Tensor,
        *,
        name: str,
        batch: int,
        pairs: int,
        rgb_size: Tuple[int, int],
    ) -> None:
        expected = (batch, pairs, 2, *rgb_size)
        if tuple(flow.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(flow.shape)}")
        if not flow.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if not torch.isfinite(flow).all():
            raise FloatingPointError(f"{name} contains non-finite values")

    def _external_feature_flow(
        self,
        raft_flow_forward_rgb: torch.Tensor,
        raft_flow_backward_rgb: torch.Tensor,
        *,
        batch: int,
        frames: int,
        rgb_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pairs = max(frames - 1, 0)
        self._validate_rgb_flow(
            raft_flow_forward_rgb,
            name="raft_flow_forward_rgb",
            batch=batch,
            pairs=pairs,
            rgb_size=rgb_size,
        )
        self._validate_rgb_flow(
            raft_flow_backward_rgb,
            name="raft_flow_backward_rgb",
            batch=batch,
            pairs=pairs,
            rgb_size=rgb_size,
        )
        if raft_flow_forward_rgb.device != device or raft_flow_backward_rgb.device != device:
            raise ValueError("External V7 flow and STC spatial features must share a device")
        # ``resize_flow_sequence`` bilinearly resizes *and* converts the vector
        # displacement from RGB pixels to feature-grid pixels.
        return (
            resize_flow_sequence(raft_flow_forward_rgb.detach().float(), feature_size).to(dtype=dtype),
            resize_flow_sequence(raft_flow_backward_rgb.detach().float(), feature_size).to(dtype=dtype),
        )

    def _backward_confidence(
        self,
        spatial: torch.Tensor,
        flow_forward: torch.Tensor,
        flow_backward: torch.Tensor,
    ) -> torch.Tensor:
        """FB confidence defined on target/current coordinates for ``t-1 -> t``."""
        batch, frames, channels, height, width = spatial.shape
        if frames <= 1:
            return spatial.new_zeros(batch, 0, 1, height, width)
        previous = spatial[:, :-1].reshape(-1, channels, height, width)
        current = spatial[:, 1:].reshape(-1, channels, height, width)
        backward = flow_backward.reshape(-1, 2, height, width)
        forward = flow_forward.reshape(-1, 2, height, width)
        _, feature_valid = backward_warp_feature(previous, backward, fallback=current)
        warped_forward, flow_valid = backward_warp_feature(forward, backward)
        error = backward + warped_forward
        scale = float(self.config.flow_confidence_scale)
        confidence = torch.exp(
            -error.square().sum(dim=1, keepdim=True) / (2.0 * scale * scale)
        ) * (feature_valid & flow_valid).to(spatial.dtype)
        return confidence.detach().reshape(batch, frames - 1, 1, height, width)

    def encode(
        self,
        rgb_sequence: torch.Tensor,
        bg_mask_sequence: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        frame_ids: Optional[torch.Tensor] = None,
        temporal_memory: Optional[TemporalMemoryState] = None,
        frame_valid_mask: Optional[torch.Tensor] = None,
        deformable_alignment_scale: float = 1.0,
        raft_flow_forward_rgb: Optional[torch.Tensor] = None,
        raft_flow_backward_rgb: Optional[torch.Tensor] = None,
    ):
        if raft_flow_forward_rgb is None or raft_flow_backward_rgb is None:
            raise ValueError("V8 requires frozen V7 RAFT forward and backward RGB flow")
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
            frame_valid_mask = frame_valid_mask.to(device=frame_ids.device, dtype=torch.bool)

        # This is the frozen legacy V5 path.  It must remain the base stream so
        # a zero-initialized V8 does not alter V5 outputs at all.
        spatial, base_aligned, legacy_forward, legacy_backward, legacy_confidence = (
            super()._spatial_and_bg_alignment(rgb_sequence, bg_mask_sequence, output_size)
        )
        _, _, channels, encoded_height, encoded_width = spatial.shape
        raft_forward, raft_backward = self._external_feature_flow(
            raft_flow_forward_rgb,
            raft_flow_backward_rgb,
            batch=batch,
            frames=frames,
            rgb_size=(height, width),
            feature_size=(encoded_height, encoded_width),
            device=spatial.device,
            dtype=spatial.dtype,
        )
        raft_confidence_backward = self._backward_confidence(
            spatial, raft_forward, raft_backward
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
            raft_forward,
            raft_backward,
            raft_confidence_backward,
            bg_mask_sequence,
            deformable_alignment_scale,
        )

        memory, overlap_count = self._validated_memory(
            temporal_memory, frame_ids, channels, encoded_height, encoded_width
        )
        tokens = aligned.permute(0, 3, 4, 1, 2).reshape(
            batch * encoded_height * encoded_width, frames, channels
        )
        token_frame_ids = self._expand_ids(frame_ids, encoded_height, encoded_width)
        memory_frame_ids = memory_valid = None
        if memory is not None:
            memory_frame_ids = self._expand_ids(memory.frame_ids, encoded_height, encoded_width)
            memory_valid = self._expand_valid(memory.valid_mask, encoded_height, encoded_width)

        memory_frames = min(int(self.config.cross_clip_memory_frames), frames)
        next_layer_features = []
        for layer_index, temporal_block in enumerate(self.stc_adapter.temporal_blocks):
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
            next_layer_features.append(
                cached.detach() if bool(self.config.detach_cross_clip_memory) else cached
            )

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
            delta, features, spatial, aligned, base_aligned,
            raft_forward, raft_backward, raft_confidence_backward,
            next_memory, overlap_count, deformed_previous, deformed_next,
            reliability_backward, reliability_forward, residual_backward,
            residual_forward, mask_backward, mask_forward, deformation_minus_base,
            legacy_forward, legacy_backward, legacy_confidence,
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
        raft_flow_forward_rgb: Optional[torch.Tensor] = None,
        raft_flow_backward_rgb: Optional[torch.Tensor] = None,
    ):
        values = self.encode(
            rgb_sequence,
            bg_mask_sequence,
            output_size=output_size,
            frame_ids=frame_ids,
            temporal_memory=temporal_memory,
            frame_valid_mask=frame_valid_mask,
            deformable_alignment_scale=deformable_alignment_scale,
            raft_flow_forward_rgb=raft_flow_forward_rgb,
            raft_flow_backward_rgb=raft_flow_backward_rgb,
        )
        (
            delta, features, spatial, aligned, base_aligned, raft_forward,
            raft_backward, raft_confidence, next_memory, overlap_count,
            deformed_previous, deformed_next, reliability_backward,
            reliability_forward, residual_backward, residual_forward,
            mask_backward, mask_forward, deformation_minus_base,
            legacy_forward, legacy_backward, legacy_confidence,
        ) = values
        latent_bg_mask = F.interpolate(
            bg_mask_sequence.flatten(0, 1).to(device=delta.device, dtype=delta.dtype),
            size=delta.shape[-2:], mode="nearest",
        ).reshape(delta.shape[0], delta.shape[1], 1, *delta.shape[-2:])
        with torch.no_grad():
            bias_values = torch.stack([
                block.attention.relative_position_bias.abs().mean()
                for block in self.stc_adapter.temporal_blocks
            ]).mean()
            gate_values = torch.stack([
                torch.tanh(block.cross_clip_gate).abs()
                for block in self.stc_adapter.temporal_blocks
            ]).mean()
        result = RAFTGuidedDeformableRGBSTCOutput(
            delta_bg=delta * latent_bg_mask,
            features=features,
            spatial_features=spatial,
            aligned_spatial_features=aligned,
            latent_bg_mask=latent_bg_mask,
            predicted_flow_forward=raft_forward if predict_flow else None,
            predicted_flow_backward=raft_backward if predict_flow else None,
            alignment_confidence=raft_confidence,
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
            raft_flow_forward_rgb=raft_flow_forward_rgb.detach(),
            raft_flow_backward_rgb=raft_flow_backward_rgb.detach(),
            legacy_flow_forward=legacy_forward,
            legacy_flow_backward=legacy_backward,
            legacy_alignment_confidence=legacy_confidence,
        )
        if return_dict:
            return result
        return (
            result.delta_bg, result.features, result.latent_bg_mask,
            result.predicted_flow_forward, result.predicted_flow_backward,
            result.alignment_confidence, result.temporal_memory,
        )


def augment_brushnet_condition_v8(
    model: RAFTGuidedDeformableBGSTCAdapter,
    base_condition_latents: torch.Tensor,
    rgb_sequence: torch.Tensor,
    bg_mask_sequence: torch.Tensor,
    *,
    raft_flow_provider,
    injection_scale: float = 1.0,
    predict_flow: bool = True,
    frame_ids: Optional[torch.Tensor] = None,
    previous_rgb_sequence: Optional[torch.Tensor] = None,
    previous_bg_mask_sequence: Optional[torch.Tensor] = None,
    previous_frame_ids: Optional[torch.Tensor] = None,
    previous_valid_mask: Optional[torch.Tensor] = None,
    deformable_alignment_scale: float = 1.0,
):
    """Build V8's condition with V7 flow for both current and predecessor clips."""
    if base_condition_latents.ndim != 4 or base_condition_latents.shape[1] != 4:
        raise ValueError("base_condition_latents must have shape [B*T,4,h,w]")
    batch, frames = rgb_sequence.shape[:2]
    if base_condition_latents.shape[0] != batch * frames:
        raise ValueError("base_condition_latents batch must equal B*T")
    current_flow = raft_flow_provider.predict_sequence(rgb_sequence)
    temporal_memory = None
    if previous_rgb_sequence is not None:
        if previous_bg_mask_sequence is None or previous_frame_ids is None:
            raise ValueError("Previous RGB requires previous masks and frame IDs")
        if previous_valid_mask is None:
            previous_valid_mask = torch.ones_like(previous_frame_ids, dtype=torch.bool)
        if bool(previous_valid_mask.any()):
            previous_flow = raft_flow_provider.predict_sequence(previous_rgb_sequence)
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
                    raft_flow_forward_rgb=previous_flow.forward,
                    raft_flow_backward_rgb=previous_flow.backward,
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
        raft_flow_forward_rgb=current_flow.forward,
        raft_flow_backward_rgb=current_flow.backward,
    )
    base_sequence = base_condition_latents.reshape(batch, frames, 4, *base_condition_latents.shape[-2:])
    delta = output.delta_bg.to(dtype=base_sequence.dtype)
    latent_bg_mask = output.latent_bg_mask.to(dtype=base_sequence.dtype)
    augmented = base_sequence + float(injection_scale) * delta
    return torch.cat((augmented, latent_bg_mask), dim=2).flatten(0, 1), output, augmented
