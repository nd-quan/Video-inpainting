"""V5 RGB-STC: relative temporal bias plus explicit cross-clip memory.

The V4++ local restoration path is preserved exactly at initialization.  A
zero-initialized relative-position table augments local temporal attention and
a separate, zero-gated memory-attention branch reads temporal features from the
overlap of the predecessor clip.  Keeping memory in a separate softmax is
important: concatenating extra keys into V4++ self-attention would change its
output even when every new bias value is zero.

Memory is an explicit value object, never mutable module state.  Training can
therefore form ``previous -> current`` pairs inside one dataset item, detach the
previous memory, and still shuffle/shard items safely with DDP.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import register_to_config

try:
    from STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (
        FlowAlignedRGBSTCOutput,
    )
    from STC_encoder_v4pp_bg_feature.bg_focused_flow_aligned_stc_adapter import (
        BGFocusedFlowAlignedRGBSTCAdapter,
    )
except ModuleNotFoundError:  # Imported through examples.brushnet.
    from ..STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (
        FlowAlignedRGBSTCOutput,
    )
    from ..STC_encoder_v4pp_bg_feature.bg_focused_flow_aligned_stc_adapter import (
        BGFocusedFlowAlignedRGBSTCAdapter,
    )


@dataclass(frozen=True)
class TemporalMemoryState:
    """Detached per-layer overlap features from one predecessor clip.

    ``layer_features[l]`` has shape ``[B,O,C,H,W]`` and contains the output of
    temporal block ``l`` for the cached frame IDs.  ``valid_mask`` permits a
    batch to contain both run-first samples (empty memory) and regular pairs.
    """

    frame_ids: torch.Tensor
    valid_mask: torch.Tensor
    layer_features: Tuple[torch.Tensor, ...]

    def detach(self) -> "TemporalMemoryState":
        return TemporalMemoryState(
            frame_ids=self.frame_ids.detach(),
            valid_mask=self.valid_mask.detach(),
            layer_features=tuple(feature.detach() for feature in self.layer_features),
        )

    def to(self, *args, **kwargs) -> "TemporalMemoryState":
        probe = self.layer_features[0].to(*args, **kwargs)
        target_device = probe.device
        return TemporalMemoryState(
            frame_ids=self.frame_ids.to(device=target_device),
            valid_mask=self.valid_mask.to(device=target_device),
            layer_features=(probe,) + tuple(
                feature.to(*args, **kwargs) for feature in self.layer_features[1:]
            ),
        )


@dataclass
class RelativeCrossClipRGBSTCOutput(FlowAlignedRGBSTCOutput):
    temporal_memory: TemporalMemoryState
    memory_overlap_count: torch.Tensor
    relative_bias_abs_mean: torch.Tensor
    cross_clip_gate_abs_mean: torch.Tensor


class RelativePositionVideoComposerAttention(nn.Module):
    """VideoComposer head layout with learned relative frame-distance bias."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        dropout: float,
        max_relative_distance: int,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_channels = int(channels)
        self.max_relative_distance = int(max_relative_distance)
        if self.num_heads < 1 or self.max_relative_distance < 1:
            raise ValueError("num_heads and max_relative_distance must be positive")
        inner_channels = self.num_heads * self.head_channels
        self.scale = self.head_channels**-0.5
        # Keep the V2 names and shapes so full V4++ weights transfer strictly.
        self.to_qkv = nn.Linear(channels, inner_channels * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_channels, channels),
            nn.Dropout(float(dropout)),
        )
        self.relative_position_bias = nn.Parameter(
            torch.zeros(self.num_heads, 2 * self.max_relative_distance + 1)
        )

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length = tensor.shape[:2]
        return tensor.reshape(
            batch, length, self.num_heads, self.head_channels
        ).permute(0, 2, 1, 3)

    def _relative_bias(
        self, query_frame_ids: torch.Tensor, key_frame_ids: torch.Tensor
    ) -> torch.Tensor:
        if query_frame_ids.ndim != 2 or key_frame_ids.ndim != 2:
            raise ValueError("frame IDs must have shape [N,L]")
        if query_frame_ids.shape[0] != key_frame_ids.shape[0]:
            raise ValueError("query/key frame-ID batches must match")
        distance = key_frame_ids[:, None, :] - query_frame_ids[:, :, None]
        indices = distance.clamp(
            -self.max_relative_distance, self.max_relative_distance
        ) + self.max_relative_distance
        # embedding: [N,Q,K,H], then attention layout [N,H,Q,K].
        table = self.relative_position_bias.transpose(0, 1)
        return F.embedding(indices.long(), table).permute(0, 3, 1, 2)

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_frame_ids: torch.Tensor,
        key_frame_ids: torch.Tensor,
        key_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        similarity = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        similarity = similarity + self._relative_bias(
            query_frame_ids, key_frame_ids
        ).to(dtype=similarity.dtype)
        has_valid_key = None
        if key_valid_mask is not None:
            if key_valid_mask.shape != key_frame_ids.shape:
                raise ValueError("key_valid_mask must match key_frame_ids")
            valid = key_valid_mask.to(device=similarity.device, dtype=torch.bool)
            similarity = similarity.masked_fill(
                ~valid[:, None, None, :], torch.finfo(similarity.dtype).min
            )
            has_valid_key = valid.any(dim=-1)
        attention = similarity.float().softmax(dim=-1).to(similarity.dtype)
        attended = torch.matmul(attention, value)
        attended = attended.permute(0, 2, 1, 3).reshape(
            query.shape[0], query.shape[2], -1
        )
        attended = self.to_out(attended)
        if has_valid_key is not None:
            attended = attended * has_valid_key[:, None, None].to(attended.dtype)
        return attended

    def forward(self, tokens: torch.Tensor, frame_ids: torch.Tensor) -> torch.Tensor:
        query, key, value = self.to_qkv(tokens).chunk(3, dim=-1)
        return self._attend(
            self._split_heads(query),
            self._split_heads(key),
            self._split_heads(value),
            frame_ids,
            frame_ids,
        )

    def attend_memory(
        self,
        query_tokens: torch.Tensor,
        memory_tokens: torch.Tensor,
        query_frame_ids: torch.Tensor,
        memory_frame_ids: torch.Tensor,
        memory_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.to_qkv(query_tokens).chunk(3, dim=-1)[0]
        _, key, value = self.to_qkv(memory_tokens).chunk(3, dim=-1)
        return self._attend(
            self._split_heads(query),
            self._split_heads(key),
            self._split_heads(value),
            query_frame_ids,
            memory_frame_ids,
            key_valid_mask=memory_valid_mask,
        )


class RelativeCrossClipTemporalBlock(nn.Module):
    """V2 temporal block plus a zero-gated predecessor-memory residual."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        max_relative_distance: int,
    ):
        super().__init__()
        self.attention_norm = nn.LayerNorm(channels)
        self.attention = RelativePositionVideoComposerAttention(
            channels, num_heads, dropout, max_relative_distance
        )
        hidden_channels = max(int(channels * mlp_ratio), channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_channels, channels),
        )
        # tanh bounds the branch and exactly disables it at warm start.
        self.cross_clip_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        tokens: torch.Tensor,
        frame_ids: torch.Tensor,
        memory_tokens: Optional[torch.Tensor] = None,
        memory_frame_ids: Optional[torch.Tensor] = None,
        memory_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        tokens = tokens + self.attention(normalized, frame_ids)
        if memory_tokens is not None:
            if memory_frame_ids is None or memory_valid_mask is None:
                raise ValueError("memory IDs and validity are required with memory")
            memory_normalized = self.attention_norm(memory_tokens)
            memory_residual = self.attention.attend_memory(
                normalized,
                memory_normalized,
                frame_ids,
                memory_frame_ids,
                memory_valid_mask,
            )
            tokens = tokens + torch.tanh(self.cross_clip_gate) * memory_residual
        return tokens + self.feed_forward(tokens)


class RelativeCrossClipBGSTCAdapter(BGFocusedFlowAlignedRGBSTCAdapter):
    """V4++ with relative temporal bias and overlap-memory attention."""

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
        )
        if int(relative_position_max_distance) < 1:
            raise ValueError("relative_position_max_distance must be positive")
        if int(cross_clip_memory_frames) < 1:
            raise ValueError("cross_clip_memory_frames must be positive")

        upgraded = nn.ModuleList()
        for old_block in self.stc_adapter.temporal_blocks:
            new_block = RelativeCrossClipTemporalBlock(
                channels=int(hidden_channels),
                num_heads=int(num_heads),
                mlp_ratio=float(mlp_ratio),
                dropout=float(dropout),
                max_relative_distance=int(relative_position_max_distance),
            )
            transfer = new_block.load_state_dict(old_block.state_dict(), strict=False)
            expected = {
                "cross_clip_gate",
                "attention.relative_position_bias",
            }
            if set(transfer.missing_keys) != expected or transfer.unexpected_keys:
                raise RuntimeError(
                    "Internal temporal-block upgrade mismatch: "
                    f"missing={transfer.missing_keys}, "
                    f"unexpected={transfer.unexpected_keys}"
                )
            upgraded.append(new_block)
        self.stc_adapter.temporal_blocks = upgraded

    @classmethod
    def from_v4pp_pretrained(
        cls,
        pretrained_model_path,
        relative_position_max_distance: int = 32,
        cross_clip_memory_frames: int = 4,
        detach_cross_clip_memory: bool = True,
        require_memory_overlap: bool = True,
    ) -> "RelativeCrossClipBGSTCAdapter":
        """Upgrade one complete V4++ component and audit every transferred key."""
        source = BGFocusedFlowAlignedRGBSTCAdapter.from_pretrained(
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
            relative_position_max_distance=int(relative_position_max_distance),
            cross_clip_memory_frames=int(cross_clip_memory_frames),
            detach_cross_clip_memory=bool(detach_cross_clip_memory),
            require_memory_overlap=bool(require_memory_overlap),
        )
        transfer = model.load_state_dict(source.state_dict(), strict=False)
        expected_missing = {
            f"stc_adapter.temporal_blocks.{index}.cross_clip_gate"
            for index in range(int(config.num_layers))
        }
        expected_missing.update(
            f"stc_adapter.temporal_blocks.{index}.attention.relative_position_bias"
            for index in range(int(config.num_layers))
        )
        if set(transfer.missing_keys) != expected_missing or transfer.unexpected_keys:
            raise RuntimeError(
                "V4++ -> V5 transfer mismatch: "
                f"missing={transfer.missing_keys}, "
                f"unexpected={transfer.unexpected_keys}"
            )
        return model

    @staticmethod
    def _frame_ids(
        frame_ids: Optional[torch.Tensor], batch: int, frames: int, device
    ) -> torch.Tensor:
        if frame_ids is None:
            return torch.arange(frames, device=device, dtype=torch.long)[None].expand(
                batch, -1
            )
        if tuple(frame_ids.shape) != (batch, frames):
            raise ValueError(
                f"frame_ids must have shape {(batch, frames)}, "
                f"got {tuple(frame_ids.shape)}"
            )
        return frame_ids.to(device=device, dtype=torch.long)

    @staticmethod
    def _expand_ids(frame_ids: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return frame_ids[:, None, None, :].expand(
            frame_ids.shape[0], height, width, frame_ids.shape[1]
        ).reshape(frame_ids.shape[0] * height * width, frame_ids.shape[1])

    @staticmethod
    def _expand_valid(valid: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return valid[:, None, None, :].expand(
            valid.shape[0], height, width, valid.shape[1]
        ).reshape(valid.shape[0] * height * width, valid.shape[1])

    def _spatial_and_bg_alignment(
        self,
        rgb_sequence: torch.Tensor,
        bg_mask_sequence: torch.Tensor,
        output_size: Tuple[int, int],
    ):
        batch, frames, _, _ = self.stc_adapter._validate_inputs(
            rgb_sequence, bg_mask_sequence
        )
        pixel_condition = self.build_pixel_condition(
            rgb_sequence, bg_mask_sequence
        ).flatten(0, 1)
        spatial_flat = self.stc_adapter.spatial_encoder(pixel_condition)
        if spatial_flat.shape[-2:] != output_size:
            spatial_flat = F.interpolate(
                spatial_flat, size=output_size, mode="bilinear", align_corners=False
            )
        channels, encoded_height, encoded_width = spatial_flat.shape[1:]
        spatial = spatial_flat.reshape(
            batch, frames, channels, encoded_height, encoded_width
        )
        flow_forward, flow_backward = self._decode_bidirectional_flow(spatial)
        fully_aligned, confidence = self._align_spatial_features(
            spatial, flow_forward, flow_backward
        )
        bg_feature_mask = F.interpolate(
            bg_mask_sequence.flatten(0, 1).to(
                device=spatial.device, dtype=spatial.dtype
            ),
            size=(encoded_height, encoded_width),
            mode="nearest",
        ).reshape(batch, frames, 1, encoded_height, encoded_width)
        bg_feature_mask = (bg_feature_mask >= 0.5).to(spatial.dtype)
        aligned_current = spatial[:, 1:] + bg_feature_mask[:, 1:] * (
            fully_aligned[:, 1:] - spatial[:, 1:]
        )
        aligned = torch.cat((spatial[:, :1], aligned_current), dim=1)
        return spatial, aligned, flow_forward, flow_backward, confidence

    def _validated_memory(
        self,
        temporal_memory: Optional[TemporalMemoryState],
        frame_ids: torch.Tensor,
        channels: int,
        height: int,
        width: int,
    ):
        if temporal_memory is None:
            return None, frame_ids.new_zeros(frame_ids.shape[0])
        if len(temporal_memory.layer_features) != len(
            self.stc_adapter.temporal_blocks
        ):
            raise ValueError("Temporal memory layer count does not match the model")
        batch, memory_frames = temporal_memory.frame_ids.shape
        if batch != frame_ids.shape[0] or temporal_memory.valid_mask.shape != (
            batch,
            memory_frames,
        ):
            raise ValueError("Temporal memory batch/validity shape mismatch")
        expected = (batch, memory_frames, channels, height, width)
        if any(tuple(feature.shape) != expected for feature in temporal_memory.layer_features):
            raise ValueError(f"Temporal memory features must have shape {expected}")
        memory_ids = temporal_memory.frame_ids.to(
            device=frame_ids.device, dtype=torch.long
        )
        memory_valid = temporal_memory.valid_mask.to(
            device=frame_ids.device, dtype=torch.bool
        )
        matches = frame_ids[:, :, None].eq(memory_ids[:, None, :])
        overlap_valid = matches.any(dim=1)
        if bool(self.config.require_memory_overlap):
            memory_valid = memory_valid & overlap_valid
        overlap_count = (memory_valid & overlap_valid).sum(dim=1)
        return (
            TemporalMemoryState(
                frame_ids=memory_ids,
                valid_mask=memory_valid,
                layer_features=tuple(
                    feature.to(device=frame_ids.device)
                    for feature in temporal_memory.layer_features
                ),
            ),
            overlap_count,
        )

    def encode(
        self,
        rgb_sequence: torch.Tensor,
        bg_mask_sequence: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        frame_ids: Optional[torch.Tensor] = None,
        temporal_memory: Optional[TemporalMemoryState] = None,
        frame_valid_mask: Optional[torch.Tensor] = None,
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
            aligned,
            flow_forward,
            flow_backward,
            confidence,
        ) = self._spatial_and_bg_alignment(
            rgb_sequence, bg_mask_sequence, output_size
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
        next_ids = frame_ids[:, -memory_frames:]
        next_valid = frame_valid_mask[:, -memory_frames:]
        next_memory = TemporalMemoryState(
            frame_ids=next_ids.detach(),
            valid_mask=next_valid.detach(),
            layer_features=tuple(next_layer_features),
        )
        return (
            delta,
            features,
            spatial,
            aligned,
            flow_forward,
            flow_backward,
            confidence,
            next_memory,
            overlap_count,
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
    ):
        values = self.encode(
            rgb_sequence,
            bg_mask_sequence,
            output_size=output_size,
            frame_ids=frame_ids,
            temporal_memory=temporal_memory,
            frame_valid_mask=frame_valid_mask,
        )
        (
            delta,
            features,
            spatial,
            aligned,
            flow_forward,
            flow_backward,
            confidence,
            next_memory,
            overlap_count,
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
        result = RelativeCrossClipRGBSTCOutput(
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


def augment_brushnet_condition_v5(
    model: RelativeCrossClipBGSTCAdapter,
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
):
    """Build predecessor memory without a diffusion graph, then run current once."""
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
            # DDP wraps only the current differentiable call. The predecessor
            # pass is deterministic read-only context (GroupNorm/LayerNorm,
            # no BN). Mixed batches may contain padded run-first items; their
            # validity remains false and is masked from memory attention.
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
    )
    base_sequence = base_condition_latents.reshape(
        batch, frames, 4, *base_condition_latents.shape[-2:]
    )
    delta = output.delta_bg.to(dtype=base_sequence.dtype)
    latent_bg_mask = output.latent_bg_mask.to(dtype=base_sequence.dtype)
    augmented = base_sequence + float(injection_scale) * delta
    brushnet_condition = torch.cat((augmented, latent_bg_mask), dim=2).flatten(0, 1)
    return brushnet_condition, output, augmented
