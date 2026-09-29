"""V8-R temporal adapter with a spatial-feature base and no light-flow head.

The only optical-flow input is the frozen distilled RAFT student's RGB flow.
Forward flow is defined on frame t and samples t+1. Backward flow is defined
on frame t+1 and samples t. All flow inputs use RGB-pixel displacement units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


def groups(channels: int) -> int:
    return next(n for n in range(min(8, channels), 0, -1) if channels % n == 0)


def resize_flow(flow: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("flow must be [N,2,H,W]")
    old_h, old_w = flow.shape[-2:]
    new_h, new_w = size
    result = F.interpolate(flow.float(), size=size, mode="bilinear", align_corners=True)
    return torch.stack((result[:, 0] * new_w / old_w, result[:, 1] * new_h / old_h), 1)


def backward_warp(source: torch.Tensor, flow: torch.Tensor, fallback=None):
    if source.shape[0] != flow.shape[0] or source.shape[-2:] != flow.shape[-2:]:
        raise ValueError("source/flow shape mismatch")
    h, w = source.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=flow.dtype),
        torch.arange(w, device=flow.device, dtype=flow.dtype), indexing="ij",
    )
    sx, sy = x[None] + flow[:, 0], y[None] + flow[:, 1]
    valid = ((sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1))[:, None]
    grid = torch.stack((2 * (sx + .5) / w - 1, 2 * (sy + .5) / h - 1), -1)
    warped = F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if fallback is not None:
        warped = torch.where(valid, warped, fallback)
    return warped, valid


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                                   nn.GroupNorm(groups(out_ch), out_ch), nn.SiLU())

    def forward(self, x):
        return self.block(x)


class RelativeAttention(nn.Module):
    def __init__(self, channels, heads, dropout, max_distance):
        super().__init__()
        self.num_heads = heads
        self.head_channels = channels
        self.max_relative_distance = max_distance
        self.scale = channels ** -.5
        self.to_qkv = nn.Linear(channels, 3 * heads * channels, bias=False)
        self.to_out = nn.Sequential(nn.Linear(heads * channels, channels), nn.Dropout(dropout))
        self.relative_position_bias = nn.Parameter(torch.zeros(heads, 2 * max_distance + 1))

    def _split(self, x):
        n, length = x.shape[:2]
        return x.reshape(n, length, self.num_heads, self.head_channels).permute(0, 2, 1, 3)

    def _attend(self, q, k, v, query_ids, key_ids, valid=None):
        distance = (key_ids[:, None, :] - query_ids[:, :, None]).clamp(
            -self.max_relative_distance, self.max_relative_distance
        ) + self.max_relative_distance
        bias = F.embedding(distance.long(), self.relative_position_bias.T).permute(0, 3, 1, 2)
        logits = q @ k.transpose(-1, -2) * self.scale + bias.to(q.dtype)
        any_valid = None
        if valid is not None:
            any_valid = valid.any(-1)
            logits = logits.masked_fill(~valid[:, None, None, :], torch.finfo(logits.dtype).min)
        weights = logits.float().softmax(-1).to(v.dtype)
        result = (weights @ v).permute(0, 2, 1, 3).reshape(q.shape[0], q.shape[2], -1)
        result = self.to_out(result)
        return result if any_valid is None else result * any_valid[:, None, None].to(result.dtype)

    def forward(self, tokens, frame_ids):
        q, k, v = self.to_qkv(tokens).chunk(3, -1)
        return self._attend(self._split(q), self._split(k), self._split(v), frame_ids, frame_ids)

    def attend_memory(self, tokens, memory, ids, memory_ids, valid):
        q = self.to_qkv(tokens).chunk(3, -1)[0]
        _, k, v = self.to_qkv(memory).chunk(3, -1)
        return self._attend(self._split(q), self._split(k), self._split(v), ids, memory_ids, valid)


class TemporalBlock(nn.Module):
    def __init__(self, channels, heads, mlp_ratio, dropout, max_distance):
        super().__init__()
        self.attention_norm = nn.LayerNorm(channels)
        self.attention = RelativeAttention(channels, heads, dropout, max_distance)
        hidden = max(channels, int(channels * mlp_ratio))
        self.feed_forward = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(hidden, channels))
        self.cross_clip_gate = nn.Parameter(torch.zeros(()))

    def forward(self, tokens, ids, memory=None, memory_ids=None, valid=None):
        normalized = self.attention_norm(tokens)
        tokens = tokens + self.attention(normalized, ids)
        if memory is not None:
            tokens = tokens + torch.tanh(self.cross_clip_gate) * self.attention.attend_memory(
                normalized, self.attention_norm(memory), ids, memory_ids, valid
            )
        else:
            # A DDP rank can receive the first clip of a sequence while other
            # ranks receive clips with predecessor memory. Keep the gate in the
            # backward graph with an exactly zero gradient on that rank.
            tokens = tokens + self.cross_clip_gate * 0.0
        return tokens + self.feed_forward(tokens)


class STCBackbone(nn.Module):
    """Keep compatible spatial/temporal parameter names without a flow head."""
    def __init__(self, channels, heads, layers, mlp_ratio, dropout, max_distance):
        super().__init__()
        half = max(channels // 2, 8)
        self.spatial_encoder = nn.Sequential(
            DownBlock(4, half), DownBlock(half, channels), DownBlock(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(groups(channels), channels), nn.SiLU(),
        )
        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(channels, heads, mlp_ratio, dropout, max_distance) for _ in range(layers)
        ])
        self.output_norm = nn.GroupNorm(groups(channels), channels)
        self.output_act = nn.SiLU()
        self.zero_conv = nn.Conv2d(channels, 4, 1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)


@dataclass(frozen=True)
class TemporalMemory:
    frame_ids: torch.Tensor
    valid_mask: torch.Tensor
    layer_features: Tuple[torch.Tensor, ...]

    def detach(self):
        return TemporalMemory(self.frame_ids.detach(), self.valid_mask.detach(),
                              tuple(x.detach() for x in self.layer_features))


@dataclass
class PairOutput:
    aligned_source: torch.Tensor
    residual_offset: torch.Tensor
    modulation_mask: torch.Tensor
    reliability: torch.Tensor
    base_difference_abs_mean: torch.Tensor


class ResidualDCN(nn.Module):
    """DCNv2 on an already prewarped source; input omits the old constant channel."""
    def __init__(self, channels=64, hidden_channels=128, kernel_size=3, deform_groups=4,
                 residual_max_displacement=2., flow_max_displacement=(8., 8.)):
        super().__init__()
        if channels % deform_groups or kernel_size % 2 != 1:
            raise ValueError("DCN groups must divide channels and kernel must be odd")
        self.kernel_size, self.deform_groups = kernel_size, deform_groups
        self.residual_max_displacement = residual_max_displacement
        self.flow_max_displacement = tuple(flow_max_displacement)
        self.offset_channels = 2 * deform_groups * kernel_size * kernel_size
        self.mask_channels = deform_groups * kernel_size * kernel_size
        self.offset_mask_head = nn.Sequential(
            nn.Conv2d(3 * channels + 4, hidden_channels, 3, padding=1),
            nn.GroupNorm(groups(hidden_channels), hidden_channels), nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(groups(hidden_channels), hidden_channels), nn.SiLU(),
            nn.Conv2d(hidden_channels, self.offset_channels + self.mask_channels, 3, padding=1),
        )
        nn.init.zeros_(self.offset_mask_head[-1].weight)
        nn.init.zeros_(self.offset_mask_head[-1].bias)
        self.weight = nn.Parameter(torch.zeros(channels, channels // deform_groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))
        with torch.no_grad():
            for channel in range(channels):
                self.weight[channel, channel % (channels // deform_groups), kernel_size // 2, kernel_size // 2] = 2.

    def forward(self, prewarped, target, flow_native, source_bg, target_bg, valid):
        bounds = flow_native.new_tensor(self.flow_max_displacement).reshape(1, 2, 1, 1)
        head = torch.cat((target, prewarped, target - prewarped, flow_native / bounds,
                          target_bg, source_bg), 1)
        raw_offset, raw_mask = self.offset_mask_head(head).split(
            (self.offset_channels, self.mask_channels), 1
        )
        offset = torch.tanh(raw_offset) * self.residual_max_displacement
        mask = torch.sigmoid(raw_mask)
        deformed = deform_conv2d(prewarped, offset, self.weight, self.bias,
                                 padding=(self.kernel_size // 2, self.kernel_size // 2), mask=mask)
        deformed = torch.where(valid, deformed, target)
        reliability = (target_bg * source_bg.clamp(0, 1) * valid.to(target.dtype)).detach()
        return PairOutput(deformed, offset, mask, reliability,
                          (deformed - prewarped).detach().float().abs().mean())


class BidirectionalFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Conv2d(5 * channels + 2, 2 * channels, 3, padding=1),
            nn.GroupNorm(groups(2 * channels), 2 * channels), nn.SiLU(),
            nn.Conv2d(2 * channels, channels, 3, padding=1),
            nn.GroupNorm(groups(channels), channels), nn.SiLU(),
        )
        self.to_residual_and_gate = nn.Conv2d(channels, channels + 1, 1)
        nn.init.zeros_(self.to_residual_and_gate.weight)
        nn.init.zeros_(self.to_residual_and_gate.bias)

    def forward(self, base, previous, following, previous_reliability, following_reliability, bg, scale=1.):
        x = torch.cat((base, previous, following, base - previous, base - following,
                       previous_reliability, following_reliability), 1)
        residual, gate = self.to_residual_and_gate(self.hidden(x)).split((base.shape[1], 1), 1)
        union = 1 - (1 - previous_reliability.clamp(0, 1)) * (1 - following_reliability.clamp(0, 1))
        return base + scale * bg * union * torch.sigmoid(gate) * residual


@dataclass
class StandaloneV8ROutput:
    delta_bg: torch.Tensor
    features: torch.Tensor
    spatial_features: torch.Tensor
    aligned_spatial_features: torch.Tensor
    latent_bg_mask: torch.Tensor
    predicted_flow_forward: torch.Tensor
    predicted_flow_backward: torch.Tensor
    alignment_confidence: torch.Tensor
    temporal_memory: TemporalMemory
    memory_overlap_count: torch.Tensor
    relative_bias_abs_mean: torch.Tensor
    cross_clip_gate_abs_mean: torch.Tensor
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
    raft_flow_forward_rgb: torch.Tensor
    raft_flow_backward_rgb: torch.Tensor

    @property
    def legacy_flow_forward(self):
        # Compatibility for the shared evaluator only; no head is instantiated.
        return torch.zeros_like(self.predicted_flow_forward)

    @property
    def legacy_flow_backward(self):
        return torch.zeros_like(self.predicted_flow_backward)


class StandaloneV8RAdapter(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(self, hidden_channels=64, num_heads=2, num_layers=1, mlp_ratio=4., dropout=0.,
                 downsample_factor=8, output_channels=4, condition_mode="full_rgb_bg_mask",
                 flow_max_displacement=(8., 8.), relative_position_max_distance=32,
                 cross_clip_memory_frames=4, detach_cross_clip_memory=True,
                 require_memory_overlap=True, deform_hidden_channels=128, deform_kernel_size=3,
                 deform_groups=4, deform_residual_max_displacement=2.,
                 deformable_alignment_direction="bidirectional", rescaled_warp_scale=4):
        super().__init__()
        if downsample_factor != 8 or output_channels != 4 or condition_mode != "full_rgb_bg_mask":
            raise ValueError("Standalone V8-R requires 8x spatial downsampling, 4 latent channels, full RGB+BG mask")
        if rescaled_warp_scale < 1 or cross_clip_memory_frames < 1:
            raise ValueError("rescaled_warp_scale and cross_clip_memory_frames must be positive")
        if deformable_alignment_direction not in ("bidirectional", "previous_only", "next_only"):
            raise ValueError("invalid deformable_alignment_direction")
        self.stc_adapter = STCBackbone(hidden_channels, num_heads, num_layers, mlp_ratio,
                                       dropout, relative_position_max_distance)
        self.deformable_alignment = ResidualDCN(hidden_channels, deform_hidden_channels,
                                                deform_kernel_size, deform_groups,
                                                deform_residual_max_displacement, flow_max_displacement)
        self.deformable_fusion = BidirectionalFusion(hidden_channels)

    def _prewarp(self, source, target, source_bg, rgb_flow):
        h, w = source.shape[-2:]
        scale = int(self.config.rescaled_warp_scale)
        high = (h * scale, w * scale)
        source_up = F.interpolate(source, size=high, mode="nearest")
        target_up = F.interpolate(target, size=high, mode="nearest")
        bg_up = F.interpolate(source_bg, size=high, mode="nearest")
        flow_high = resize_flow(rgb_flow.detach(), high).to(source.dtype)
        warped, valid = backward_warp(source_up, flow_high, fallback=target_up)
        bg_warped, _ = backward_warp(bg_up, flow_high)
        return (F.interpolate(warped, size=(h, w), mode="nearest"),
                F.interpolate(bg_warped, size=(h, w), mode="nearest").clamp(0, 1),
                F.interpolate(valid.to(source.dtype), size=(h, w), mode="nearest") >= .5)

    def forward(self, rgb_sequence, bg_mask_sequence, output_size=None, predict_flow=True,
                return_dict=True, frame_ids=None, temporal_memory=None, frame_valid_mask=None,
                deformable_alignment_scale=1., raft_flow_forward_rgb=None,
                raft_flow_backward_rgb=None):
        if rgb_sequence.ndim != 5 or rgb_sequence.shape[2] != 3:
            raise ValueError("rgb_sequence must be [B,T,3,H,W]")
        batch, frames, _, rgb_h, rgb_w = rgb_sequence.shape
        if tuple(bg_mask_sequence.shape) != (batch, frames, 1, rgb_h, rgb_w):
            raise ValueError("BG mask must be [B,T,1,H,W]")
        expected_flow = (batch, frames - 1, 2, rgb_h, rgb_w)
        if raft_flow_forward_rgb is None or tuple(raft_flow_forward_rgb.shape) != expected_flow:
            raise ValueError(f"forward RAFT flow must be {expected_flow}")
        if raft_flow_backward_rgb is None or tuple(raft_flow_backward_rgb.shape) != expected_flow:
            raise ValueError(f"backward RAFT flow must be {expected_flow}")
        if output_size is None:
            output_size = (rgb_h // 8, rgb_w // 8)
        if frame_ids is None:
            frame_ids = torch.arange(frames, device=rgb_sequence.device)[None].expand(batch, -1)
        if tuple(frame_ids.shape) != (batch, frames):
            raise ValueError("frame_ids must be [B,T]")
        if frame_valid_mask is None:
            frame_valid_mask = torch.ones_like(frame_ids, dtype=torch.bool)
        flat = torch.cat((rgb_sequence, bg_mask_sequence.to(rgb_sequence.dtype)), 2).flatten(0, 1)
        spatial_flat = self.stc_adapter.spatial_encoder(flat)
        if spatial_flat.shape[-2:] != tuple(output_size):
            spatial_flat = F.interpolate(spatial_flat, size=output_size, mode="bilinear", align_corners=False)
        channels, h, w = spatial_flat.shape[1:]
        spatial = spatial_flat.reshape(batch, frames, channels, h, w)
        bg = F.interpolate(bg_mask_sequence.flatten(0, 1).to(spatial.dtype),
                           size=(h, w), mode="nearest").reshape(batch, frames, 1, h, w)
        bg = (bg >= .5).to(spatial.dtype)
        if frames > 1:
            flow_forward = resize_flow(raft_flow_forward_rgb.flatten(0, 1), (h, w)).to(spatial.dtype).reshape(batch, frames-1, 2, h, w)
            flow_backward = resize_flow(raft_flow_backward_rgb.flatten(0, 1), (h, w)).to(spatial.dtype).reshape(batch, frames-1, 2, h, w)
        else:
            flow_forward = flow_backward = spatial.new_empty(batch, 0, 2, h, w)
        empty_pair = spatial.new_empty(batch, 0, channels, h, w)
        empty_scalar = spatial.new_empty(batch, 0, 1, h, w)
        empty_offset = spatial.new_empty(batch, 0, self.deformable_alignment.offset_channels, h, w)
        empty_mask = spatial.new_empty(batch, 0, self.deformable_alignment.mask_channels, h, w)
        if frames > 1:
            pair_shape = (-1, channels, h, w)
            source_prev, target_next = spatial[:, :-1].reshape(pair_shape), spatial[:, 1:].reshape(pair_shape)
            bg_prev = bg[:, :-1].reshape(-1, 1, h, w)
            bg_next = bg[:, 1:].reshape(-1, 1, h, w)
            back_rgb = raft_flow_backward_rgb.flatten(0, 1)
            fwd_rgb = raft_flow_forward_rgb.flatten(0, 1)
            prev_warp, prev_bg_warp, prev_valid = self._prewarp(source_prev, target_next, bg_prev, back_rgb)
            next_warp, next_bg_warp, next_valid = self._prewarp(target_next, source_prev, bg_next, fwd_rgb)
            previous_pair = self.deformable_alignment(prev_warp, target_next, flow_backward.flatten(0, 1),
                                                       prev_bg_warp, bg_next, prev_valid)
            following_pair = self.deformable_alignment(next_warp, source_prev, flow_forward.flatten(0, 1),
                                                        next_bg_warp, bg_prev, next_valid)
            prev = spatial.clone()
            nxt = spatial.clone()
            prev_rel = spatial.new_zeros(batch, frames, 1, h, w)
            next_rel = spatial.new_zeros(batch, frames, 1, h, w)
            prev[:, 1:] = previous_pair.aligned_source.reshape(batch, frames-1, channels, h, w)
            nxt[:, :-1] = following_pair.aligned_source.reshape(batch, frames-1, channels, h, w)
            prev_rel[:, 1:] = previous_pair.reliability.reshape(batch, frames-1, 1, h, w)
            next_rel[:, :-1] = following_pair.reliability.reshape(batch, frames-1, 1, h, w)
            direction = self.config.deformable_alignment_direction
            if direction == "previous_only":
                next_rel.zero_()
                nxt = spatial
            elif direction == "next_only":
                prev_rel.zero_()
                prev = spatial
            aligned = self.deformable_fusion(spatial_flat, prev.flatten(0, 1), nxt.flatten(0, 1),
                                             prev_rel.flatten(0, 1), next_rel.flatten(0, 1),
                                             bg.flatten(0, 1), scale=float(deformable_alignment_scale))
            aligned = aligned.reshape_as(spatial)
            diff = .5 * (previous_pair.base_difference_abs_mean + following_pair.base_difference_abs_mean)
            deformed_previous = previous_pair.aligned_source.reshape(batch, frames-1, channels, h, w)
            deformed_next = following_pair.aligned_source.reshape(batch, frames-1, channels, h, w)
            backward_rel = previous_pair.reliability.reshape(batch, frames-1, 1, h, w)
            forward_rel = following_pair.reliability.reshape(batch, frames-1, 1, h, w)
            off_b = previous_pair.residual_offset.reshape(batch, frames-1, -1, h, w)
            off_f = following_pair.residual_offset.reshape(batch, frames-1, -1, h, w)
            mask_b = previous_pair.modulation_mask.reshape(batch, frames-1, -1, h, w)
            mask_f = following_pair.modulation_mask.reshape(batch, frames-1, -1, h, w)
        else:
            aligned = spatial
            deformed_previous = deformed_next = empty_pair
            backward_rel = forward_rel = empty_scalar
            off_b = off_f = empty_offset
            mask_b = mask_f = empty_mask
            diff = spatial.new_zeros(())
        overlap_count = frame_ids.new_zeros(batch)
        memory = None
        if temporal_memory is not None:
            if len(temporal_memory.layer_features) != len(self.stc_adapter.temporal_blocks):
                raise ValueError("memory layer count mismatch")
            matches = frame_ids[:, :, None].eq(temporal_memory.frame_ids[:, None, :])
            valid = temporal_memory.valid_mask.to(frame_ids.device)
            overlap = matches.any(1)
            if self.config.require_memory_overlap:
                valid = valid & overlap
            overlap_count = (valid & overlap).sum(1)
            memory = (temporal_memory, valid)
        tokens = aligned.permute(0, 3, 4, 1, 2).reshape(batch*h*w, frames, channels)
        ids = frame_ids[:, None, None, :].expand(batch, h, w, frames).reshape(batch*h*w, frames)
        memory_frames = min(int(self.config.cross_clip_memory_frames), frames)
        layer_cache = []
        for layer_index, block in enumerate(self.stc_adapter.temporal_blocks):
            mem_tokens = mem_ids = mem_valid = None
            if memory is not None:
                state, valid = memory
                m = state.layer_features[layer_index].to(device=tokens.device, dtype=tokens.dtype)
                if tuple(m.shape[:2]) != (batch, state.frame_ids.shape[1]) or tuple(m.shape[2:]) != (channels, h, w):
                    raise ValueError("memory feature shape mismatch")
                mem_tokens = m.permute(0, 3, 4, 1, 2).reshape(batch*h*w, m.shape[1], channels)
                mem_ids = state.frame_ids[:, None, None, :].expand(batch, h, w, -1).reshape(batch*h*w, -1)
                mem_valid = valid[:, None, None, :].expand(batch, h, w, -1).reshape(batch*h*w, -1)
            tokens = block(tokens, ids, mem_tokens, mem_ids, mem_valid)
            layer = tokens.reshape(batch, h, w, frames, channels).permute(0, 3, 4, 1, 2)
            cached = layer[:, -memory_frames:]
            layer_cache.append(cached.detach() if self.config.detach_cross_clip_memory else cached)
        features = tokens.reshape(batch, h, w, frames, channels).permute(0, 3, 4, 1, 2)
        flat_features = self.stc_adapter.output_act(self.stc_adapter.output_norm(features.flatten(0, 1)))
        delta = self.stc_adapter.zero_conv(flat_features).reshape(batch, frames, 4, h, w)
        bg_latent = bg
        next_memory = TemporalMemory(frame_ids[:, -memory_frames:].detach(),
                                     frame_valid_mask[:, -memory_frames:].detach(), tuple(layer_cache))
        confidence = spatial.new_ones(batch, max(frames-1, 0), 1, h, w)
        result = StandaloneV8ROutput(
            delta * bg_latent, features, spatial, aligned, bg_latent,
            flow_forward if predict_flow else None, flow_backward if predict_flow else None,
            confidence, next_memory, overlap_count,
            torch.stack([b.attention.relative_position_bias.detach().abs().mean() for b in self.stc_adapter.temporal_blocks]).mean(),
            torch.stack([torch.tanh(b.cross_clip_gate.detach()).abs() for b in self.stc_adapter.temporal_blocks]).mean(),
            spatial, deformed_previous, deformed_next, backward_rel, forward_rel,
            off_b, off_f, mask_b, mask_f, diff,
            raft_flow_forward_rgb.detach(), raft_flow_backward_rgb.detach(),
        )
        if return_dict:
            return result
        return (result.delta_bg, result.features, result.latent_bg_mask,
                result.predicted_flow_forward, result.predicted_flow_backward,
                result.alignment_confidence, result.temporal_memory)


def augment_brushnet_condition(model, base_condition_latents, rgb_sequence, bg_mask_sequence,
                               *, raft_flow_provider, injection_scale=1., predict_flow=True,
                               frame_ids=None, previous_rgb_sequence=None,
                               previous_bg_mask_sequence=None, previous_frame_ids=None,
                               previous_valid_mask=None, deformable_alignment_scale=1.,
                               external_current_flow=None, external_previous_flow=None):
    """Add BG-only 4-channel correction; preserve the fifth BrushNet mask channel."""
    batch, frames = rgb_sequence.shape[:2]
    if tuple(base_condition_latents.shape[:2]) != (batch*frames, 4):
        raise ValueError("base_condition_latents must be [B*T,4,h,w]")
    current = external_current_flow or raft_flow_provider.predict_sequence(rgb_sequence)
    memory = None
    if previous_rgb_sequence is not None and previous_valid_mask is not None and bool(previous_valid_mask.any()):
        previous = external_previous_flow or raft_flow_provider.predict_sequence(previous_rgb_sequence)
        bare = model.module if hasattr(model, "module") else model
        with torch.no_grad():
            old = bare(previous_rgb_sequence, previous_bg_mask_sequence,
                       output_size=base_condition_latents.shape[-2:],
                       predict_flow=False, frame_ids=previous_frame_ids,
                       frame_valid_mask=previous_valid_mask,
                       deformable_alignment_scale=deformable_alignment_scale,
                       raft_flow_forward_rgb=previous.forward,
                       raft_flow_backward_rgb=previous.backward)
        memory = old.temporal_memory.detach()
    output = model(rgb_sequence, bg_mask_sequence,
                   output_size=base_condition_latents.shape[-2:],
                   predict_flow=predict_flow, frame_ids=frame_ids, temporal_memory=memory,
                   deformable_alignment_scale=deformable_alignment_scale,
                   raft_flow_forward_rgb=current.forward,
                   raft_flow_backward_rgb=current.backward)
    base = base_condition_latents.reshape(batch, frames, 4, *base_condition_latents.shape[-2:])
    mask = output.latent_bg_mask.to(base.dtype)
    augmented = base + float(injection_scale) * output.delta_bg.to(base.dtype)
    return torch.cat((augmented, mask), 2).flatten(0, 1), output, augmented
