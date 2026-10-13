"""Oracle teacher-flow warping of BrushNet residuals before diffusion U-Net.

The model contains no learned optical-flow head, DCN, or latent correction.
Teacher flow is computed from clean RGB outside this model and is required for
both training and oracle evaluation. The spatial encoder and temporal blocks
can be initialized from a complete V8-R checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from safetensors.torch import load_file

try:
    from v8r_standalone.model import STCBackbone, TemporalMemory, backward_warp, groups, resize_flow
except ModuleNotFoundError:  # Imported through examples.brushnet.proposed.
    from ..v8r_standalone.model import STCBackbone, TemporalMemory, backward_warp, groups, resize_flow


class PortProjection(nn.Module):
    """Warp this BrushNet residual, fuse TCE context, emit a zero-start delta."""

    def __init__(self, residual_channels: int, guide_channels: int, bottleneck: int):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(residual_channels, bottleneck, 1),
            nn.GroupNorm(groups(bottleneck), bottleneck),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(bottleneck + guide_channels + 2, bottleneck, 3, padding=1),
            nn.GroupNorm(groups(bottleneck), bottleneck),
            nn.SiLU(),
        )
        self.output = nn.Conv2d(bottleneck, residual_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _warp_pair(source, source_bg, flow_rgb, valid_rgb):
        h, w = source.shape[-2:]
        high = (h * 4, w * 4)
        source_up = F.interpolate(source, size=high, mode="nearest")
        bg_up = F.interpolate(source_bg, size=high, mode="nearest")
        flow_high = resize_flow(flow_rgb.detach().float(), high).to(source.dtype)
        warped, geometric_valid = backward_warp(source_up, flow_high)
        warped_bg, _ = backward_warp(bg_up, flow_high)
        warped = F.interpolate(warped, size=(h, w), mode="nearest")
        warped_bg = F.interpolate(warped_bg, size=(h, w), mode="nearest")
        geometric_valid = F.interpolate(geometric_valid.float(), size=(h, w), mode="nearest") >= 0.5
        teacher_valid = F.interpolate(valid_rgb.detach().float(), size=(h, w), mode="nearest") >= 0.5
        return warped, warped_bg, geometric_valid & teacher_valid

    def forward(
        self, residual, guide, bg_rgb, flow_f, flow_b, valid_f, valid_b,
        warp_gain, warp_strength,
    ):
        batch, frames = guide.shape[:2]
        if residual.shape[0] != batch * frames:
            raise ValueError("BrushNet residual batch must equal B*T")
        height, width = residual.shape[-2:]
        compressed = self.compress(residual).reshape(batch, frames, -1, height, width)
        bg = F.interpolate(bg_rgb.flatten(0, 1).float(), (height, width), mode="nearest")
        bg = (bg >= 0.5).to(compressed.dtype).reshape(batch, frames, 1, height, width)
        difference = torch.zeros_like(compressed)
        prev_weight = compressed.new_zeros(batch, frames, 1, height, width)
        next_weight = torch.zeros_like(prev_weight)
        if frames > 1:
            # Work on one adjacent pair at a time so 64x64 -> 256x256 never
            # expands all T-1 high-channel residuals in a single allocation.
            differences = []
            prev_weights = []
            next_weights = []
            for frame in range(frames):
                target = compressed[:, frame]
                target_bg = bg[:, frame]
                numerator = torch.zeros_like(target)
                denominator = torch.ones_like(target_bg)
                p_weight = torch.zeros_like(target_bg)
                n_weight = torch.zeros_like(target_bg)
                if frame > 0:
                    warped, warped_bg, valid = self._warp_pair(
                        compressed[:, frame - 1], bg[:, frame - 1],
                        flow_b[:, frame - 1], valid_b[:, frame - 1],
                    )
                    p_weight = target_bg * warped_bg.clamp(0, 1) * valid.to(target.dtype)
                    numerator = numerator + p_weight * (warped - target)
                    denominator = denominator + p_weight
                if frame + 1 < frames:
                    warped, warped_bg, valid = self._warp_pair(
                        compressed[:, frame + 1], bg[:, frame + 1],
                        flow_f[:, frame], valid_f[:, frame],
                    )
                    n_weight = target_bg * warped_bg.clamp(0, 1) * valid.to(target.dtype)
                    numerator = numerator + n_weight * (warped - target)
                    denominator = denominator + n_weight
                differences.append(numerator / denominator)
                prev_weights.append(p_weight)
                next_weights.append(n_weight)
            difference = torch.stack(differences, dim=1)
            prev_weight = torch.stack(prev_weights, dim=1)
            next_weight = torch.stack(next_weights, dim=1)
        aligned = compressed + torch.tanh(warp_gain) * float(warp_strength) * difference
        resized_guide = F.interpolate(guide.flatten(0, 1), (height, width), mode="bilinear", align_corners=False)
        inputs = torch.cat((aligned.flatten(0, 1), resized_guide,
                            prev_weight.flatten(0, 1), next_weight.flatten(0, 1)), dim=1)
        correction = self.output(self.fuse(inputs)) * bg.flatten(0, 1)
        return correction, prev_weight, next_weight


class SparseBrushNetPortAdapter(nn.Module):
    def __init__(
        self,
        guide_channels: int,
        down_indices: Sequence[int],
        down_channels: Sequence[int],
        mid_channels: int,
        bottleneck: int,
    ):
        super().__init__()
        if len(down_indices) != len(down_channels) or not down_indices:
            raise ValueError("down_indices and down_channels must be nonempty and aligned")
        if sorted(set(map(int, down_indices))) != list(map(int, down_indices)):
            raise ValueError("down_indices must be unique and sorted")
        self.down_indices = tuple(int(i) for i in down_indices)
        self.down = nn.ModuleList(
            [PortProjection(int(c), guide_channels, bottleneck) for c in down_channels]
        )
        self.mid = PortProjection(int(mid_channels), guide_channels, bottleneck)

    def forward(self, down_residuals, mid_residual, guide, bg_rgb,
                flow_f, flow_b, valid_f, valid_b, warp_gain, warp_strength):
        corrections = []
        reliability = []
        for index, head in zip(self.down_indices, self.down):
            correction, prev, following = head(
                down_residuals[index], guide, bg_rgb,
                flow_f, flow_b, valid_f, valid_b, warp_gain, warp_strength,
            )
            corrections.append(correction)
            reliability.append((prev.mean(), following.mean()))
        mid_correction, prev, following = self.mid(
            mid_residual, guide, bg_rgb,
            flow_f, flow_b, valid_f, valid_b, warp_gain, warp_strength,
        )
        reliability.append((prev.mean(), following.mean()))
        return tuple(corrections), mid_correction, reliability


@dataclass
class TeacherWarpPortOutput:
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
    port_down_corrections: Tuple[torch.Tensor, ...]
    port_mid_correction: torch.Tensor
    reliability_backward: torch.Tensor
    reliability_forward: torch.Tensor
    teacher_flow_forward_rgb: torch.Tensor
    teacher_flow_backward_rgb: torch.Tensor
    teacher_valid_forward_rgb: torch.Tensor
    teacher_valid_backward_rgb: torch.Tensor
    bg_mask_sequence: torch.Tensor
    warp_strength: float
    brushnet_residuals: Optional[tuple]


class TeacherWarpPortModel(ModelMixin, ConfigMixin):
    """Compute TCE context; warp selected BrushNet residual ports on an x4 grid."""

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        hidden_channels: int = 64,
        num_heads: int = 2,
        num_layers: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        relative_position_max_distance: int = 32,
        cross_clip_memory_frames: int = 4,
        detach_cross_clip_memory: bool = True,
        require_memory_overlap: bool = True,
        rescaled_warp_scale: int = 4,
        down_port_indices: Sequence[int] = (2, 5, 8),
        down_port_channels: Sequence[int] = (320, 640, 1280),
        mid_port_channels: int = 1280,
        port_bottleneck_channels: int = 64,
        initial_warp_gain: float = 1.5,
    ):
        super().__init__()
        if int(rescaled_warp_scale) != 4:
            raise ValueError("The V9 teacher-warp experiment requires x4 rescaling")
        if tuple(down_port_indices) != (2, 5, 8):
            raise ValueError("The first V9 experiment selects down ports 2, 5, 8")
        self.stc_adapter = STCBackbone(
            int(hidden_channels), int(num_heads), int(num_layers), float(mlp_ratio),
            float(dropout), int(relative_position_max_distance),
        )
        # The old 64->4 latent correction has no role in this feature-port path.
        del self.stc_adapter.zero_conv
        self.warp_gain = nn.Parameter(torch.tensor(float(initial_warp_gain)))
        self.port_adapter = SparseBrushNetPortAdapter(
            guide_channels=int(hidden_channels),
            down_indices=down_port_indices,
            down_channels=down_port_channels,
            mid_channels=int(mid_port_channels),
            bottleneck=int(port_bottleneck_channels),
        )

    @classmethod
    def from_v8r_pretrained(cls, component: str | Path, **overrides):
        """Copy only spatial/temporal/norm tensors; reject missing source keys."""
        path = Path(component).expanduser().resolve()
        if path.name in {"best.json", "latest.json"}:
            pointer = json.loads(path.read_text(encoding="utf-8"))["checkpoint"]
            candidate = Path(pointer)
            path = (candidate if candidate.is_absolute() else path.parent / candidate).resolve()
        for name in ("stc_v8r_model", "stc_v8_model", "stc_v8r_standalone_model"):
            if (path / name / "config.json").is_file():
                path = path / name
                break
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        keys = (
            "hidden_channels", "num_heads", "num_layers", "mlp_ratio", "dropout",
            "relative_position_max_distance", "cross_clip_memory_frames",
            "detach_cross_clip_memory", "require_memory_overlap",
        )
        kwargs = {key: config[key] for key in keys if key in config}
        kwargs.update(overrides)
        model = cls(**kwargs)
        weights_path = path / "diffusion_pytorch_model.safetensors"
        if weights_path.is_file():
            source = load_file(str(weights_path), device="cpu")
        else:
            try:
                source = torch.load(path / "diffusion_pytorch_model.bin", map_location="cpu", weights_only=True)
            except TypeError:
                source = torch.load(path / "diffusion_pytorch_model.bin", map_location="cpu")
        prefixes = (
            "stc_adapter.spatial_encoder.",
            "stc_adapter.temporal_blocks.",
            "stc_adapter.output_norm.",
        )
        transferred = {}
        for key, target in model.state_dict().items():
            if not key.startswith(prefixes):
                continue
            value = source.get(key)
            if value is None or value.shape != target.shape:
                raise ValueError(f"Incompatible V8-R initialization tensor: {key}")
            transferred[key] = value
        if not transferred:
            raise ValueError("No V8-R spatial/temporal weights were transferred")
        incompatible = model.load_state_dict(transferred, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected transferred keys: {incompatible.unexpected_keys}")
        return model

    def forward(
        self,
        rgb_sequence: torch.Tensor,
        bg_mask_sequence: torch.Tensor,
        *,
        teacher_flow_forward_rgb: torch.Tensor,
        teacher_flow_backward_rgb: torch.Tensor,
        teacher_valid_forward_rgb: torch.Tensor,
        teacher_valid_backward_rgb: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        frame_ids: Optional[torch.Tensor] = None,
        temporal_memory: Optional[TemporalMemory] = None,
        frame_valid_mask: Optional[torch.Tensor] = None,
        warp_strength: float = 1.0,
        compute_port_corrections: bool = True,
        brushnet_residuals: Optional[tuple] = None,
    ) -> TeacherWarpPortOutput:
        if rgb_sequence.ndim != 5 or rgb_sequence.shape[2] != 3:
            raise ValueError("rgb_sequence must be [B,T,3,H,W]")
        batch, frames, _, rgb_h, rgb_w = rgb_sequence.shape
        if tuple(bg_mask_sequence.shape) != (batch, frames, 1, rgb_h, rgb_w):
            raise ValueError("BG mask must match RGB sequence")
        flow_shape = (batch, frames - 1, 2, rgb_h, rgb_w)
        valid_shape = (batch, frames - 1, 1, rgb_h, rgb_w)
        for name, value in (("forward", teacher_flow_forward_rgb), ("backward", teacher_flow_backward_rgb)):
            if tuple(value.shape) != flow_shape or not torch.isfinite(value).all():
                raise ValueError(f"teacher {name} flow must be finite {flow_shape}")
        for name, value in (("forward", teacher_valid_forward_rgb), ("backward", teacher_valid_backward_rgb)):
            if tuple(value.shape) != valid_shape:
                raise ValueError(f"teacher {name} validity must be {valid_shape}")
        if output_size is None:
            output_size = (rgb_h // 8, rgb_w // 8)
        output_size = tuple(map(int, output_size))
        if min(output_size) < 8 or any(n % 8 for n in output_size):
            raise ValueError("TCE feature dimensions must be divisible by eight")
        if frame_ids is None:
            frame_ids = torch.arange(frames, device=rgb_sequence.device)[None].expand(batch, -1)
        if tuple(frame_ids.shape) != (batch, frames):
            raise ValueError("frame_ids must be [B,T]")
        if frame_valid_mask is None:
            frame_valid_mask = torch.ones_like(frame_ids, dtype=torch.bool)
        elif tuple(frame_valid_mask.shape) != (batch, frames):
            raise ValueError("frame_valid_mask must be [B,T]")

        pixel_condition = torch.cat((rgb_sequence, bg_mask_sequence.to(rgb_sequence.dtype)), dim=2)
        spatial_flat = self.stc_adapter.spatial_encoder(pixel_condition.flatten(0, 1))
        if spatial_flat.shape[-2:] != output_size:
            spatial_flat = F.interpolate(spatial_flat, size=output_size, mode="bilinear", align_corners=False)
        channels, h, w = spatial_flat.shape[1:]
        spatial = spatial_flat.reshape(batch, frames, channels, h, w)
        bg = F.interpolate(bg_mask_sequence.flatten(0, 1).to(spatial.dtype), size=(h, w), mode="nearest")
        bg = (bg >= 0.5).to(spatial.dtype).reshape(batch, frames, 1, h, w)
        empty_flow = spatial.new_empty(batch, 0, 2, h, w)
        if frames > 1:
            flow_f = resize_flow(teacher_flow_forward_rgb.flatten(0, 1).detach(), (h, w)).to(spatial.dtype).reshape(batch, frames - 1, 2, h, w)
            flow_b = resize_flow(teacher_flow_backward_rgb.flatten(0, 1).detach(), (h, w)).to(spatial.dtype).reshape(batch, frames - 1, 2, h, w)
        else:
            flow_f = flow_b = empty_flow
        aligned = spatial

        overlap_count = frame_ids.new_zeros(batch)
        memory = None
        if temporal_memory is not None:
            if len(temporal_memory.layer_features) != len(self.stc_adapter.temporal_blocks):
                raise ValueError("cross-clip memory layer count mismatch")
            if tuple(temporal_memory.frame_ids.shape[:1]) != (batch,):
                raise ValueError("cross-clip memory batch mismatch")
            matches = frame_ids[:, :, None].eq(temporal_memory.frame_ids[:, None, :])
            overlap = matches.any(dim=1)
            valid = temporal_memory.valid_mask.to(frame_ids.device)
            if self.config.require_memory_overlap:
                valid = valid & overlap
            overlap_count = (valid & overlap).sum(dim=1)
            memory = (temporal_memory, valid)
        tokens = aligned.permute(0, 3, 4, 1, 2).reshape(batch * h * w, frames, channels)
        ids = frame_ids[:, None, None, :].expand(batch, h, w, frames).reshape(batch * h * w, frames)
        memory_frames = min(int(self.config.cross_clip_memory_frames), frames)
        layer_cache = []
        for layer_index, block in enumerate(self.stc_adapter.temporal_blocks):
            mem_tokens = mem_ids = mem_valid = None
            if memory is not None:
                state, valid = memory
                source = state.layer_features[layer_index].to(device=tokens.device, dtype=tokens.dtype)
                expected = (batch, state.frame_ids.shape[1], channels, h, w)
                if tuple(source.shape) != expected:
                    raise ValueError(f"cross-clip memory feature must be {expected}")
                mem_tokens = source.permute(0, 3, 4, 1, 2).reshape(batch * h * w, source.shape[1], channels)
                mem_ids = state.frame_ids[:, None, None, :].expand(batch, h, w, -1).reshape(batch * h * w, -1)
                mem_valid = valid[:, None, None, :].expand(batch, h, w, -1).reshape(batch * h * w, -1)
            tokens = block(tokens, ids, mem_tokens, mem_ids, mem_valid)
            layer = tokens.reshape(batch, h, w, frames, channels).permute(0, 3, 4, 1, 2)
            cached = layer[:, -memory_frames:]
            layer_cache.append(cached.detach() if self.config.detach_cross_clip_memory else cached)
        features = tokens.reshape(batch, h, w, frames, channels).permute(0, 3, 4, 1, 2)
        port_feature = self.stc_adapter.output_act(self.stc_adapter.output_norm(features.flatten(0, 1)))
        port_feature = port_feature.reshape(batch, frames, channels, h, w)
        next_memory = TemporalMemory(
            frame_ids[:, -memory_frames:].detach(),
            frame_valid_mask[:, -memory_frames:].detach(),
            tuple(layer_cache),
        )
        if brushnet_residuals is not None and compute_port_corrections:
            down_corrections, mid_correction, reliability = self.port_adapter(
                brushnet_residuals[0], brushnet_residuals[1], port_feature,
                bg_mask_sequence, teacher_flow_forward_rgb,
                teacher_flow_backward_rgb, teacher_valid_forward_rgb,
                teacher_valid_backward_rgb, self.warp_gain, warp_strength,
            )
            reliability_backward = torch.stack([p for p, _ in reliability]).mean()
            reliability_forward = torch.stack([n for _, n in reliability]).mean()
        else:
            down_corrections = ()
            mid_correction = spatial.new_empty(0)
            reliability_backward = spatial.new_zeros(())
            reliability_forward = spatial.new_zeros(())
        zero_delta = spatial.new_zeros(batch, frames, 4, h, w)
        zero_rel = spatial.new_zeros(batch, max(frames - 1, 0), 1, h, w)
        with torch.no_grad():
            relative_bias = torch.stack([
                block.attention.relative_position_bias.abs().mean()
                for block in self.stc_adapter.temporal_blocks
            ]).mean()
            memory_gate = torch.stack([
                torch.tanh(block.cross_clip_gate).abs()
                for block in self.stc_adapter.temporal_blocks
            ]).mean()
        return TeacherWarpPortOutput(
            delta_bg=zero_delta,
            features=port_feature,
            spatial_features=spatial,
            aligned_spatial_features=aligned,
            latent_bg_mask=bg,
            predicted_flow_forward=flow_f,
            predicted_flow_backward=flow_b,
            alignment_confidence=zero_rel,
            temporal_memory=next_memory,
            memory_overlap_count=overlap_count,
            relative_bias_abs_mean=relative_bias,
            cross_clip_gate_abs_mean=memory_gate,
            port_down_corrections=down_corrections,
            port_mid_correction=mid_correction,
            reliability_backward=reliability_backward,
            reliability_forward=reliability_forward,
            teacher_flow_forward_rgb=teacher_flow_forward_rgb,
            teacher_flow_backward_rgb=teacher_flow_backward_rgb,
            teacher_valid_forward_rgb=teacher_valid_forward_rgb,
            teacher_valid_backward_rgb=teacher_valid_backward_rgb,
            bg_mask_sequence=bg_mask_sequence,
            warp_strength=float(warp_strength),
            brushnet_residuals=brushnet_residuals,
        )
