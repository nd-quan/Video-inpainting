"""Exact frozen V8/IP-Adapter helpers used by RGB-STC phase 1."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file

from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from ip_adapter.attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from ip_adapter.attention_processor import AttnProcessor, IPAttnProcessor


class FGBGFeatureFusion(nn.Module):
    """The unchanged fusion module from the V8 shared-noise trainer."""

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.fg_to_bg_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.bg_to_fg_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, historical_fg: torch.Tensor, historical_bg: torch.Tensor):
        # Historical V8 naming is intentionally retained: historical_fg is
        # actually degraded-BG-only and historical_bg is HQ-ROI-only.
        historical_fg = historical_fg.unsqueeze(1)
        historical_bg = historical_bg.unsqueeze(1)
        # Preserve the original V8 call contract, including MultiheadAttention's
        # default kernel path, for the closest checkpoint reproduction.
        fg_to_bg, _ = self.fg_to_bg_attn(
            historical_fg, historical_bg, historical_bg
        )
        bg_to_fg, _ = self.bg_to_fg_attn(
            historical_bg, historical_fg, historical_fg
        )
        return self.out_proj(torch.cat((fg_to_bg, bg_to_fg), dim=-1)).squeeze(1)


def _attention_hidden_size(unet, name: str) -> int:
    if name.startswith("mid_block"):
        return int(unet.config.block_out_channels[-1])
    if name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        return int(list(reversed(unet.config.block_out_channels))[block_id])
    if name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        return int(unet.config.block_out_channels[block_id])
    raise ValueError(f"Cannot infer attention hidden size for {name}")


def install_and_load_ip_adapter(
    unet,
    image_proj_model: ImageProjModel,
    checkpoint_path,
) -> Dict[str, int]:
    """Install IP processors and strictly consume every checkpoint tensor."""
    checkpoint_path = Path(checkpoint_path)
    state = load_file(str(checkpoint_path), device="cpu")
    all_keys = set(state)

    image_prefix = "image_proj_model."
    image_state = {
        key[len(image_prefix) :]: value
        for key, value in state.items()
        if key.startswith(image_prefix)
    }
    image_proj_model.load_state_dict(image_state, strict=True)
    consumed = {image_prefix + key for key in image_state}

    processors = {}
    cross_attention_count = 0
    for name in unet.attn_processors.keys():
        if name.endswith("attn1.processor"):
            processors[name] = AttnProcessor()
        else:
            processors[name] = IPAttnProcessor(
                hidden_size=_attention_hidden_size(unet, name),
                cross_attention_dim=unet.config.cross_attention_dim,
            )
            cross_attention_count += 1
    unet.set_attn_processor(processors)

    for name, processor in unet.attn_processors.items():
        if not name.endswith("attn2.processor"):
            continue
        prefix = f"unet.{name}."
        processor_state = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        processor.load_state_dict(processor_state, strict=True)
        consumed.update(prefix + key for key in processor_state)

    unused = sorted(all_keys - consumed)
    if unused:
        raise ValueError(
            "IP-Adapter checkpoint contains tensors not consumed by the exact "
            f"V8 mapping: {unused[:5]} (total={len(unused)})"
        )
    if cross_attention_count != 16:
        raise ValueError(
            f"Expected 16 SD1.5 attn2 processors, got {cross_attention_count}"
        )
    return {
        "image_projection_tensors": len(image_state),
        "cross_attention_processors": cross_attention_count,
        "checkpoint_tensors": len(all_keys),
    }


def load_fusion_module(checkpoint_path, embed_dim: int = 1024):
    fusion = FGBGFeatureFusion(embed_dim=embed_dim, num_heads=8)
    fusion.load_state_dict(load_file(str(checkpoint_path), device="cpu"), strict=True)
    return fusion


@torch.no_grad()
def build_frozen_v8_context(
    batch,
    image_encoder,
    fusion_module,
    image_proj_model,
    text_encoder,
    fusion_scale: float,
    device,
    dtype,
    autocast_context,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the checkpoint's null-text + full/fused CLIP conditioning."""
    base_image_embeds = image_encoder(
        batch["clip_images"].to(device=device, dtype=dtype)
    ).image_embeds
    historical_fg = image_encoder(
        batch["fg_clip_images"].to(device=device, dtype=dtype)
    ).image_embeds
    historical_bg = image_encoder(
        batch["bg_clip_images"].to(device=device, dtype=dtype)
    ).image_embeds
    # CLIP encoders above keep permanently-cast inference weights and run
    # outside autocast, exactly as V8. Only prepared FP32 modules enter AMP.
    with autocast_context():
        fused = fusion_module(historical_fg, historical_bg)
    # Prepared fusion's output wrapper cast back to FP32 before this addition.
    fused = fused.float()
    image_embeds = base_image_embeds + float(fusion_scale) * fused

    drop_flags = batch["drop_image_embeds"].to(device=device).bool()
    if drop_flags.any():
        image_embeds = image_embeds.masked_fill(drop_flags[:, None], 0.0)
    with autocast_context():
        ip_tokens = image_proj_model(image_embeds)

    text_hidden_states = text_encoder(
        batch["input_ids"].to(device=device), return_dict=False
    )[0]
    unet_hidden_states = torch.cat((text_hidden_states, ip_tokens), dim=1)
    return text_hidden_states, unet_hidden_states


def frozen_v8_predict(
    brushnet,
    unet,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    brushnet_condition: torch.Tensor,
    brushnet_text_hidden_states: torch.Tensor,
    unet_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Run frozen BrushNet and U-Net while retaining gradients to condition."""
    down_samples, mid_sample, up_samples = brushnet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=brushnet_text_hidden_states,
        brushnet_cond=brushnet_condition,
        return_dict=False,
    )
    return unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=unet_hidden_states,
        # Prepared BrushNet converted its outputs back to FP32 in V8, followed
        # by these explicit casts before the prepared/autocast U-Net.
        down_block_add_samples=[
            sample.to(dtype=noisy_latents.dtype) for sample in down_samples
        ],
        mid_block_add_sample=mid_sample.to(dtype=noisy_latents.dtype),
        up_block_add_samples=[
            sample.to(dtype=noisy_latents.dtype) for sample in up_samples
        ],
        return_dict=False,
    )[0]
