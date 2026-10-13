"""Build oracle STC features and inject them at BrushNet residual ports."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def precompute_brushnet_residuals(
    *, brushnet, noisy_latents, timesteps, base_condition_latents,
    bg_mask_sequence, brushnet_text_hidden_states,
):
    bg = F.interpolate(
        bg_mask_sequence.flatten(0, 1).to(base_condition_latents.dtype),
        size=base_condition_latents.shape[-2:], mode="nearest",
    )
    condition = torch.cat((base_condition_latents, (bg >= 0.5).to(bg.dtype)), dim=1)
    with torch.no_grad():
        residuals = brushnet(
            noisy_latents, timesteps,
            encoder_hidden_states=brushnet_text_hidden_states,
            brushnet_cond=condition,
            return_dict=False,
        )
    return {"brushnet_residuals": residuals}


def augment_brushnet_condition(
    model,
    base_condition_latents: torch.Tensor,
    rgb_sequence: torch.Tensor,
    bg_mask_sequence: torch.Tensor,
    *,
    teacher_flow_forward_rgb: torch.Tensor,
    teacher_flow_backward_rgb: torch.Tensor,
    teacher_valid_forward_rgb: torch.Tensor,
    teacher_valid_backward_rgb: torch.Tensor,
    frame_ids: Optional[torch.Tensor] = None,
    previous_rgb_sequence: Optional[torch.Tensor] = None,
    previous_bg_mask_sequence: Optional[torch.Tensor] = None,
    previous_frame_ids: Optional[torch.Tensor] = None,
    previous_valid_mask: Optional[torch.Tensor] = None,
    previous_teacher_flow_forward_rgb: Optional[torch.Tensor] = None,
    previous_teacher_flow_backward_rgb: Optional[torch.Tensor] = None,
    previous_teacher_valid_forward_rgb: Optional[torch.Tensor] = None,
    previous_teacher_valid_backward_rgb: Optional[torch.Tensor] = None,
    warp_strength: float = 1.0,
    brushnet_residuals=None,
    injection_scale: float = 1.0,
    predict_flow: bool = True,
):
    del injection_scale, predict_flow  # Port scale is applied after BrushNet.
    batch, frames = rgb_sequence.shape[:2]
    if base_condition_latents.ndim != 4 or base_condition_latents.shape[:2] != (batch * frames, 4):
        raise ValueError("base_condition_latents must be [B*T,4,h,w]")
    memory = None
    if previous_rgb_sequence is not None:
        required = (
            previous_bg_mask_sequence, previous_frame_ids, previous_valid_mask,
            previous_teacher_flow_forward_rgb, previous_teacher_flow_backward_rgb,
            previous_teacher_valid_forward_rgb, previous_teacher_valid_backward_rgb,
        )
        if any(value is None for value in required):
            raise ValueError("Predecessor RGB requires masks, IDs and teacher flow/validity")
        if bool(previous_valid_mask.any()):
            bare = model.module if hasattr(model, "module") else model
            with torch.no_grad():
                previous_output = bare(
                    previous_rgb_sequence,
                    previous_bg_mask_sequence,
                    teacher_flow_forward_rgb=previous_teacher_flow_forward_rgb,
                    teacher_flow_backward_rgb=previous_teacher_flow_backward_rgb,
                    teacher_valid_forward_rgb=previous_teacher_valid_forward_rgb,
                    teacher_valid_backward_rgb=previous_teacher_valid_backward_rgb,
                    output_size=base_condition_latents.shape[-2:],
                    frame_ids=previous_frame_ids,
                    frame_valid_mask=previous_valid_mask,
                    warp_strength=warp_strength,
                    compute_port_corrections=False,
                )
            memory = previous_output.temporal_memory.detach()
    output = model(
        rgb_sequence,
        bg_mask_sequence,
        teacher_flow_forward_rgb=teacher_flow_forward_rgb,
        teacher_flow_backward_rgb=teacher_flow_backward_rgb,
        teacher_valid_forward_rgb=teacher_valid_forward_rgb,
        teacher_valid_backward_rgb=teacher_valid_backward_rgb,
        output_size=base_condition_latents.shape[-2:],
        frame_ids=frame_ids,
        temporal_memory=memory,
        warp_strength=warp_strength,
        brushnet_residuals=brushnet_residuals,
    )
    bg_latent = F.interpolate(
        bg_mask_sequence.flatten(0, 1).to(device=base_condition_latents.device, dtype=base_condition_latents.dtype),
        size=base_condition_latents.shape[-2:],
        mode="nearest",
    )
    bg_latent = (bg_latent >= 0.5).to(base_condition_latents.dtype)
    condition = torch.cat((base_condition_latents, bg_latent), dim=1)
    return condition, output, base_condition_latents.reshape(batch, frames, 4, *base_condition_latents.shape[-2:])


def predict_with_feature_ports(
    *, model, stc_output, args, brushnet, unet, noisy_latents,
    timesteps, brushnet_condition, brushnet_text_hidden_states, unet_hidden_states,
):
    del brushnet, brushnet_condition, brushnet_text_hidden_states
    if stc_output.brushnet_residuals is None:
        raise ValueError("V9 predictor requires BrushNet residuals computed before TCE")
    down, mid, up = stc_output.brushnet_residuals
    bare = model.module if hasattr(model, "module") else model
    indices = tuple(int(value) for value in bare.config.down_port_indices)
    corrections = stc_output.port_down_corrections
    if len(indices) != len(corrections):
        raise RuntimeError("Feature-port correction count does not match the checkpoint")
    down = list(down)
    scale = float(args.stc_injection_scale)
    for index, correction in zip(indices, corrections):
        if tuple(correction.shape) != tuple(down[index].shape):
            raise ValueError(f"down[{index}] correction shape {tuple(correction.shape)} != {tuple(down[index].shape)}")
        down[index] = down[index] + scale * correction.to(device=down[index].device, dtype=down[index].dtype)
    correction = stc_output.port_mid_correction
    if tuple(correction.shape) != tuple(mid.shape):
        raise ValueError(f"mid correction shape {tuple(correction.shape)} != {tuple(mid.shape)}")
    mid = mid + scale * correction.to(device=mid.device, dtype=mid.dtype)
    return unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=unet_hidden_states,
        down_block_add_samples=[value.to(dtype=noisy_latents.dtype) for value in down],
        mid_block_add_sample=mid.to(dtype=noisy_latents.dtype),
        up_block_add_samples=[value.to(dtype=noisy_latents.dtype) for value in up],
        return_dict=False,
    )[0]
