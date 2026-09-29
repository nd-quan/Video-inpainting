"""Isolated Stage-3 helpers for joint STC-condition and U-Net temporal adapters.

This module intentionally lives beside, rather than inside,
``stc_condition_adapter_training.py``.  The condition-only Stage-3 experiment
therefore keeps its original architecture, optimizer groups, and checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch

from diffusers.models.unet_temporal_adapter import DiffusionUNetTemporalAdapter
from stc_condition_adapter_training import FrozenBrushNetSTCConditionModel


class FrozenBrushNetSTCConditionTemporalModel(FrozenBrushNetSTCConditionModel):
    """Stage 3 with a separate temporal residual path inside the frozen U-Net."""

    def __init__(
        self,
        noise_shaper,
        condition_adapter,
        temporal_adapter: DiffusionUNetTemporalAdapter,
        brushnet,
        unet,
        alphas_cumprod: torch.Tensor,
        noise_strength: float = 1.0,
        injection_scale: Optional[float] = None,
        temporal_adapter_scale: float = 1.0,
    ):
        super().__init__(
            noise_shaper=noise_shaper,
            condition_adapter=condition_adapter,
            brushnet=brushnet,
            unet=unet,
            alphas_cumprod=alphas_cumprod,
            noise_strength=noise_strength,
            injection_scale=injection_scale,
        )
        self.temporal_adapter = temporal_adapter
        self.temporal_adapter_scale = float(temporal_adapter_scale)

    def forward(
        self,
        clean_latents: torch.Tensor,
        independent_noise: torch.Tensor,
        decoded_latents: torch.Tensor,
        bg_mask: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        brushnet_condition: torch.Tensor,
    ):
        if clean_latents.ndim != 5 or clean_latents.shape != independent_noise.shape:
            raise ValueError("clean_latents/noise must match [B,T,C,H,W]")
        if decoded_latents.shape != clean_latents.shape:
            raise ValueError("decoded_latents must match clean_latents")
        batch, frames = clean_latents.shape[:2]
        if timesteps.shape != (batch,):
            raise ValueError("timesteps must have one value per video clip")

        shaped = self.noise_shaper(
            independent_noise=independent_noise,
            decoded_latents=decoded_latents,
            bg_mask=bg_mask,
            strength=self.noise_strength,
        )
        # Preserve the selected Stage-1/2 noise law. Gradients reach the STC
        # encoder only through feature injection, exactly as in original Stage 3.
        shaped_noise = shaped["noise"].detach()
        noisy_latents = self._add_noise(clean_latents, shaped_noise, timesteps)
        noisy_flat = noisy_latents.flatten(0, 1)
        condition_flat = brushnet_condition.flatten(0, 1)
        timestep_flat = timesteps.repeat_interleave(frames)
        text_flat = self._flatten_text(encoder_hidden_states, frames)

        down, mid, up = self.brushnet(
            noisy_flat,
            timestep_flat,
            encoder_hidden_states=text_flat,
            brushnet_cond=condition_flat,
            return_dict=False,
        )
        adapted = self.condition_adapter(
            shaped["stc_features"],
            down,
            mid,
            up,
            injection_scale=self.injection_scale,
            return_dict=True,
        )
        prediction = self.unet(
            noisy_flat,
            timestep_flat,
            encoder_hidden_states=text_flat,
            down_block_add_samples=list(adapted["down"]),
            mid_block_add_sample=adapted["mid"],
            up_block_add_samples=list(adapted["up"]),
            temporal_adapter=self.temporal_adapter,
            temporal_num_frames=frames,
            temporal_adapter_scale=self.temporal_adapter_scale,
            return_dict=False,
        )[0]
        return {
            "prediction": prediction.reshape(batch, frames, *prediction.shape[1:]),
            "noisy_latents": noisy_latents,
            "noise": shaped_noise,
            "predicted_flow_backward": shaped[
                "predicted_flow_backward"
            ].detach(),
            "beta_mean": shaped["beta_mean"].detach(),
            "effective_beta_mean": shaped["effective_beta_mean"].detach(),
            "correction_energy": adapted["correction_energy"],
            "correction_rms": adapted["correction_rms"],
            "correction_abs": adapted["correction_abs"],
        }


def build_temporal_adapter(unet, config) -> DiffusionUNetTemporalAdapter:
    """Build a zero-initialized adapter matching the deployed diffusion U-Net."""

    return DiffusionUNetTemporalAdapter(
        block_out_channels=tuple(int(value) for value in unet.config.block_out_channels),
        down_block_indices=tuple(int(value) for value in config.get("down_block_indices", (0, 1, 2))),
        use_mid=bool(config.get("use_mid", True)),
        up_block_indices=tuple(int(value) for value in config.get("up_block_indices", ())),
        bottleneck_channels=int(config.get("bottleneck_channels", 64)),
        temporal_kernel_size=int(config.get("temporal_kernel_size", 3)),
        dropout=float(config.get("dropout", 0.0)),
    )


def resolve_temporal_adapter_component(checkpoint: Path) -> Path:
    component = Path(checkpoint).expanduser().resolve() / "temporal_adapter"
    if not (component / "config.json").is_file():
        raise FileNotFoundError(
            f"Temporal adapter is missing from joint checkpoint: {component}"
        )
    return component


def load_or_build_temporal_adapter(
    unet,
    config,
    resume: Optional[Path] = None,
    initialization: Optional[Path] = None,
) -> DiffusionUNetTemporalAdapter:
    """Load joint weights, or initialize a new zero-residual temporal adapter.

    A condition-only Stage-3 initialization legitimately has no temporal folder;
    in that case the temporal output projections remain exactly zero initially.
    A true joint-training resume is strict and must contain the component.
    """

    if resume is not None:
        model = DiffusionUNetTemporalAdapter.from_pretrained(
            resolve_temporal_adapter_component(resume)
        )
    elif initialization is not None and (
        Path(initialization).expanduser().resolve() / "temporal_adapter" / "config.json"
    ).is_file():
        model = DiffusionUNetTemporalAdapter.from_pretrained(
            Path(initialization).expanduser().resolve() / "temporal_adapter"
        )
    else:
        model = build_temporal_adapter(unet, config)

    expected_channels = tuple(int(value) for value in unet.config.block_out_channels)
    actual_channels = tuple(int(value) for value in model.config.block_out_channels)
    if actual_channels != expected_channels:
        raise ValueError(
            "Temporal adapter/U-Net channel mismatch: "
            f"adapter={actual_channels}, unet={expected_channels}"
        )
    return model


def configure_joint_stage3_trainable(
    model: FrozenBrushNetSTCConditionTemporalModel,
    train_stc_encoder: bool = False,
) -> None:
    """Train only condition/temporal adapters and optional STC encoder."""

    model.requires_grad_(False)
    model.condition_adapter.requires_grad_(True)
    model.temporal_adapter.requires_grad_(True)
    if train_stc_encoder:
        model.noise_shaper.stc_encoder.requires_grad_(True)

    allowed = ["condition_adapter.", "temporal_adapter."]
    if train_stc_encoder:
        allowed.append("noise_shaper.stc_encoder.")
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not any(name.startswith(prefix) for prefix in allowed)
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected joint Stage-3 parameters: {unexpected}")


def joint_stage3_parameter_groups(
    model: FrozenBrushNetSTCConditionTemporalModel,
    condition_lr: float,
    temporal_lr: float,
    train_stc_encoder: bool = False,
    stc_lr: Optional[float] = None,
):
    configure_joint_stage3_trainable(model, train_stc_encoder)
    groups = [
        {
            "params": list(model.condition_adapter.parameters()),
            "lr": float(condition_lr),
            "name": "condition_adapter",
        },
        {
            "params": list(model.temporal_adapter.parameters()),
            "lr": float(temporal_lr),
            "name": "temporal_adapter",
        },
    ]
    if train_stc_encoder:
        groups.append(
            {
                "params": list(model.noise_shaper.stc_encoder.parameters()),
                "lr": float(stc_lr if stc_lr is not None else condition_lr * 0.1),
                "name": "stc_encoder",
            }
        )
    return groups


def joint_stage3_trainable_parameters(
    model: FrozenBrushNetSTCConditionTemporalModel,
) -> Iterable[torch.nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def set_joint_stage3_train_mode(
    model,
    train_stc_encoder: bool,
    gradient_checkpointing: bool,
) -> None:
    bare = model.module if hasattr(model, "module") else model
    bare.eval()
    bare.condition_adapter.train()
    bare.temporal_adapter.train()
    bare.noise_shaper.eval()
    if train_stc_encoder:
        bare.noise_shaper.stc_encoder.train()
    if gradient_checkpointing:
        bare.brushnet.train()
        bare.unet.train()
