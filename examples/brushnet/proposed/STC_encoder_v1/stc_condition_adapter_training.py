"""Stage-3 helpers for STC feature injection into frozen BrushNet diffusion.

These helpers intentionally contain no diffusion U-Net temporal adapter. The
joint experimental variant lives in ``stc_condition_temporal_adapter_training``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from diffusers.models.stc_condition_adapter import STCBrushNetConditionAdapter
from diffusers.models.stc_noise_shaper import STCConditionedNoiseShaper


def _resolve_pointer(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.name in {"latest.json", "best.json"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "checkpoint" not in payload:
            raise KeyError(f"Checkpoint pointer has no 'checkpoint' field: {path}")
        path = Path(payload["checkpoint"]).expanduser().resolve()
    return path


def resolve_stage2_noise_shaper(path: Path) -> Path:
    """Resolve a Stage-2 checkpoint/pointer or a direct noise-shaper folder."""

    path = _resolve_pointer(path)
    nested = path / "noise_shaper"
    if nested.is_dir():
        path = nested
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"Stage-2 noise shaper not found at {path}")
    return path


def load_stage2_noise_shaper(path: Path) -> Tuple[STCConditionedNoiseShaper, Path]:
    component = resolve_stage2_noise_shaper(path)
    model = STCConditionedNoiseShaper.from_pretrained(component)
    if str(model.config.flow_prediction_mode).lower() != "full":
        raise ValueError("Stage 3 requires a full-flow Stage-2 noise shaper")
    if str(model.config.beta_mode).lower() != "learned":
        raise ValueError(
            "Stage 3 expects the learned-beta Stage-2 checkpoint; "
            "run train_stc_noise_fusion_vcm.py first"
        )
    return model, component


def resolve_stage1_flow_predictor(path: Path) -> Path:
    """Resolve a Stage-1 checkpoint/pointer or direct flow-predictor folder."""

    path = _resolve_pointer(path)
    nested = path / "flow_predictor"
    if nested.is_dir():
        path = nested
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(f"Stage-1 flow predictor not found at {path}")
    return path


def load_stage1_fixed_noise_shaper(
    path: Path,
    fixed_beta: float,
) -> Tuple[STCConditionedNoiseShaper, Path]:
    """Reuse Stage-1 flow weights with an immutable fixed-beta noise law.

    The saved Stage-1 checkpoint is never modified. A new model is rebuilt
    from its architecture config, fixed-beta deployment options are applied,
    and every trained flow weight is transferred strictly.
    """

    beta = float(fixed_beta)
    if not 0.0 <= beta <= 1.0:
        raise ValueError("model.fixed_beta must be in [0,1]")
    component = resolve_stage1_flow_predictor(path)
    source = STCConditionedNoiseShaper.from_pretrained(component)
    if str(source.config.flow_prediction_mode).lower() != "full":
        raise ValueError("Fixed-beta Stage 3 requires a full-flow Stage-1 predictor")
    if str(source.config.beta_mode).lower() != "fixed":
        raise ValueError(
            "Fixed-beta Stage 3 requires a fixed-beta Stage-1 checkpoint"
        )
    model = STCConditionedNoiseShaper.from_config(
        source.config,
        beta_mode="fixed",
        fixed_beta=beta,
        warp_region="all",
    )
    model.load_state_dict(source.state_dict(), strict=True)
    del source
    model.requires_grad_(False).eval()
    validate_stage3_noise_shaper(
        model,
        expected_mode="fixed_stage1",
        fixed_beta=beta,
    )
    return model, component


def validate_stage3_noise_shaper(
    model: STCConditionedNoiseShaper,
    expected_mode: str,
    fixed_beta: Optional[float] = None,
) -> None:
    """Validate fresh, initialized, and resumed Stage-3 noise sources."""

    mode = str(expected_mode).lower()
    if mode not in {"learned_stage2", "fixed_stage1"}:
        raise ValueError(
            "model.noise_source must be 'learned_stage2' or 'fixed_stage1'"
        )
    if str(model.config.flow_prediction_mode).lower() != "full":
        raise ValueError("Stage 3 requires full-flow STC prediction")
    if str(model.config.warp_region).lower() != "all":
        raise ValueError("Stage 3 requires whole-frame noise warping")

    beta_mode = str(model.config.beta_mode).lower()
    beta_head = getattr(model, "beta_head", None)
    if mode == "learned_stage2":
        if beta_mode != "learned" or beta_head is None:
            raise ValueError(
                "learned_stage2 requires a learned beta_head checkpoint"
            )
        return

    if beta_mode != "fixed" or beta_head is not None:
        raise ValueError("fixed_stage1 requires fixed beta and no beta_head")
    if fixed_beta is None:
        raise ValueError("model.fixed_beta is required for fixed_stage1")
    expected_beta = float(fixed_beta)
    actual_beta = float(model.config.fixed_beta)
    if not 0.0 <= expected_beta <= 1.0:
        raise ValueError("model.fixed_beta must be in [0,1]")
    if abs(actual_beta - expected_beta) > 1e-8:
        raise ValueError(
            "Fixed-beta checkpoint/config mismatch: "
            f"checkpoint={actual_beta}, config={expected_beta}"
        )


def build_stage3_adapter(
    brushnet,
    noise_shaper: STCConditionedNoiseShaper,
    adapter_config: Optional[Mapping] = None,
) -> STCBrushNetConditionAdapter:
    config = dict(adapter_config or {})
    input_channels = int(noise_shaper.config.hidden_channels)
    configured_input = int(config.get("input_channels", input_channels))
    if configured_input != input_channels:
        raise ValueError(
            "adapter.input_channels must match the noise-source STC hidden channels: "
            f"{configured_input} vs {input_channels}"
        )
    return STCBrushNetConditionAdapter.from_brushnet(
        brushnet,
        input_channels=input_channels,
        bottleneck_channels=int(config.get("bottleneck_channels", 64)),
        use_down=bool(config.get("use_down", True)),
        use_mid=bool(config.get("use_mid", True)),
        use_up=bool(config.get("use_up", False)),
        injection_scale=float(config.get("injection_scale", 1.0)),
    )


def validate_stage3_adapter(
    adapter: STCBrushNetConditionAdapter,
    brushnet,
    noise_shaper: STCConditionedNoiseShaper,
) -> None:
    """Reject initialized/resumed adapters built for another deployment."""

    expected_input = int(noise_shaper.config.hidden_channels)
    if int(adapter.config.input_channels) != expected_input:
        raise ValueError(
            "Stage-3 adapter/STC channel mismatch: "
            f"adapter={adapter.config.input_channels}, STC={expected_input}"
        )
    expected_down = tuple(
        int(module.out_channels) for module in brushnet.brushnet_down_blocks
    )
    expected_mid = int(brushnet.brushnet_mid_block.out_channels)
    expected_up = tuple(
        int(module.out_channels) for module in brushnet.brushnet_up_blocks
    )
    if tuple(adapter.config.down_channels) != expected_down:
        raise ValueError("Stage-3 adapter down ports do not match deployed BrushNet")
    if int(adapter.config.mid_channel) != expected_mid:
        raise ValueError("Stage-3 adapter mid port does not match deployed BrushNet")
    if tuple(adapter.config.up_channels) != expected_up:
        raise ValueError("Stage-3 adapter up ports do not match deployed BrushNet")


class FrozenBrushNetSTCConditionModel(nn.Module):
    """Run fixed/learned noise shaping and Stage-3 injection end to end.

    BrushNet, the diffusion U-Net and the noise-shaping law are frozen.
    Gradients traverse the frozen denoiser to the condition adapter and, when
    explicitly enabled, to the STC encoder. Fixed-beta deployments contain no
    beta head; learned-beta deployments keep their Stage-2 beta head frozen.
    """

    def __init__(
        self,
        noise_shaper: STCConditionedNoiseShaper,
        condition_adapter: STCBrushNetConditionAdapter,
        brushnet: nn.Module,
        unet: nn.Module,
        alphas_cumprod: torch.Tensor,
        noise_strength: float = 1.0,
        injection_scale: Optional[float] = None,
    ):
        super().__init__()
        if not 0.0 <= float(noise_strength) <= 1.0:
            raise ValueError("noise_strength must be in [0,1]")
        self.noise_shaper = noise_shaper
        self.condition_adapter = condition_adapter
        self.brushnet = brushnet
        self.unet = unet
        self.noise_strength = float(noise_strength)
        self.injection_scale = injection_scale
        self.register_buffer(
            "alphas_cumprod",
            torch.as_tensor(alphas_cumprod, dtype=torch.float32).clone(),
            persistent=False,
        )

    @staticmethod
    def _flatten_text(text: torch.Tensor, frames: int) -> torch.Tensor:
        if text.ndim == 3:
            return text.repeat_interleave(frames, dim=0)
        if text.ndim == 4:
            return text.flatten(0, 1)
        raise ValueError("encoder_hidden_states must be [B,L,D] or [B,T,L,D]")

    def _add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self.alphas_cumprod.to(
            device=clean.device, dtype=clean.dtype
        )[timesteps].reshape(-1, 1, 1, 1, 1)
        return alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise

    def forward(
        self,
        clean_latents: torch.Tensor,
        independent_noise: torch.Tensor,
        decoded_latents: torch.Tensor,
        bg_mask: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        brushnet_condition: torch.Tensor,
    ) -> Dict[str, object]:
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
        # Stage 3 must not change the selected fixed/learned noise law through
        # its diffusion target. Optional STC fine-tuning is driven only by the
        # feature-injection path below.
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
            return_dict=False,
        )[0]
        return {
            "prediction": prediction.reshape(
                batch, frames, *prediction.shape[1:]
            ),
            "noisy_latents": noisy_latents,
            "noise": shaped_noise,
            "predicted_flow_backward": shaped["predicted_flow_backward"].detach(),
            "beta_mean": shaped["beta_mean"].detach(),
            "effective_beta_mean": shaped["effective_beta_mean"].detach(),
            "correction_energy": adapted["correction_energy"],
            "correction_rms": adapted["correction_rms"],
            "correction_abs": adapted["correction_abs"],
        }


def configure_stage3_trainable(
    model: FrozenBrushNetSTCConditionModel,
    train_stc_encoder: bool = False,
) -> None:
    """Expose only Stage-3 parameters and optionally the pretrained STC encoder."""

    model.requires_grad_(False)
    model.condition_adapter.requires_grad_(True)
    if train_stc_encoder:
        model.noise_shaper.stc_encoder.requires_grad_(True)

    allowed_prefixes = ["condition_adapter."]
    if train_stc_encoder:
        allowed_prefixes.append("noise_shaper.stc_encoder.")
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not any(name.startswith(prefix) for prefix in allowed_prefixes)
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected Stage-3 trainable parameters: {unexpected}")


def stage3_parameter_groups(
    model: FrozenBrushNetSTCConditionModel,
    adapter_lr: float,
    train_stc_encoder: bool = False,
    stc_lr: Optional[float] = None,
):
    configure_stage3_trainable(model, train_stc_encoder=train_stc_encoder)
    adapter_parameters = list(model.condition_adapter.parameters())
    if not adapter_parameters:
        raise ValueError("Stage-3 condition adapter has no parameters")
    groups = [{"params": adapter_parameters, "lr": float(adapter_lr), "name": "adapter"}]
    if train_stc_encoder:
        stc_parameters = list(model.noise_shaper.stc_encoder.parameters())
        groups.append(
            {
                "params": stc_parameters,
                "lr": float(stc_lr if stc_lr is not None else adapter_lr * 0.1),
                "name": "stc_encoder",
            }
        )
    return groups


def stage3_trainable_parameters(
    model: FrozenBrushNetSTCConditionModel,
) -> Iterable[torch.nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def set_stage3_train_mode(
    model,
    train_stc_encoder: bool,
    gradient_checkpointing: bool,
) -> None:
    bare = model.module if hasattr(model, "module") else model
    bare.eval()
    bare.condition_adapter.train()
    bare.noise_shaper.eval()
    if train_stc_encoder:
        bare.noise_shaper.stc_encoder.train()
    # Diffusers checkpointing is active only in train mode. These modules stay
    # parameter-frozen, so this changes activation handling rather than weights.
    if gradient_checkpointing:
        bare.brushnet.train()
        bare.unet.train()
