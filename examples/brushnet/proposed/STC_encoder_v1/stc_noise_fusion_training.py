"""Stage-2 helpers for training STC-guided noise fusion.

Stage 1 stores a full-flow :class:`STCConditionedNoiseShaper` with a fixed
beta.  Stage 2 rebuilds the same model with a learned beta head, transfers the
STC encoder and bidirectional flow decoder exactly, and freezes every
parameter except the new beta head.

This module deliberately has no dependency on the legacy motion adapter or
its raw/refined-flow dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

from diffusers.models.stc_noise_shaper import STCConditionedNoiseShaper


def resolve_component_checkpoint(
    path: Path,
    component: str = "flow_predictor",
) -> Path:
    """Resolve a checkpoint directory, component directory, or JSON pointer."""

    path = Path(path).expanduser().resolve()
    if path.name in {"latest.json", "best.json"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "checkpoint" not in payload:
            raise KeyError(f"Checkpoint pointer has no 'checkpoint' field: {path}")
        path = Path(payload["checkpoint"]).expanduser().resolve()
    nested = path / component
    if nested.is_dir():
        path = nested
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"Cannot find {component!r} pretrained component at {path}"
        )
    return path


def build_stage2_noise_shaper(
    stage1_checkpoint: Path,
    fusion_config: Optional[Mapping] = None,
) -> Tuple[STCConditionedNoiseShaper, Path]:
    """Create learned-beta Stage 2 model from frozen Stage-1 flow weights.

    The only expected missing weights during transfer are the newly created
    ``beta_head`` parameters.  Any other mismatch is treated as a hard error so
    Stage 2 cannot silently start from an incomplete flow predictor.
    """

    fusion = dict(fusion_config or {})
    component = resolve_component_checkpoint(stage1_checkpoint, "flow_predictor")
    stage1 = STCConditionedNoiseShaper.from_pretrained(component)
    if str(stage1.config.flow_prediction_mode).lower() != "full":
        raise ValueError("Stage 2 requires a Stage-1 flow_prediction_mode='full' model")

    beta_min = float(fusion.get("beta_min", 0.05))
    beta_max = float(fusion.get("beta_max", 0.95))
    initial_beta = float(fusion.get("initial_beta", 0.5))
    model = STCConditionedNoiseShaper.from_config(
        stage1.config,
        beta_mode="learned",
        beta_min=beta_min,
        beta_max=beta_max,
        initial_beta=initial_beta,
        warp_region="all",
        channel_normalize=bool(fusion.get("channel_normalize", False)),
        global_normalize=bool(fusion.get("global_normalize", False)),
        norm_eps=float(fusion.get("norm_eps", 1e-5)),
    )
    incompatible = model.load_state_dict(stage1.state_dict(), strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    expected_missing = {
        "beta_head.0.weight",
        "beta_head.0.bias",
        "beta_head.2.weight",
        "beta_head.2.bias",
    }
    if missing != expected_missing or unexpected:
        raise RuntimeError(
            "Unexpected Stage-1 transfer mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    set_beta_only_trainable(model)
    return model, component


def set_beta_only_trainable(model: STCConditionedNoiseShaper) -> None:
    """Freeze STC/flow prediction and expose only learned beta parameters."""

    if str(model.config.flow_prediction_mode).lower() != "full":
        raise ValueError("Beta-only Stage 2 requires full-flow mode")
    if str(model.config.beta_mode).lower() != "learned" or model.beta_head is None:
        raise ValueError("Beta-only Stage 2 requires beta_mode='learned'")
    model.requires_grad_(False)
    model.beta_head.requires_grad_(True)


def beta_parameters(model: STCConditionedNoiseShaper) -> Iterable[torch.nn.Parameter]:
    """Yield and validate the complete Stage-2 trainable parameter set."""

    set_beta_only_trainable(model)
    parameters = [parameter for parameter in model.beta_head.parameters()]
    if not parameters or not all(parameter.requires_grad for parameter in parameters):
        raise ValueError("The learned beta head has no trainable parameters")
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if not trainable_names or any(
        not name.startswith("beta_head.") for name in trainable_names
    ):
        raise RuntimeError(
            f"Stage 2 exposed non-beta trainable parameters: {sorted(trainable_names)}"
        )
    return parameters


def backward_warp_sequence(
    source: torch.Tensor,
    backward_flow: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Warp ``[B,P,C,H,W]`` source features with latent-pixel backward flow."""

    if source.ndim != 5 or backward_flow.ndim != 5:
        raise ValueError("source and backward_flow must be five-dimensional")
    if source.shape[:2] != backward_flow.shape[:2] or backward_flow.shape[2] != 2:
        raise ValueError("source/flow must match [B,P] and flow must have two channels")
    batch, pairs, channels, height, width = source.shape
    flow = backward_flow
    if flow.shape[-2:] != (height, width):
        old_height, old_width = flow.shape[-2:]
        flow = F.interpolate(
            flow.flatten(0, 1),
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        ).reshape(batch, pairs, 2, height, width)
        flow = flow.clone()
        flow[:, :, 0].mul_(width / max(old_width, 1))
        flow[:, :, 1].mul_(height / max(old_height, 1))

    dtype = source.dtype
    yy, xx = torch.meshgrid(
        torch.arange(height, device=source.device, dtype=dtype),
        torch.arange(width, device=source.device, dtype=dtype),
        indexing="ij",
    )
    sample_x = xx[None, None] + flow[:, :, 0].to(dtype)
    sample_y = yy[None, None] + flow[:, :, 1].to(dtype)
    valid = (
        (sample_x >= 0.0)
        & (sample_x <= max(width - 1, 0))
        & (sample_y >= 0.0)
        & (sample_y <= max(height - 1, 0))
        & torch.isfinite(sample_x)
        & torch.isfinite(sample_y)
    )[:, :, None]
    grid = torch.stack(
        (
            2.0 * sample_x / max(width - 1, 1) - 1.0,
            2.0 * sample_y / max(height - 1, 1) - 1.0,
        ),
        dim=-1,
    )
    warped = F.grid_sample(
        source.reshape(batch * pairs, channels, height, width),
        grid.reshape(batch * pairs, height, width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(batch, pairs, channels, height, width)
    return warped, valid.to(source.dtype)


def temporal_latent_loss(
    predicted_clean: torch.Tensor,
    backward_flow: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Motion-compensated temporal loss over every geometrically valid pixel."""

    if predicted_clean.ndim != 5 or predicted_clean.shape[1] < 2:
        raise ValueError("predicted_clean must have shape [B,T,C,H,W] with T>=2")
    previous = predicted_clean[:, :-1]
    current = predicted_clean[:, 1:]
    warped_previous, valid = backward_warp_sequence(previous, backward_flow)
    error = ((current.float() - warped_previous.float()).square() + eps**2).sqrt()
    weight = valid.float().expand_as(error)
    return (error * weight).sum() / weight.sum().clamp_min(1e-6)


def beta_spatial_smoothness(beta: torch.Tensor) -> torch.Tensor:
    """Small total-variation prior for the learned beta maps."""

    if beta.ndim != 5:
        raise ValueError("beta must have shape [B,T-1,1,H,W]")
    horizontal = (
        (beta[..., :, 1:] - beta[..., :, :-1]).abs().mean()
        if beta.shape[-1] > 1
        else beta.new_zeros(())
    )
    vertical = (
        (beta[..., 1:, :] - beta[..., :-1, :]).abs().mean()
        if beta.shape[-2] > 1
        else beta.new_zeros(())
    )
    return 0.5 * (horizontal + vertical)
