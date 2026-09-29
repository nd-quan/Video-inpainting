"""Explicit, audited initialization; never load the legacy V5 flow head."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .model import StandaloneV8RAdapter


CONFIG_KEYS = (
    "hidden_channels", "num_heads", "num_layers", "mlp_ratio", "dropout",
    "downsample_factor", "output_channels", "condition_mode",
    "flow_max_displacement", "relative_position_max_distance",
    "cross_clip_memory_frames", "detach_cross_clip_memory", "require_memory_overlap",
    "deform_hidden_channels", "deform_kernel_size", "deform_groups",
    "deform_residual_max_displacement", "deformable_alignment_direction",
    "rescaled_warp_scale",
)


def resolve_component(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    candidates = (path, path / "stc_v8r_model", path / "stc_v8_model")
    for candidate in candidates:
        if (candidate / "config.json").is_file() and any(
            (candidate / name).is_file()
            for name in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin")
        ):
            return candidate
    raise FileNotFoundError(f"No complete V8/V8-R adapter component at {path}")


def initialize_from_legacy(path: str | Path, **overrides):
    """Transfer spatial/temporal layers; initialize DCN/fusion afresh.

    The old V8-R model was trained with a V5-aligned base, so its fusion and
    offset network must not be silently transferred into a spatial-base model.
    A detailed key report is returned for checkpoint provenance.
    """
    component = resolve_component(path)
    config = json.loads((component / "config.json").read_text(encoding="utf-8"))
    kwargs = {key: config[key] for key in CONFIG_KEYS if key in config}
    kwargs.update(overrides)
    model = StandaloneV8RAdapter(**kwargs)
    weights_file = component / "diffusion_pytorch_model.safetensors"
    if weights_file.is_file():
        weights = load_file(str(weights_file), device="cpu")
    else:
        try:
            weights = torch.load(component / "diffusion_pytorch_model.bin", map_location="cpu", weights_only=True)
        except TypeError:
            weights = torch.load(component / "diffusion_pytorch_model.bin", map_location="cpu")
    destination = model.state_dict()
    copied = {}
    for key, tensor in destination.items():
        if not key.startswith(("stc_adapter.spatial_encoder.", "stc_adapter.temporal_blocks.",
                               "stc_adapter.output_norm.", "stc_adapter.zero_conv.")):
            continue
        if key not in weights or weights[key].shape != tensor.shape:
            raise ValueError(f"Required migration tensor missing or incompatible: {key}")
        copied[key] = weights[key]
    missing, unexpected = model.load_state_dict(copied, strict=False)
    if unexpected or not copied:
        raise RuntimeError(f"Migration failure: unexpected={unexpected}, copied={len(copied)}")
    report = {
        "source": str(component),
        "copied_count": len(copied),
        "copied_keys": sorted(copied),
        "new_keys": sorted(missing),
        "discarded_legacy_flow_keys": sorted(k for k in weights if "flow" in k and k not in copied),
        "fusion_and_dcn_reset": True,
    }
    return model, report
