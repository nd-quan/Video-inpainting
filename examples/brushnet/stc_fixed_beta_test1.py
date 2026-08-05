"""Utilities for the fixed-beta STC noise-shaping Test-1 experiment.

Test 1 reuses the Stage-1 STC flow predictor to shape only the initial
Gaussian latent.  It does not train beta and it does not install a Stage-3
condition adapter.  All diffusion, BrushNet and IP-Adapter parameters remain
frozen during evaluation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch

from diffusers.models.stc_noise_shaper import STCConditionedNoiseShaper


def resolve_json_checkpoint(path: Path) -> Tuple[Path, Dict]:
    """Resolve a checkpoint directory or a JSON checkpoint pointer."""

    path = Path(path).expanduser().resolve()
    metadata: Dict = {}
    if path.suffix.lower() == ".json":
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint pointer not found: {path}")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = metadata.get("checkpoint")
        if not checkpoint:
            raise ValueError(f"Checkpoint pointer has no 'checkpoint': {path}")
        path = Path(checkpoint).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {path}")
    return path, metadata


def resolve_stage1_component(path: Path) -> Tuple[Path, Path, Dict]:
    """Return ``(checkpoint, flow_predictor, pointer_metadata)`` for Stage 1."""

    checkpoint, metadata = resolve_json_checkpoint(path)
    component = checkpoint / "flow_predictor"
    if not component.is_dir() or not (component / "config.json").is_file():
        raise FileNotFoundError(
            f"Stage-1 checkpoint has no flow_predictor component: {checkpoint}"
        )
    return checkpoint, component, metadata


def resolve_v8_components(checkpoint: Path) -> Dict[str, Path]:
    """Resolve the three deployable components of a V8 checkpoint.

    V8 keeps the diffusion U-Net frozen, so there is deliberately no U-Net
    component here.  The caller must load the base Stable-Diffusion U-Net.
    """

    checkpoint = Path(checkpoint).expanduser().resolve()
    components = {
        "checkpoint": checkpoint,
        "brushnet": checkpoint / "brushnet",
        "ip_adapter": checkpoint / "ipadapter" / "model.safetensors",
        "fusion": checkpoint / "ipadapter" / "fusion_module.safetensors",
    }
    if not (components["brushnet"] / "config.json").is_file():
        raise FileNotFoundError(
            f"V8 BrushNet component not found: {components['brushnet']}"
        )
    for key in ("ip_adapter", "fusion"):
        if not components[key].is_file():
            raise FileNotFoundError(f"V8 {key} weights not found: {components[key]}")
    return components


def load_fixed_beta_stage1(
    checkpoint: Path,
    fixed_beta: float,
    torch_dtype: Optional[torch.dtype] = None,
) -> Tuple[STCConditionedNoiseShaper, Dict]:
    """Load Stage 1 and rebuild it as an immutable fixed-beta noise shaper.

    Rebuilding from config makes beta overrides explicit without modifying the
    saved Stage-1 config or weights.  Strict state transfer guarantees that the
    complete trained flow predictor was recovered.
    """

    beta = float(fixed_beta)
    if not 0.0 <= beta <= 1.0:
        raise ValueError("fixed_beta must be in [0, 1]")
    resolved, component, pointer_metadata = resolve_stage1_component(checkpoint)
    source = STCConditionedNoiseShaper.from_pretrained(component)
    if str(source.config.flow_prediction_mode).lower() != "full":
        raise ValueError("Fixed-beta Test 1 requires Stage-1 full-flow mode")

    model = STCConditionedNoiseShaper.from_config(
        source.config,
        beta_mode="fixed",
        fixed_beta=beta,
        warp_region="all",
    )
    model.load_state_dict(source.state_dict(), strict=True)
    del source
    model.requires_grad_(False).eval()
    if torch_dtype is not None:
        model.to(dtype=torch_dtype)
    if getattr(model, "beta_head", None) is not None:
        raise RuntimeError("Fixed-beta Test 1 must not contain a beta_head")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Stage-1 noise shaper was not completely frozen")

    metadata = {
        "stage1_checkpoint": str(resolved),
        "stage1_component": str(component),
        "stage1_step": pointer_metadata.get("step"),
        "stage1_valid_epe": pointer_metadata.get("valid_epe"),
        "fixed_beta": beta,
        "flow_prediction_mode": str(model.config.flow_prediction_mode),
        "warp_region": str(model.config.warp_region),
        "full_flow_max_displacement": list(
            model.config.full_flow_max_displacement
        ),
    }
    return model, metadata


def deterministic_clip_seed(
    seed: int, class_name: str, sequence: str, first_frame: int
) -> int:
    """Derive a stable seed shared by all beta ablations for one clip."""

    identity = f"{int(seed)}|{class_name}|{sequence}|{int(first_frame)}"
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little") % (2**31 - 1)


def inspect_manifest_split(manifest_path: Path, split: str) -> Dict:
    """Summarize sequence coverage before a long evaluation starts."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if split not in manifest.get("splits", []):
        raise ValueError(f"Split {split!r} is not declared in {manifest_path}")
    empty, available, frame_count = [], [], 0
    for sequence in manifest.get("sequences", []):
        entry = sequence.get("splits", {}).get(split, {})
        count = int(entry.get("frame_count", 0))
        item = f"{sequence['class']}/{sequence['name']}"
        if count > 0:
            available.append(item)
            frame_count += count
        else:
            empty.append(item)
    return {
        "split": split,
        "available_sequences": available,
        "empty_sequences": empty,
        "frame_count": frame_count,
    }


def validate_fixed_test1_config(config: Mapping) -> Dict:
    """Validate all local artifacts without loading the large diffusion model."""

    model = config["model"]
    data = config["data"]
    base_model = Path(model["base_model"]).expanduser().resolve()
    for component in ("unet", "vae", "text_encoder", "tokenizer", "scheduler"):
        if not (base_model / component).is_dir():
            raise FileNotFoundError(
                f"Base-model component {component!r} not found under {base_model}"
            )
    stage1, stage1_component, pointer = resolve_stage1_component(
        Path(model["stage1_checkpoint"])
    )
    v8 = resolve_v8_components(Path(model["v8_checkpoint"]))
    dataset_root = Path(data["dataset_root"]).expanduser().resolve()
    manifest = Path(data.get("manifest", dataset_root / "manifest.json"))
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    coverage = inspect_manifest_split(manifest, str(data.get("split", "test")))
    if bool(data.get("require_all_sequences", False)) and coverage["empty_sequences"]:
        raise ValueError(
            "The requested split has empty sequences: "
            + ", ".join(coverage["empty_sequences"])
        )
    beta = float(config.get("noise_shaping", {}).get("fixed_beta", 0.5))
    if not 0.0 <= beta <= 1.0:
        raise ValueError("noise_shaping.fixed_beta must be in [0, 1]")
    noise = config.get("noise_shaping", {})
    if str(noise.get("warp_region", "all")).lower() != "all":
        raise ValueError("Fixed-beta Test 1 requires noise_shaping.warp_region='all'")
    strength = float(noise.get("strength", 1.0))
    if not 0.0 <= strength <= 1.0:
        raise ValueError("noise_shaping.strength must be in [0, 1]")
    ip = config.get("ip_adapter", {})
    if not bool(ip.get("include_base_image", False)):
        raise ValueError("V8 Test 1 requires ip_adapter.include_base_image=true")
    if not bool(ip.get("v8_mask_order", False)):
        raise ValueError("V8 Test 1 requires ip_adapter.v8_mask_order=true")
    if int(ip.get("num_tokens", 4)) != 4:
        raise ValueError("The V8 checkpoint was trained with exactly 4 IP tokens")

    inference = config.get("inference", {})
    if bool(inference.get("composite_input_roi", True)):
        blending = str(
            inference.get("roi_blending", "gaussian_soft")
        ).lower()
        if blending not in {"hard", "gaussian_soft"}:
            raise ValueError(
                "inference.roi_blending must be 'hard' or 'gaussian_soft'"
            )
        if blending == "gaussian_soft":
            kernel_size = int(inference.get("gaussian_kernel_size", 21))
            if kernel_size < 1 or kernel_size % 2 == 0:
                raise ValueError(
                    "inference.gaussian_kernel_size must be a positive odd integer"
                )
            if float(inference.get("gaussian_sigma", 0.0)) < 0:
                raise ValueError(
                    "inference.gaussian_sigma must be non-negative"
                )
    codec = str(inference.get("video_codec", "mp4v"))
    if bool(inference.get("save_video", True)) and len(codec) != 4:
        raise ValueError("inference.video_codec must contain four characters")

    stage1_config = json.loads(
        (stage1_component / "config.json").read_text(encoding="utf-8")
    )
    if str(stage1_config.get("flow_prediction_mode", "")).lower() != "full":
        raise ValueError("Fixed-beta Test 1 requires a full-flow Stage-1 checkpoint")
    if str(stage1_config.get("beta_mode", "")).lower() != "fixed":
        raise ValueError("Fixed-beta Test 1 requires a fixed-beta Stage-1 checkpoint")
    return {
        "base_model": str(base_model),
        "stage1_checkpoint": str(stage1),
        "stage1_component": str(stage1_component),
        "stage1_step": pointer.get("step"),
        "stage1_valid_epe": pointer.get("valid_epe"),
        "v8_components": {key: str(value) for key, value in v8.items()},
        "coverage": coverage,
        "fixed_beta": beta,
    }
