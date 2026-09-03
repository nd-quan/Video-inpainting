#!/usr/bin/env python
"""Train V8: frozen V7 RAFT base flow + V6-style feature DCN.

V7 is loaded as a per-process, inference-only RGB flow provider.  V8 does
not back-propagate into RAFT and does not train an additional flow loss: the
cached clean flow remains only a diagnostic reported by the inherited trainer.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v3_rgb_flow import train_rgb_stc_flow_shared_noise as trainer
from STC_encoder_v4pp_bg_feature.feature_alignment import compute_feature_alignment_loss
from STC_encoder_v5_relative_crossclip import train_v5_relative_crossclip as v5
from STC_encoder_v5_relative_crossclip.cross_clip_data import collate_cross_clip_teacher_flow
from STC_encoder_v5_relative_crossclip.relative_crossclip_stc_adapter import (
    RelativeCrossClipBGSTCAdapter,
)
from STC_encoder_v6_flow_deformable import train_v6_flow_deformable as v6
from STC_encoder_v6_flow_deformable.deformable_alignment_loss import (
    WeightedV6ExtraLossOutput,
    compute_deformable_alignment_loss,
)
from STC_encoder_v8_raft_deformable.raft_flow_provider import (
    FrozenV7RAFTFlowProvider,
    resolve_raft_student_component,
)
from STC_encoder_v8_raft_deformable.raft_guided_deformable_stc_adapter import (
    RAFTGuidedDeformableBGSTCAdapter,
    augment_brushnet_condition_v8,
)


EXPERIMENT_NAME = "rgb_stc_v8_frozen_v7_raft_deformable"
DEFAULT_OUTPUT = (
    trainer.REPO_ROOT
    / "experiments"
    / "train_stc_v8_raft_deform_only_T16_S12_sharedNoise_0.95"
)

_base_checkpoint_metadata = trainer.checkpoint_metadata
_base_resume_contract = trainer._resume_contract
_FLOW_PROVIDER_BY_DEVICE: Dict[str, FrozenV7RAFTFlowProvider] = {}


def resolve_complete_component(path_value: str, expected: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if path.name in {"best.json", "latest.json"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate = Path(payload["checkpoint"])
        path = candidate if candidate.is_absolute() else path.parent / candidate
        path = path.resolve()
    nested = path / expected
    if (nested / "config.json").is_file():
        path = nested
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"No {expected} config below {path}")
    if not any(
        (path / name).is_file()
        for name in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin")
    ):
        raise FileNotFoundError(f"No complete {expected} weights below {path}")
    return path


def _add_variant_arguments(parser) -> None:
    parser.add_argument(
        "--init_v5_model",
        default=None,
        help="Full V5 component/checkpoint/latest.json for Stage-A V8.",
    )
    parser.add_argument(
        "--init_v8_model",
        default=None,
        help="Full V8 component/checkpoint for Stage-B joint fine-tuning.",
    )
    parser.add_argument("--training_stage", choices=("deform_only", "joint"), default="deform_only")
    parser.add_argument("--relative_position_max_distance", type=int, default=32)
    parser.add_argument("--cross_clip_memory_frames", type=int, default=4)
    parser.add_argument("--detach_cross_clip_memory", action="store_true", default=True)
    parser.add_argument("--deform_hidden_channels", type=int, default=128)
    parser.add_argument("--deform_kernel_size", type=int, default=3)
    parser.add_argument("--deform_groups", type=int, default=4)
    parser.add_argument("--deform_residual_max_displacement", type=float, default=2.0)
    parser.add_argument("--deform_alignment_loss_weight", type=float, default=0.05)
    parser.add_argument("--deform_alignment_charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--deform_alignment_warmup_steps", type=int, default=500)
    parser.add_argument("--deform_offset_loss_weight", type=float, default=1e-3)
    parser.add_argument(
        "--raft_student_path",
        required=True,
        help="V7 best/latest.json, V7 checkpoint root, or its raft_student component.",
    )
    parser.add_argument("--raft_pair_batch_size", type=int, default=1)
    parser.add_argument(
        "--raft_mixed_precision",
        dest="raft_mixed_precision",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--raft_no_mixed_precision",
        dest="raft_mixed_precision",
        action="store_false",
    )


def _model_config_matches(model, args) -> None:
    expected = {
        "relative_position_max_distance": args.relative_position_max_distance,
        "cross_clip_memory_frames": args.cross_clip_memory_frames,
        "deform_hidden_channels": args.deform_hidden_channels,
        "deform_kernel_size": args.deform_kernel_size,
        "deform_groups": args.deform_groups,
        "deform_residual_max_displacement": args.deform_residual_max_displacement,
    }
    for key, value in expected.items():
        saved = getattr(model.config, key)
        matches = (
            math.isclose(float(saved), float(value), rel_tol=0.0, abs_tol=1e-12)
            if isinstance(value, float)
            else int(saved) == int(value)
        )
        if not matches:
            raise ValueError(f"V8 config mismatch {key}: model={saved}, args={value}")


def _model_factory(args):
    if args.training_stage == "deform_only":
        source = resolve_complete_component(args.init_v5_model, "stc_v5_model")
        model = RAFTGuidedDeformableBGSTCAdapter.from_v5_pretrained(
            source,
            deform_hidden_channels=args.deform_hidden_channels,
            deform_kernel_size=args.deform_kernel_size,
            deform_groups=args.deform_groups,
            deform_residual_max_displacement=args.deform_residual_max_displacement,
            detach_deform_reliability=True,
        )
    else:
        source = resolve_complete_component(args.init_v8_model, "stc_v8_model")
        model = RAFTGuidedDeformableBGSTCAdapter.from_pretrained(str(source))
    _model_config_matches(model, args)
    return model


def _configure_trainable_parameters(*, model, args) -> None:
    model.requires_grad_(False)
    model.deformable_alignment.requires_grad_(True)
    model.deformable_fusion.requires_grad_(True)
    if args.training_stage == "joint":
        # RAFT stays outside the model and frozen.  Preserve the spatial and
        # legacy flow encoders; adapt only temporal/condition correction paths.
        model.alignment_fusion.requires_grad_(True)
        model.stc_adapter.temporal_blocks.requires_grad_(True)
        model.stc_adapter.output_norm.requires_grad_(True)
        model.stc_adapter.zero_conv.requires_grad_(True)


def _trainable_component_names(args):
    names = ["deformable_alignment", "deformable_fusion"]
    if args.training_stage == "joint":
        names.extend((
            "alignment_fusion",
            "stc_adapter.temporal_blocks",
            "stc_adapter.output_norm",
            "stc_adapter.zero_conv",
        ))
    return names


def _provider(args, device: torch.device) -> FrozenV7RAFTFlowProvider:
    component = resolve_raft_student_component(args.raft_student_path)
    key = "|".join((str(device), str(component), str(args.raft_pair_batch_size), str(args.raft_mixed_precision)))
    provider = _FLOW_PROVIDER_BY_DEVICE.get(key)
    if provider is None:
        provider = FrozenV7RAFTFlowProvider(
            component,
            device=device,
            pair_batch_size=args.raft_pair_batch_size,
            mixed_precision=args.raft_mixed_precision,
        )
        _FLOW_PROVIDER_BY_DEVICE[key] = provider
    return provider


def _augment_extra_kwargs(*, batch, device, num_clips, num_frames, resolution):
    values = v5._augment_extra_kwargs(
        batch=batch,
        device=device,
        num_clips=num_clips,
        num_frames=num_frames,
        resolution=resolution,
    )
    # The provider is process-local, frozen, and never put in DDP/optimizer.
    values["raft_flow_provider"] = _provider(_ACTIVE_ARGS, device)
    return values


def _build_v8_extra_train_loss(*, stc_output, batch, bg_mask_sequence, args, global_step: int):
    """Train DCN candidates on V7-valid geometry, never through frozen RAFT."""
    del batch, bg_mask_sequence
    output = compute_deformable_alignment_loss(
        spatial_features=stc_output.spatial_features,
        deformed_previous=stc_output.deformed_previous_features,
        deformed_next=stc_output.deformed_next_features,
        residual_offset_backward=stc_output.residual_offset_backward,
        residual_offset_forward=stc_output.residual_offset_forward,
        reliability_backward=stc_output.deform_reliability_backward,
        reliability_forward=stc_output.deform_reliability_forward,
        # These tensors are V7 flow already resized to the feature grid.  The
        # V6 loss uses them solely for finite/in-bounds DCN supervision support.
        teacher_forward=stc_output.predicted_flow_forward,
        teacher_backward=stc_output.predicted_flow_backward,
        valid_forward=None,
        valid_backward=None,
        charbonnier_eps=args.deform_alignment_charbonnier_eps,
    )
    ramp = (
        min(1.0, float(global_step + 1) / float(args.deform_alignment_warmup_steps))
        if args.deform_alignment_warmup_steps > 0
        else 1.0
    )
    deform_weight = float(args.deform_alignment_loss_weight) * ramp
    offset_weight = float(args.deform_offset_loss_weight)
    weighted_deform = deform_weight * output.loss
    weighted_offset = offset_weight * output.loss_offset
    total = weighted_deform + weighted_offset
    return WeightedV6ExtraLossOutput(
        loss=total,
        metrics={
            "train/deform_alignment_effective_weight": total.new_tensor(deform_weight),
            "train/deform_offset_effective_weight": total.new_tensor(offset_weight),
            "train/loss_deform_alignment": output.loss,
            "train/loss_deform_alignment_weighted": weighted_deform,
            "train/loss_deform_backward": output.loss_backward,
            "train/loss_deform_forward": output.loss_forward,
            "train/loss_deform_offset": output.loss_offset,
            "train/loss_deform_offset_weighted": weighted_offset,
            "train/loss_deform_offset_backward": output.loss_offset_backward,
            "train/loss_deform_offset_forward": output.loss_offset_forward,
            "train/deform_valid_backward": output.valid_backward_ratio,
            "train/deform_valid_forward": output.valid_forward_ratio,
            "train/deform_reliability_backward": output.reliability_backward_mean,
            "train/deform_reliability_forward": output.reliability_forward_mean,
        },
    )


def _extra_train_metrics(*, stc_output, batch):
    values = v6._extra_train_metrics(stc_output=stc_output, batch=batch)
    raft_forward = stc_output.predicted_flow_forward.detach().float()
    raft_backward = stc_output.predicted_flow_backward.detach().float()
    legacy_forward = stc_output.legacy_flow_forward.detach().float()
    legacy_backward = stc_output.legacy_flow_backward.detach().float()
    values.update({
        "train/v7_raft_flow_forward_magnitude_feature_px": raft_forward.square().sum(dim=2).sqrt().mean(),
        "train/v7_raft_flow_backward_magnitude_feature_px": raft_backward.square().sum(dim=2).sqrt().mean(),
        "train/v7_raft_flow_forward_magnitude_rgb_px": stc_output.raft_flow_forward_rgb.detach().float().square().sum(dim=2).sqrt().mean(),
        "train/v7_raft_flow_backward_magnitude_rgb_px": stc_output.raft_flow_backward_rgb.detach().float().square().sum(dim=2).sqrt().mean(),
        "train/legacy_flow_forward_magnitude_feature_px": legacy_forward.square().sum(dim=2).sqrt().mean(),
        "train/legacy_flow_backward_magnitude_feature_px": legacy_backward.square().sum(dim=2).sqrt().mean(),
        "train/v7_raft_confidence_backward_mean": stc_output.alignment_confidence.detach().float().mean(),
    })
    return values


def _resolved_initialization(args) -> str:
    if args.training_stage == "deform_only":
        return str(resolve_complete_component(args.init_v5_model, "stc_v5_model"))
    return str(resolve_complete_component(args.init_v8_model, "stc_v8_model"))


def _checkpoint_metadata(args, accelerator, global_step, epoch, next_batch_index):
    metadata = _base_checkpoint_metadata(args, accelerator, global_step, epoch, next_batch_index)
    metadata.update({
        "experiment": EXPERIMENT_NAME,
        "model_variant": "v8_frozen_v7_raft_guided_modulated_deformable_alignment",
        "loss": "L_diff + lambda_feature*L_feature(diagnostic when frozen) + lambda_deform*L_deform + lambda_offset*L_offset",
        "inference_component": "stc_v8_model",
        "initialization_component": _resolved_initialization(args),
        "training_stage": args.training_stage,
        "trainable_components": _trainable_component_names(args),
        "legacy_flow_head_frozen": True,
        "spatial_encoder_frozen": True,
        "v7_raft_flow_frozen": True,
        "v7_raft_student_component": str(resolve_raft_student_component(args.raft_student_path)),
        "v7_raft_pair_batch_size": args.raft_pair_batch_size,
        "v7_raft_mixed_precision": args.raft_mixed_precision,
        "v7_flow_input": "degraded RGB [-1,1] only",
        "v7_flow_units_before_resize": "RGB [dx,dy] pixels",
        "v7_flow_units_for_dcn": "STC feature-grid [dx,dy] pixels via resize_flow_sequence",
        "deform_base_stream": "frozen legacy V5 light-flow alignment",
        "deform_offset_prior": "frozen V7 RAFT student flow",
        "deform_alignment_loss_weight": args.deform_alignment_loss_weight,
        "deform_alignment_charbonnier_eps": args.deform_alignment_charbonnier_eps,
        "deform_alignment_warmup_steps": args.deform_alignment_warmup_steps,
        "deform_offset_loss_weight": args.deform_offset_loss_weight,
        "deformation_scope": "first_order_bidirectional_raw_adjacent_features",
        "deformation_region": "BG_to_BG_reliable_support",
        "cross_clip_memory_deformed": False,
        "teacher_flow_required_at_inference": False,
        "v7_raft_required_at_inference": True,
        "legacy_stc_adapter_export_complete": False,
    })
    return metadata


def _resume_contract(args):
    contract = _base_resume_contract(args)
    contract.update({
        "experiment": EXPERIMENT_NAME,
        "initialization_component": _resolved_initialization(args),
        "training_stage": args.training_stage,
        "relative_position_max_distance": args.relative_position_max_distance,
        "cross_clip_memory_frames": args.cross_clip_memory_frames,
        "detach_cross_clip_memory": args.detach_cross_clip_memory,
        "deform_hidden_channels": args.deform_hidden_channels,
        "deform_kernel_size": args.deform_kernel_size,
        "deform_groups": args.deform_groups,
        "deform_residual_max_displacement": args.deform_residual_max_displacement,
        "deform_alignment_loss_weight": args.deform_alignment_loss_weight,
        "deform_alignment_charbonnier_eps": args.deform_alignment_charbonnier_eps,
        "deform_alignment_warmup_steps": args.deform_alignment_warmup_steps,
        "deform_offset_loss_weight": args.deform_offset_loss_weight,
        "v7_raft_student_component": str(resolve_raft_student_component(args.raft_student_path)),
        "v7_raft_pair_batch_size": args.raft_pair_batch_size,
        "v7_raft_mixed_precision": args.raft_mixed_precision,
        "trainable_components": _trainable_component_names(args),
    })
    return contract


def _install_variant() -> None:
    trainer.RGBSTCFlowAdapter = RAFTGuidedDeformableBGSTCAdapter
    trainer.augment_brushnet_condition = augment_brushnet_condition_v8
    trainer.FEATURE_ALIGNMENT_LOSS_FN = compute_feature_alignment_loss
    trainer.EXTRA_TRAIN_LOSS_FN = _build_v8_extra_train_loss
    trainer.CONFIGURE_TRAINABLE_PARAMETERS_FN = _configure_trainable_parameters
    trainer.ADD_VARIANT_ARGUMENTS_FN = _add_variant_arguments
    trainer.MODEL_FACTORY_FN = _model_factory
    trainer.AUGMENT_EXTRA_KWARGS_FN = _augment_extra_kwargs
    trainer.EXTRA_TRAIN_METRICS_FN = _extra_train_metrics
    trainer.POST_DATASET_VALIDATION_FN = v5._post_dataset_validation
    trainer.TRAIN_DATALOADER_DROP_LAST = True
    trainer.FULL_MODEL_COMPONENT_NAME = "stc_v8_model"
    trainer.SAVE_LEGACY_STC_ADAPTER = False
    trainer.DDP_FIND_UNUSED_PARAMETERS = True
    trainer.make_dataset = v5._make_dataset
    trainer.collate_teacher_flow_clips = collate_cross_clip_teacher_flow
    trainer.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    trainer.EXPERIMENT_NAME = EXPERIMENT_NAME
    trainer.FLOW_INFERENCE_DEPENDENCY = True
    trainer.INFERENCE_COMPONENT = "stc_v8_model"
    trainer.TRAINING_LOG_TITLE = "RGB-STC V8: frozen V7 RAFT-flow-guided deformable alignment"
    trainer.checkpoint_metadata = _checkpoint_metadata
    trainer._resume_contract = _resume_contract


def _rewrite_zero_flow_loss(argv: Iterable[str]):
    """Let V8 expose a true zero legacy-flow weight without editing V3's parser."""
    values = list(argv)
    rewritten = list(values)
    explicit_value: Optional[float] = None
    for index, token in enumerate(values):
        if token == "--flow_loss_weight" and index + 1 < len(values):
            explicit_value = float(values[index + 1])
            rewritten[index + 1] = "1e-12" if explicit_value == 0.0 else values[index + 1]
        elif token.startswith("--flow_loss_weight="):
            explicit_value = float(token.split("=", 1)[1])
            rewritten[index] = "--flow_loss_weight=1e-12" if explicit_value == 0.0 else token
    if explicit_value is not None and explicit_value != 0.0:
        raise ValueError("V8 freezes V7 RAFT; --flow_loss_weight must be exactly 0")
    return rewritten


def parse_args(input_args=None):
    global _ACTIVE_ARGS
    _install_variant()
    argv = list(sys.argv[1:] if input_args is None else input_args)
    args = trainer.parse_args(_rewrite_zero_flow_loss(argv))
    # The base trainer requires a strictly positive legacy-flow coefficient.
    # We used an infinitesimal parser sentinel above, then restore V8's actual
    # objective before any preflight, metadata, or optimization is executed.
    args.flow_loss_weight = 0.0
    if args.init_stc_adapter:
        raise ValueError("V8 rejects --init_stc_adapter")
    if args.mixed_precision == "bf16":
        raise ValueError("torchvision deform_conv2d does not support BF16 here")
    for name in ("init_v5_model", "init_v8_model"):
        value = getattr(args, name)
        if value:
            setattr(args, name, str(Path(value).expanduser().resolve()))
    args.raft_student_path = str(resolve_raft_student_component(args.raft_student_path))
    if args.training_stage == "deform_only":
        if not args.init_v5_model or args.init_v8_model:
            raise ValueError("deform_only requires --init_v5_model and forbids --init_v8_model")
    elif not args.init_v8_model or args.init_v5_model:
        raise ValueError("joint requires --init_v8_model and forbids --init_v5_model")
    if args.relative_position_max_distance < args.clip_length - 1:
        raise ValueError("relative_position_max_distance must cover T-1")
    overlap = args.clip_length - args.clip_stride
    if overlap <= 0 or not 1 <= args.cross_clip_memory_frames <= overlap:
        raise ValueError("cross-clip memory must fit the positive T-S overlap")
    if args.deform_hidden_channels < 1:
        raise ValueError("deform_hidden_channels must be positive")
    if args.deform_kernel_size < 1 or args.deform_kernel_size % 2 == 0:
        raise ValueError("deform_kernel_size must be a positive odd integer")
    if args.deform_groups < 1 or args.stc_hidden_channels % args.deform_groups:
        raise ValueError("deform_groups must divide stc_hidden_channels")
    if args.raft_pair_batch_size < 1:
        raise ValueError("raft_pair_batch_size must be positive")
    for name in ("deform_residual_max_displacement", "deform_alignment_charbonnier_eps"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("deform_alignment_loss_weight", "deform_offset_loss_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if args.deform_alignment_loss_weight == 0.0:
        raise ValueError("V8 requires a positive deform alignment loss")
    if args.deform_alignment_warmup_steps < 0:
        raise ValueError("deform_alignment_warmup_steps must be non-negative")
    _ACTIVE_ARGS = args
    return args


def run_preflight(args) -> None:
    _install_variant()
    trainer.run_preflight(args)
    model = _model_factory(args).eval()
    _configure_trainable_parameters(model=model, args=args)
    final_head = model.deformable_alignment.offset_mask_head[-1]
    final_fusion = model.deformable_fusion.to_residual_and_gate
    if args.training_stage == "deform_only":
        source = RelativeCrossClipBGSTCAdapter.from_pretrained(
            resolve_complete_component(args.init_v5_model, "stc_v5_model")
        ).eval()
        generator = torch.Generator().manual_seed(97)
        rgb = torch.randn(1, 4, 3, 32, 32, generator=generator)
        bg = torch.ones(1, 4, 1, 32, 32)
        ids = torch.arange(4)[None]
        # V7's RGB flow normally arrives at 512 pixels.  Any valid external
        # flow must retain exact V5 output while the V8 DCN fusion is zero.
        raft_forward = torch.randn(1, 3, 2, 32, 32, generator=generator)
        raft_backward = torch.randn(1, 3, 2, 32, 32, generator=generator)
        with torch.no_grad():
            source_output = source(rgb, bg, output_size=(4, 4), frame_ids=ids)
            v8_output = model(
                rgb, bg, output_size=(4, 4), frame_ids=ids,
                raft_flow_forward_rgb=raft_forward,
                raft_flow_backward_rgb=raft_backward,
            )
        if not torch.equal(source_output.features, v8_output.features):
            raise RuntimeError("Fresh V8 is not exact V5 identity")
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    report = {
        "v8_preflight": "ok",
        "training_stage": args.training_stage,
        "initialization_component": _resolved_initialization(args),
        "v7_raft_student_component": args.raft_student_path,
        "v7_raft_frozen": True,
        "v7_flow_input": "degraded RGB only",
        "v7_flow_resize": "resize_flow_sequence RGB->STC feature grid",
        "legacy_v5_base_stream_preserved": True,
        "fresh_v8_exact_v5_identity": args.training_stage == "deform_only",
        "offset_mask_final_zero": bool(torch.count_nonzero(final_head.weight) == 0 and torch.count_nonzero(final_head.bias) == 0) if args.training_stage == "deform_only" else None,
        "fusion_final_zero": bool(torch.count_nonzero(final_fusion.weight) == 0 and torch.count_nonzero(final_fusion.bias) == 0) if args.training_stage == "deform_only" else None,
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "trainable_parameter_prefix_sample": trainable[:12],
        "flow_loss_weight": args.flow_loss_weight,
        "mixed_precision": args.mixed_precision,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def main(args) -> None:
    _install_variant()
    trainer.main(args)


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.preflight_only:
        run_preflight(parsed)
    else:
        main(parsed)
