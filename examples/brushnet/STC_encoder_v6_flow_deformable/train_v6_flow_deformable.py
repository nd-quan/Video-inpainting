#!/usr/bin/env python
"""Train V6: V5 plus flow-guided modulated deformable alignment."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v3_rgb_flow import train_rgb_stc_flow_shared_noise as trainer
from STC_encoder_v4pp_bg_feature.feature_alignment import (
    compute_feature_alignment_loss,
)
from STC_encoder_v5_relative_crossclip import train_v5_relative_crossclip as v5
from STC_encoder_v5_relative_crossclip.cross_clip_data import (
    collate_cross_clip_teacher_flow,
)
from STC_encoder_v5_relative_crossclip.relative_crossclip_stc_adapter import (
    RelativeCrossClipBGSTCAdapter,
)
from STC_encoder_v6_flow_deformable.deformable_alignment_loss import (
    build_v6_extra_train_loss,
)
from STC_encoder_v6_flow_deformable.flow_guided_deformable_stc_adapter import (
    FlowGuidedDeformableBGSTCAdapter,
    augment_brushnet_condition_v6,
)


EXPERIMENT_NAME = "rgb_stc_v6_flow_guided_modulated_deformable"
DEFAULT_OUTPUT = (
    trainer.REPO_ROOT
    / "experiments"
    / "train_stc_v6_deform_only_T16_S12_sharedNoise_0.95"
)

_base_checkpoint_metadata = trainer.checkpoint_metadata
_base_resume_contract = trainer._resume_contract


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
        for name in (
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.bin",
        )
    ):
        raise FileNotFoundError(f"No complete {expected} weights below {path}")
    return path


def _add_variant_arguments(parser) -> None:
    parser.add_argument(
        "--init_v5_model",
        default=None,
        help="Full V5 component/checkpoint/latest.json for deform_only Stage A.",
    )
    parser.add_argument(
        "--init_v6_model",
        default=None,
        help="Full Stage-A V6 component/checkpoint for joint Stage B.",
    )
    parser.add_argument(
        "--training_stage",
        choices=("deform_only", "joint"),
        default="deform_only",
    )
    parser.add_argument("--relative_position_max_distance", type=int, default=32)
    parser.add_argument("--cross_clip_memory_frames", type=int, default=4)
    parser.add_argument(
        "--detach_cross_clip_memory",
        action="store_true",
        default=True,
    )
    parser.add_argument("--deform_hidden_channels", type=int, default=128)
    parser.add_argument("--deform_kernel_size", type=int, default=3)
    parser.add_argument("--deform_groups", type=int, default=4)
    parser.add_argument(
        "--deform_residual_max_displacement", type=float, default=2.0
    )
    parser.add_argument(
        "--deform_alignment_loss_weight", type=float, default=0.05
    )
    parser.add_argument(
        "--deform_alignment_charbonnier_eps", type=float, default=1e-3
    )
    parser.add_argument(
        "--deform_alignment_warmup_steps", type=int, default=500
    )
    parser.add_argument("--deform_offset_loss_weight", type=float, default=1e-3)


def _model_factory(args):
    if args.training_stage == "deform_only":
        source = resolve_complete_component(args.init_v5_model, "stc_v5_model")
        model = FlowGuidedDeformableBGSTCAdapter.from_v5_pretrained(
            source,
            deform_hidden_channels=args.deform_hidden_channels,
            deform_kernel_size=args.deform_kernel_size,
            deform_groups=args.deform_groups,
            deform_residual_max_displacement=(
                args.deform_residual_max_displacement
            ),
            detach_deform_reliability=True,
        )
    else:
        source = resolve_complete_component(args.init_v6_model, "stc_v6_model")
        model = FlowGuidedDeformableBGSTCAdapter.from_pretrained(source)
    expected = {
        "relative_position_max_distance": args.relative_position_max_distance,
        "cross_clip_memory_frames": args.cross_clip_memory_frames,
        "deform_hidden_channels": args.deform_hidden_channels,
        "deform_kernel_size": args.deform_kernel_size,
        "deform_groups": args.deform_groups,
        "deform_residual_max_displacement": (
            args.deform_residual_max_displacement
        ),
    }
    for key, value in expected.items():
        saved = getattr(model.config, key)
        if isinstance(value, float):
            matches = math.isclose(float(saved), value, rel_tol=0.0, abs_tol=1e-12)
        else:
            matches = int(saved) == int(value)
        if not matches:
            raise ValueError(f"V6 config mismatch {key}: model={saved}, args={value}")
    return model


def _configure_trainable_parameters(*, model, args) -> None:
    model.requires_grad_(False)
    model.deformable_alignment.requires_grad_(True)
    model.deformable_fusion.requires_grad_(True)
    if args.training_stage == "joint":
        # Preserve the repaired motion prior and spatial representation.  Joint
        # Stage B adapts the inherited temporal/output path plus both fusions.
        model.alignment_fusion.requires_grad_(True)
        model.stc_adapter.temporal_blocks.requires_grad_(True)
        model.stc_adapter.output_norm.requires_grad_(True)
        model.stc_adapter.zero_conv.requires_grad_(True)


def _trainable_component_names(args):
    names = ["deformable_alignment", "deformable_fusion"]
    if args.training_stage == "joint":
        names.extend(
            (
                "alignment_fusion",
                "stc_adapter.temporal_blocks",
                "stc_adapter.output_norm",
                "stc_adapter.zero_conv",
            )
        )
    return names


def _extra_train_metrics(*, stc_output, batch):
    values = v5._extra_train_metrics(stc_output=stc_output, batch=batch)
    offsets = torch.cat(
        (
            stc_output.residual_offset_backward.detach().float().flatten(),
            stc_output.residual_offset_forward.detach().float().flatten(),
        )
    ).abs()
    masks = torch.cat(
        (
            stc_output.modulation_mask_backward.detach().float().flatten(),
            stc_output.modulation_mask_forward.detach().float().flatten(),
        )
    )
    if offsets.numel():
        offset_mean = offsets.mean()
        offset_p95 = torch.quantile(offsets, 0.95)
        offset_max = offsets.max()
    else:
        offset_mean = offset_p95 = offset_max = offsets.new_zeros(())
    if masks.numel():
        mask_mean = masks.mean()
        mask_std = masks.std(unbiased=False)
        mask_saturation = ((masks < 0.05) | (masks > 0.95)).float().mean()
    else:
        mask_mean = mask_std = mask_saturation = masks.new_zeros(())
    values.update(
        {
            "train/deform_offset_abs_mean": offset_mean,
            "train/deform_offset_abs_p95": offset_p95,
            "train/deform_offset_abs_max": offset_max,
            "train/deform_mask_mean": mask_mean,
            "train/deform_mask_std": mask_std,
            "train/deform_mask_saturation": mask_saturation,
            "train/deformation_minus_base_abs_mean": (
                stc_output.deformation_minus_base_abs_mean.detach().float()
            ),
        }
    )
    return values


def _resolved_initialization(args) -> str:
    if args.training_stage == "deform_only":
        return str(resolve_complete_component(args.init_v5_model, "stc_v5_model"))
    return str(resolve_complete_component(args.init_v6_model, "stc_v6_model"))


def _checkpoint_metadata(args, accelerator, global_step, epoch, next_batch_index):
    metadata = _base_checkpoint_metadata(
        args, accelerator, global_step, epoch, next_batch_index
    )
    metadata.update(
        {
            "experiment": EXPERIMENT_NAME,
            "model_variant": "v6_flow_guided_modulated_deformable_alignment",
            "loss": (
                "L_diff + flow_loss_weight*L_flow + feature_weight*L_feature "
                "+ deform_weight*L_deform + offset_weight*L_offset"
            ),
            "inference_component": "stc_v6_model",
            "initialization_component": _resolved_initialization(args),
            "training_stage": args.training_stage,
            "trainable_components": _trainable_component_names(args),
            "flow_head_frozen": True,
            "spatial_encoder_frozen": True,
            "relative_position_max_distance": args.relative_position_max_distance,
            "cross_clip_memory_frames": args.cross_clip_memory_frames,
            "detach_cross_clip_memory": args.detach_cross_clip_memory,
            "cross_clip_pair_index_sha256": getattr(
                args, "cross_clip_pair_index_sha256", None
            ),
            "deform_hidden_channels": args.deform_hidden_channels,
            "deform_kernel_size": args.deform_kernel_size,
            "deform_groups": args.deform_groups,
            "deform_residual_max_displacement": (
                args.deform_residual_max_displacement
            ),
            "deform_alignment_loss_weight": args.deform_alignment_loss_weight,
            "deform_alignment_charbonnier_eps": (
                args.deform_alignment_charbonnier_eps
            ),
            "deform_alignment_warmup_steps": (
                args.deform_alignment_warmup_steps
            ),
            "deform_offset_loss_weight": args.deform_offset_loss_weight,
            "deformation_scope": "first_order_bidirectional_raw_adjacent_features",
            "deformation_region": "BG_to_BG_reliable_support",
            "cross_clip_memory_deformed": False,
            "teacher_flow_required_at_inference": False,
            "legacy_stc_adapter_export_complete": False,
        }
    )
    return metadata


def _resume_contract(args):
    contract = _base_resume_contract(args)
    contract.update(
        {
            "experiment": EXPERIMENT_NAME,
            "initialization_component": _resolved_initialization(args),
            "training_stage": args.training_stage,
            "relative_position_max_distance": args.relative_position_max_distance,
            "cross_clip_memory_frames": args.cross_clip_memory_frames,
            "detach_cross_clip_memory": args.detach_cross_clip_memory,
            "deform_hidden_channels": args.deform_hidden_channels,
            "deform_kernel_size": args.deform_kernel_size,
            "deform_groups": args.deform_groups,
            "deform_residual_max_displacement": (
                args.deform_residual_max_displacement
            ),
            "deform_alignment_loss_weight": args.deform_alignment_loss_weight,
            "deform_alignment_charbonnier_eps": (
                args.deform_alignment_charbonnier_eps
            ),
            "deform_alignment_warmup_steps": (
                args.deform_alignment_warmup_steps
            ),
            "deform_offset_loss_weight": args.deform_offset_loss_weight,
            "trainable_components": _trainable_component_names(args),
        }
    )
    return contract


def _install_variant() -> None:
    trainer.RGBSTCFlowAdapter = FlowGuidedDeformableBGSTCAdapter
    trainer.augment_brushnet_condition = augment_brushnet_condition_v6
    trainer.FEATURE_ALIGNMENT_LOSS_FN = compute_feature_alignment_loss
    trainer.EXTRA_TRAIN_LOSS_FN = build_v6_extra_train_loss
    trainer.CONFIGURE_TRAINABLE_PARAMETERS_FN = _configure_trainable_parameters
    trainer.ADD_VARIANT_ARGUMENTS_FN = _add_variant_arguments
    trainer.MODEL_FACTORY_FN = _model_factory
    trainer.AUGMENT_EXTRA_KWARGS_FN = v5._augment_extra_kwargs
    trainer.EXTRA_TRAIN_METRICS_FN = _extra_train_metrics
    trainer.POST_DATASET_VALIDATION_FN = v5._post_dataset_validation
    trainer.TRAIN_DATALOADER_DROP_LAST = True
    trainer.FULL_MODEL_COMPONENT_NAME = "stc_v6_model"
    trainer.SAVE_LEGACY_STC_ADAPTER = False
    trainer.DDP_FIND_UNUSED_PARAMETERS = True
    trainer.make_dataset = v5._make_dataset
    trainer.collate_teacher_flow_clips = collate_cross_clip_teacher_flow
    trainer.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    trainer.EXPERIMENT_NAME = EXPERIMENT_NAME
    trainer.FLOW_INFERENCE_DEPENDENCY = True
    trainer.INFERENCE_COMPONENT = "stc_v6_model"
    trainer.TRAINING_LOG_TITLE = (
        "RGB-STC V6: V5 + flow-guided modulated deformable alignment"
    )
    trainer.checkpoint_metadata = _checkpoint_metadata
    trainer._resume_contract = _resume_contract


def parse_args(input_args=None):
    _install_variant()
    args = trainer.parse_args(input_args)
    if args.init_stc_adapter:
        raise ValueError("V6 rejects --init_stc_adapter")
    if args.mixed_precision == "bf16":
        raise ValueError("torchvision deform_conv2d does not support BF16 here")
    for name in ("init_v5_model", "init_v6_model"):
        value = getattr(args, name)
        if value:
            setattr(args, name, str(Path(value).expanduser().resolve()))
    if args.training_stage == "deform_only":
        if not args.init_v5_model or args.init_v6_model:
            raise ValueError(
                "deform_only requires --init_v5_model and forbids --init_v6_model"
            )
    elif not args.init_v6_model or args.init_v5_model:
        raise ValueError(
            "joint requires --init_v6_model and forbids --init_v5_model"
        )
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
    for name in (
        "deform_residual_max_displacement",
        "deform_alignment_charbonnier_eps",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("deform_alignment_loss_weight", "deform_offset_loss_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if args.deform_alignment_loss_weight == 0.0:
        raise ValueError("V6 requires a positive deform alignment loss")
    if args.deform_alignment_warmup_steps < 0:
        raise ValueError("deform_alignment_warmup_steps must be non-negative")
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
        generator = torch.Generator().manual_seed(91)
        rgb = torch.randn(1, 4, 3, 32, 32, generator=generator)
        bg = torch.ones(1, 4, 1, 32, 32)
        ids = torch.arange(4)[None]
        with torch.no_grad():
            source_output = source(rgb, bg, output_size=(4, 4), frame_ids=ids)
            v6_output = model(rgb, bg, output_size=(4, 4), frame_ids=ids)
        if not torch.equal(source_output.features, v6_output.features):
            raise RuntimeError("Fresh V6 is not exact V5 identity")
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    report = {
        "v6_preflight": "ok",
        "training_stage": args.training_stage,
        "initialization_component": _resolved_initialization(args),
        "first_order": True,
        "bidirectional": True,
        "deforms_cross_clip_memory": False,
        "deform_groups": int(model.config.deform_groups),
        "deform_kernel_size": int(model.config.deform_kernel_size),
        "deform_residual_max_displacement": float(
            model.config.deform_residual_max_displacement
        ),
        "offset_mask_final_zero": bool(
            torch.count_nonzero(final_head.weight) == 0
            and torch.count_nonzero(final_head.bias) == 0
        )
        if args.training_stage == "deform_only"
        else None,
        "fusion_final_zero": bool(
            torch.count_nonzero(final_fusion.weight) == 0
            and torch.count_nonzero(final_fusion.bias) == 0
        )
        if args.training_stage == "deform_only"
        else None,
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "trainable_parameter_prefix_sample": trainable[:12],
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
