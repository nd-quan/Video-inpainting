#!/usr/bin/env python
"""Train V8-R joint: frozen RAFT, rescaled pre-warp, native DCNv2.

One optimizer owns a high-LR freshly reset deformable branch and a low-LR
pretrained temporal branch.  Temporal gradients/LR stay disabled for the
first ``joint_unfreeze_step`` optimizer updates; the DDP graph and optimizer
schema remain fixed, so exact checkpoint resume remains supported.
"""

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
from STC_encoder_v4pp_bg_feature.feature_alignment import compute_feature_alignment_loss
from STC_encoder_v5_relative_crossclip import train_v5_relative_crossclip as v5
from STC_encoder_v5_relative_crossclip.cross_clip_data import collate_cross_clip_teacher_flow
from STC_encoder_v8_raft_deformable import train_v8_raft_deformable as v8
from STC_encoder_v8_raft_deformable.raft_flow_provider import (
    FrozenV7RAFTFlowProvider,
    resolve_raft_student_component,
)
from STC_encoder_v8_rescaled_deformable.rescaled_raft_deformable_stc_adapter import (
    RescaledRAFTGuidedDeformableBGSTCAdapter,
    augment_brushnet_condition_v8_rescaled,
)


EXPERIMENT_NAME = "rgb_stc_v8_rescaled_prewarp_joint"
DEFAULT_OUTPUT = (
    trainer.REPO_ROOT
    / "experiments"
    / "train_stc_v8_rescaled_joint_T16_S12_sharedNoise_0.95"
)
_base_checkpoint_metadata = trainer.checkpoint_metadata
_base_resume_contract = trainer._resume_contract


def _add_variant_arguments(parser) -> None:
    parser.add_argument(
        "--init_v8_model",
        required=True,
        help="Existing native V8 component/checkpoint used for shared weights.",
    )
    parser.add_argument(
        "--training_stage", choices=("rescaled_joint",), default="rescaled_joint"
    )
    parser.add_argument(
        "--deformable_alignment_direction",
        choices=("bidirectional", "previous_only", "next_only"),
        default="bidirectional",
    )
    parser.add_argument("--relative_position_max_distance", type=int, default=32)
    parser.add_argument("--cross_clip_memory_frames", type=int, default=4)
    parser.add_argument("--detach_cross_clip_memory", action="store_true", default=True)
    parser.add_argument("--deform_hidden_channels", type=int, default=128)
    parser.add_argument("--deform_kernel_size", type=int, default=3)
    parser.add_argument("--deform_groups", type=int, default=4)
    parser.add_argument("--deform_residual_max_displacement", type=float, default=2.0)
    parser.add_argument("--deform_alignment_loss_weight", type=float, default=0.1)
    parser.add_argument("--deform_alignment_charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--deform_alignment_warmup_steps", type=int, default=500)
    parser.add_argument("--deform_offset_loss_weight", type=float, default=5e-4)
    parser.add_argument("--rescaled_warp_scale", type=int, default=4)
    parser.add_argument("--joint_unfreeze_step", type=int, default=500)
    parser.add_argument("--deform_learning_rate", type=float, default=2e-5)
    parser.add_argument("--temporal_learning_rate", type=float, default=5e-6)
    parser.add_argument("--temporal_lr_warmup_steps", type=int, default=100)
    parser.add_argument(
        "--reset_deformable_branch",
        dest="reset_deformable_branch",
        action="store_true",
        default=True,
        help="Reset native-V8 DCN/fusion because V8-R changes sampling geometry.",
    )
    parser.add_argument(
        "--keep_deformable_branch",
        dest="reset_deformable_branch",
        action="store_false",
        help="Ablation only: transfer native-V8 DCN/fusion weights.",
    )
    parser.add_argument(
        "--raft_student_path",
        required=True,
        help="Frozen V7 ProPainter-RAFT student component or best/latest pointer.",
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


def _model_factory(args):
    source = v8.resolve_complete_component(args.init_v8_model, "stc_v8_model")
    model = RescaledRAFTGuidedDeformableBGSTCAdapter.from_v8_pretrained(
        source,
        rescaled_warp_scale=args.rescaled_warp_scale,
        reset_deformable_branch=args.reset_deformable_branch,
    )
    expected = {
        "relative_position_max_distance": args.relative_position_max_distance,
        "cross_clip_memory_frames": args.cross_clip_memory_frames,
        "deform_hidden_channels": args.deform_hidden_channels,
        "deform_kernel_size": args.deform_kernel_size,
        "deform_groups": args.deform_groups,
        "deform_residual_max_displacement": args.deform_residual_max_displacement,
        "rescaled_warp_scale": args.rescaled_warp_scale,
    }
    for key, value in expected.items():
        saved = getattr(model.config, key)
        matches = (
            math.isclose(float(saved), float(value), rel_tol=0.0, abs_tol=1e-12)
            if isinstance(value, float)
            else int(saved) == int(value)
        )
        if not matches:
            raise ValueError(f"V8-R config mismatch {key}: model={saved}, args={value}")
    model.register_to_config(
        deformable_alignment_direction=args.deformable_alignment_direction
    )
    return model


def _temporal_modules(model):
    return (
        model.alignment_fusion,
        model.stc_adapter.temporal_blocks,
        model.stc_adapter.output_norm,
        model.stc_adapter.zero_conv,
    )


def _configure_trainable_parameters(*, model, args) -> None:
    del args
    model.requires_grad_(False)
    model.deformable_alignment.requires_grad_(True)
    model.deformable_fusion.requires_grad_(True)
    for module in _temporal_modules(model):
        module.requires_grad_(True)


def _optimizer_param_groups(*, model, args):
    deform = list(model.deformable_alignment.parameters()) + list(
        model.deformable_fusion.parameters()
    )
    temporal = [
        parameter
        for module in _temporal_modules(model)
        for parameter in module.parameters()
    ]
    deform_ids = {id(parameter) for parameter in deform}
    temporal_ids = {id(parameter) for parameter in temporal}
    if deform_ids & temporal_ids:
        raise RuntimeError("V8-R optimizer parameter groups overlap")
    selected = deform_ids | temporal_ids
    expected = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if selected != expected:
        raise RuntimeError("V8-R optimizer groups do not cover trainable parameters")
    return [
        {
            "params": deform,
            "lr": float(args.deform_learning_rate),
            "name": "deform",
        },
        {
            "params": temporal,
            "lr": float(args.temporal_learning_rate),
            "name": "temporal",
        },
    ]


def _lr_scheduler_factory(*, optimizer, args, accelerator):
    del accelerator

    def deform_schedule(step: int) -> float:
        warmup = int(args.lr_warmup_steps)
        return 1.0 if warmup <= 0 else min(1.0, float(step + 1) / warmup)

    def temporal_schedule(step: int) -> float:
        if step < int(args.joint_unfreeze_step):
            return 0.0
        warmup = int(args.temporal_lr_warmup_steps)
        if warmup <= 0:
            return 1.0
        local_step = step - int(args.joint_unfreeze_step)
        return min(1.0, float(local_step + 1) / warmup)

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=(deform_schedule, temporal_schedule)
    )


def _before_optimizer_step(*, model, optimizer, global_step: int, args) -> None:
    del optimizer
    if global_step >= int(args.joint_unfreeze_step):
        return
    bare = model.module if hasattr(model, "module") else model
    for module in _temporal_modules(bare):
        for parameter in module.parameters():
            parameter.grad = None


def _optimizer_metrics(*, optimizer, lr_scheduler, global_step: int):
    del lr_scheduler
    values = {group.get("name", str(index)): float(group["lr"]) for index, group in enumerate(optimizer.param_groups)}
    return {
        "train/lr_deform": values["deform"],
        "train/lr_temporal": values["temporal"],
        "train/temporal_branch_active": float(global_step >= _ACTIVE_ARGS.joint_unfreeze_step),
    }


def _resolved_initialization(args) -> str:
    return str(v8.resolve_complete_component(args.init_v8_model, "stc_v8_model"))


def _checkpoint_metadata(args, accelerator, global_step, epoch, next_batch_index):
    metadata = _base_checkpoint_metadata(
        args, accelerator, global_step, epoch, next_batch_index
    )
    metadata.update(
        {
            "experiment": EXPERIMENT_NAME,
            "model_variant": "v8_rescaled_prewarp_native_dcn_geometric_only",
            "inference_component": "stc_v8r_model",
            "initialization_component": _resolved_initialization(args),
            "training_stage": args.training_stage,
            "trainable_components": [
                "deformable_alignment",
                "deformable_fusion",
                "alignment_fusion",
                "stc_adapter.temporal_blocks",
                "stc_adapter.output_norm",
                "stc_adapter.zero_conv",
            ],
            "joint_unfreeze_step": args.joint_unfreeze_step,
            "deform_learning_rate": args.deform_learning_rate,
            "temporal_learning_rate": args.temporal_learning_rate,
            "temporal_lr_warmup_steps": args.temporal_lr_warmup_steps,
            "rescaled_warp_scale": args.rescaled_warp_scale,
            "rescaled_warp_up_mode": "nearest",
            "rescaled_warp_down_mode": "nearest",
            "deform_reliability_mode": "geometric_only",
            "fb_confidence_role": "detached_diagnostic_only",
            "dcn_source": "nearest_downsample(warp(nearest_upsample(raw_spatial), resized_RGB_RAFT_flow))",
            "dcn_offset": "learned_residual_only_no_second_flow",
            "deformable_alignment_direction": args.deformable_alignment_direction,
            "deform_alignment_loss_weight": args.deform_alignment_loss_weight,
            "deform_alignment_warmup_steps": args.deform_alignment_warmup_steps,
            "deform_offset_loss_weight": args.deform_offset_loss_weight,
            "reset_deformable_branch": args.reset_deformable_branch,
            "v7_raft_flow_frozen": True,
            "v7_raft_student_component": str(
                resolve_raft_student_component(args.raft_student_path)
            ),
            "v7_raft_architecture": "propainter_raft_large",
            "loss": "L_diff + ramp(0.1)*L_deform + 0.0005*L_offset",
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
            "joint_unfreeze_step": args.joint_unfreeze_step,
            "deform_learning_rate": args.deform_learning_rate,
            "temporal_learning_rate": args.temporal_learning_rate,
            "temporal_lr_warmup_steps": args.temporal_lr_warmup_steps,
            "rescaled_warp_scale": args.rescaled_warp_scale,
            "deform_reliability_mode": "geometric_only",
            "deformable_alignment_direction": args.deformable_alignment_direction,
            "deform_hidden_channels": args.deform_hidden_channels,
            "deform_kernel_size": args.deform_kernel_size,
            "deform_groups": args.deform_groups,
            "deform_residual_max_displacement": args.deform_residual_max_displacement,
            "deform_alignment_loss_weight": args.deform_alignment_loss_weight,
            "deform_alignment_charbonnier_eps": args.deform_alignment_charbonnier_eps,
            "deform_alignment_warmup_steps": args.deform_alignment_warmup_steps,
            "deform_offset_loss_weight": args.deform_offset_loss_weight,
            "reset_deformable_branch": args.reset_deformable_branch,
            "v7_raft_student_component": str(
                resolve_raft_student_component(args.raft_student_path)
            ),
            "v7_raft_architecture": "propainter_raft_large",
            "v7_raft_pair_batch_size": args.raft_pair_batch_size,
            "v7_raft_mixed_precision": args.raft_mixed_precision,
        }
    )
    return contract


def _install_variant() -> None:
    v8._install_variant()
    trainer.RGBSTCFlowAdapter = RescaledRAFTGuidedDeformableBGSTCAdapter
    trainer.augment_brushnet_condition = augment_brushnet_condition_v8_rescaled
    trainer.FEATURE_ALIGNMENT_LOSS_FN = compute_feature_alignment_loss
    trainer.EXTRA_TRAIN_LOSS_FN = v8._build_v8_extra_train_loss
    trainer.CONFIGURE_TRAINABLE_PARAMETERS_FN = _configure_trainable_parameters
    trainer.OPTIMIZER_PARAM_GROUPS_FN = _optimizer_param_groups
    trainer.LR_SCHEDULER_FACTORY_FN = _lr_scheduler_factory
    trainer.EXTRA_OPTIMIZER_METRICS_FN = _optimizer_metrics
    trainer.BEFORE_OPTIMIZER_STEP_FN = _before_optimizer_step
    trainer.ADD_VARIANT_ARGUMENTS_FN = _add_variant_arguments
    trainer.MODEL_FACTORY_FN = _model_factory
    trainer.AUGMENT_EXTRA_KWARGS_FN = v8._augment_extra_kwargs
    trainer.EXTRA_TRAIN_METRICS_FN = v8._extra_train_metrics
    trainer.POST_DATASET_VALIDATION_FN = v5._post_dataset_validation
    trainer.TRAIN_DATALOADER_DROP_LAST = True
    trainer.FULL_MODEL_COMPONENT_NAME = "stc_v8r_model"
    trainer.SAVE_LEGACY_STC_ADAPTER = False
    trainer.DDP_FIND_UNUSED_PARAMETERS = True
    trainer.make_dataset = v5._make_dataset
    trainer.collate_teacher_flow_clips = collate_cross_clip_teacher_flow
    trainer.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    trainer.EXPERIMENT_NAME = EXPERIMENT_NAME
    trainer.FLOW_INFERENCE_DEPENDENCY = True
    trainer.INFERENCE_COMPONENT = "stc_v8r_model"
    trainer.TRAINING_LOG_TITLE = (
        "V8-R joint: frozen RAFT + xS pre-warp + geometric-only native DCN"
    )
    trainer.checkpoint_metadata = _checkpoint_metadata
    trainer._resume_contract = _resume_contract


def parse_args(input_args=None):
    global _ACTIVE_ARGS
    _install_variant()
    argv = list(sys.argv[1:] if input_args is None else input_args)
    args = trainer.parse_args(v8._rewrite_zero_flow_loss(argv))
    args.flow_loss_weight = 0.0
    if args.init_stc_adapter:
        raise ValueError("V8-R rejects --init_stc_adapter")
    if args.mixed_precision == "bf16":
        raise ValueError("torchvision deform_conv2d requires fp16/float32 here")
    args.init_v8_model = str(Path(args.init_v8_model).expanduser().resolve())
    args.raft_student_path = str(
        resolve_raft_student_component(args.raft_student_path)
    )
    raft_config_path = Path(args.raft_student_path) / "config.json"
    raft_config = json.loads(raft_config_path.read_text(encoding="utf-8"))
    if raft_config.get("architecture") != "propainter_raft_large":
        raise ValueError(
            "V8-R is configured for the V7 ProPainter-RAFT student, but "
            f"{raft_config_path} declares architecture="
            f"{raft_config.get('architecture')!r}"
        )
    v8.resolve_complete_component(args.init_v8_model, "stc_v8_model")
    if args.rescaled_warp_scale < 1:
        raise ValueError("rescaled_warp_scale must be >= 1")
    if not 0 <= args.joint_unfreeze_step < args.max_train_steps:
        raise ValueError("joint_unfreeze_step must be in [0,max_train_steps)")
    if args.lr_scheduler != "constant":
        raise ValueError("V8-R staged joint currently requires --lr_scheduler constant")
    for name in (
        "deform_learning_rate",
        "temporal_learning_rate",
        "deform_alignment_loss_weight",
        "deform_alignment_charbonnier_eps",
        "deform_offset_loss_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if args.deform_alignment_warmup_steps < 0 or args.temporal_lr_warmup_steps < 0:
        raise ValueError("warmup steps must be non-negative")
    if args.relative_position_max_distance < args.clip_length - 1:
        raise ValueError("relative_position_max_distance must cover T-1")
    overlap = args.clip_length - args.clip_stride
    if overlap <= 0 or not 1 <= args.cross_clip_memory_frames <= overlap:
        raise ValueError("cross-clip memory must fit the positive T-S overlap")
    if args.feature_alignment_loss_weight != 0.0:
        raise ValueError("First V8-R run requires feature_alignment_loss_weight=0")
    v8._ACTIVE_ARGS = args
    _ACTIVE_ARGS = args
    return args


def run_preflight(args) -> None:
    _install_variant()
    trainer.run_preflight(args)
    model = _model_factory(args).eval()
    _configure_trainable_parameters(model=model, args=args)
    final_head = model.deformable_alignment.offset_mask_head[-1]
    final_fusion = model.deformable_fusion.to_residual_and_gate
    if args.reset_deformable_branch:
        if torch.count_nonzero(final_head.weight) or torch.count_nonzero(final_head.bias):
            raise RuntimeError("Reset V8-R offset/mask output is not zero initialized")
        if torch.count_nonzero(final_fusion.weight) or torch.count_nonzero(final_fusion.bias):
            raise RuntimeError("Reset V8-R fusion output is not zero initialized")
    groups = _optimizer_param_groups(model=model, args=args)
    report = {
        "v8_rescaled_preflight": "ok",
        "initialization_component": _resolved_initialization(args),
        "v7_raft_student_component": args.raft_student_path,
        "v7_raft_architecture": "propainter_raft_large",
        "v7_raft_frozen": True,
        "rescaled_warp_scale": args.rescaled_warp_scale,
        "rescaled_warp_path": "RGB flow -> xS grid (direct); up -> warp -> down",
        "dcn_source": "rescaled prewarped spatial feature",
        "dcn_offset": "bounded learned residual only",
        "double_flow_warp": False,
        "deform_reliability_mode": "geometric_only",
        "fb_confidence_role": "diagnostic_only",
        "deformable_alignment_direction": args.deformable_alignment_direction,
        "joint_unfreeze_step": args.joint_unfreeze_step,
        "optimizer_groups": {
            "deform": {
                "lr": args.deform_learning_rate,
                "parameters": sum(p.numel() for p in groups[0]["params"]),
            },
            "temporal": {
                "lr": args.temporal_learning_rate,
                "parameters": sum(p.numel() for p in groups[1]["params"]),
            },
        },
        "loss": (
            f"L_diff + ramp({args.deform_alignment_loss_weight})*L_deform "
            f"+ {args.deform_offset_loss_weight}*L_offset"
        ),
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
