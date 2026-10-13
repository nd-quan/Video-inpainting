#!/usr/bin/env python
"""Stage 2: train standalone spatial-base V8-R; freeze RAFT/BrushNet/U-Net.

The shared diffusion data loop is reused, but its model, optimizer, condition,
metrics and checkpoint contract are replaced here. No legacy flow head runs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
PROPOSED = THIS_DIR.parent
REPO = THIS_DIR.parents[3]
sys.path.insert(0, str(PROPOSED))
sys.path.insert(0, str(THIS_DIR.parent.parent))

from STC_encoder_v3_rgb_flow import train_rgb_stc_flow_shared_noise as trainer
from STC_encoder_v8_raft_deformable import train_v8_raft_deformable as v8
from STC_encoder_v8_raft_deformable.raft_flow_provider import resolve_raft_student_component
from STC_encoder_v8_rescaled_deformable import train_v8_rescaled_joint as legacy
from v8r_standalone.migration import initialize_from_legacy, resolve_component
from v8r_standalone.model import StandaloneV8RAdapter, augment_brushnet_condition
from flow_refinement.provider import install_v8r_provider

_BASE_METADATA = trainer.checkpoint_metadata
_BASE_CONTRACT = trainer._resume_contract


def model_factory(args):
    kwargs = {
        "rescaled_warp_scale": args.rescaled_warp_scale,
        "deformable_alignment_direction": args.deformable_alignment_direction,
        "cross_clip_memory_frames": args.cross_clip_memory_frames,
    }
    # The shared loop loads the exact adapter component when resuming;
    # this factory is only used for fresh initialization.
    model, report = initialize_from_legacy(args.init_v8_model, **kwargs)
    print(json.dumps({"standalone_migration": report}, sort_keys=True))
    return model


def configure_trainable(*, model, args):
    model.requires_grad_(False)
    model.deformable_alignment.requires_grad_(True)
    model.deformable_fusion.requires_grad_(True)
    if args.joint_unfreeze_step < args.max_train_steps:
        model.stc_adapter.temporal_blocks.requires_grad_(True)
        model.stc_adapter.output_norm.requires_grad_(True)
        model.stc_adapter.zero_conv.requires_grad_(True)


def optimizer_groups(*, model, args):
    deform = list(model.deformable_alignment.parameters()) + list(model.deformable_fusion.parameters())
    temporal = (list(model.stc_adapter.temporal_blocks.parameters())
                + list(model.stc_adapter.output_norm.parameters())
                + list(model.stc_adapter.zero_conv.parameters()))
    selected = {id(p) for p in deform + temporal}
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if selected != expected or len(deform) + len(temporal) != len(selected):
        raise RuntimeError("Standalone optimizer groups are incomplete or overlapping")
    return [{"params": deform, "lr": args.deform_learning_rate, "name": "deform"},
            {"params": temporal, "lr": args.temporal_learning_rate, "name": "temporal"}]


def before_optimizer_step(*, model, optimizer, global_step, args):
    del optimizer
    if global_step >= args.joint_unfreeze_step:
        return
    bare = model.module if hasattr(model, "module") else model
    for module in (bare.stc_adapter.temporal_blocks, bare.stc_adapter.output_norm,
                   bare.stc_adapter.zero_conv):
        for parameter in module.parameters():
            parameter.grad = None


def train_metrics(*, stc_output, batch):
    offsets = torch.cat((stc_output.residual_offset_backward.detach().float().flatten(),
                         stc_output.residual_offset_forward.detach().float().flatten())).abs()
    return {
        "train/memory_overlap_mean": stc_output.memory_overlap_count.float().mean(),
        "train/cross_clip_gate_abs_mean": stc_output.cross_clip_gate_abs_mean,
        "train/deform_offset_abs_mean": offsets.mean() if offsets.numel() else offsets.new_zeros(()),
        "train/deform_reliability_backward": stc_output.deform_reliability_backward.float().mean(),
        "train/deform_reliability_forward": stc_output.deform_reliability_forward.float().mean(),
        "train/raft_flow_magnitude_feature_px": stc_output.predicted_flow_backward.float().square().sum(2).sqrt().mean(),
    }


def metadata(args, accelerator, global_step, epoch, next_batch_index):
    result = _BASE_METADATA(args, accelerator, global_step, epoch, next_batch_index)
    result.update({"experiment": "v8r_standalone", "model_variant": "spatial_base_raft_x4_residual_dcn",
                   "inference_component": "stc_v8r_standalone_model",
                   "initialization_component": str(resolve_component(args.init_v8_model)),
                   "legacy_lightweight_flow": False, "legacy_v5_alignment": False,
                   "raft_student_component": str(resolve_raft_student_component(args.raft_student_path)),
                   "joint_unfreeze_step": args.joint_unfreeze_step,
                   "rescaled_warp_scale": args.rescaled_warp_scale})
    result["flow_refiner_checkpoint"] = os.environ.get("V8R_FLOW_REFINER_PATH", "")
    return result


def resume_contract(args):
    result = _BASE_CONTRACT(args)
    result.update({"experiment": "v8r_standalone", "model_variant": "spatial_base_raft_x4_residual_dcn",
                   "initialization_component": str(resolve_component(args.init_v8_model)),
                   "raft_student_component": str(resolve_raft_student_component(args.raft_student_path)),
                   "rescaled_warp_scale": args.rescaled_warp_scale,
                   "joint_unfreeze_step": args.joint_unfreeze_step})
    result["flow_refiner_checkpoint"] = os.environ.get("V8R_FLOW_REFINER_PATH", "")
    return result


def install():
    legacy._install_variant()
    install_v8r_provider(v8)
    trainer.REPO_ROOT = REPO
    trainer.DEFAULT_OUTPUT = REPO / "experiments" / "train_v8r_standalone"
    trainer.EXPERIMENT_NAME = "v8r_standalone"
    trainer.TRAINING_LOG_TITLE = "Standalone V8-R: spatial base, frozen RAFT, x4 prewarp, residual DCN"
    trainer.RGBSTCFlowAdapter = StandaloneV8RAdapter
    trainer.augment_brushnet_condition = augment_brushnet_condition
    trainer.MODEL_FACTORY_FN = model_factory
    trainer.CONFIGURE_TRAINABLE_PARAMETERS_FN = configure_trainable
    trainer.OPTIMIZER_PARAM_GROUPS_FN = optimizer_groups
    trainer.BEFORE_OPTIMIZER_STEP_FN = before_optimizer_step
    trainer.EXTRA_TRAIN_METRICS_FN = train_metrics
    trainer.checkpoint_metadata = metadata
    trainer._resume_contract = resume_contract
    trainer.FULL_MODEL_COMPONENT_NAME = "stc_v8r_standalone_model"
    trainer.INFERENCE_COMPONENT = "stc_v8r_standalone_model"


def parse_args(argv=None):
    install()
    args = trainer.parse_args(v8._rewrite_zero_flow_loss(list(sys.argv[1:] if argv is None else argv)))
    args.flow_loss_weight = 0.0
    if args.mixed_precision == "bf16":
        raise ValueError("DCNv2 path requires fp16 or float32, not bf16")
    if args.independent_frame_timesteps:
        raise ValueError("Use a shared diffusion timestep per clip")
    if args.feature_alignment_loss_weight != 0 or args.flow_loss_weight != 0:
        raise ValueError("Standalone stage 2 requires feature_alignment_loss_weight=flow_loss_weight=0")
    if not 0 <= args.joint_unfreeze_step < args.max_train_steps:
        raise ValueError("joint_unfreeze_step must be in [0,max_train_steps)")
    if args.lr_scheduler != "constant":
        raise ValueError("Staged joint optimizer requires constant scheduler")
    args.init_v8_model = str(resolve_component(args.init_v8_model))
    args.raft_student_path = str(resolve_raft_student_component(args.raft_student_path))
    v8._ACTIVE_ARGS = args
    legacy._ACTIVE_ARGS = args
    return args


def main():
    args = parse_args()
    if args.preflight_only:
        trainer.run_preflight(args)
        model = model_factory(args)
        assert not any("flow_predictor" in key or "alignment_fusion" in key for key in model.state_dict())
        print(json.dumps({"standalone_preflight": "ok", "spatial_base": True,
                          "legacy_flow_head": False, "student_frozen": True}))
    else:
        trainer.main(args)


if __name__ == "__main__":
    main()
