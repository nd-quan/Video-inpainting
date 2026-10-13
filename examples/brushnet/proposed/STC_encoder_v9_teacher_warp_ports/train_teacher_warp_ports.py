#!/usr/bin/env python
"""Train x4 clean-teacher feature warping at sparse BrushNet residual ports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import CLIPImageProcessor


THIS_DIR = Path(__file__).resolve().parent
PROPOSED = THIS_DIR.parent
BRUSHNET_EXAMPLES = PROPOSED.parent
if str(PROPOSED) not in sys.path:
    sys.path.insert(0, str(PROPOSED))
if str(BRUSHNET_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_EXAMPLES))

from STC_encoder_v3_rgb_flow import train_rgb_stc_flow_shared_noise as trainer
from STC_encoder_v5_relative_crossclip.train_v5_relative_crossclip import _dataset_signature
from STC_encoder_v9_teacher_warp_ports.condition import (
    augment_brushnet_condition,
    precompute_brushnet_residuals,
    predict_with_feature_ports,
)
from STC_encoder_v9_teacher_warp_ports.data import (
    CrossClipOracleFlowDataset,
    collate_oracle_flow_clips,
)
from STC_encoder_v9_teacher_warp_ports.model import TeacherWarpPortModel


EXPERIMENT_NAME = "stc_v9_clean_teacher_x4_warp_feature_ports"
DEFAULT_OUTPUT = trainer.REPO_ROOT / "experiments" / "train_stc_v9_teacher_x4_ports_T16_S12"
COMPONENT_NAME = "stc_v9_teacher_warp_ports_model"
_BASE_METADATA = trainer.checkpoint_metadata
_BASE_CONTRACT = trainer._resume_contract
_ACTIVE_ARGS = None


def _add_arguments(parser) -> None:
    parser.add_argument("--init_v8r_model", required=True,
                        help="Complete V8-R or standalone component/checkpoint/pointer")
    parser.add_argument("--warp_strength", type=float, default=1.0,
                        help="1: teacher x4 warp; 0: matched no-warp ablation")
    parser.add_argument("--port_learning_rate", type=float, default=2e-5)
    parser.add_argument("--port_bottleneck_channels", type=int, default=64)


def _brushnet_port_channels(args):
    config_file = Path(args.baseline_checkpoint).expanduser().resolve() / "brushnet" / "config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    channels = tuple(int(c) for c in config["block_out_channels"])
    layers = int(config["layers_per_block"])
    down = [channels[0]]
    for stage, width in enumerate(channels):
        down.extend([width] * layers)
        if stage < len(channels) - 1:
            down.append(width)
    indices = (2, 5, 8)
    if len(down) <= max(indices) or len(channels) != 4 or layers != 2:
        raise ValueError("V9 selected ports require the deployed four-stage, two-layer BrushNet")
    return indices, tuple(down[i] for i in indices), channels[-1]


def _model_factory(args):
    indices, channels, mid_channel = _brushnet_port_channels(args)
    return TeacherWarpPortModel.from_v8r_pretrained(
        args.init_v8r_model,
        rescaled_warp_scale=4,
        down_port_indices=indices,
        down_port_channels=channels,
        mid_port_channels=mid_channel,
        port_bottleneck_channels=args.port_bottleneck_channels,
    )


def _make_dataset(args, tokenizer):
    dataset = CrossClipOracleFlowDataset(
        dataset_root=args.dataset_root,
        split=args.train_split,
        teacher_flow_root=args.teacher_flow_root,
        tokenizer=tokenizer,
        clip_image_processor=CLIPImageProcessor(),
        clip_length=args.clip_length,
        stride=args.clip_stride,
        resolution=args.resolution,
    )
    args.cross_clip_pair_index_sha256 = _dataset_signature(dataset)
    args.cross_clip_transition_count = dataset.cross_clip_transition_count
    args.cross_clip_run_count = dataset.cross_clip_run_count
    return dataset


def _extra_kwargs(*, batch, device, num_clips, num_frames, resolution):
    expected_flow = (num_clips, num_frames - 1, 2, resolution, resolution)
    expected_valid = (num_clips, num_frames - 1, 1, resolution, resolution)
    result = {
        "frame_ids": batch["frame_ids"].to(device=device, non_blocking=True),
        "previous_rgb_sequence": batch["previous_conditioning_pixel_values"].to(device=device, non_blocking=True),
        "previous_bg_mask_sequence": batch["previous_masks"].to(device=device, non_blocking=True),
        "previous_frame_ids": batch["previous_frame_ids"].to(device=device, non_blocking=True),
        "previous_valid_mask": batch["previous_valid_mask"].to(device=device, non_blocking=True),
        "warp_strength": float(_ACTIVE_ARGS.warp_strength),
    }
    names = {
        "teacher_flow_forward_rgb": "teacher_flow_forward",
        "teacher_flow_backward_rgb": "teacher_flow_backward",
        "teacher_valid_forward_rgb": "teacher_valid_forward",
        "teacher_valid_backward_rgb": "teacher_valid_backward",
        "previous_teacher_flow_forward_rgb": "previous_teacher_flow_forward",
        "previous_teacher_flow_backward_rgb": "previous_teacher_flow_backward",
        "previous_teacher_valid_forward_rgb": "previous_teacher_valid_forward",
        "previous_teacher_valid_backward_rgb": "previous_teacher_valid_backward",
    }
    for argument, key in names.items():
        value = batch[key]
        expected = expected_valid if "valid" in key else expected_flow
        if tuple(value.shape) != expected:
            raise ValueError(f"{key} shape {tuple(value.shape)} != {expected}")
        result[argument] = value.to(device=device, non_blocking=True)
    return result


def _configure_trainable(*, model, args) -> None:
    del args
    model.requires_grad_(False)
    model.port_adapter.requires_grad_(True)
    model.warp_gain.requires_grad_(True)


def _optimizer_groups(*, model, args):
    port = list(model.port_adapter.parameters()) + [model.warp_gain]
    selected = {id(p) for p in port}
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if selected != expected or len(port) != len(selected):
        raise RuntimeError("V9 optimizer groups must cover every trainable parameter exactly once")
    return (
        {"params": port, "lr": args.port_learning_rate, "name": "port_and_warp"},
    )


def _metrics(*, stc_output, batch):
    del batch
    values = [value.detach().float().square().mean() for value in stc_output.port_down_corrections]
    values.append(stc_output.port_mid_correction.detach().float().square().mean())
    return {
        "train/port_correction_rms": torch.stack(values).mean().sqrt(),
        "train/teacher_warp_reliability_backward": stc_output.reliability_backward.detach().float().mean(),
        "train/teacher_warp_reliability_forward": stc_output.reliability_forward.detach().float().mean(),
        "train/memory_overlap_mean": stc_output.memory_overlap_count.detach().float().mean(),
        "train/relative_bias_abs_mean": stc_output.relative_bias_abs_mean,
        "train/cross_clip_gate_abs_mean": stc_output.cross_clip_gate_abs_mean,
    }


def _metadata(args, accelerator, global_step, epoch, next_batch_index):
    result = _BASE_METADATA(args, accelerator, global_step, epoch, next_batch_index)
    result.update({
        "experiment": EXPERIMENT_NAME,
        "inference_component": COMPONENT_NAME,
        "initialization_component": str(Path(args.init_v8r_model).expanduser().resolve()),
        "flow_source": "clean_GT_cached_ProPainter_RAFT_teacher",
        "flow_role": "active_feature_warp",
        "flow_grid": "RGB_512_to_each_BrushNet_port_x4_then_nearest_down",
        "rescaled_warp_scale": 4,
        "legacy_v5_flow": False,
        "deformable_convolution": False,
        "latent_delta": False,
        "warp_strength": args.warp_strength,
        "selected_down_ports": [2, 5, 8],
        "selected_mid_port": True,
        "port_learning_rate": args.port_learning_rate,
        "port_bottleneck_channels": args.port_bottleneck_channels,
        "loss": "L_diff",
        "cross_clip_pair_index_sha256": getattr(args, "cross_clip_pair_index_sha256", None),
        "cross_clip_transition_count": getattr(args, "cross_clip_transition_count", None),
    })
    return result


def _resume_contract(args):
    result = _BASE_CONTRACT(args)
    result.update({
        "experiment": EXPERIMENT_NAME,
        "initialization_component": str(Path(args.init_v8r_model).expanduser().resolve()),
        "flow_source": "clean_GT_cached_ProPainter_RAFT_teacher",
        "rescaled_warp_scale": 4,
        "legacy_v5_flow": False,
        "deformable_convolution": False,
        "warp_strength": args.warp_strength,
        "port_learning_rate": args.port_learning_rate,
        "port_bottleneck_channels": args.port_bottleneck_channels,
        "selected_down_ports": [2, 5, 8],
    })
    return result


def install_variant() -> None:
    trainer.ADD_VARIANT_ARGUMENTS_FN = _add_arguments
    trainer.MODEL_FACTORY_FN = _model_factory
    trainer.AUGMENT_EXTRA_KWARGS_FN = _extra_kwargs
    trainer.PRE_AUGMENT_FN = precompute_brushnet_residuals
    trainer.PREDICT_FN = predict_with_feature_ports
    trainer.ALLOW_ZERO_FLOW_LOSS_WEIGHT = True
    trainer.CONFIGURE_TRAINABLE_PARAMETERS_FN = _configure_trainable
    trainer.OPTIMIZER_PARAM_GROUPS_FN = _optimizer_groups
    trainer.EXTRA_TRAIN_METRICS_FN = _metrics
    trainer.FEATURE_ALIGNMENT_LOSS_FN = None
    trainer.EXTRA_TRAIN_LOSS_FN = None
    trainer.POST_PREDICTION_LOSS_FN = None
    trainer.RGBSTCFlowAdapter = TeacherWarpPortModel
    trainer.augment_brushnet_condition = augment_brushnet_condition
    trainer.make_dataset = _make_dataset
    trainer.collate_teacher_flow_clips = collate_oracle_flow_clips
    trainer.FULL_MODEL_COMPONENT_NAME = COMPONENT_NAME
    trainer.INFERENCE_COMPONENT = COMPONENT_NAME
    trainer.SAVE_LEGACY_STC_ADAPTER = False
    trainer.FLOW_INFERENCE_DEPENDENCY = True
    trainer.TRAIN_DATALOADER_DROP_LAST = True
    trainer.DDP_FIND_UNUSED_PARAMETERS = False
    trainer.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    trainer.EXPERIMENT_NAME = EXPERIMENT_NAME
    trainer.TRAINING_LOG_TITLE = "V9: clean teacher x4 feature warp to BrushNet ports"
    trainer.checkpoint_metadata = _metadata
    trainer._resume_contract = _resume_contract


def parse_args(argv=None):
    global _ACTIVE_ARGS
    install_variant()
    args = trainer.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.condition_mode != "full_rgb_bg_mask":
        raise ValueError("V9 requires full_rgb_bg_mask")
    if args.resolution != 512:
        raise ValueError("The selected clean teacher cache and x4 geometry require 512x512")
    if args.flow_loss_weight != 0 or args.feature_alignment_loss_weight != 0:
        raise ValueError("V9 uses teacher flow as input; flow/feature supervision weights must be zero")
    if args.independent_frame_timesteps:
        raise ValueError("V9 requires a shared diffusion timestep across each clip")
    if not 0.0 <= args.warp_strength <= 1.0:
        raise ValueError("warp_strength must be in [0,1]")
    if args.port_learning_rate <= 0:
        raise ValueError("V9 port learning rate must be positive")
    _ACTIVE_ARGS = args
    return args


def main():
    args = parse_args()
    if args.preflight_only:
        trainer.run_preflight(args)
        model = _model_factory(args)
        state = model.state_dict()
        if any("deformable" in key or "flow_head" in key or "zero_conv" in key for key in state):
            raise RuntimeError("V9 model unexpectedly contains a legacy alignment or latent module")
        print(json.dumps({"v9_preflight": "ok", "teacher_warp_scale": 4,
                          "port_indices": [2, 5, 8], "model_tensors": len(state)}))
    else:
        trainer.main(args)


if __name__ == "__main__":
    main()
