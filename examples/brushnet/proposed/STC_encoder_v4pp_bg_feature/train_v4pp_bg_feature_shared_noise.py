#!/usr/bin/env python
"""Train V4++: BG-focused V4 alignment plus adjacent feature loss."""

from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v3_rgb_flow import train_rgb_stc_flow_shared_noise as trainer
from STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (
    augment_brushnet_condition,
)
from STC_encoder_v4pp_bg_feature.bg_focused_flow_aligned_stc_adapter import (
    BGFocusedFlowAlignedRGBSTCAdapter,
)
from STC_encoder_v4pp_bg_feature.feature_alignment import (
    compute_feature_alignment_loss,
)


EXPERIMENT_NAME = "rgb_stc_v4pp_bg_focused_feature_alignment"
DEFAULT_OUTPUT = (
    trainer.REPO_ROOT
    / "experiments"
    / "train_stc_v4pp_bg_feature_T16_S12_sharedNoise_0.95"
)

_base_checkpoint_metadata = trainer.checkpoint_metadata
_base_resume_contract = trainer._resume_contract


def _checkpoint_metadata(args, accelerator, global_step, epoch, next_batch_index):
    metadata = _base_checkpoint_metadata(
        args, accelerator, global_step, epoch, next_batch_index
    )
    metadata.update(
        {
            "experiment": EXPERIMENT_NAME,
            "model_variant": "v4pp_bg_focused_flow_aligned_rgb_stc",
            "loss": (
                "L_diff + flow_loss_weight*L_flow + "
                "effective_feature_weight*L_feature_alignment"
            ),
            "flow_inference_dependency": True,
            "inference_component": "stc_flow_model",
            "alignment_stage": "raw_spatial_features_before_temporal_attention",
            "alignment_scope": "adjacent_frames_within_each_clip",
            "alignment_region": "internal_M_BG_only",
            "roi_alignment_policy": "exact_raw_v2_spatial_feature_path",
            "alignment_recurrence": "none_previous_raw_feature_only",
            "feature_alignment_warp": "predicted_bidirectional_flow",
            "feature_alignment_target": "stop_gradient_raw_spatial_feature",
            "feature_alignment_validity": "teacher_valid_and_in_bounds",
            "feature_alignment_confidence": "detached_with_configured_floor",
            "feature_alignment_oob_fallback": "none_zero_padding_not_current_target",
            "temporal_position_embedding": False,
            "recommended_initialization": "warm_start_trained_rgb_stc_v2",
        }
    )
    return metadata


def _resume_contract(args):
    contract = _base_resume_contract(args)
    contract["experiment"] = EXPERIMENT_NAME
    contract["alignment_region"] = "internal_M_BG_only"
    return contract


def _install_variant():
    trainer.RGBSTCFlowAdapter = BGFocusedFlowAlignedRGBSTCAdapter
    trainer.augment_brushnet_condition = augment_brushnet_condition
    trainer.FEATURE_ALIGNMENT_LOSS_FN = compute_feature_alignment_loss
    trainer.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    trainer.EXPERIMENT_NAME = EXPERIMENT_NAME
    trainer.FLOW_INFERENCE_DEPENDENCY = True
    trainer.INFERENCE_COMPONENT = "stc_flow_model"
    trainer.TRAINING_LOG_TITLE = (
        "RGB-STC v4++: BG-focused alignment + feature-alignment loss"
    )
    trainer.checkpoint_metadata = _checkpoint_metadata
    trainer._resume_contract = _resume_contract


def parse_args(input_args=None):
    _install_variant()
    args = trainer.parse_args(input_args)
    if not args.resume_from_checkpoint and not args.init_stc_adapter:
        raise ValueError(
            "A fresh V4++ run requires --init_stc_adapter pointing to a "
            "trained RGB-STC v2 component."
        )
    if args.feature_alignment_loss_weight <= 0.0:
        raise ValueError("V4++ requires --feature_alignment_loss_weight > 0")
    if args.feature_alignment_region != "bg":
        raise ValueError("V4++ BG-focused contract requires feature region 'bg'")
    return args


def run_preflight(args):
    _install_variant()
    trainer.run_preflight(args)


def main(args):
    _install_variant()
    trainer.main(args)


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.preflight_only:
        run_preflight(parsed)
    else:
        main(parsed)

