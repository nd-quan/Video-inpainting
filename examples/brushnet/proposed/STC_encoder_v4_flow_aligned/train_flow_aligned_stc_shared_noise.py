#!/usr/bin/env python
"""Train clip-local flow-aligned RGB-STC by reusing the audited V3 trainer.

The data, frozen V8 path, shared-noise law, teacher-flow loss, DDP handling and
checkpoint state are intentionally inherited from V3.  Only the trainable
model and its inference contract change.
"""

from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v3_rgb_flow import train_rgb_stc_flow_shared_noise as trainer
from STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (
    FlowAlignedRGBSTCAdapter,
    augment_brushnet_condition,
)


EXPERIMENT_NAME = "rgb_stc_v4_flow_aligned_clip_local"
DEFAULT_OUTPUT = (
    trainer.REPO_ROOT
    / "experiments"
    / "train_stc_v4_flow_aligned_T16_S12_sharedNoise_0.9"
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
            "model_variant": "flow_aligned_rgb_stc",
            "flow_inference_dependency": True,
            "inference_component": "stc_flow_model",
            "alignment_stage": "raw_spatial_features_before_temporal_attention",
            "alignment_scope": "adjacent_frames_within_each_clip",
            "alignment_recurrence": "none_previous_raw_feature_only",
            "temporal_position_embedding": False,
            "recommended_initialization": "warm_start_rgb_stc_v2_T8_then_finetune_T16",
        }
    )
    return metadata


def _resume_contract(args):
    contract = _base_resume_contract(args)
    contract["experiment"] = EXPERIMENT_NAME
    return contract


def _install_variant():
    trainer.RGBSTCFlowAdapter = FlowAlignedRGBSTCAdapter
    trainer.augment_brushnet_condition = augment_brushnet_condition
    trainer.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    trainer.EXPERIMENT_NAME = EXPERIMENT_NAME
    trainer.FLOW_INFERENCE_DEPENDENCY = True
    trainer.INFERENCE_COMPONENT = "stc_flow_model"
    trainer.TRAINING_LOG_TITLE = "RGB-STC v4: flow-aligned T16 features"
    trainer.checkpoint_metadata = _checkpoint_metadata
    trainer._resume_contract = _resume_contract


def parse_args(input_args=None):
    _install_variant()
    args = trainer.parse_args(input_args)
    if not args.resume_from_checkpoint and not args.init_stc_adapter:
        raise ValueError(
            "A fresh V4 run requires --init_stc_adapter pointing to the trained "
            "RGB-STC v2 component. Use --resume_from_checkpoint for continuation."
        )
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
