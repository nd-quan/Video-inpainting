#!/usr/bin/env python
"""Standalone V8-R evaluation, including optional latent/RGB DDIM guidance."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
sys.path.insert(0, str(THIS_DIR.parent.parent))

from STC_encoder_v8_raft_deformable import evaluate_v8_raft_deformable as v8_eval
from STC_encoder_v8_raft_deformable import evaluate_v8_raft_deformable_temporal_student as temporal_eval
from v8r_standalone.model import StandaloneV8RAdapter, augment_brushnet_condition
from flow_refinement.provider import install_v8r_provider

_BASE_BUILD = v8_eval.build_v8_condition
_BASE_BEFORE_PIPELINE_CALL = temporal_eval.before_pipeline_call


def preflight(args):
    if args.condition_flow_source != "student":
        raise ValueError("Standalone evaluation currently supports only frozen student flow")
    if args.deformable_alignment_scale < 0:
        raise ValueError("deformable_alignment_scale must be nonnegative")
    args.raft_student_path = str(v8_eval.resolve_raft_student_component(args.raft_student_path))
    refiner_path = os.environ.get("V8R_FLOW_REFINER_PATH", "").strip()
    if refiner_path and not Path(refiner_path).expanduser().is_file():
        raise FileNotFoundError(f"V8R_FLOW_REFINER_PATH does not exist: {refiner_path}")
    v8_eval._ACTIVE_EVAL_ARGS = args
    v8_eval._ORACLE_CACHE = None
    v8_eval.v5_evaluator.RelativeCrossClipBGSTCAdapter = StandaloneV8RAdapter
    dataset, paths = v8_eval._base_v5_preflight(args)
    adapter = StandaloneV8RAdapter.from_pretrained(str(args.stc_adapter_path))
    if args.deformable_alignment_direction is None:
        args.deformable_alignment_direction = adapter.config.deformable_alignment_direction
    print(json.dumps({"standalone_preflight": "ok", "legacy_flow_head": False,
                      "legacy_v5_alignment": False,
                      "raft_student_component": args.raft_student_path,
                      "flow_refiner_checkpoint": refiner_path,
                      "rescaled_warp_scale": adapter.config.rescaled_warp_scale}))
    return dataset, paths


def build_condition(*args, **kwargs):
    # The shared V8 condition builder handles VAE encoding and cross-clip
    # memory. Its legacy-flow stats are removed: this variant has no such head.
    condition, stats = _BASE_BUILD(*args, **kwargs)
    stats.pop("legacy_flow_forward_magnitude_feature_px", None)
    stats.pop("legacy_flow_backward_magnitude_feature_px", None)
    stats.pop("v7_raft_confidence_backward_mean", None)
    stats["legacy_flow_head_present"] = 0
    stats["flow_backend"] = ("v7_student+two_scale_refinement"
                             if os.environ.get("V8R_FLOW_REFINER_PATH", "").strip()
                             else "v7_student")
    return condition, stats


def before_pipeline_call(*args, **kwargs):
    stats = _BASE_BEFORE_PIPELINE_CALL(*args, **kwargs)
    if stats.get("temporal_guidance_enabled") and os.environ.get("V8R_FLOW_REFINER_PATH", "").strip():
        stats["temporal_flow_backend"] = "v7_student+two_scale_refinement"
    return stats


def main():
    install_v8r_provider(v8_eval)
    v8_eval.RAFTGuidedDeformableBGSTCAdapter = StandaloneV8RAdapter
    v8_eval.augment_brushnet_condition_v8 = augment_brushnet_condition
    v8_eval.preflight = preflight
    v8_eval.build_v8_condition = build_condition
    temporal_eval.RAFTGuidedDeformableBGSTCAdapter = StandaloneV8RAdapter
    temporal_eval.before_pipeline_call = before_pipeline_call
    temporal_eval.main()


if __name__ == "__main__":
    main()
