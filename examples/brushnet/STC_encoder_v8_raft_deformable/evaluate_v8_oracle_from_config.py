#!/usr/bin/env python
"""Replay one saved student shard with ONLY V8 condition flow replaced.

Temporal-guidance flow stays student-derived, isolating the conditioning/DCN
ablation. This is an oracle diagnostic using clean data, not deployable inference.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def replay_arguments(config, output_dir, teacher_root, teacher_split, preflight=False):
    if config.get("condition_flow_source", "student") != "student":
        raise ValueError("Reference must be a student evaluation")
    if Path(output_dir).resolve() == Path(config["output_dir"]).resolve():
        raise ValueError("Oracle output must not overwrite student output")
    replay = dict(config)
    replay.update(output_dir=str(output_dir), overwrite=False,
                  preflight_only=preflight, condition_flow_source="teacher",
                  teacher_flow_root=str(teacher_root), teacher_flow_split=teacher_split)
    argv = []
    false_flags = {"raft_mixed_precision": "--raft_no_mixed_precision",
                   "temporal_detach_previous": "--no_temporal_detach_previous"}
    for key, value in replay.items():
        if value is None:
            continue
        flag = "--" + key
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            elif key in false_flags:
                argv.append(false_flags[key])
        elif isinstance(value, list):
            if value:
                argv.extend([flag, *map(str, value)])
        else:
            argv.extend([flag, str(value)])
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_run_config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--teacher_flow_root", type=Path, required=True)
    parser.add_argument("--teacher_flow_split", default="test")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--oracle_available_clips_only", action="store_true")
    parser.add_argument("--disable_cge", action="store_true")
    args = parser.parse_args()
    raw = args.reference_run_config.read_bytes()
    config = json.loads(raw)
    if args.disable_cge:
        config.update(cge_guidance_scale=0.0, cge_max_evals=0)
    argv = replay_arguments(config, args.output_dir, args.teacher_flow_root,
                            args.teacher_flow_split, args.preflight_only)
    if args.oracle_available_clips_only:
        argv.append("--oracle_available_clips_only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "reference_run_config": str(args.reference_run_config.resolve()),
        "reference_sha256": hashlib.sha256(raw).hexdigest(),
        "condition_flow_source": "cached_clean_teacher",
        "temporal_guidance_flow_source": "unchanged_frozen_V7_student",
        "teacher_flow_root": str(args.teacher_flow_root.resolve()),
        "teacher_flow_split": args.teacher_flow_split,
        "oracle_available_clips_only": args.oracle_available_clips_only,
        "cge_disabled": args.disable_cge,
        "changed_model_weights": False,
        "metric_names_preserved": True,
        "note": "Legacy v7_raft_* condition metric names describe teacher flow in this run.",
    }
    target = args.output_dir / "oracle_provenance.json"
    if target.exists() and json.loads(target.read_text()) != provenance:
        raise ValueError("Output contains a different oracle experiment")
    target.write_text(json.dumps(provenance, indent=2) + "\n")
    import evaluate_v8_raft_deformable_cge_temporal_bg_only as evaluator
    sys.argv = [str(Path(evaluator.__file__)), *argv]
    evaluator.main()


if __name__ == "__main__":
    main()
