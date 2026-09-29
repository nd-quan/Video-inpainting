#!/usr/bin/env python3
"""Run matched RGB- and latent-space temporal-guidance evaluations for one sequence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


BRUSHNET = Path(__file__).resolve().parents[1]
EXAMPLES = BRUSHNET / "examples" / "brushnet"
DATASET_ROOT = EXAMPLES / "dataset"
CHECKPOINT = BRUSHNET / "experiments" / "train_sharedNoise_sameBG_0.95_T8" / "checkpoint-2250"
STUDENT = BRUSHNET / "experiments" / "train_v7_raft_student_flow" / "checkpoint-0004750" / "raft_student"
INFERENCE = EXAMPLES / "test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion_fixedBG_temporal_v0.py"
METRICS = BRUSHNET / "Quan_test" / "evaluate_flat_sequence_metrics.py"


def run(command, env, log_path):
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=EXAMPLES, env=env, stdout=log, stderr=subprocess.STDOUT)
    return {"command": command, "started_at_unix": started, "elapsed_seconds": time.time() - started,
            "returncode": completed.returncode, "log": str(log_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--dataset-split", choices=("test_1", "test_2"), default="test_1")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse complete 150-frame inference outputs.")
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    dataset = DATASET_ROOT / args.dataset_split
    if not (dataset / args.sequence / "inputs").is_dir():
        raise FileNotFoundError(dataset / args.sequence / "inputs")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), LONG_TEST_ROOT=str(dataset),
               SEQUENCES=args.sequence, MAX_IMAGES="150", CHECKPOINT_DIR=str(CHECKPOINT),
               TEMPORAL_CLIP_SIZE="4", NUM_INFERENCE_STEPS="50", MAIN_NOISE_SEED="1234",
               SHARED_NOISE_SEED="6789", SHARED_BG_NOISE_STRENGTH="1",
               VARIANCE_PRESERVING_SHARED_NOISE="1", TEMPORAL_BG_MASK_MODE="pair_intersection",
               TEMPORAL_FLOW_BACKEND="v7_student", V7_RAFT_STUDENT_PATH=str(STUDENT),
               GUIDANCE_MODE="temporal", TEMPORAL_START_STEP="0", TEMPORAL_END_STEP="50",
               TEMPORAL_EVERY_N_STEPS="1", TEMPORAL_GUIDANCE_SCALE="0.1",
               TEMPORAL_DETACH_PREVIOUS="1", FORCE_REGENERATE="1", OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4")
    result_path = root / "comparison_run.json"
    results = json.loads(result_path.read_text()) if args.resume and result_path.is_file() else {
               "sequence": args.sequence, "dataset_split": args.dataset_split,
               "physical_gpu": str(args.gpu), "frame_count": 150,
               "shared_noise": {"main_seed": 1234, "shared_seed": 6789, "strength": 1.0,
                                "variance_preserving": True},
               "temporal_guidance": {"flow_backend": "v7_student", "flow_path": str(STUDENT),
                                     "clip_size": 4, "steps": 50, "window": [0, 50], "scale": 0.1,
                                     "bg_mask_mode": "pair_intersection", "detach_previous": True},
               "runs": {}}
    for space in ("latent", "rgb"):
        output = root / space
        run_env = dict(env, OUTPUT_ROOT=str(output), TEMPORAL_GUIDANCE_SPACE=space)
        generated = output / args.dataset_split / args.sequence
        frames = sorted(generated.glob("[0-9]*.png")) if generated.is_dir() else []
        reuse_inference = args.resume and len(frames) == 150
        if reuse_inference:
            results["runs"].setdefault(space, {"output_root": str(output)})
            results["runs"][space]["output_root"] = str(output)
            results["runs"][space].setdefault("inference", {"reused": True, "elapsed_seconds": None})
        else:
            inference = run([sys.executable, "-u", str(INFERENCE)], run_env,
                            root / f"{args.sequence}_{space}_inference.log")
            results["runs"][space] = {"inference": inference, "output_root": str(output)}
            if inference["returncode"]:
                result_path.write_text(json.dumps(results, indent=2) + "\n")
                raise SystemExit(inference["returncode"])
        metric_dir = output / "test_1" / "metrics_full_reference"
        if not metric_dir.is_dir():
            metric = run([sys.executable, str(METRICS), "--eval-root", str(output / args.dataset_split),
                          "--gt-root", str(dataset), "--sequences", args.sequence, "--allow-partial"], run_env,
                         root / f"{args.sequence}_{space}_metrics.log")
            results["runs"][space]["metrics"] = metric
            if metric["returncode"]:
                result_path.write_text(json.dumps(results, indent=2) + "\n")
                raise SystemExit(metric["returncode"])
    result_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
