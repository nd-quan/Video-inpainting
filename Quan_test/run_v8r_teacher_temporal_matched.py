#!/usr/bin/env python3
"""Run teacher-only latent temporal guidance against the saved V8-R student cases."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "examples/brushnet/proposed/STC_encoder_v8_rescaled_deformable/evaluate_v8_rescaled_deformable_temporal_student.py"
METRICS = ROOT / "Quan_test/evaluate_v8_full_sequence_metrics.py"
DEFAULT_OUTPUT = ROOT / "experiments/eval_v8r_ckpt2500_temporal_teacher_matched_20260929"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def make_config(record, root, *, flow="teacher", max_clips=None, name=None, stc_adapter_path=None):
    config = json.loads(Path(record["student_config"]).read_text())
    if config["condition_flow_source"] != "student" or config["temporal_guidance_space"] != "latent":
        raise ValueError(f"Unexpected student control settings: {record['sequence']}")
    config["dataset_root"] = str(root / "original_reference" / record["split"])
    config["output_dir"] = str(root / (name or f"{record['sequence']}_teacher"))
    config["condition_flow_source"] = flow
    config["temporal_flow_source"] = flow
    if stc_adapter_path is not None:
        config["stc_adapter_path"] = str(stc_adapter_path)
    if flow == "teacher":
        config["teacher_flow_root"] = str(root / "teacher_cache")
        config["teacher_flow_split"] = "matched"
        config["temporal_teacher_flow_root"] = str(root / "teacher_cache")
        config["temporal_teacher_flow_split"] = "matched"
    if max_clips is not None:
        config["max_clips"] = max_clips
    return config


def cli(config, *, preflight=False):
    args = [sys.executable, "-u", str(EVALUATOR)]
    for key, value in config.items():
        if key in {"preflight_only", "overwrite"} or value is None or value is False:
            continue
        if isinstance(value, bool):
            args.append(f"--{key}")
        elif isinstance(value, list):
            args.extend([f"--{key}", *map(str, value)])
        else:
            args.extend([f"--{key}", str(value)])
    if preflight:
        args.append("--preflight_only")
    return args


def run_logged(command, log_path, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    env["PYTHONPATH"] = str(ROOT / "examples/brushnet") + os.pathsep + env.get("PYTHONPATH", "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} GPU {gpu} ===\n")
        log.write("Command: " + " ".join(command) + "\n")
        log.flush()
        return subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def run_case(record, root, gpu, stc_adapter_path=None):
    sequence = record["sequence"]
    output = root / f"{sequence}_teacher"
    config = make_config(record, root, stc_adapter_path=stc_adapter_path)
    clips = list(output.glob("*/clip_metrics.json"))
    status = {"sequence": sequence, "gpu": gpu, "expected_clips": record["clips"],
              "student_metrics": record["student_metrics"], "teacher_output": str(output)}
    if len(clips) != record["clips"]:
        status.update(stage="evaluating", completed_clips=len(clips))
        save_json(output / "run_status.json", status)
        code = run_logged(cli(config), output / "evaluation.log", gpu)
        clips = list(output.glob("*/clip_metrics.json"))
        if code or len(clips) != record["clips"]:
            status.update(stage="failed_evaluation", exit_code=code, completed_clips=len(clips))
            save_json(output / "run_status.json", status)
            raise RuntimeError(f"{sequence}: evaluation failed ({code}, {len(clips)}/{record['clips']} clips)")
    metric_csv = output / "metrics_full_reference/metrics.csv"
    if not metric_csv.is_file():
        status.update(stage="metrics", completed_clips=len(clips))
        save_json(output / "run_status.json", status)
        code = run_logged([sys.executable, "-u", str(METRICS), str(output)],
                          output / "metrics.log", gpu)
        if code or not metric_csv.is_file():
            status.update(stage="failed_metrics", exit_code=code)
            save_json(output / "run_status.json", status)
            raise RuntimeError(f"{sequence}: metrics failed ({code})")
    status.update(stage="complete", completed_clips=len(clips), teacher_metrics=str(metric_csv))
    save_json(output / "run_status.json", status)
    print(f"GPU {gpu}: {sequence} complete ({len(clips)} clips)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--stc-adapter-path", type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    manifest = json.loads((root / "reference_manifest.json").read_text())
    records = manifest["records"]
    if not (root / "original_reference_manifest.json").is_file():
        raise FileNotFoundError("Validated original-size student reference is missing")
    if not (root / "teacher_cache/build_report.json").is_file():
        raise FileNotFoundError("Teacher flow cache is not complete")
    for record in records:
        config = make_config(record, root, stc_adapter_path=args.stc_adapter_path)
        code = run_logged(cli(config, preflight=True), root / "preflight" / f"{record['sequence']}.log", args.gpus[0])
        if code:
            raise RuntimeError(f"Preflight failed for {record['sequence']}: {code}")
    print(f"All {len(records)} teacher preflights passed", flush=True)
    if args.preflight_only:
        return
    buckets = [[] for _ in args.gpus]
    loads = [0] * len(args.gpus)
    for record in sorted(records, key=lambda r: r["clips"], reverse=True):
        idx = min(range(len(loads)), key=lambda x: loads[x])
        buckets[idx].append(record)
        loads[idx] += record["clips"]
    assignment = {str(gpu): [record["sequence"] for record in bucket]
                  for gpu, bucket in zip(args.gpus, buckets)}
    save_json(root / "gpu_assignment.json", assignment)
    print("Assignments:", assignment, flush=True)
    def work(item):
        gpu, bucket = item
        for record in bucket:
            run_case(record, root, gpu, args.stc_adapter_path)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(work, item) for item in zip(args.gpus, buckets)]
        for future in futures:
            future.result()
    print(f"All {len(records)} teacher cases and metrics complete", flush=True)


if __name__ == "__main__":
    main()
