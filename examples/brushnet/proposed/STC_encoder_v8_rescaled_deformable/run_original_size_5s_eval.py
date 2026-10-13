#!/usr/bin/env python3
"""Resume the 13 V8-R latent-guidance evaluations at native video sizes.

Each complete sequence stays on one GPU so cross-clip state remains ordered.
The arguments for each run come from its previous 512x512 run_config.json;
only dataset_root, output_dir, and original_size change.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time


BRUSHNET = Path(__file__).resolve().parents[4]
DATASET = BRUSHNET / "examples/brushnet/dataset/original_size"
EXPERIMENTS = BRUSHNET / "experiments"
REFERENCES = EXPERIMENTS / "eval_tg_stc"
EVALUATOR = Path(__file__).resolve().with_name("evaluate_v8_rescaled_deformable_temporal_student.py")
METRICS = BRUSHNET / "Quan_test/evaluate_v8_full_sequence_metrics.py"
DEFAULT_OUTPUT = EXPERIMENTS / "eval_v8r_original_size_5s_ckpt2500_latent_tg0p1"
PYTHON = Path("/home/cilab/ndquan/envs/guided_diff/bin/python")


def reference_config(sequence: str) -> Path:
    suffix = "" if sequence == "BasketballPass" else "_stc1p0"
    name = f"eval_v8_rescaled_x4_ckpt2500_{sequence.lower()}_latent_tg0.1{suffix}"
    return REFERENCES / name / "run_config.json"


def command_from_config(config: dict, dataset_root: Path, output_dir: Path) -> list[str]:
    config = dict(config)
    config["dataset_root"] = str(dataset_root)
    config["output_dir"] = str(output_dir)
    config["original_size"] = True
    command = [str(PYTHON), str(EVALUATOR)]
    for key, value in config.items():
        if value is None or value is False:
            continue
        if key == "raft_mixed_precision":
            # True is already the CLI default.
            continue
        if key == "temporal_detach_previous":
            command.append("--temporal_detach_previous")
            continue
        command.append("--" + key)
        if value is True:
            continue
        if isinstance(value, list):
            command.extend(map(str, value))
        else:
            command.append(str(value))
    return command


def run_logged(command: list[str], log: Path, gpu: str) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str(BRUSHNET / "examples/brushnet") + os.pathsep + env.get("PYTHONPATH", "")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n$ {' '.join(command)}\n")
        stream.flush()
        return subprocess.run(command, cwd=BRUSHNET, env=env, stdout=stream,
                              stderr=subprocess.STDOUT, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Use only after a complete successful preflight of this dataset")
    parser.add_argument("--max-clips", type=int, help="Diagnostic clip limit; skips full metrics")
    args = parser.parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        parser.error("--gpus must contain distinct GPU indices")
    manifest = json.loads((DATASET / "manifest.json").read_text())
    records = manifest["sequences"]
    if args.sequences:
        records = [record for record in records if record["sequence"] in args.sequences]
        if {record["sequence"] for record in records} != set(args.sequences):
            parser.error("Unknown sequence in --sequences")
    if not records:
        parser.error("No sequences selected")
    for record in records:
        if record["frames"] != 5 * record["fps"]:
            raise ValueError(f"Not exactly 5 seconds: {record['sequence']}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for record in records:
        sequence = record["sequence"]
        config_path = reference_config(sequence)
        config = json.loads(config_path.read_text())
        if config["include_branches"] != [sequence]:
            raise ValueError(f"Reference configuration selects wrong sequence: {config_path}")
        if config["temporal_guidance_space"] != "latent" or config["temporal_guidance_scale"] != 0.1:
            raise ValueError(f"Unexpected temporal guidance in {config_path}")
        output = output_root / sequence
        output.mkdir(exist_ok=True)
        command = command_from_config(config, DATASET / record["split"], output)
        if max(record["width"], record["height"]) > 1920:
            # ProPainter RAFT builds an all-pairs correlation volume. At 2560x1600
            # its feature grid exceeds the memory of an A6000; flow vectors are
            # scaled back to native RGB pixel units by the provider.
            command.extend(["--raft_max_side", "1024"])
            command.extend(["--denoising_batch_size", "4"])
        if args.max_clips is not None:
            command.extend(["--max_clips", str(args.max_clips)])
        jobs.append((record, config_path, command, output))
    (output_root / f"run_manifest_{os.getpid()}.json").write_text(json.dumps({
        "dataset_manifest": str(DATASET / "manifest.json"),
        "gpus": gpus,
        "jobs": [{"sequence": record["sequence"], "reference_config": str(reference),
                  "output_dir": str(output), "command": command}
                 for record, reference, command, output in jobs],
    }, indent=2) + "\n")

    # Preflight validates all dataset branches and model contracts before GPUs are occupied.
    if not args.skip_preflight:
        for record, _, command, output in jobs:
            preflight_log = output / "preflight.log"
            status = run_logged(command + ["--preflight_only"], preflight_log, gpus[0])
            print(f"PREFLIGHT {record['sequence']}: {status}", flush=True)
            if status:
                return status
    if args.preflight_only:
        return 0

    pending = queue.Queue()
    # Large clips start first so shorter runs fill the idle GPU slots later.
    for job in sorted(jobs, key=lambda j: j[0]["width"] * j[0]["height"] * j[0]["frames"], reverse=True):
        pending.put(job)

    def worker(gpu: str) -> list[dict]:
        results = []
        while True:
            try:
                record, _, command, output = pending.get_nowait()
            except queue.Empty:
                break
            sequence = record["sequence"]
            if args.max_clips is None and (output / "metrics_full_reference/metrics.csv").is_file():
                result = {"sequence": sequence, "gpu": gpu, "eval_status": 0,
                          "metric_status": 0, "elapsed_seconds": 0.0, "already_complete": True}
                results.append(result)
                pending.task_done()
                print(f"SKIP complete: {sequence}", flush=True)
                continue
            print(f"START GPU {gpu}: {sequence} ({record['width']}x{record['height']}, {record['frames']} frames)", flush=True)
            start = time.time()
            eval_status = run_logged(command, output / "eval.log", gpu)
            metric_status = None
            if eval_status == 0 and args.max_clips is None:
                metric_status = run_logged(
                    [str(PYTHON), str(METRICS), str(output)],
                    output / "metrics.log", gpu,
                )
            result = {"sequence": sequence, "gpu": gpu, "eval_status": eval_status,
                      "metric_status": metric_status, "elapsed_seconds": round(time.time() - start, 1)}
            (output / "job_status.json").write_text(json.dumps(result, indent=2) + "\n")
            results.append(result)
            print(f"FINISH GPU {gpu}: {sequence}, eval={eval_status}, metrics={metric_status}", flush=True)
            pending.task_done()
        return results

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in gpus]
        results = [item for future in as_completed(futures) for item in future.result()]
    (output_root / f"job_summary_{os.getpid()}.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0 if all(item["eval_status"] == 0 and
                    (args.max_clips is not None or item["metric_status"] == 0)
                    for item in results) else 1


if __name__ == "__main__":
    sys.exit(main())
