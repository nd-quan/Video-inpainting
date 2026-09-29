#!/usr/bin/env python3
"""Run paired latent temporal guidance with student and clean teacher flow."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples/brushnet"
DATASET = EXAMPLES / "dataset"
REFERENCE = ROOT / "experiments/eval_temporal_guidance_rgb_vs_latent_student_s0p1_20260918"
DEFAULT_OUT = ROOT / "experiments/eval_temporal_guidance_student_vs_teacher_latent_s0p1_20260929"
DEFAULT_CKPT = ROOT / "experiments/train_sharedNoise_sameBG_0.95_T8/checkpoint-2250"
STUDENT = ROOT / "experiments/train_v7_raft_student_flow/checkpoint-0004750/raft_student"
INFERENCE = EXAMPLES / "test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion_fixedBG_temporal_v0.py"
METRICS = ROOT / "Quan_test/evaluate_flat_sequence_metrics.py"


def run(command, env, log_path):
    started = time.time()
    with log_path.open("w") as log:
        process = subprocess.run(command, cwd=EXAMPLES, env=env, stdout=log, stderr=subprocess.STDOUT)
    return {"command": command, "started_at_unix": started,
            "elapsed_seconds": time.time() - started, "returncode": process.returncode,
            "log": str(log_path)}


def execute(case, gpu, root, checkpoint, teacher):
    seq, split, flow = case
    case_dir = root / checkpoint.name / seq / flow
    case_dir.mkdir(parents=True, exist_ok=True)
    status_path = case_dir / "run.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    output = case_dir / split / seq
    metadata = output / "source_metadata.json"
    images = list(output.glob("[0-9]*.png"))
    expected = json.loads(next((REFERENCE / seq).glob(f"latent/{split}/{seq}/source_metadata.json")).read_text())["frame_count"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), LONG_TEST_ROOT=str(DATASET / split),
               SEQUENCES=seq, MAX_IMAGES="150", OUTPUT_ROOT=str(case_dir),
               CHECKPOINT_DIR=str(checkpoint), TEMPORAL_CLIP_SIZE="4", NUM_INFERENCE_STEPS="50",
               MAIN_NOISE_SEED="1234", SHARED_NOISE_SEED="6789",
               SHARED_BG_NOISE_STRENGTH="1", VARIANCE_PRESERVING_SHARED_NOISE="1",
               TEMPORAL_BG_MASK_MODE="pair_intersection", TEMPORAL_FLOW_BACKEND=("v7_student" if flow == "student" else "teacher_cache"),
               V7_RAFT_STUDENT_PATH=str(STUDENT), TEACHER_FLOW_ROOT=str(teacher),
               TEACHER_FLOW_SPLIT="matched", GUIDANCE_MODE="temporal",
               TEMPORAL_START_STEP="0", TEMPORAL_END_STEP="50", TEMPORAL_EVERY_N_STEPS="1",
               TEMPORAL_GUIDANCE_SCALE="0.1", TEMPORAL_DETACH_PREVIOUS="1",
               TEMPORAL_GUIDANCE_SPACE="latent", FORCE_REGENERATE="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    env["PYTHONPATH"] = str(EXAMPLES / "proposed") + os.pathsep + env.get("PYTHONPATH", "")
    status.update(sequence=seq, dataset_split=split, flow=flow, physical_gpu=gpu,
                  checkpoint=str(checkpoint), student_checkpoint=str(STUDENT),
                  teacher_cache=str(teacher), expected_frames=expected,
                  settings={"space": "latent", "steps": 50, "window": [0, 50],
                            "scale": 0.1, "clip_size": 4, "bg_mask": "pair_intersection",
                            "main_seed": 1234, "shared_seed": 6789})
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    if len(images) != expected or not metadata.is_file():
        print(f"GPU {gpu} START {seq} {flow}", flush=True)
        status["inference"] = run([sys.executable, "-u", str(INFERENCE)], env, case_dir / "inference.log")
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        if status["inference"]["returncode"]:
            raise RuntimeError(f"Inference failed: {case_dir}")
    if len(list(output.glob("[0-9]*.png"))) != expected:
        raise RuntimeError(f"Incomplete frames: {output}")
    if not (case_dir / split / "metrics_full_reference/metrics.csv").is_file():
        status["metrics"] = run([sys.executable, str(METRICS), "--eval-root", str(case_dir / split),
                                 "--gt-root", str(DATASET / split), "--sequences", seq,
                                 "--allow-partial"], env, case_dir / "metrics.log")
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        if status["metrics"]["returncode"]:
            raise RuntimeError(f"Metrics failed: {case_dir}")
    status["complete"] = True
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(f"GPU {gpu} DONE {seq} {flow}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--flows", nargs="+", choices=("student", "teacher"),
                        default=["student", "teacher"])
    parser.add_argument("--sequences", nargs="+", default=None)
    args = parser.parse_args()
    root = args.output_root.resolve()
    teacher = root / "teacher_cache"
    if "teacher" in args.flows and not (teacher / "build_report.json").is_file():
        raise FileNotFoundError(f"Complete teacher cache required: {teacher}")
    if not args.checkpoint.is_dir() or not STUDENT.is_dir():
        raise FileNotFoundError("Model checkpoint or student checkpoint missing")
    cases = queue.Queue()
    for seq_dir in sorted(REFERENCE.iterdir()):
        if seq_dir.is_dir():
            if args.sequences is not None and seq_dir.name not in args.sequences:
                continue
            match = list(seq_dir.glob(f"latent/test_*/{seq_dir.name}/source_metadata.json"))
            if len(match) != 1:
                raise ValueError(seq_dir)
            split = json.loads(match[0].read_text())["split"]
            for flow in args.flows:
                cases.put((seq_dir.name, split, flow))
    def worker(gpu):
        failures = []
        while not cases.empty():
            try:
                case = cases.get_nowait()
            except queue.Empty:
                break
            try:
                execute(case, gpu, root, args.checkpoint.resolve(), teacher)
            except Exception as error:
                failures.append((case, str(error)))
                print(f"GPU {gpu} FAILED {case}: {error}", flush=True)
            finally:
                cases.task_done()
        return failures
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        failures = [item for batch in pool.map(worker, args.gpus) for item in batch]
    if failures:
        raise RuntimeError(f"{len(failures)} failed cases: {failures}")


if __name__ == "__main__":
    main()
