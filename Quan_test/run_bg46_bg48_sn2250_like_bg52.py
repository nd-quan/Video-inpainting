#!/usr/bin/env python3
"""Evaluate one SFU BG split using the saved final BG52 configuration."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_latent_tg0p1_v7student_512/run_config.json"
GROUPS = (
    "Traffic,BQTerrace,BQMall,BQSquare",
    "BasketballDrive,Cactus,BasketballDrill,PartyScene,Kimono",
    "BasketballPass,BlowingBubbles,RaceHorsesC,RaceHorsesD,ParkScene",
)


def save_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bg", type=int, choices=(46, 48), required=True)
    parser.add_argument("--gpus", type=int, nargs=3, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(REFERENCE.read_text())
    dataset = Path(config["dataset"].replace("/bg52/", "/bg{}/".format(args.bg)))
    output = REFERENCE.parent.parent / "eval_bg{}_sn2250_latent_tg0p1_v7student_512".format(args.bg)
    config.update(dataset=str(dataset), gpus=args.gpus,
                  launcher=str(Path(__file__).resolve().relative_to(ROOT)),
                  source_config=str(REFERENCE),
                  temporal_clip_stride=config.get("temporal_clip_stride", config["temporal_clip_size"]))
    expected = {}
    for source in sorted((dataset / "input").glob("Class_*/*")):
        relative = source.relative_to(dataset / "input")
        names = {p.name for p in source.glob("*.png")}
        if not names:
            raise ValueError("Empty input: " + str(source))
        for kind in ("mask", "GT"):
            if names != {p.name for p in (dataset / kind / relative).glob("*.png")}:
                raise ValueError("Input/{} mismatch: {}".format(kind, relative))
        expected[relative] = names
    if len(expected) != config["expected_sequences"] or sum(map(len, expected.values())) != config["expected_frames"]:
        raise ValueError("Dataset counts do not match reference")
    if {p.name for p in expected} != set(",".join(GROUPS).split(",")):
        raise ValueError("Sequence assignment mismatch")
    for path in (ROOT / config["evaluator"], Path(config["checkpoint"]) / "brushnet",
                 Path(config["checkpoint"]) / "ipadapter", Path(config["flow_checkpoint"])):
        if not path.exists():
            raise FileNotFoundError(path)
    logs = output / "terminal_logs"
    logs.mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing run config differs: " + str(config_path))
    save_json(config_path, config)
    # Remove inherited evaluator overrides so defaults match the original launcher.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("TEMPORAL_", "CGE_", "TEACHER_FLOW_"))
           and k not in ("BASE_MODEL_PATH", "SEQUENCES")}
    env.update({
        "LONG_TEST_ROOT": str(dataset), "OUTPUT_ROOT": str(output),
        "CHECKPOINT_DIR": config["checkpoint"], "V7_RAFT_STUDENT_PATH": config["flow_checkpoint"],
        "DISCOVER_HIERARCHICAL_SEQUENCES": "1", "MAX_IMAGES": str(config["max_images"]),
        "FORCE_REGENERATE": "0", "GUIDANCE_MODE": config["guidance_mode"],
        "TEMPORAL_FLOW_BACKEND": config["temporal_flow_backend"],
        "TEMPORAL_GUIDANCE_SPACE": config["temporal_guidance_space"],
        "TEMPORAL_GUIDANCE_SCALE": str(config["temporal_guidance_scale"]),
        "TEMPORAL_START_STEP": str(config["temporal_window"][0]),
        "TEMPORAL_END_STEP": str(config["temporal_window"][1]),
        "TEMPORAL_EVERY_N_STEPS": str(config["temporal_every_n_steps"]),
        "TEMPORAL_CLIP_SIZE": str(config["temporal_clip_size"]),
        "TEMPORAL_CLIP_STRIDE": str(config["temporal_clip_stride"]),
        "TEMPORAL_BG_MASK_MODE": config["temporal_bg_mask_mode"],
        "TEMPORAL_DETACH_PREVIOUS": str(int(config["temporal_detach_previous"])),
        "NUM_INFERENCE_STEPS": str(config["inference_steps"]),
        "SHARED_BG_NOISE_STRENGTH": str(config["shared_bg_noise_strength"]),
        "VARIANCE_PRESERVING_SHARED_NOISE": str(int(config["variance_preserving_shared_noise"])),
        "MAIN_NOISE_SEED": str(config["main_noise_seed"]),
        "SHARED_NOISE_SEED": str(config["shared_noise_seed"]),
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(ROOT / "examples/brushnet/proposed") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "PREFLIGHT_ONLY": "0",
    })
    evaluator = [sys.executable, "-u", str(ROOT / config["evaluator"])]
    with (logs / "preflight.log").open("w") as log:
        subprocess.run(evaluator, cwd=ROOT, env=dict(env, PREFLIGHT_ONLY="1"),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print("Preflight OK: bg{}, {} sequences, {} frames".format(
        args.bg, len(expected), sum(map(len, expected.values()))), flush=True)
    if args.preflight_only:
        return 0
    status = {"state": "inference", "pid": os.getpid(), "gpus": args.gpus,
              "started_at": datetime.now().astimezone().isoformat()}
    status_path = output / "status.json"
    save_json(status_path, status)
    try:
        def worker(gpu, sequences):
            with (logs / "gpu{}.log".format(gpu)).open("a") as log:
                result = subprocess.run(evaluator, cwd=ROOT,
                                        env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu), SEQUENCES=sequences),
                                        stdout=log, stderr=subprocess.STDOUT)
            print("GPU {} exit={}: {}".format(gpu, result.returncode, sequences), flush=True)
            return result.returncode

        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(worker, args.gpus, GROUPS))
        if any(results):
            raise RuntimeError("Worker failures: " + str(results))
        for relative, names in expected.items():
            folder = output / "test" / relative
            if {p.name for p in folder.glob("*.png")} != names:
                raise ValueError("Output frame mismatch: " + str(relative))
            for name in names:
                with Image.open(folder / name) as image:
                    if image.size != tuple(config["output_size"]):
                        raise ValueError("Output size mismatch: " + str(folder / name))
        status.update(state="metrics", verified_sequences=len(expected),
                      verified_frames=sum(map(len, expected.values())))
        save_json(status_path, status)
        metrics_dir = output / "test/metrics_full_reference"
        if not (metrics_dir / "metrics.csv").is_file():
            with (logs / "metrics.log").open("a") as log:
                subprocess.run([
                    sys.executable, "-u", str(ROOT / "Quan_test/evaluate_flat_sequence_metrics.py"),
                    "--eval-root", str(output / "test"), "--gt-root", str(dataset), "--hierarchical",
                    "--sequences", *sorted(p.name for p in expected),
                ], cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=str(args.gpus[-1])),
                    stdout=log, stderr=subprocess.STDOUT, check=True)
        status.update(state="complete", metrics=str(metrics_dir / "metrics.csv"))
        print("Inference and metrics complete: " + str(output), flush=True)
    except Exception as error:
        status.update(state="failed", error=str(error))
        raise
    finally:
        status["updated_at"] = datetime.now().astimezone().isoformat()
        save_json(status_path, status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
