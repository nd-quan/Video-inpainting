#!/usr/bin/env python3
"""Recompute student metrics where only a subset of old V8-R clips remains."""

from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from imgToVideo_rgb_stc_eval import discover_sources, select_unique_frames


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/eval_v8r_ckpt2500_temporal_teacher_matched_20260929"
EVALUATOR = ROOT / "Quan_test/evaluate_v8_full_sequence_metrics.py"


def student_frames(path, sequence):
    with path.open(newline="") as file:
        return int(next(row for row in csv.DictReader(file) if row["sequence"] == sequence)["frames"])


def prepare(record):
    sequence, split = record["sequence"], record["split"]
    source = Path(record["source_dir"])
    selected = select_unique_frames(discover_sources(source, "final")[sequence], "first")
    if len(selected) != record["unique_frames"]:
        raise ValueError(f"Saved student frame count changed: {sequence}")
    first, last = record["frame_range"]
    if [frame for frame, _, _ in selected] != list(range(first, last + 1)):
        raise ValueError(f"Student frame range changed: {sequence}")
    stage = OUT / "matched_student_subset" / sequence
    clip = stage / f"{sequence}__{first:06d}-{last:06d}"
    for kind in ("final", "gt"):
        (clip / kind).mkdir(parents=True, exist_ok=True)
    for frame, final, _ in selected:
        name = f"{frame:06d}.png"
        gt = OUT / "frozen_reference" / split / sequence / "gt" / name
        if not gt.is_file():
            raise FileNotFoundError(gt)
        for kind, source_file in (("final", final), ("gt", gt)):
            target = clip / kind / name
            if target.is_symlink():
                if target.resolve() != source_file.resolve():
                    raise ValueError(f"Staged link changed: {target}")
            elif target.exists():
                raise FileExistsError(target)
            else:
                target.symlink_to(source_file.resolve())
    (clip / "clip_metrics.json").write_text(json.dumps({"video": sequence,
        "frame_ids": list(range(first, last + 1))}, indent=2) + "\n")
    (stage / "source.json").write_text(json.dumps({"student_source_dir": str(source),
        "student_full_metrics": record["student_metrics"], "selection": "first",
        "matched_teacher_frame_range": [first, last], "gt": str(OUT / "frozen_reference" / split / sequence / "gt")}, indent=2) + "\n")
    return stage


def evaluate(stage, gpu):
    csv_path = stage / "metrics_full_reference/metrics.csv"
    if csv_path.is_file():
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    with (stage / "metrics.log").open("w") as log:
        code = subprocess.run([sys.executable, "-u", str(EVALUATOR), str(stage)],
                              cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if code or not csv_path.is_file():
        raise RuntimeError(f"Matched student metrics failed: {stage} (exit {code})")


def main():
    records = json.loads((OUT / "reference_manifest.json").read_text())["records"]
    missing = [record for record in records if student_frames(Path(record["student_metrics"]), record["sequence"])
               != record["unique_frames"]]
    print("Recomputing matched student metrics:", [r["sequence"] for r in missing], flush=True)
    stages = [prepare(record) for record in missing]
    with ThreadPoolExecutor(max_workers=min(3, len(stages))) as pool:
        futures = [pool.submit(evaluate, stage, gpu) for stage, gpu in zip(stages, (0, 1, 2))]
        for future in futures:
            future.result()
    print("Matched student subset metrics complete", flush=True)


if __name__ == "__main__":
    main()
