#!/usr/bin/env python3
"""Detached, resumable teacher-only rerun on the existing matched cache."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/eval_temporal_guidance_student_vs_teacher_latent_s0p1_20260929"
STATUS = OUT / "retest_status.json"
RUNNER = ROOT / "Quan_test/run_student_teacher_latent_comparison.py"


def write_status(**values):
    STATUS.write_text(json.dumps(values, indent=2) + "\n")


def main():
    started = time.time()
    command = [sys.executable, "-u", str(RUNNER), "--flows", "teacher",
               "--gpus", "0", "1", "2", "3", "--min-free-mib", "9000"]
    write_status(status="running", started_at_unix=started, pid=os.getpid(),
                 command=command, teacher_cache=str(OUT / "teacher_cache"))
    exit_code = 1
    for attempt in (1, 2):
        print(f"Teacher rerun attempt {attempt}", flush=True)
        exit_code = subprocess.run(command, cwd=ROOT).returncode
        if exit_code == 0:
            break
    output = OUT / "checkpoint-2250"
    complete = []
    for case in output.glob("*/teacher/run.json"):
        record = json.loads(case.read_text())
        if record.get("complete") and list(case.parent.glob("test_*/metrics_full_reference/metrics.csv")):
            complete.append(record["sequence"])
    if len(complete) != 13:
        exit_code = 1
    write_status(status="complete" if exit_code == 0 else "failed",
                 started_at_unix=started, finished_at_unix=time.time(),
                 pid=os.getpid(), returncode=exit_code,
                 completed_sequences=sorted(complete), completed_count=len(complete),
                 teacher_cache=str(OUT / "teacher_cache"))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
