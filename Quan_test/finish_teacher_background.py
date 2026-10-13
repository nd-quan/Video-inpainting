#!/usr/bin/env python3
"""Detached continuation after the interactive teacher runner exits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/eval_temporal_guidance_student_vs_teacher_latent_s0p1_20260929"
INTERACTIVE_PID = int(sys.argv[1])


def alive(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        return state != "Z"
    except FileNotFoundError:
        return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state = OUT / "background_status.json"
    state.write_text(json.dumps({"status": "waiting_for_interactive_runner",
                                 "interactive_pid": INTERACTIVE_PID,
                                 "supervisor_pid": os.getpid()}, indent=2) + "\n")
    while alive(INTERACTIVE_PID):
        time.sleep(20)
    python = "/home/cilab/ndquan/envs/guided_diff/bin/python"
    runner = ROOT / "Quan_test/run_student_teacher_latent_comparison.py"
    summary = ROOT / "Quan_test/summarize_student_teacher_latent_comparison.py"
    state.write_text(json.dumps({"status": "resuming_teacher", "supervisor_pid": os.getpid()}, indent=2) + "\n")
    ret = subprocess.run([python, "-u", str(runner), "--flows", "teacher",
                          "--gpus", "1", "2", "3", "4"], cwd=ROOT).returncode
    if ret == 0:
        ret = subprocess.run([python, str(summary), str(OUT)], cwd=ROOT).returncode
    state.write_text(json.dumps({"status": "complete" if ret == 0 else "failed",
                                 "returncode": ret, "finished_at_unix": time.time(),
                                 "supervisor_pid": os.getpid()}, indent=2) + "\n")
    raise SystemExit(ret)


if __name__ == "__main__":
    main()
