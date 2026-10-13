#!/usr/bin/env python3
"""Resume the teacher evaluation if the attached launcher ends early."""

import json
from pathlib import Path
import subprocess
import sys
import time

from run_v8r_teacher_temporal_matched import DEFAULT_OUTPUT


def running(pid):
    stat = Path(f"/proc/{pid}/stat")
    return stat.is_file() and stat.read_text().split()[2] != "Z"


def complete(root):
    records = json.loads((root / "reference_manifest.json").read_text())["records"]
    return all((root / f"{r['sequence']}_teacher/metrics_full_reference/metrics.csv").is_file()
               for r in records)


def main():
    root = DEFAULT_OUTPUT
    pid = int((root / "run.pid").read_text())
    print(f"Watching evaluation PID {pid}", flush=True)
    while running(pid):
        time.sleep(20)
    if complete(root):
        print("All cases complete; no resume required", flush=True)
        return
    print("Evaluation stopped before completion; resuming unfinished cases", flush=True)
    command = [sys.executable, "-u", str(Path(__file__).with_name("run_v8r_teacher_temporal_matched.py"))]
    for attempt in range(1, 4):
        result = subprocess.run(command)
        if result.returncode == 0 and complete(root):
            print("All cases complete after resume", flush=True)
            return
        print(f"Resume attempt {attempt} failed with exit {result.returncode}", flush=True)
        if attempt < 3:
            time.sleep(60)
    raise RuntimeError("Teacher evaluation incomplete after three resume attempts")


if __name__ == "__main__":
    main()
