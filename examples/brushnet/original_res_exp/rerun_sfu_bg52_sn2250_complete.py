#!/usr/bin/env python3
"""Regenerate BG52 SN2250 PNGs, compute metrics, and make videos on GPUs 0/1."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


BRUSHNET = Path(__file__).resolve().parents[2]
OUTPUT = BRUSHNET / "experiments/eval_sfu_bg52_table_window_sn2250_rgb_tg"
RUNNER = Path(__file__).with_name("run_sfu_bg52_table_window_sn2250_rgb_tg.py")
POSTPROCESS = Path(__file__).with_name("postprocess_sfu_bg52_sn2250.py")
PYTHON = Path("/home/cilab/ndquan/envs/guided_diff/bin/python")
STATUS = OUTPUT / "logs/rerun_status.json"


def save_status(state: str, **extra) -> None:
    data = {"state": state, "updated_at": datetime.now().isoformat(timespec="seconds"),
            "gpus": [0, 1], **extra}
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def run_stage(name: str, command: list[str], env: dict[str, str]) -> None:
    log = OUTPUT / "logs" / f"rerun_{name}.log"
    save_status(name, log=str(log))
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n=== {datetime.now().isoformat()} {name} ===\n")
        stream.flush()
        result = subprocess.run(command, cwd=BRUSHNET, env=env,
                                stdout=stream, stderr=subprocess.STDOUT,
                                check=False)
    if result.returncode:
        save_status("failed", stage=name, exit_code=result.returncode,
                    log=str(log))
        raise RuntimeError(f"{name} exited {result.returncode}; see {log}")


def main() -> int:
    for path in (PYTHON, RUNNER, POSTPROCESS):
        if not path.is_file():
            raise FileNotFoundError(path)
    (OUTPUT / "logs").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SFU_EVAL_GPUS"] = "0,1"
    env["PYTHONUNBUFFERED"] = "1"
    run_stage("evaluation", [str(PYTHON), "-u", str(RUNNER)], env)
    run_stage("postprocess", [str(PYTHON), "-u", str(POSTPROCESS), "--gpu", "0"], env)
    summary = OUTPUT / "metrics_full_reference_no_we/summary.csv"
    videos = OUTPUT / "videos/video_manifest.json"
    if not summary.is_file() or not videos.is_file():
        raise RuntimeError("Metrics or video manifest missing after postprocessing")
    save_status("complete", summary=str(summary), video_manifest=str(videos),
                finished_at=datetime.now().isoformat(timespec="seconds"))
    print(f"BG52 SN2250 rerun complete: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
