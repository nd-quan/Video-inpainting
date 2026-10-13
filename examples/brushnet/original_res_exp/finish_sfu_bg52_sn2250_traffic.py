#!/usr/bin/env python3
"""Finish BG52 SN2250 Traffic on GPU 0, then score and encode all sequences."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import resume_sfu_bg48_traffic_clips as resume
import run_sfu_bg52_table_window_sn2250_rgb_tg as bg52


OUTPUT = bg52.OUTPUT
STATUS = OUTPUT / "logs/rerun_status.json"
POSTPROCESS = Path(__file__).with_name("postprocess_sfu_bg52_sn2250.py")


def save_status(state: str, **extra) -> None:
    payload = {"state": state, "updated_at": datetime.now().isoformat(timespec="seconds"),
               "gpus": [0], **extra}
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def main() -> int:
    resume.BRUSHNET = bg52.BRUSHNET
    resume.DATASET = bg52.DATASET
    resume.CHECKPOINT = bg52.CHECKPOINT
    resume.STUDENT = bg52.STUDENT
    resume.OUTPUT = OUTPUT
    resume.EVALUATOR = bg52.EVALUATOR
    resume.PYTHON = bg52.PYTHON
    resume.LOG_DIR = OUTPUT / "logs"
    resume.STATUS_PATH = resume.LOG_DIR / "status.json"
    resume.RESUME_STATUS_PATH = resume.LOG_DIR / "traffic_resume_status.json"
    save_status("traffic_resume", traffic_status=str(resume.RESUME_STATUS_PATH))
    try:
        result = resume.main()
        if result:
            save_status("failed", stage="traffic_resume", exit_code=result)
            return result
        save_status("postprocess", traffic_status=str(resume.RESUME_STATUS_PATH))
        log_path = OUTPUT / "logs/rerun_postprocess.log"
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"\n=== {datetime.now().isoformat()} BG52 postprocess ===\n")
            stream.flush()
            result = subprocess.run(
                [str(bg52.PYTHON), "-u", str(POSTPROCESS), "--gpu", "0"],
                cwd=bg52.BRUSHNET, stdout=stream, stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            save_status("failed", stage="postprocess", exit_code=result.returncode,
                        log=str(log_path))
            return result.returncode
        save_status("complete", frames=1198,
                    summary=str(OUTPUT / "metrics_full_reference_no_we/summary.csv"),
                    video_manifest=str(OUTPUT / "videos/video_manifest.json"))
        print("BG52 SN2250 Traffic, metrics, and videos complete", flush=True)
        return 0
    except Exception as exc:
        save_status("failed", stage="traffic_resume", error=repr(exc))
        raise


if __name__ == "__main__":
    sys.exit(main())
