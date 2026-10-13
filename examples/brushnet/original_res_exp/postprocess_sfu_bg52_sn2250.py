#!/usr/bin/env python3
"""Score native BG52 SN2250 PNGs without WE, then encode H.264 videos."""

from __future__ import annotations

from pathlib import Path
import sys

import postprocess_sfu_bg48_bvi as workflow


BRUSHNET = Path(__file__).resolve().parents[2]
EVAL = BRUSHNET / "experiments/eval_sfu_bg52_table_window_sn2250_rgb_tg"


def video_path(job: dict) -> Path:
    return EVAL / "videos" / job["class"] / (
        f"{job['sequence']}_bg52_sn2250_sharednoise_rgb_tg.mp4"
    )


def main() -> int:
    workflow.EVAL = EVAL
    workflow.DATASET = Path(
        "/home/cilab/ndquan/vcm-rs/output/sfu_qp0_bg_table_window_test/bg52/test"
    )
    workflow.OUTPUT = EVAL / "metrics_full_reference_no_we"
    workflow.WORKFLOW_STATUS = workflow.OUTPUT / "workflow_status.json"
    workflow.METHOD_LABEL = "BG52 SN checkpoint-2250 + RGB TG"
    workflow.video_path = video_path
    return workflow.main()


if __name__ == "__main__":
    sys.exit(main())
