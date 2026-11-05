#!/usr/bin/env python3
"""Evaluate SFU BG52 with the BVI BG52 checkpoint at step 2000."""

from __future__ import annotations

import run_sfu_bg52_table_window_sn2250_rgb_tg as evaluation


evaluation.CHECKPOINT = (
    evaluation.BRUSHNET
    / "experiments/finetune_shared_noise_bvi_dvc_bg52_original_size/checkpoint-2000"
)
evaluation.OUTPUT = (
    evaluation.BRUSHNET
    / "experiments/eval_sfu_bg52_table_window_bvi_ckpt2000_rgb_tg"
)


if __name__ == "__main__":
    raise SystemExit(evaluation.main())
