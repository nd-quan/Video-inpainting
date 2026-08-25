#!/usr/bin/env python
"""Evaluate the complete V4++ checkpoint with the audited V4/V2 pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v4_flow_aligned import evaluate_flow_aligned_stc as v4_evaluator
from STC_encoder_v4pp_bg_feature.bg_focused_flow_aligned_stc_adapter import (
    BGFocusedFlowAlignedRGBSTCAdapter,
)


def main():
    # V4's evaluator already loads the complete stc_flow_model and reports
    # flow/alignment diagnostics.  Replace only its checkpoint class.
    v4_evaluator.FlowAlignedRGBSTCAdapter = BGFocusedFlowAlignedRGBSTCAdapter
    v4_evaluator.main()


if __name__ == "__main__":
    main()

