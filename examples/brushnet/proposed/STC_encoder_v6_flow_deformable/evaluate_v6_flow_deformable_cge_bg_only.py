#!/usr/bin/env python
"""Evaluate V6 with fast CGE: direct ROI fidelity + VCM-RS QP52 on BG.

This is an intentionally separate entry point from ``evaluate_v6_flow_deformable_cge``:
the latter remains the exact train-style ROI-QP20/BG-QP52 composite.  This
variant reproduces the faster legacy CGE structure while retaining VCM-RS for
the background:

    L = ||M_ROI * (x_hat - y)||^2
        + ||M_BG * (VCM_RS_QP52(x_hat) - y)||^2

Each finite-difference CGE evaluation invokes VCM-RS three times per frame
(``x``, ``x+h``, ``x-h``), rather than six times in the exact dual-region
operator.  It is an ablation/speed-quality trade-off, not an exact match to
the two-QP training degradation.
"""

from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v6_flow_deformable import evaluate_v6_flow_deformable_cge as dual_cge
from vcmrs_codec_adapter import VCMRSBackgroundOnlyCodec


DEFAULT_OUTPUT = (
    Path(dual_cge.evaluator.PROJECT_ROOT)
    / "experiments"
    / "eval_stc_v6_deformable_cge_bg_only"
)


def main() -> None:
    # Reuse the tested V6 evaluator wiring.  Only its injected codec class and
    # default output root change, so this file cannot alter existing dual-CGE
    # jobs or their outputs.
    dual_cge.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    dual_cge.VCMRSDualRegionCodec = VCMRSBackgroundOnlyCodec
    dual_cge.main()


if __name__ == "__main__":
    main()
