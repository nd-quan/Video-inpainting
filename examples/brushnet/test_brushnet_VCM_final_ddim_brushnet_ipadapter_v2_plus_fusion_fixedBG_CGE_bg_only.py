#!/usr/bin/env python
"""Single-frame fast CGE: direct ROI fidelity + VCM-RS QP52 on background.

This entry point reuses the null-text fixed-background single-frame runner,
including its ``--max_images`` argument.  It is intended for timing a literal
one-image CGE call; V6's STC evaluator always operates on a multi-frame clip.

The dual-region runner remains untouched.  Here, each CGE evaluation invokes
only the three finite-difference QP52 background round trips.  ROI is held by
the scheduler's direct image-space fidelity term.
"""

from __future__ import annotations

from pathlib import Path

import test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion_fixedBG_CGE_v0_2_ as dual_cge
from vcmrs_codec_adapter import VCMRSBackgroundOnlyCodec


DEFAULT_OUTPUT_ROOT = Path(
    "/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/"
    "eval_sharednoise_cge_bg_only/test_1_checkpoint-2250"
)


def main() -> None:
    dual_cge.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    dual_cge.VCMRSDualRegionCodec = VCMRSBackgroundOnlyCodec
    dual_cge.main()


if __name__ == "__main__":
    main()
