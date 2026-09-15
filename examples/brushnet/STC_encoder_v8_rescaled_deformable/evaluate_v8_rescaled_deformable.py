#!/usr/bin/env python
"""Evaluate V8-R while reusing the established non-CGE V8 protocol.

The native V8 evaluator owns dataset construction, cross-clip state, frozen
V7 RAFT provisioning, diffusion sampling, output writing, and metrics.  This
entry point changes only the STC model class/condition implementation so a
``stc_v8r_model`` checkpoint cannot silently run with native V8 warping.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v8_raft_deformable import (  # noqa: E402
    evaluate_v8_raft_deformable as native_v8_evaluator,
)
from STC_encoder_v8_rescaled_deformable.rescaled_raft_deformable_stc_adapter import (  # noqa: E402
    RescaledRAFTGuidedDeformableBGSTCAdapter,
    augment_brushnet_condition_v8_rescaled,
)


_native_preflight = native_v8_evaluator.preflight


def preflight(args):
    """Run native V8 checks, then assert the rescaled checkpoint contract."""
    dataset, paths = _native_preflight(args)
    adapter = RescaledRAFTGuidedDeformableBGSTCAdapter.from_pretrained(
        str(args.stc_adapter_path)
    )
    if int(adapter.config.rescaled_warp_scale) < 1:
        raise ValueError("V8-R checkpoint has invalid rescaled_warp_scale")
    if adapter.config.rescaled_warp_up_mode != "nearest":
        raise ValueError("V8-R checkpoint must use nearest upsampling")
    if adapter.config.rescaled_warp_down_mode != "nearest":
        raise ValueError("V8-R checkpoint must use nearest downsampling")
    if adapter.config.deform_reliability_mode != "geometric_only":
        raise ValueError("V8-R checkpoint must use geometric_only reliability")
    report = {
        "v8_rescaled_preflight": "ok",
        "checkpoint_component": str(Path(args.stc_adapter_path).resolve()),
        "rescaled_warp_scale": int(adapter.config.rescaled_warp_scale),
        "rescaled_warp_up_mode": adapter.config.rescaled_warp_up_mode,
        "rescaled_warp_down_mode": adapter.config.rescaled_warp_down_mode,
        "dcn_source": "rescaled prewarped spatial feature",
        "dcn_offset": "learned residual only; RAFT motion is not applied twice",
        "deform_reliability_mode": adapter.config.deform_reliability_mode,
        "fb_confidence_role": "diagnostic_only",
        "cge_enabled": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return dataset, paths


def main() -> None:
    # The native evaluator resolves these globals at call time. Replacing them
    # keeps its complete evaluation protocol while selecting V8-R geometry.
    native_v8_evaluator.RAFTGuidedDeformableBGSTCAdapter = (
        RescaledRAFTGuidedDeformableBGSTCAdapter
    )
    native_v8_evaluator.augment_brushnet_condition_v8 = (
        augment_brushnet_condition_v8_rescaled
    )
    native_v8_evaluator.preflight = preflight
    native_v8_evaluator.main()


if __name__ == "__main__":
    main()
