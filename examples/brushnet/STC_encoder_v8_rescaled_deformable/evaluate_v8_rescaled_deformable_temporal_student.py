#!/usr/bin/env python
"""Evaluate V8-R with frozen-V7 latent/RGB temporal DDIM guidance.

This wrapper preserves the established temporal-student evaluation protocol
and replaces only the native V8 condition adapter with the V8-R rescaled
pre-warp adapter.
"""

from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v8_raft_deformable import (  # noqa: E402
    evaluate_v8_raft_deformable as native_v8_evaluator,
)
from STC_encoder_v8_raft_deformable import (  # noqa: E402
    evaluate_v8_raft_deformable_temporal_student as temporal_v8_evaluator,
)
from STC_encoder_v8_rescaled_deformable import (  # noqa: E402
    evaluate_v8_rescaled_deformable as v8r_evaluator,
)
from STC_encoder_v8_rescaled_deformable.rescaled_raft_deformable_stc_adapter import (  # noqa: E402
    RescaledRAFTGuidedDeformableBGSTCAdapter,
    augment_brushnet_condition_v8_rescaled,
)


def main() -> None:
    # Patch the condition path consumed by the temporal evaluator. The
    # diffusion scheduler and guidance implementation remain unchanged.
    native_v8_evaluator.RAFTGuidedDeformableBGSTCAdapter = (
        RescaledRAFTGuidedDeformableBGSTCAdapter
    )
    native_v8_evaluator.augment_brushnet_condition_v8 = (
        augment_brushnet_condition_v8_rescaled
    )
    native_v8_evaluator.preflight = v8r_evaluator.preflight
    temporal_v8_evaluator.RAFTGuidedDeformableBGSTCAdapter = (
        RescaledRAFTGuidedDeformableBGSTCAdapter
    )
    temporal_v8_evaluator.main()


if __name__ == "__main__":
    main()
