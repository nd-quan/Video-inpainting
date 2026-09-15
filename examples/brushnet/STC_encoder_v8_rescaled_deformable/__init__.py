"""V8-R: rescaled RAFT pre-warp followed by native-grid DCN refinement."""

from .rescaled_raft_deformable_stc_adapter import (
    RescaledRAFTGuidedDeformableBGSTCAdapter,
    augment_brushnet_condition_v8_rescaled,
)

__all__ = [
    "RescaledRAFTGuidedDeformableBGSTCAdapter",
    "augment_brushnet_condition_v8_rescaled",
]
