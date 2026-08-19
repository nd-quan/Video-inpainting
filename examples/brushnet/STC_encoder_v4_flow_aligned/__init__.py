"""Flow-aligned RGB-STC for clip-local T=16 training."""

from .flow_aligned_stc_adapter import (
    FlowAlignedRGBSTCAdapter,
    FlowAlignedRGBSTCOutput,
    augment_brushnet_condition,
)

__all__ = [
    "FlowAlignedRGBSTCAdapter",
    "FlowAlignedRGBSTCOutput",
    "augment_brushnet_condition",
]
