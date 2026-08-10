"""RGB STC condition adapter for the frozen V8 BrushNet baseline."""

from .rgb_stc_adapter import (
    RGBSTCConditionAdapter,
    RGBSTCConditionOutput,
    augment_brushnet_condition,
)

__all__ = [
    "RGBSTCConditionAdapter",
    "RGBSTCConditionOutput",
    "augment_brushnet_condition",
]
