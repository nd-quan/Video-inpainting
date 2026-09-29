"""Standalone two-stage V8-R experiment: RAFT distillation and temporal STC."""

from .model import StandaloneV8RAdapter, augment_brushnet_condition

__all__ = ["StandaloneV8RAdapter", "augment_brushnet_condition"]
