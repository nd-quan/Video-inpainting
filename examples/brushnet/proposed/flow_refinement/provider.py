"""Frozen refined-flow provider with the existing V7 sequence-flow contract."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

PROPOSED = Path(__file__).resolve().parent.parent
if str(PROPOSED) not in sys.path:
    sys.path.insert(0, str(PROPOSED))

from flow_refinement.model import FlowRefinementBlock, RefinerConfig
from STC_encoder_v8_raft_deformable.raft_flow_provider import (
    FrozenV7RAFTFlowProvider, V7RAFTFlowOutput, resolve_raft_student_component,
)


class FrozenRefinedFlowProvider:
    """Compose V7 coarse flow and a separately trained two-scale refiner."""

    def __init__(self, base: FrozenV7RAFTFlowProvider, checkpoint: Path,
                 *, pair_batch_size: int = 1):
        self.base = base
        self.device = base.device
        self.pair_batch_size = int(pair_batch_size)
        if self.pair_batch_size < 1:
            raise ValueError("pair_batch_size must be positive")
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu")
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported flow-refiner checkpoint format")
        actual_base = resolve_raft_student_component(base.raft_student_path)
        saved_base = resolve_raft_student_component(payload["base_raft_component"])
        if actual_base != saved_base:
            raise ValueError(f"Refiner was trained with {saved_base}, not {actual_base}")
        self.refiner = FlowRefinementBlock(RefinerConfig(**payload["model_config"]))
        self.refiner.load_state_dict(payload["model_state"], strict=True)
        self.refiner = self.refiner.to(self.device).float().eval().requires_grad_(False)
        self.checkpoint = checkpoint
        self.step = int(payload["step"])

    def metadata(self):
        result = dict(self.base.metadata())
        result.update({"flow_refiner_checkpoint": str(self.checkpoint),
                       "flow_refiner_step": self.step,
                       "flow_refiner_frozen": True})
        return result

    @torch.inference_mode()
    def predict_sequence(self, rgb_sequence: torch.Tensor) -> V7RAFTFlowOutput:
        coarse = self.base.predict_sequence(rgb_sequence)
        batch, pairs = coarse.forward.shape[:2]
        if pairs == 0:
            return coarse
        _, _, _, height, width = rgb_sequence.shape
        frames0 = rgb_sequence[:, :-1].reshape(-1, 3, height, width)
        frames1 = rgb_sequence[:, 1:].reshape(-1, 3, height, width)
        flow_f = coarse.forward.reshape(-1, 2, height, width)
        flow_b = coarse.backward.reshape(-1, 2, height, width)
        output_f, output_b = [], []
        for start in range(0, len(frames0), self.pair_batch_size):
            end = start+self.pair_batch_size
            refined_f, refined_b = self.refiner(frames0[start:end], frames1[start:end],
                                                flow_f[start:end], flow_b[start:end])
            output_f.append(refined_f)
            output_b.append(refined_b)
        forward = torch.cat(output_f).reshape(batch, pairs, 2, height, width)
        backward = torch.cat(output_b).reshape(batch, pairs, 2, height, width)
        if not torch.isfinite(forward).all() or not torch.isfinite(backward).all():
            raise FloatingPointError("Flow refiner produced non-finite values")
        return V7RAFTFlowOutput(forward, backward)


def optional_provider(base: FrozenV7RAFTFlowProvider):
    """Install checkpoint selected by V8R_FLOW_REFINER_PATH, if configured."""
    value = os.environ.get("V8R_FLOW_REFINER_PATH", "").strip()
    if not value:
        return base
    return FrozenRefinedFlowProvider(base, Path(value), pair_batch_size=base.pair_batch_size)


def install_v8r_provider(v8_module):
    """Wrap the shared V8 provider once; both condition and DDIM request it."""
    current = v8_module._provider
    if getattr(current, "_v8r_flow_refinement_wrapper", False):
        return
    cache = {}

    def provider(args, device):
        base = current(args, device)
        key = (id(base), os.environ.get("V8R_FLOW_REFINER_PATH", "").strip())
        if key not in cache:
            cache[key] = optional_provider(base)
        return cache[key]

    provider._v8r_flow_refinement_wrapper = True
    v8_module._provider = provider
