"""Frozen V7 RAFT-student flow provider for V8.

The V7 student is deliberately outside the checkpointable STC module.  It is
loaded once per process, receives only degraded RGB clips, and produces a
detached optical-flow prior.  This keeps V8's trainable state limited to the
STC/DCN modules and makes the inference dependency explicit.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import torch


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
V7_DIR = BRUSHNET_DIR / "STC_encoder_v7_raft_flow_distillation"
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))
if str(V7_DIR) not in sys.path:
    sys.path.insert(0, str(V7_DIR))

from raft_student import RAFTStudentFlowPredictor  # noqa: E402


@dataclass(frozen=True)
class V7RAFTFlowOutput:
    """Adjacent full-RGB flow in the common STC ``[dx,dy]`` convention."""

    forward: torch.Tensor
    backward: torch.Tensor


def resolve_raft_student_component(path_value: Union[str, Path]) -> Path:
    """Resolve a V7 run pointer, checkpoint root, or ``raft_student`` folder."""
    path = Path(path_value).expanduser().resolve()
    if path.name in {"best.json", "latest.json"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate = Path(payload["checkpoint"])
        path = candidate if candidate.is_absolute() else path.parent / candidate
        path = path.resolve()
    nested = path / "raft_student"
    if (nested / "config.json").is_file():
        path = nested
    if not (path / "config.json").is_file() or not (
        path / "pytorch_model.bin"
    ).is_file():
        raise FileNotFoundError(
            "Expected V7 RAFT student config.json and pytorch_model.bin below "
            f"{path}"
        )
    return path


class FrozenV7RAFTFlowProvider:
    """One frozen V7 student used repeatedly by one training/eval process."""

    def __init__(
        self,
        raft_student_path: Union[str, Path],
        *,
        device: Union[str, torch.device],
        pair_batch_size: int = 1,
        mixed_precision: bool = True,
    ):
        self.raft_student_path = resolve_raft_student_component(raft_student_path)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("V8's frozen V7 RAFT provider requires a CUDA device")
        # ``torch.device('cuda')`` and ``torch.device('cuda:0')`` are not
        # equal even when they name the same visible accelerator.  Materialize
        # the current index once so the strict input-device contract is stable.
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if int(pair_batch_size) < 1:
            raise ValueError("pair_batch_size must be positive")
        self.pair_batch_size = int(pair_batch_size)
        self.student = RAFTStudentFlowPredictor.from_pretrained(
            self.raft_student_path,
            map_location="cpu",
            mixed_precision=bool(mixed_precision),
        ).to(device=self.device, dtype=torch.float32)
        self.student.requires_grad_(False)
        self.student.eval()

    def metadata(self):
        config = json.loads(
            (self.raft_student_path / "config.json").read_text(encoding="utf-8")
        )
        return {
            "raft_student_component": str(self.raft_student_path),
            "raft_architecture": config.get("architecture"),
            "raft_iterations": int(config["iterations"]),
            "raft_input_normalization": config.get("input_normalization"),
            "flow_convention": config.get("flow_convention"),
            "pair_batch_size": self.pair_batch_size,
            "frozen": True,
        }

    @staticmethod
    def _validate_rgb(rgb_sequence: torch.Tensor) -> Tuple[int, int]:
        if rgb_sequence.ndim != 5 or rgb_sequence.shape[2] != 3:
            raise ValueError("rgb_sequence must have shape [B,T,3,H,W]")
        if not rgb_sequence.is_floating_point():
            raise TypeError("rgb_sequence must be floating point in [-1,1]")
        height, width = rgb_sequence.shape[-2:]
        if height % 8 or width % 8:
            raise ValueError("V7 RAFT input resolution must be divisible by eight")
        return int(height), int(width)

    @torch.inference_mode()
    def predict_sequence(self, rgb_sequence: torch.Tensor) -> V7RAFTFlowOutput:
        """Predict all local adjacent flows of a degraded RGB clip.

        Inputs remain in the V7 training normalization ``[-1,1]``.  Flow is
        returned in RGB-pixel units; V8 converts it to STC feature pixels only
        after the spatial encoder has fixed its actual output resolution.
        """
        height, width = self._validate_rgb(rgb_sequence)
        batch, frames = rgb_sequence.shape[:2]
        if rgb_sequence.device != self.device:
            raise ValueError(
                f"RGB is on {rgb_sequence.device}, but V7 RAFT is on {self.device}"
            )
        if frames <= 1:
            empty = torch.zeros(
                batch, 0, 2, height, width, device=self.device, dtype=torch.float32
            )
            return V7RAFTFlowOutput(empty, empty)
        frame0 = rgb_sequence[:, :-1].reshape(-1, 3, height, width).float()
        frame1 = rgb_sequence[:, 1:].reshape(-1, 3, height, width).float()
        # Retain fixed BatchNorm statistics even when the surrounding STC is in
        # ``train`` mode.  No gradient path is created through frozen V7.
        self.student.eval()
        forward, backward = self.student.predict_bidirectional(
            frame0,
            frame1,
            return_all=False,
            pair_batch_size=self.pair_batch_size,
        )
        pairs = frames - 1
        forward = forward.reshape(batch, pairs, 2, height, width).detach().float()
        backward = backward.reshape(batch, pairs, 2, height, width).detach().float()
        if not torch.isfinite(forward).all() or not torch.isfinite(backward).all():
            raise FloatingPointError("Frozen V7 RAFT produced non-finite flow")
        return V7RAFTFlowOutput(forward, backward)
