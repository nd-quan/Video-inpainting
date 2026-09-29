"""Trainable ProPainter-RAFT student used before V6 feature deformation.

The module intentionally accepts only degraded RGB image pairs in ``[-1, 1]``.
It does not consume STC features or masks: those are used by the loss and the
later V6 integration, while RGB context remains available to RAFT matching.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn


def _import_propainter_raft(propainter_root: Path):
    """Import ProPainter's RAFT without depending on the launch CWD."""
    root = Path(propainter_root).expanduser().resolve()
    if not (root / "RAFT" / "raft.py").is_file():
        raise FileNotFoundError(f"No ProPainter RAFT implementation below {root}")
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    from RAFT.raft import RAFT  # pylint: disable=import-outside-toplevel

    return RAFT


def _state_dict(checkpoint: Path):
    """Load either a plain RAFT state dict or a wrapped training checkpoint."""
    try:
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        payload = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"RAFT checkpoint must contain a state dict: {checkpoint}")
    state = {}
    for key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        key = str(key)
        if key.startswith("module."):
            key = key[len("module.") :]
        state[key] = value
    if not state:
        raise ValueError(f"No tensor parameters found in {checkpoint}")
    return state


class RAFTStudentFlowPredictor(nn.Module):
    """A pretrained RAFT made trainable on degraded RGB pairs.

    ``predict_bidirectional`` follows the shared STC convention:

    * forward is defined on frame ``t`` and samples ``t + 1``;
    * backward is defined on frame ``t + 1`` and samples ``t``;
    * channel order is ``[dx, dy]`` in image pixels.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        propainter_root: Union[str, Path],
        raft_checkpoint: Union[str, Path],
        *,
        iterations: int = 20,
        mixed_precision: bool = True,
        freeze_batchnorm: bool = True,
    ):
        super().__init__()
        self.propainter_root = Path(propainter_root).expanduser().resolve()
        self.raft_checkpoint = Path(raft_checkpoint).expanduser().resolve()
        if not self.raft_checkpoint.is_file():
            raise FileNotFoundError(self.raft_checkpoint)
        if int(iterations) < 1:
            raise ValueError("iterations must be positive")

        raft_class = _import_propainter_raft(self.propainter_root)
        raft_args = argparse.Namespace(
            small=False,
            mixed_precision=bool(mixed_precision),
            alternate_corr=False,
            dropout=0.0,
        )
        self.raft = raft_class(raft_args)
        report = self.raft.load_state_dict(_state_dict(self.raft_checkpoint), strict=True)
        if report.missing_keys or report.unexpected_keys:
            raise RuntimeError(
                "RAFT checkpoint transfer mismatch: "
                f"missing={report.missing_keys}, unexpected={report.unexpected_keys}"
            )
        self.iterations = int(iterations)
        self.freeze_batchnorm = bool(freeze_batchnorm)
        if self.freeze_batchnorm:
            self.raft.freeze_bn()

    def train(self, mode: bool = True):
        super().train(mode)
        # RAFT's checkpoint was trained with much larger batches.  Updating
        # BatchNorm running statistics from one or two VCM pairs is unstable.
        if self.freeze_batchnorm:
            self.raft.freeze_bn()
        return self

    def config_dict(self):
        return {
            "format_version": self.FORMAT_VERSION,
            "architecture": "propainter_raft_large",
            "propainter_root": str(self.propainter_root),
            "source_checkpoint": str(self.raft_checkpoint),
            "iterations": int(self.iterations),
            "freeze_batchnorm": bool(self.freeze_batchnorm),
            "input_normalization": "RGB [0,1] -> [-1,1]",
            "flow_convention": "forward=t->t+1 on t; backward=t+1->t on t+1; [dx,dy]",
        }

    def save_pretrained(self, output_dir: Union[str, Path]) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output / "pytorch_model.bin")
        (output / "config.json").write_text(
            json.dumps(self.config_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        *,
        propainter_root: Union[str, Path, None] = None,
        raft_checkpoint: Union[str, Path, None] = None,
        map_location: str = "cpu",
        mixed_precision: bool = False,
    ) -> "RAFTStudentFlowPredictor":
        model_dir = Path(model_dir).expanduser().resolve()
        config_path = model_dir / "config.json"
        weight_path = model_dir / "pytorch_model.bin"
        if not config_path.is_file() or not weight_path.is_file():
            raise FileNotFoundError(f"Expected config.json and pytorch_model.bin in {model_dir}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model = cls(
            propainter_root=propainter_root or config["propainter_root"],
            raft_checkpoint=raft_checkpoint or config["source_checkpoint"],
            iterations=int(config["iterations"]),
            mixed_precision=bool(mixed_precision),
            freeze_batchnorm=bool(config.get("freeze_batchnorm", True)),
        )
        try:
            state = torch.load(str(weight_path), map_location=map_location, weights_only=True)
        except TypeError:
            state = torch.load(str(weight_path), map_location=map_location)
        report = model.load_state_dict(state, strict=True)
        if report.missing_keys or report.unexpected_keys:
            raise RuntimeError(
                "RAFT-student transfer mismatch: "
                f"missing={report.missing_keys}, unexpected={report.unexpected_keys}"
            )
        return model

    @staticmethod
    def _validate_pair_inputs(frame0: torch.Tensor, frame1: torch.Tensor) -> None:
        if frame0.ndim != 4 or frame1.ndim != 4 or frame0.shape != frame1.shape:
            raise ValueError("frame0/frame1 must share shape [N,3,H,W]")
        if frame0.shape[1] != 3:
            raise ValueError("RAFT student requires three-channel RGB inputs")
        if frame0.shape[-2] % 8 or frame0.shape[-1] % 8:
            raise ValueError("RAFT image height and width must be divisible by 8")
        if not frame0.is_floating_point() or not frame1.is_floating_point():
            raise TypeError("RAFT inputs must be floating-point tensors")

    def _predict_direction(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        return_all: bool,
        pair_batch_size: int,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        self._validate_pair_inputs(source, target)
        if pair_batch_size < 1:
            raise ValueError("pair_batch_size must be positive")
        chunks = []
        for start in range(0, source.shape[0], int(pair_batch_size)):
            source_chunk = source[start : start + pair_batch_size]
            target_chunk = target[start : start + pair_batch_size]
            output = self.raft(
                source_chunk,
                target_chunk,
                iters=self.iterations,
                test_mode=not return_all,
            )
            chunks.append(output if return_all else output[1])
        if not return_all:
            return torch.cat(chunks, dim=0)
        prediction_count = len(chunks[0])
        if any(len(chunk) != prediction_count for chunk in chunks):
            raise RuntimeError("RAFT returned different iteration counts across chunks")
        return [torch.cat([chunk[index] for chunk in chunks], dim=0) for index in range(prediction_count)]

    def predict_bidirectional(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        *,
        return_all: bool = False,
        pair_batch_size: int = 1,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]:
        """Predict `t->t+1` and `t+1->t` flow for a batch of image pairs."""
        forward = self._predict_direction(
            frame0, frame1, return_all=return_all, pair_batch_size=pair_batch_size
        )
        backward = self._predict_direction(
            frame1, frame0, return_all=return_all, pair_batch_size=pair_batch_size
        )
        return forward, backward

    def forward(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        *,
        return_all: bool = False,
        pair_batch_size: int = 1,
    ):
        """DDP-compatible alias for :meth:`predict_bidirectional`."""
        return self.predict_bidirectional(
            frame0,
            frame1,
            return_all=return_all,
            pair_batch_size=pair_batch_size,
        )
