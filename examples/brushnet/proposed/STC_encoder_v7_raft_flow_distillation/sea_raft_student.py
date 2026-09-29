"""Trainable SEA-RAFT student for V7 clean-to-degraded flow distillation."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn


def _import_sea_raft(sea_raft_root: Path):
    """Import SEA-RAFT's top-level core module from an explicit repository."""
    core_root = sea_raft_root / "core"
    raft_path = core_root / "raft.py"
    if not raft_path.is_file():
        raise FileNotFoundError(raft_path)
    core_string = str(core_root)
    if core_string not in sys.path:
        sys.path.insert(0, core_string)
    module = importlib.import_module("raft")
    module_path = Path(module.__file__).resolve()
    if module_path != raft_path.resolve():
        raise RuntimeError(
            "Top-level module 'raft' was already imported from another repository: "
            f"{module_path}; expected {raft_path}"
        )
    return module.RAFT


def _state_dict(checkpoint: Path) -> Dict[str, torch.Tensor]:
    if checkpoint.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError("SEA-RAFT safetensors checkpoints require `pip install safetensors`.") from exc
        payload = load_file(str(checkpoint), device="cpu")
    else:
        try:
            payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.0
            payload = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"SEA-RAFT checkpoint must contain a state dict: {checkpoint}")
    state = {
        str(key)[len("module.") :] if str(key).startswith("module.") else str(key): value
        for key, value in payload.items()
        if torch.is_tensor(value)
    }
    # SEA-RAFT's BasicBlock registers shortcut BatchNorm under both ``bn3``
    # and ``downsample.1``.  Official safetensors checkpoints retain only
    # ``bn3`` to avoid duplicating shared tensors.
    for key, value in tuple(state.items()):
        if ".bn3." in key:
            state.setdefault(key.replace(".bn3.", ".downsample.1."), value)
    if not state:
        raise ValueError(f"No tensor parameters found in {checkpoint}")
    return state


class SEAStudentFlowPredictor(nn.Module):
    """A V7-compatible, trainable SEA-RAFT student.

    V7 images are normalized to ``[-1, 1]``.  SEA-RAFT instead accepts RGB
    values in ``[0, 255]`` and normalizes internally, so the conversion occurs
    at this module boundary.  Both returned directions use V7's [dx,dy]
    pixel-flow convention.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        sea_raft_root: Union[str, Path],
        sea_raft_cfg: Union[str, Path],
        sea_raft_checkpoint: Union[str, Path],
        *,
        iterations: int = 4,
        freeze_batchnorm: bool = True,
    ):
        super().__init__()
        self.sea_raft_root = Path(sea_raft_root).expanduser().resolve()
        self.sea_raft_cfg = Path(sea_raft_cfg).expanduser().resolve()
        self.sea_raft_checkpoint = Path(sea_raft_checkpoint).expanduser().resolve()
        if not self.sea_raft_cfg.is_file():
            raise FileNotFoundError(self.sea_raft_cfg)
        if not self.sea_raft_checkpoint.is_file():
            raise FileNotFoundError(self.sea_raft_checkpoint)
        if int(iterations) < 1:
            raise ValueError("iterations must be positive")

        config = json.loads(self.sea_raft_cfg.read_text(encoding="utf-8"))
        self.sea_raft_config = dict(config)
        model_args = SimpleNamespace(**config)
        # The full flow checkpoint below initializes every parameter.  Do not
        # trigger torchvision's unrelated ImageNet download during V7 setup.
        model_args.init_weight = False
        raft_class = _import_sea_raft(self.sea_raft_root)
        self.raft = raft_class(model_args)
        report = self.raft.load_state_dict(_state_dict(self.sea_raft_checkpoint), strict=True)
        if report.missing_keys or report.unexpected_keys:
            raise RuntimeError(
                "SEA-RAFT checkpoint transfer mismatch: "
                f"missing={report.missing_keys}, unexpected={report.unexpected_keys}"
            )
        self.iterations = int(iterations)
        self.freeze_batchnorm = bool(freeze_batchnorm)
        if self.freeze_batchnorm:
            self._freeze_batchnorm()

    def _freeze_batchnorm(self) -> None:
        for module in self.raft.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                # Keep affine parameters trainable; only freeze unstable
                # running-statistic updates from V7's small pair batches.
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_batchnorm:
            self._freeze_batchnorm()
        return self

    def config_dict(self) -> Dict:
        return {
            "format_version": self.FORMAT_VERSION,
            "architecture": "sea_raft",
            "sea_raft_root": str(self.sea_raft_root),
            "sea_raft_cfg": str(self.sea_raft_cfg),
            "sea_raft_config": self.sea_raft_config,
            "source_checkpoint": str(self.sea_raft_checkpoint),
            "iterations": int(self.iterations),
            "freeze_batchnorm": bool(self.freeze_batchnorm),
            "input_normalization": "V7 RGB [-1,1] -> SEA-RAFT RGB [0,255]",
            "flow_convention": "forward=t->t+1 on t; backward=t+1->t on t+1; [dx,dy]",
        }

    def save_pretrained(self, output_dir: Union[str, Path]) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output / "pytorch_model.bin")
        (output / "config.json").write_text(
            json.dumps(self.config_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Union[str, Path],
        *,
        sea_raft_root: Union[str, Path, None] = None,
        sea_raft_cfg: Union[str, Path, None] = None,
        sea_raft_checkpoint: Union[str, Path, None] = None,
        map_location: str = "cpu",
    ) -> "SEAStudentFlowPredictor":
        model_dir = Path(model_dir).expanduser().resolve()
        config_path = model_dir / "config.json"
        weight_path = model_dir / "pytorch_model.bin"
        if not config_path.is_file() or not weight_path.is_file():
            raise FileNotFoundError(f"Expected config.json and pytorch_model.bin in {model_dir}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("architecture") != "sea_raft":
            raise ValueError(f"Not a saved SEA-RAFT V7 student: {config_path}")
        model = cls(
            sea_raft_root=sea_raft_root or config["sea_raft_root"],
            sea_raft_cfg=sea_raft_cfg or config["sea_raft_cfg"],
            sea_raft_checkpoint=sea_raft_checkpoint or config["source_checkpoint"],
            iterations=int(config["iterations"]),
            freeze_batchnorm=bool(config.get("freeze_batchnorm", True)),
        )
        try:
            state = torch.load(str(weight_path), map_location=map_location, weights_only=True)
        except TypeError:  # PyTorch < 2.0
            state = torch.load(str(weight_path), map_location=map_location)
        report = model.load_state_dict(state, strict=True)
        if report.missing_keys or report.unexpected_keys:
            raise RuntimeError(
                "Saved SEA-RAFT student transfer mismatch: "
                f"missing={report.missing_keys}, unexpected={report.unexpected_keys}"
            )
        return model

    @staticmethod
    def _validate_pair_inputs(frame0: torch.Tensor, frame1: torch.Tensor) -> None:
        if frame0.ndim != 4 or frame1.ndim != 4 or frame0.shape != frame1.shape:
            raise ValueError("frame0/frame1 must share shape [N,3,H,W]")
        if frame0.shape[1] != 3:
            raise ValueError("SEA-RAFT student requires three-channel RGB inputs")
        if frame0.shape[-2] % 8 or frame0.shape[-1] % 8:
            raise ValueError("SEA-RAFT image height and width must be divisible by 8")
        if not frame0.is_floating_point() or not frame1.is_floating_point():
            raise TypeError("SEA-RAFT inputs must be floating-point tensors")

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
            # Clamp only for numerical robustness at the data/model boundary;
            # normal V7 inputs already lie exactly in [-1, 1].
            source_chunk = (source[start : start + pair_batch_size].clamp(-1, 1) + 1.0) * 127.5
            target_chunk = (target[start : start + pair_batch_size].clamp(-1, 1) + 1.0) * 127.5
            output = self.raft(source_chunk, target_chunk, iters=self.iterations, test_mode=True)
            chunks.append(output["flow"] if return_all else output["final"])
        if not return_all:
            return torch.cat(chunks, dim=0)
        prediction_count = len(chunks[0])
        if any(len(chunk) != prediction_count for chunk in chunks):
            raise RuntimeError("SEA-RAFT returned different iteration counts across chunks")
        return [torch.cat([chunk[index] for chunk in chunks], dim=0) for index in range(prediction_count)]

    def predict_bidirectional(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        *,
        return_all: bool = False,
        pair_batch_size: int = 1,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]:
        forward = self._predict_direction(frame0, frame1, return_all=return_all, pair_batch_size=pair_batch_size)
        backward = self._predict_direction(frame1, frame0, return_all=return_all, pair_batch_size=pair_batch_size)
        return forward, backward

    def forward(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        *,
        return_all: bool = False,
        pair_batch_size: int = 1,
    ):
        return self.predict_bidirectional(
            frame0, frame1, return_all=return_all, pair_batch_size=pair_batch_size
        )
