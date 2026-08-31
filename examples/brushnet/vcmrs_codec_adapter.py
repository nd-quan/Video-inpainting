#!/usr/bin/env python3
"""One-frame VCM-RS round trip for CGE.

The adapter deliberately runs VCM-RS in a child process.  VCM-RS owns global
codec/model state and starts helper Python processes, so importing its encoder
inside the BrushNet process is both fragile and likely to contaminate CUDA
state.

Input/output tensors use RGB in [0, 1].  Two profiles are available:

* ``vtm_only`` explicitly bypasses every optional VCM tool and is the closest
  replacement for the old VVC-only CGE operator.
* ``train_match`` loads one frozen ROI/BG descriptor and retains the spatial
  retargeting/restoration plus NNLF used to create the independent-image
  training data.

EncFormatAdapter and DecFormatAdapter stay enabled in both profiles because
they provide the required PNG <-> YUV conversion around VTM.
"""

from __future__ import annotations

import argparse
import configparser
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from PIL import Image


DEFAULT_VCMRS_ROOT = Path("/home/cilab/ndquan/vcm-rs")
DEFAULT_VCMRS_PYTHON = Path("/home/cilab/ndquan/envs/vcm/bin/python")


class VCMRSCodecError(RuntimeError):
    """Raised when a VCM-RS one-frame round trip cannot be completed."""


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _tail(text: str, line_count: int = 80) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-line_count:])


@dataclass
class VCMRSOneFrameCodec:
    """Callable VVC-like degradation operator backed by VCM-RS/VTM.

    ``roundtrip01`` accepts one RGB CHW tensor (or a batch of exactly one) in
    [0, 1] and returns the encoder-side reconstruction with the same
    shape/device/dtype.  The encoder already reconstructs the coded frame, so
    a separate invocation of ``vcmrs.decoder`` is intentionally omitted.
    """

    vcmrs_root: Path = DEFAULT_VCMRS_ROOT
    python_executable: Path = DEFAULT_VCMRS_PYTHON
    quality: int = 52
    frame_rate: int = 30
    profile: str = "vtm_only"
    roi_descriptor: Optional[Path] = None
    timeout_seconds: float = 600.0
    keep_artifacts: bool = False
    artifact_root: Optional[Path] = None
    max_parallel: int = 1
    cuda_visible_devices: Optional[str] = None
    _semaphore: threading.BoundedSemaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.vcmrs_root = Path(self.vcmrs_root).expanduser().resolve()
        self.python_executable = Path(self.python_executable).expanduser().resolve()
        if self.artifact_root is not None:
            self.artifact_root = Path(self.artifact_root).expanduser().resolve()
        if self.roi_descriptor is not None:
            self.roi_descriptor = Path(self.roi_descriptor).expanduser().resolve()
        if not self.vcmrs_root.is_dir():
            raise FileNotFoundError(f"VCM-RS root does not exist: {self.vcmrs_root}")
        if not self.python_executable.is_file():
            raise FileNotFoundError(f"VCM-RS Python does not exist: {self.python_executable}")
        if not 0 <= self.quality <= 100:
            raise ValueError(f"quality must be in [0, 100], got {self.quality}")
        if self.profile not in {"vtm_only", "train_match"}:
            raise ValueError(
                f"profile must be 'vtm_only' or 'train_match', got {self.profile!r}"
            )
        if self.roi_descriptor is not None and not self.roi_descriptor.is_file():
            raise FileNotFoundError(f"ROI/BG descriptor does not exist: {self.roi_descriptor}")
        if self.max_parallel < 1:
            raise ValueError(f"max_parallel must be >= 1, got {self.max_parallel}")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._semaphore = threading.BoundedSemaphore(self.max_parallel)

    def set_roi_descriptor(self, descriptor: Path) -> None:
        """Select the frozen key-0 descriptor before processing one source frame."""

        descriptor = Path(descriptor).expanduser().resolve()
        if not descriptor.is_file():
            raise FileNotFoundError(f"ROI/BG descriptor does not exist: {descriptor}")
        self.roi_descriptor = descriptor

    @classmethod
    def from_env(cls) -> "VCMRSOneFrameCodec":
        """Build the adapter from ``CGE_VCMRS_*`` environment variables."""

        artifact_root = os.environ.get("CGE_VCMRS_ARTIFACT_ROOT")
        cuda_visible_devices = os.environ.get("CGE_VCMRS_CUDA_VISIBLE_DEVICES")
        roi_descriptor = os.environ.get("CGE_VCMRS_DESCRIPTOR")
        return cls(
            vcmrs_root=Path(os.environ.get("CGE_VCMRS_ROOT", str(DEFAULT_VCMRS_ROOT))),
            python_executable=Path(
                os.environ.get("CGE_VCMRS_PYTHON", str(DEFAULT_VCMRS_PYTHON))
            ),
            quality=int(os.environ.get("CGE_VCMRS_QUALITY", "52")),
            frame_rate=int(os.environ.get("CGE_VCMRS_FRAME_RATE", "30")),
            profile=os.environ.get("CGE_VCMRS_PROFILE", "vtm_only"),
            roi_descriptor=Path(roi_descriptor) if roi_descriptor else None,
            timeout_seconds=float(os.environ.get("CGE_VCMRS_TIMEOUT", "600")),
            keep_artifacts=_env_flag("CGE_VCMRS_KEEP_ARTIFACTS"),
            artifact_root=Path(artifact_root) if artifact_root else None,
            max_parallel=int(os.environ.get("CGE_VCMRS_MAX_PARALLEL", "1")),
            cuda_visible_devices=cuda_visible_devices,
        )

    def _new_job_dir(self, call_key: str) -> Path:
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in call_key)
        safe_key = safe_key[:80] or "frame"
        parent = self.artifact_root
        if parent is not None:
            parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=f"cge_vcmrs_{safe_key}_", dir=parent))

    def _encoder_options(self, job_dir: Path) -> Dict[str, str]:
        """Return a fully pinned profile; do not rely on mutable VCM defaults."""

        options = {
            "output_dir": str(job_dir / "output"),
            "working_dir": str(job_dir / "work"),
            "output_bitstream_fname": "bitstream/coded.bin",
            "output_recon_fname": "recon/coded",
            "Configuration": "AllIntra",
            "IntraPeriod": "1",
            "FrameRate": str(self.frame_rate),
            "FramesToBeEncoded": "1",
            "FrameSkip": "0",
            "InputBitDepth": "8",
            "OutputBitDepth": "10",
            "InputChromaFormat": "420",
            "OutputChromaFormat": "input",
            "quality": str(self.quality),
            "InnerCodec": "VTM",
            "IntraHumanAdapter": "0" if self.profile == "vtm_only" else "1",
            "InterMachineAdapter": "0",
            "TemporalResamplingAdaptiveFlag": "0",
            "InterpolationControlEnableFlag": "0",
            "RoIRetargetingMode": "off" if self.profile == "vtm_only" else "sequence",
            "SpatialDescriptorMode": "NoDescriptor",
            "DecNeuralNetworkBitDepthRestoration": (
                "Bypass"
                if self.profile == "vtm_only"
                else "Neural_Network_Bit_Depth_Restoration"
            ),
            "DecChromaSynthesis": "Bypass",
            "EncChromaAnalysis": "Bypass",
            "EncChromaRemoval": "Bypass",
            "DecBitDepthRestoration": "Bypass",
            "EncBitDepthTruncation": "Bypass",
            "DecTemporalSynthesis": "Bypass",
            "EncTemporalDownsampling": "Bypass",
            "DecSpatialResampling": "Bypass",
            "EncSpatialDownsampling": "Bypass",
            "DecSpatialRestoration": (
                "Bypass" if self.profile == "vtm_only" else "Spatial_Restoration"
            ),
            "EncSpatialRetargeting": (
                "Bypass" if self.profile == "vtm_only" else "Spatial_Retargeting"
            ),
            "EncSpatialRetargetingScaleAdjust": "Bypass",
            "NnlfSwitch": "Bypass" if self.profile == "vtm_only" else "NnlfSliceBased",
            "PostFilter": "Bypass",
            "VCMBitStructOn": "1",
        }
        if self.profile == "train_match":
            if self.roi_descriptor is None:
                raise ValueError(
                    "train_match requires a frozen one-frame descriptor. Set "
                    "CGE_VCMRS_DESCRIPTOR or call set_roi_descriptor() before inference."
                )
            options.update(
                {
                    "RoIDescriptorMode": "load",
                    "RoIDescriptor": str(self.roi_descriptor),
                    "RoIAccumulationPeriod": "-1",
                }
            )
        return options

    def _write_encoder_ini(self, ini_path: Path, input_path: Path, job_dir: Path) -> None:
        # VCM-RS treats every non-default INI section name as an input argv.
        # Quote it because io_utils.parse_ini_file applies shlex.split(section).
        input_section = shlex.quote(str(input_path))
        parser = configparser.ConfigParser(interpolation=None)
        parser.optionxform = str
        parser["default"] = self._encoder_options(job_dir)
        parser[input_section] = {}
        with ini_path.open("w", encoding="utf-8") as stream:
            parser.write(stream)

    def _subprocess_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        vcm_bin = str(self.python_executable.parent)
        old_path = env.get("PATH", "")
        env["PATH"] = vcm_bin if not old_path else vcm_bin + os.pathsep + old_path
        old_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(self.vcmrs_root)
            if not old_pythonpath
            else str(self.vcmrs_root) + os.pathsep + old_pythonpath
        )
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        return env

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _run_encoder(self, ini_path: Path, job_dir: Path) -> None:
        command = [
            str(self.python_executable),
            "-m",
            "vcmrs.encoder",
            "--logfile",
            str(job_dir / "encoder.log"),
            str(ini_path),
        ]
        process = subprocess.Popen(
            command,
            cwd=self.vcmrs_root,
            env=self._subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            self._terminate_process_group(process)
            stdout, stderr = process.communicate()
            raise VCMRSCodecError(
                "VCM-RS encoder timed out. Artifacts were preserved.\n"
                f"timeout={self.timeout_seconds}s\n"
                f"job_dir={job_dir}\n"
                f"command={shlex.join(command)}\n"
                f"stdout_tail:\n{_tail(stdout)}\n"
                f"stderr_tail:\n{_tail(stderr)}"
            ) from exc

        (job_dir / "stdout.log").write_text(stdout, encoding="utf-8", errors="replace")
        (job_dir / "stderr.log").write_text(stderr, encoding="utf-8", errors="replace")
        if process.returncode != 0:
            raise VCMRSCodecError(
                "VCM-RS encoder failed. Artifacts were preserved.\n"
                f"returncode={process.returncode}\n"
                f"cwd={self.vcmrs_root}\n"
                f"job_dir={job_dir}\n"
                f"command={shlex.join(command)}\n"
                f"stdout_tail:\n{_tail(stdout)}\n"
                f"stderr_tail:\n{_tail(stderr)}"
            )

    @staticmethod
    def _save_tensor_png(image_chw: torch.Tensor, path: Path) -> None:
        array = (
            image_chw.detach()
            .to(device="cpu", dtype=torch.float32)
            .permute(1, 2, 0)
            .mul(255.0)
            .round()
            .clamp_(0, 255)
            .to(torch.uint8)
            .numpy()
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array, mode="RGB").save(path)

    @staticmethod
    def _find_reconstruction(job_dir: Path) -> Path:
        recon_root = job_dir / "output" / "recon"
        candidates = sorted(path for path in recon_root.rglob("*.png") if path.is_file())
        if len(candidates) != 1:
            raise VCMRSCodecError(
                "Expected exactly one reconstructed PNG from one-frame VCM-RS encoding, "
                f"found {len(candidates)} under {recon_root}: {candidates}. "
                f"Artifacts were preserved in {job_dir}."
            )
        return candidates[0]

    def roundtrip01(self, image: torch.Tensor, *, call_key: str = "frame") -> torch.Tensor:
        """Encode/reconstruct one RGB tensor and return RGB in [0, 1]."""

        was_batched = image.ndim == 4
        if was_batched:
            if image.shape[0] != 1:
                raise ValueError(f"VCMRSOneFrameCodec requires batch size 1, got {image.shape[0]}")
            image_chw = image[0]
        elif image.ndim == 3:
            image_chw = image
        else:
            raise ValueError(f"Expected CHW or 1CHW tensor, got shape {tuple(image.shape)}")

        if image_chw.shape[0] != 3:
            raise ValueError(f"Expected three RGB channels, got shape {tuple(image_chw.shape)}")
        if not image_chw.is_floating_point():
            raise TypeError(f"Expected floating-point tensor, got dtype {image_chw.dtype}")
        if not torch.isfinite(image_chw).all():
            raise ValueError("Input contains NaN or infinity")
        min_value = float(image_chw.detach().amin().cpu())
        max_value = float(image_chw.detach().amax().cpu())
        if min_value < -1e-6 or max_value > 1.0 + 1e-6:
            raise ValueError(
                f"Input must be RGB in [0, 1], observed range [{min_value}, {max_value}]"
            )

        job_dir = self._new_job_dir(call_key)
        input_path = job_dir / "input" / "frame.png"
        ini_path = job_dir / "encoder.ini"
        succeeded = False
        try:
            self._save_tensor_png(image_chw.clamp(0, 1), input_path)
            self._write_encoder_ini(ini_path, input_path, job_dir)
            with self._semaphore:
                self._run_encoder(ini_path, job_dir)
            recon_path = self._find_reconstruction(job_dir)
            with Image.open(recon_path) as recon_image:
                recon_array = np.asarray(recon_image.convert("RGB"), dtype=np.uint8).copy()
            if tuple(recon_array.shape[:2]) != tuple(image_chw.shape[1:]):
                raise VCMRSCodecError(
                    f"Reconstruction size {recon_array.shape[1]}x{recon_array.shape[0]} does not "
                    f"match input size {image_chw.shape[2]}x{image_chw.shape[1]}. "
                    f"Artifacts were preserved in {job_dir}."
                )
            reconstructed = torch.from_numpy(recon_array).permute(2, 0, 1).float().div_(255.0)
            reconstructed = reconstructed.to(device=image.device, dtype=image.dtype)
            succeeded = True
            return reconstructed.unsqueeze(0) if was_batched else reconstructed
        finally:
            if succeeded and not self.keep_artifacts:
                shutil.rmtree(job_dir, ignore_errors=True)

    def __call__(self, image: torch.Tensor, *, call_key: str = "frame") -> torch.Tensor:
        return self.roundtrip01(image, call_key=call_key)


def _image_to_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def _save_output_tensor(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    VCMRSOneFrameCodec._save_tensor_png(image, path)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one PNG through VCM-RS/VTM once")
    parser.add_argument("input", type=Path, help="Input PNG/JPEG")
    parser.add_argument("output", type=Path, help="Output reconstructed PNG")
    parser.add_argument("--vcmrs-root", type=Path, default=DEFAULT_VCMRS_ROOT)
    parser.add_argument("--python", dest="python_executable", type=Path, default=DEFAULT_VCMRS_PYTHON)
    parser.add_argument("--quality", type=int, default=52, help="VTM QP; higher is more compressed")
    parser.add_argument("--frame-rate", type=int, default=30)
    parser.add_argument("--profile", choices=["vtm_only", "train_match"], default="vtm_only")
    parser.add_argument(
        "--roi-descriptor",
        type=Path,
        help="Frozen one-frame (key 0) descriptor required by --profile train_match",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--cuda-visible-devices")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    codec = VCMRSOneFrameCodec(
        vcmrs_root=args.vcmrs_root,
        python_executable=args.python_executable,
        quality=args.quality,
        frame_rate=args.frame_rate,
        profile=args.profile,
        roi_descriptor=args.roi_descriptor,
        timeout_seconds=args.timeout,
        keep_artifacts=args.keep_artifacts,
        artifact_root=args.artifact_root,
        cuda_visible_devices=args.cuda_visible_devices,
    )
    input_tensor = _image_to_tensor(args.input)
    output_tensor = codec.roundtrip01(input_tensor, call_key=args.input.stem)
    _save_output_tensor(output_tensor, args.output)
    print(f"VCM-RS one-frame reconstruction: {args.output.resolve()}")
    if args.keep_artifacts:
        root = args.artifact_root or Path(tempfile.gettempdir())
        print(f"VCM-RS artifacts kept under: {root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
