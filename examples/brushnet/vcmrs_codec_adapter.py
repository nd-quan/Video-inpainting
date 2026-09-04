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

``VCMRSDualRegionCodec`` implements the degradation used by the restoration
training data.  It encodes the same full image twice and hard-composites the
two reconstructions with a binary mask::

    D_M(x) = M * VCM_RS_QP20(x) + (1 - M) * VCM_RS_QP52(x)

White/non-zero mask pixels are ROI.  They select the QP20 reconstruction;
black pixels select the QP52 background reconstruction.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    configuration: str = "AllIntra"
    intra_period: int = 1
    nn_intra_qp_offset: Optional[int] = None
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
        if not 0 <= self.quality <= 63:
            raise ValueError(f"VTM QP/quality must be in [0, 63], got {self.quality}")
        if self.configuration not in {"AllIntra", "RandomAccess"}:
            raise ValueError(
                "configuration must be 'AllIntra' or 'RandomAccess', got "
                f"{self.configuration!r}"
            )
        if self.intra_period < 1:
            raise ValueError(f"intra_period must be >= 1, got {self.intra_period}")
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
            configuration=os.environ.get("CGE_VCMRS_CONFIGURATION", "AllIntra"),
            intra_period=int(os.environ.get("CGE_VCMRS_INTRA_PERIOD", "1")),
            nn_intra_qp_offset=(
                int(os.environ["CGE_VCMRS_NN_INTRA_QP_OFFSET"])
                if "CGE_VCMRS_NN_INTRA_QP_OFFSET" in os.environ
                else None
            ),
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

    def _encoder_options(
        self,
        job_dir: Path,
        descriptor_override: Optional[Path] = None,
    ) -> Dict[str, str]:
        """Return a fully pinned profile; do not rely on mutable VCM defaults."""

        descriptor = descriptor_override or self.roi_descriptor
        nn_intra_qp_offset = self.nn_intra_qp_offset
        if nn_intra_qp_offset is None:
            nn_intra_qp_offset = 0 if self.configuration == "AllIntra" else -5

        options = {
            "output_dir": str(job_dir / "output"),
            "working_dir": str(job_dir / "work"),
            "output_bitstream_fname": "bitstream/coded.bin",
            "output_recon_fname": "recon/coded",
            "Configuration": self.configuration,
            "IntraPeriod": str(self.intra_period),
            "NNIntraQPOffset": str(nn_intra_qp_offset),
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
            "RoIInflation": "auto",
            "RoIRetargetingMaxNumRoIs": "11",
            "RoIAdaptiveMarginDilation": "0",
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
            if descriptor is None:
                raise ValueError(
                    "train_match requires a frozen one-frame descriptor. Set "
                    "a descriptor on the codec or pass descriptor_override."
                )
            options.update(
                {
                    "RoIDescriptorMode": "load",
                    "RoIDescriptor": str(descriptor),
                    "RoIAccumulationPeriod": "-1",
                }
            )
        return options

    def _write_encoder_ini(
        self,
        ini_path: Path,
        input_path: Path,
        job_dir: Path,
        descriptor_override: Optional[Path] = None,
    ) -> None:
        # VCM-RS treats every non-default INI section name as an input argv.
        # Quote it because io_utils.parse_ini_file applies shlex.split(section).
        input_section = shlex.quote(str(input_path))
        parser = configparser.ConfigParser(interpolation=None)
        parser.optionxform = str
        parser["default"] = self._encoder_options(job_dir, descriptor_override)
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

    def roundtrip01(
        self,
        image: torch.Tensor,
        *,
        call_key: str = "frame",
        descriptor_override: Optional[Path] = None,
    ) -> torch.Tensor:
        """Encode/reconstruct one RGB tensor and return RGB in [0, 1]."""

        if descriptor_override is not None:
            descriptor_override = Path(descriptor_override).expanduser().resolve()
            if not descriptor_override.is_file():
                raise FileNotFoundError(
                    f"ROI/BG descriptor does not exist: {descriptor_override}"
                )

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
            self._write_encoder_ini(
                ini_path,
                input_path,
                job_dir,
                descriptor_override=descriptor_override,
            )
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


def _binary_mask_to_rectangles(binary_mask: np.ndarray) -> List[List[int]]:
    """Convert a 2-D binary mask to exact inclusive scanline rectangles."""

    if binary_mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {binary_mask.shape}")

    binary = np.asarray(binary_mask, dtype=np.bool_)
    active: Dict[Tuple[int, int], List[int]] = {}
    finished: List[List[int]] = []

    for y, row in enumerate(binary):
        padded = np.pad(row.astype(np.int8), (1, 1))
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1) - 1
        current: Dict[Tuple[int, int], List[int]] = {}

        for x1, x2 in zip(starts.tolist(), ends.tolist()):
            key = (int(x1), int(x2))
            if key in active:
                rectangle = active[key]
                rectangle[3] = y
            else:
                rectangle = [key[0], y, key[1], y]
            current[key] = rectangle

        for key, rectangle in active.items():
            if key not in current:
                finished.append(rectangle)
        active = current

    finished.extend(active.values())
    return finished


def _rectangles_to_mask(
    rectangles: Sequence[Sequence[int]], width: int, height: int
) -> np.ndarray:
    raster = np.zeros((height, width), dtype=np.bool_)
    for x1, y1, x2, y2 in rectangles:
        if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
            raise ValueError(
                f"Descriptor rectangle {(x1, y1, x2, y2)} is outside {width}x{height}"
            )
        raster[y1 : y2 + 1, x1 : x2 + 1] = True
    return raster


def _write_key0_descriptor(path: Path, rectangles: Sequence[Sequence[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["{", "  0:["]
    lines.extend(f"    {list(map(int, rectangle))}," for rectangle in rectangles)
    lines.extend(['    "scaling_method=1",', "  ],", "}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalize_roi_mask(
    roi_mask: torch.Tensor,
    image: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Return a boolean 1xHxW mask; white/non-zero is ROI."""

    if not isinstance(roi_mask, torch.Tensor):
        raise TypeError(f"roi_mask must be a torch.Tensor, got {type(roi_mask)!r}")
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"Dual-region codec requires batch size 1, got {image.shape[0]}")
        height, width = image.shape[-2:]
    elif image.ndim == 3:
        height, width = image.shape[-2:]
    else:
        raise ValueError(f"Expected CHW or 1CHW image, got shape {tuple(image.shape)}")

    mask = roi_mask.detach()
    if mask.ndim == 4:
        if mask.shape[0] != 1:
            raise ValueError(f"roi_mask batch size must be 1, got shape {tuple(mask.shape)}")
        mask = mask[0]
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 3 and mask.shape[0] == 3:
        if not torch.equal(mask[0], mask[1]) or not torch.equal(mask[0], mask[2]):
            raise ValueError("Three-channel roi_mask must have identical channels")
        mask = mask[:1]
    elif mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(
            "roi_mask must have shape HxW, 1xHxW, 3xHxW, or 1x1xHxW; "
            f"got {tuple(roi_mask.shape)}"
        )
    if tuple(mask.shape[-2:]) != (height, width):
        raise ValueError(
            f"roi_mask size {tuple(mask.shape[-2:])} does not match image {(height, width)}"
        )
    if mask.is_floating_point():
        if not torch.isfinite(mask).all():
            raise ValueError("roi_mask contains NaN or infinity")
        min_value = float(mask.amin().cpu())
        max_value = float(mask.amax().cpu())
        if min_value < -1e-6 or max_value > 1.0 + 1e-6:
            raise ValueError(
                f"roi_mask must be in [0, 1], observed range [{min_value}, {max_value}]"
            )
    return mask.to(device=image.device).gt(float(threshold))


@dataclass
class VCMRSDualRegionCodec:
    """Train-style one-frame VCM-RS degradation with separate ROI/BG QPs.

    Both child encoders receive the same full RGB image.  Their decoded
    reconstructions are merged only afterwards with a hard binary ROI mask.
    ``train_match`` generates complementary key-0 descriptors directly from
    that mask unless two explicit descriptor files are supplied.
    """

    vcmrs_root: Path = DEFAULT_VCMRS_ROOT
    python_executable: Path = DEFAULT_VCMRS_PYTHON
    roi_quality: int = 20
    bg_quality: int = 52
    frame_rate: int = 30
    configuration: str = "RandomAccess"
    intra_period: int = 64
    nn_intra_qp_offset: Optional[int] = None
    profile: str = "train_match"
    roi_descriptor: Optional[Path] = None
    bg_descriptor: Optional[Path] = None
    auto_descriptors: bool = True
    mask_threshold: float = 0.5
    timeout_seconds: float = 600.0
    keep_artifacts: bool = False
    artifact_root: Optional[Path] = None
    max_parallel: int = 1
    cuda_visible_devices: Optional[str] = None
    _roi_codec: VCMRSOneFrameCodec = field(init=False, repr=False)
    _bg_codec: VCMRSOneFrameCodec = field(init=False, repr=False)
    _semaphore: threading.BoundedSemaphore = field(init=False, repr=False)
    _descriptor_lock: threading.Lock = field(init=False, repr=False)
    _descriptor_cache: Dict[str, Tuple[Path, Path, Path]] = field(
        init=False, repr=False
    )
    _temporary_descriptor_root: Optional[tempfile.TemporaryDirectory] = field(
        init=False, default=None, repr=False
    )
    cge_operator: str = field(init=False, default="dual_region_qp20_qp52")

    def __post_init__(self) -> None:
        if not 0.0 <= self.mask_threshold < 1.0:
            raise ValueError(
                f"mask_threshold must be in [0, 1), got {self.mask_threshold}"
            )
        self.vcmrs_root = Path(self.vcmrs_root).expanduser().resolve()
        self.python_executable = Path(self.python_executable).expanduser().resolve()
        if self.artifact_root is not None:
            self.artifact_root = Path(self.artifact_root).expanduser().resolve()
        self.roi_descriptor = self._resolve_descriptor(self.roi_descriptor, "ROI")
        self.bg_descriptor = self._resolve_descriptor(self.bg_descriptor, "BG")

        common = dict(
            vcmrs_root=self.vcmrs_root,
            python_executable=self.python_executable,
            frame_rate=self.frame_rate,
            configuration=self.configuration,
            intra_period=self.intra_period,
            nn_intra_qp_offset=self.nn_intra_qp_offset,
            profile=self.profile,
            timeout_seconds=self.timeout_seconds,
            keep_artifacts=self.keep_artifacts,
            artifact_root=self.artifact_root,
            max_parallel=self.max_parallel,
            cuda_visible_devices=self.cuda_visible_devices,
        )
        self._roi_codec = VCMRSOneFrameCodec(
            quality=self.roi_quality,
            roi_descriptor=self.roi_descriptor,
            **common,
        )
        self._bg_codec = VCMRSOneFrameCodec(
            quality=self.bg_quality,
            roi_descriptor=self.bg_descriptor,
            **common,
        )
        # One global limit covers both branches and any scheduler batch threads.
        self._semaphore = threading.BoundedSemaphore(self.max_parallel)
        self._roi_codec._semaphore = self._semaphore
        self._bg_codec._semaphore = self._semaphore
        self._descriptor_lock = threading.Lock()
        self._descriptor_cache = {}

    @staticmethod
    def _resolve_descriptor(
        descriptor: Optional[Path], region_name: str
    ) -> Optional[Path]:
        if descriptor is None:
            return None
        resolved = Path(descriptor).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"{region_name} descriptor does not exist: {resolved}")
        return resolved

    @classmethod
    def from_env(cls) -> "VCMRSDualRegionCodec":
        """Build a QP20/QP52 regional codec from ``CGE_VCMRS_*`` variables."""

        artifact_root = os.environ.get("CGE_VCMRS_ARTIFACT_ROOT")
        roi_descriptor = os.environ.get("CGE_VCMRS_ROI_DESCRIPTOR")
        bg_descriptor = os.environ.get("CGE_VCMRS_BG_DESCRIPTOR")
        return cls(
            vcmrs_root=Path(os.environ.get("CGE_VCMRS_ROOT", str(DEFAULT_VCMRS_ROOT))),
            python_executable=Path(
                os.environ.get("CGE_VCMRS_PYTHON", str(DEFAULT_VCMRS_PYTHON))
            ),
            roi_quality=int(os.environ.get("CGE_VCMRS_ROI_QUALITY", "20")),
            bg_quality=int(os.environ.get("CGE_VCMRS_BG_QUALITY", "52")),
            frame_rate=int(os.environ.get("CGE_VCMRS_FRAME_RATE", "30")),
            configuration=os.environ.get("CGE_VCMRS_CONFIGURATION", "RandomAccess"),
            intra_period=int(os.environ.get("CGE_VCMRS_INTRA_PERIOD", "64")),
            nn_intra_qp_offset=(
                int(os.environ["CGE_VCMRS_NN_INTRA_QP_OFFSET"])
                if "CGE_VCMRS_NN_INTRA_QP_OFFSET" in os.environ
                else None
            ),
            profile=os.environ.get("CGE_VCMRS_PROFILE", "train_match"),
            roi_descriptor=Path(roi_descriptor) if roi_descriptor else None,
            bg_descriptor=Path(bg_descriptor) if bg_descriptor else None,
            auto_descriptors=_env_flag("CGE_VCMRS_AUTO_DESCRIPTORS", "1"),
            mask_threshold=float(os.environ.get("CGE_VCMRS_MASK_THRESHOLD", "0.5")),
            timeout_seconds=float(os.environ.get("CGE_VCMRS_TIMEOUT", "600")),
            keep_artifacts=_env_flag("CGE_VCMRS_KEEP_ARTIFACTS"),
            artifact_root=Path(artifact_root) if artifact_root else None,
            max_parallel=int(os.environ.get("CGE_VCMRS_MAX_PARALLEL", "1")),
            cuda_visible_devices=os.environ.get("CGE_VCMRS_CUDA_VISIBLE_DEVICES"),
        )

    def set_descriptors(self, roi_descriptor: Path, bg_descriptor: Path) -> None:
        """Set matching key-0 descriptors for one inference frame."""

        self.roi_descriptor = self._resolve_descriptor(roi_descriptor, "ROI")
        self.bg_descriptor = self._resolve_descriptor(bg_descriptor, "BG")
        self._roi_codec.roi_descriptor = self.roi_descriptor
        self._bg_codec.roi_descriptor = self.bg_descriptor

    def _new_descriptor_dir(self, call_key: str) -> Path:
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in call_key)
        safe_key = safe_key[:80] or "frame"
        parent = self.artifact_root
        if parent is not None:
            parent.mkdir(parents=True, exist_ok=True)
        if not self.keep_artifacts:
            if self._temporary_descriptor_root is None:
                self._temporary_descriptor_root = tempfile.TemporaryDirectory(
                    prefix="cge_vcmrs_descriptor_cache_",
                    dir=parent,
                )
            parent = Path(self._temporary_descriptor_root.name)
        return Path(tempfile.mkdtemp(prefix=f"cge_vcmrs_{safe_key}_descriptors_", dir=parent))

    def _auto_descriptor_pair(
        self, mask_1hw: torch.Tensor, call_key: str
    ) -> Tuple[Path, Path, Path]:
        descriptor_dir = self._new_descriptor_dir(call_key)
        mask_np = mask_1hw[0].to(device="cpu").numpy().astype(np.bool_)
        roi_rectangles = _binary_mask_to_rectangles(mask_np)
        bg_rectangles = _binary_mask_to_rectangles(np.logical_not(mask_np))

        roi_raster = _rectangles_to_mask(roi_rectangles, mask_np.shape[1], mask_np.shape[0])
        bg_raster = _rectangles_to_mask(bg_rectangles, mask_np.shape[1], mask_np.shape[0])
        if not np.array_equal(roi_raster, mask_np):
            raise RuntimeError("Generated ROI descriptor does not reproduce the binary mask")
        if not np.array_equal(bg_raster, np.logical_not(mask_np)):
            raise RuntimeError("Generated BG descriptor does not reproduce the mask complement")
        if np.any(roi_raster & bg_raster) or not np.all(roi_raster | bg_raster):
            raise RuntimeError("Generated ROI/BG descriptors overlap or leave uncovered pixels")

        roi_path = descriptor_dir / "roi_key0.txt"
        bg_path = descriptor_dir / "bg_key0.txt"
        _write_key0_descriptor(roi_path, roi_rectangles)
        _write_key0_descriptor(bg_path, bg_rectangles)
        return roi_path, bg_path, descriptor_dir

    @staticmethod
    def _mask_signature(mask_1hw: torch.Tensor) -> str:
        mask_np = mask_1hw.to(device="cpu").numpy().astype(np.uint8, copy=False)
        digest = hashlib.sha256()
        digest.update(str(tuple(mask_np.shape)).encode("ascii"))
        digest.update(mask_np.tobytes())
        return digest.hexdigest()

    def _descriptor_pair_for_mask(
        self, mask_1hw: torch.Tensor, call_key: str
    ) -> Tuple[Path, Path]:
        signature = self._mask_signature(mask_1hw)
        with self._descriptor_lock:
            cached = self._descriptor_cache.get(signature)
            if cached is None:
                cached = self._auto_descriptor_pair(mask_1hw, call_key)
                self._descriptor_cache[signature] = cached
            return cached[0], cached[1]

    def prepare_region_mask(self, roi_mask: torch.Tensor, image: torch.Tensor) -> None:
        """Validate and freeze auto descriptors before an expensive CGE call."""

        mask = _normalize_roi_mask(roi_mask, image, self.mask_threshold)
        if self.profile == "train_match" and self.auto_descriptors and (
            self.roi_descriptor is None or self.bg_descriptor is None
        ):
            self._descriptor_pair_for_mask(mask, "prepared")

    def roundtrip_regions01(
        self,
        image: torch.Tensor,
        *,
        roi_mask: torch.Tensor,
        call_key: str = "frame",
    ) -> torch.Tensor:
        """Run both full-frame codecs and hard-composite their reconstructions."""

        mask = _normalize_roi_mask(roi_mask, image, self.mask_threshold)
        has_roi = bool(mask.any().item())
        has_bg = bool(torch.logical_not(mask).any().item())

        roi_descriptor = self.roi_descriptor
        bg_descriptor = self.bg_descriptor
        if self.profile == "train_match" and (
            (has_roi and roi_descriptor is None) or (has_bg and bg_descriptor is None)
        ):
            if not self.auto_descriptors:
                raise ValueError(
                    "train_match needs both ROI and BG descriptors. Set "
                    "CGE_VCMRS_ROI_DESCRIPTOR/CGE_VCMRS_BG_DESCRIPTOR or enable "
                    "CGE_VCMRS_AUTO_DESCRIPTORS=1."
                )
            auto_roi, auto_bg = self._descriptor_pair_for_mask(mask, call_key)
            roi_descriptor = roi_descriptor or auto_roi
            bg_descriptor = bg_descriptor or auto_bg

        roi_reconstruction = None
        bg_reconstruction = None
        if has_roi:
            roi_reconstruction = self._roi_codec.roundtrip01(
                image,
                call_key=f"{call_key}_roi_qp{self.roi_quality}",
                descriptor_override=roi_descriptor,
            )
        if has_bg:
            bg_reconstruction = self._bg_codec.roundtrip01(
                image,
                call_key=f"{call_key}_bg_qp{self.bg_quality}",
                descriptor_override=bg_descriptor,
            )

        if roi_reconstruction is None:
            merged = bg_reconstruction
        elif bg_reconstruction is None:
            merged = roi_reconstruction
        else:
            broadcast_mask = mask
            if image.ndim == 4:
                broadcast_mask = broadcast_mask.unsqueeze(0)
            merged = torch.where(broadcast_mask, roi_reconstruction, bg_reconstruction)
        if merged is None:
            raise RuntimeError("Binary ROI/BG mask unexpectedly selected no pixels")
        return merged

    def roundtrip_background01(
        self,
        image: torch.Tensor,
        *,
        roi_mask: torch.Tensor,
        call_key: str = "frame",
    ) -> torch.Tensor:
        """Run only the QP52 background branch of the regional degradation.

        This is the deliberately *approximate* fast-CGE operator.  Its caller
        applies the codec residual only on background pixels and supplies a
        direct image-space fidelity loss for ROI.  We still pass the ROI mask
        here because the ``train_match`` background encoder needs its
        complementary key-0 descriptor.

        Returning ``image`` when a frame has no background is safe: the caller
        masks the codec residual to zero in that case, while the direct ROI
        term remains active.
        """

        mask = _normalize_roi_mask(roi_mask, image, self.mask_threshold)
        has_bg = bool(torch.logical_not(mask).any().item())
        if not has_bg:
            return image

        bg_descriptor = self.bg_descriptor
        if self.profile == "train_match" and bg_descriptor is None:
            if not self.auto_descriptors:
                raise ValueError(
                    "train_match background-only CGE needs a BG descriptor. "
                    "Set CGE_VCMRS_BG_DESCRIPTOR or enable "
                    "CGE_VCMRS_AUTO_DESCRIPTORS=1."
                )
            _, auto_bg = self._descriptor_pair_for_mask(mask, call_key)
            bg_descriptor = auto_bg

        return self._bg_codec.roundtrip01(
            image,
            call_key=f"{call_key}_bg_qp{self.bg_quality}",
            descriptor_override=bg_descriptor,
        )

    def roundtrip01(
        self,
        image: torch.Tensor,
        *,
        roi_mask: torch.Tensor,
        call_key: str = "frame",
    ) -> torch.Tensor:
        return self.roundtrip_regions01(image, roi_mask=roi_mask, call_key=call_key)


@dataclass
class VCMRSBackgroundOnlyCodec:
    """Fast-CGE façade over :class:`VCMRSDualRegionCodec`.

    It intentionally does *not* expose ``roundtrip_regions01``.  The CGE
    scheduler therefore retains its legacy residual layout: direct ROI
    fidelity plus a codec residual on background only.  The underlying VCM-RS
    invocation remains the QP52, train-match-compatible background branch.
    """

    regional_codec: VCMRSDualRegionCodec
    cge_operator: str = field(init=False, default="direct_roi_plus_bg_qp52")

    @classmethod
    def from_env(cls) -> "VCMRSBackgroundOnlyCodec":
        return cls(regional_codec=VCMRSDualRegionCodec.from_env())

    @property
    def profile(self) -> str:
        return self.regional_codec.profile

    @property
    def roi_quality(self) -> int:
        """Input ROI quality, retained only for run metadata (not encoded)."""

        return self.regional_codec.roi_quality

    @property
    def bg_quality(self) -> int:
        return self.regional_codec.bg_quality

    @property
    def roi_descriptor(self) -> Optional[Path]:
        return self.regional_codec.roi_descriptor

    @property
    def bg_descriptor(self) -> Optional[Path]:
        return self.regional_codec.bg_descriptor

    @property
    def max_parallel(self) -> int:
        return self.regional_codec.max_parallel

    def __getattr__(self, name):
        """Expose read-only VCM-RS metadata used by existing CGE runners."""

        return getattr(self.regional_codec, name)

    def prepare_region_mask(self, roi_mask: torch.Tensor, image: torch.Tensor) -> None:
        self.regional_codec.prepare_region_mask(roi_mask, image)

    def roundtrip_background01(
        self,
        image: torch.Tensor,
        *,
        roi_mask: torch.Tensor,
        call_key: str = "frame",
    ) -> torch.Tensor:
        return self.regional_codec.roundtrip_background01(
            image,
            roi_mask=roi_mask,
            call_key=call_key,
        )


def _image_to_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def _mask_to_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    return torch.from_numpy(array).unsqueeze(0).float().div_(255.0)


def _save_output_tensor(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    VCMRSOneFrameCodec._save_tensor_png(image, path)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one image through VCM-RS. With --mask, encode the same full image "
            "at ROI QP20 and BG QP52 and hard-composite the reconstructions."
        )
    )
    parser.add_argument("input", type=Path, help="Input PNG/JPEG")
    parser.add_argument("output", type=Path, help="Output reconstructed PNG")
    parser.add_argument(
        "--mask",
        type=Path,
        help="ROI mask; white/non-zero selects ROI QP20, black selects BG QP52",
    )
    parser.add_argument("--vcmrs-root", type=Path, default=DEFAULT_VCMRS_ROOT)
    parser.add_argument("--python", dest="python_executable", type=Path, default=DEFAULT_VCMRS_PYTHON)
    parser.add_argument("--quality", type=int, default=52, help="VTM QP; higher is more compressed")
    parser.add_argument("--roi-quality", type=int, default=20)
    parser.add_argument("--bg-quality", type=int, default=52)
    parser.add_argument("--frame-rate", type=int, default=30)
    parser.add_argument("--profile", choices=["vtm_only", "train_match"])
    parser.add_argument("--configuration", choices=["AllIntra", "RandomAccess"])
    parser.add_argument("--intra-period", type=int)
    parser.add_argument(
        "--roi-descriptor",
        type=Path,
        help="Frozen ROI key-0 descriptor (or the single-codec descriptor without --mask)",
    )
    parser.add_argument("--bg-descriptor", type=Path, help="Frozen BG key-0 descriptor")
    parser.add_argument("--no-auto-descriptors", action="store_true")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--cuda-visible-devices")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    input_tensor = _image_to_tensor(args.input)
    common = dict(
        vcmrs_root=args.vcmrs_root,
        python_executable=args.python_executable,
        frame_rate=args.frame_rate,
        timeout_seconds=args.timeout,
        keep_artifacts=args.keep_artifacts,
        artifact_root=args.artifact_root,
        cuda_visible_devices=args.cuda_visible_devices,
    )
    if args.mask is not None:
        codec = VCMRSDualRegionCodec(
            roi_quality=args.roi_quality,
            bg_quality=args.bg_quality,
            configuration=args.configuration or "RandomAccess",
            intra_period=args.intra_period or 64,
            profile=args.profile or "train_match",
            roi_descriptor=args.roi_descriptor,
            bg_descriptor=args.bg_descriptor,
            auto_descriptors=not args.no_auto_descriptors,
            mask_threshold=args.mask_threshold,
            **common,
        )
        mask_tensor = _mask_to_tensor(args.mask)
        if tuple(mask_tensor.shape[-2:]) != tuple(input_tensor.shape[-2:]):
            raise ValueError(
                f"Mask size {tuple(mask_tensor.shape[-2:])} does not match input "
                f"{tuple(input_tensor.shape[-2:])}; resize it with nearest-neighbor first."
            )
        output_tensor = codec.roundtrip_regions01(
            input_tensor,
            roi_mask=mask_tensor,
            call_key=args.input.stem,
        )
        completion_message = (
            f"VCM-RS dual reconstruction: ROI QP{args.roi_quality} + "
            f"BG QP{args.bg_quality} -> {args.output.resolve()}"
        )
    else:
        codec = VCMRSOneFrameCodec(
            quality=args.quality,
            configuration=args.configuration or "AllIntra",
            intra_period=args.intra_period or 1,
            profile=args.profile or "vtm_only",
            roi_descriptor=args.roi_descriptor,
            **common,
        )
        output_tensor = codec.roundtrip01(input_tensor, call_key=args.input.stem)
        completion_message = f"VCM-RS one-frame reconstruction: {args.output.resolve()}"
    _save_output_tensor(output_tensor, args.output)
    print(completion_message)
    if args.keep_artifacts:
        root = args.artifact_root or Path(tempfile.gettempdir())
        print(f"VCM-RS artifacts kept under: {root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
