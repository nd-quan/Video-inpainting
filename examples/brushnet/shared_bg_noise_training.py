"""Clip grouping and shared-background noise utilities for BrushNet training.

The sampling pipeline represents a video clip as a flattened image batch, but
uses one Gaussian background-noise field for all frames in a source sequence.
These helpers keep the clip dimension explicit and deterministically recover
the sequence field across shuffled batches, then let the 2D BrushNet/UNet
trainer flatten ``B x T`` as before.

Mask convention throughout this module is ``1 = background`` and ``0 = ROI``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CLIP_TENSOR_KEYS: Tuple[str, ...] = (
    "pixel_values",
    "masks",
    "conditioning_pixel_values",
    "input_ids",
    "clip_images",
    "fg_clip_images",
    "bg_clip_images",
)


def _randn_like(tensor: torch.Tensor, generator=None) -> torch.Tensor:
    return torch.randn(
        tensor.shape,
        generator=generator,
        device=tensor.device,
        dtype=tensor.dtype,
    )


def make_sequence_shared_background_noise(
    latent_template: torch.Tensor,
    sequence_keys: Sequence[str],
    base_seed: int,
    refresh_index: int = 0,
) -> torch.Tensor:
    """Create one deterministic shared-noise field per source sequence.

    The same ``sequence_key`` and ``refresh_index`` produce exactly the same
    field even when clips are shuffled into different batches or DDP ranks.
    Changing ``refresh_index`` (normally once per epoch) resets the field while
    retaining sequence-wide consistency within that refresh interval.
    """
    if latent_template.ndim != 5:
        raise ValueError("latent_template must have shape [B,T,C,H,W]")
    batch, _, channels, height, width = latent_template.shape
    if len(sequence_keys) != batch:
        raise ValueError(
            f"Expected {batch} sequence keys, received {len(sequence_keys)}"
        )

    cached_cpu_noise: Dict[str, torch.Tensor] = {}
    clip_noise = []
    for raw_key in sequence_keys:
        key = str(raw_key)
        if not key:
            raise ValueError("sequence keys must be non-empty")
        if key not in cached_cpu_noise:
            payload = f"{int(base_seed)}:{int(refresh_index)}:{key}".encode("utf-8")
            digest = hashlib.blake2b(
                payload, digest_size=8, person=b"BrushNetSeq"
            ).digest()
            sequence_seed = int.from_bytes(digest, byteorder="little") % (
                2**63 - 1
            )
            cpu_generator = torch.Generator(device="cpu").manual_seed(
                sequence_seed
            )
            cached_cpu_noise[key] = torch.randn(
                (channels, height, width),
                generator=cpu_generator,
                device="cpu",
                dtype=torch.float32,
            )
        clip_noise.append(cached_cpu_noise[key])

    return torch.stack(clip_noise, dim=0).unsqueeze(1).to(
        device=latent_template.device, dtype=latent_template.dtype
    )


def mix_shared_background_noise(
    independent_noise: torch.Tensor,
    shared_bg_noise: torch.Tensor,
    bg_mask: torch.Tensor,
    strength: float = 1.0,
    variance_preserving: bool = True,
) -> torch.Tensor:
    """Mix per-frame and per-clip Gaussian noise on background pixels.

    Args:
        independent_noise: Tensor shaped ``[B,T,C,H,W]``.
        shared_bg_noise: Tensor shaped ``[B,1,C,H,W]`` (or ``[B,C,H,W]``).
        bg_mask: Binary tensor shaped ``[B,T,1,H,W]`` with one on background.
        strength: For variance-preserving mixing, the target inter-frame
            background correlation ``rho``. For linear mixing, the lerp weight.
        variance_preserving: Mirror the sampling pipeline's corresponding flag.

    Returns:
        Noise shaped ``[B,T,C,H,W]``. ROI noise always remains independent.
    """
    if independent_noise.ndim != 5:
        raise ValueError("independent_noise must have shape [B,T,C,H,W]")
    batch, frames, channels, height, width = independent_noise.shape
    if bg_mask.shape != (batch, frames, 1, height, width):
        raise ValueError(
            "bg_mask must have shape [B,T,1,H,W] matching independent_noise"
        )
    if shared_bg_noise.ndim == 4:
        shared_bg_noise = shared_bg_noise.unsqueeze(1)
    if shared_bg_noise.shape != (batch, 1, channels, height, width):
        raise ValueError(
            "shared_bg_noise must have shape [B,1,C,H,W] matching "
            "independent_noise"
        )
    strength = float(strength)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("shared background noise strength must be in [0, 1]")

    if strength == 0.0:
        return independent_noise

    mask = bg_mask.to(
        device=independent_noise.device, dtype=independent_noise.dtype
    )
    shared = shared_bg_noise.to(
        device=independent_noise.device, dtype=independent_noise.dtype
    )

    if variance_preserving:
        # Here strength is rho: Var(epsilon_t)=1 and Corr(i,j)=rho on BG.
        mixed_bg = (
            math.sqrt(1.0 - strength) * independent_noise
            + math.sqrt(strength) * shared
        )
    else:
        # Exact legacy sampling behavior. At partial strength this reduces BG
        # variance to (1-s)^2 + s^2.
        mixed_bg = (1.0 - strength) * independent_noise + strength * shared

    return independent_noise * (1.0 - mask) + mixed_bg * mask


def sample_shared_background_noise(
    latent_template: torch.Tensor,
    bg_mask: torch.Tensor,
    strength: float = 1.0,
    variance_preserving: bool = True,
    generator=None,
    shared_bg_noise: torch.Tensor = None,
) -> torch.Tensor:
    """Sample independent frame noise and mix a supplied/generated BG field."""
    if latent_template.ndim != 5:
        raise ValueError("latent_template must have shape [B,T,C,H,W]")

    independent = _randn_like(latent_template, generator=generator)
    if float(strength) == 0.0:
        return independent

    if shared_bg_noise is None:
        shared_shape = (
            latent_template.shape[0],
            1,
            latent_template.shape[2],
            latent_template.shape[3],
            latent_template.shape[4],
        )
        shared_bg_noise = torch.randn(
            shared_shape,
            generator=generator,
            device=latent_template.device,
            dtype=latent_template.dtype,
        )
    return mix_shared_background_noise(
        independent,
        shared_bg_noise,
        bg_mask,
        strength=strength,
        variance_preserving=variance_preserving,
    )


def sample_clip_timesteps(
    num_clips: int,
    clip_length: int,
    num_train_timesteps: int,
    device,
    share_across_clip: bool = True,
    generator=None,
) -> torch.Tensor:
    """Sample flattened timesteps, sharing one timestep within each clip."""
    num_clips = int(num_clips)
    clip_length = int(clip_length)
    num_train_timesteps = int(num_train_timesteps)
    if num_clips <= 0 or clip_length <= 0:
        raise ValueError("num_clips and clip_length must be positive")
    if num_train_timesteps <= 0:
        raise ValueError("num_train_timesteps must be positive")

    count = num_clips if share_across_clip else num_clips * clip_length
    timesteps = torch.randint(
        0,
        num_train_timesteps,
        (count,),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    if share_across_clip:
        timesteps = timesteps.repeat_interleave(clip_length)
    return timesteps


class HierarchicalV8ClipDataset(Dataset):
    """Read V8 clips from three aligned hierarchical image trees.

    Expected layout::

        root/<split>/GT/<class>/<sequence>/<frame>.png
        root/<split>/input/<class>/<sequence>/<frame>.png
        root/<split>/mask/<class>/<sequence>/<frame>.png

    Only those three trees are inspected. Clips stay inside a leaf branch and
    inside a numerically contiguous frame run. Source masks are interpreted as
    ``0=background, 255=ROI`` and returned in V8 convention ``1=background``.
    """

    def __init__(
        self,
        dataset_root,
        split: str,
        tokenizer,
        clip_image_processor,
        clip_length: int = 4,
        stride: int = 1,
        resolution: int = 512,
    ):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = str(split)
        self.split_root = self.dataset_root / self.split
        self.clip_length = int(clip_length)
        self.stride = int(stride)
        self.resolution = int(resolution)
        self.clip_image_processor = clip_image_processor
        if self.clip_length < 2:
            raise ValueError("clip_length must be at least 2 for shared noise")
        if self.stride <= 0:
            raise ValueError("clip stride must be positive")
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")

        self.roots = {
            kind: self.split_root / kind for kind in ("GT", "input", "mask")
        }
        for kind, root in self.roots.items():
            if not root.is_dir():
                raise FileNotFoundError(
                    f"Hierarchical {kind} directory not found: {root}"
                )

        relative_files = {
            kind: {
                path.relative_to(root)
                for path in root.rglob("*.png")
                if path.is_file()
            }
            for kind, root in self.roots.items()
        }
        gt_files = relative_files["GT"]
        if not gt_files:
            raise ValueError(f"No PNG frames found below {self.roots['GT']}")
        for kind in ("input", "mask"):
            missing = sorted(gt_files - relative_files[kind])
            extra = sorted(relative_files[kind] - gt_files)
            if missing or extra:
                raise ValueError(
                    f"{kind} tree is not aligned with GT: missing={len(missing)}, "
                    f"extra={len(extra)}, first_missing={missing[:1]}, "
                    f"first_extra={extra[:1]}"
                )

        branches: Dict[Path, List[Tuple[int, Path]]] = {}
        for relative_path in gt_files:
            try:
                frame_index = int(relative_path.stem)
            except ValueError as error:
                raise ValueError(
                    f"Frame filename must have a numeric stem: {relative_path}"
                ) from error
            branches.setdefault(relative_path.parent, []).append(
                (frame_index, relative_path)
            )

        self.clips: List[Tuple[str, Tuple[Path, ...], Tuple[int, ...]]] = []
        covered_files = set()
        for branch in sorted(branches, key=lambda path: path.as_posix()):
            frames = sorted(branches[branch], key=lambda item: item[0])
            indices = [item[0] for item in frames]
            if len(indices) != len(set(indices)):
                raise ValueError(f"Duplicate numeric frame id in branch {branch}")

            runs: List[List[Tuple[int, Path]]] = []
            current_run: List[Tuple[int, Path]] = []
            for frame in frames:
                if current_run and frame[0] != current_run[-1][0] + 1:
                    runs.append(current_run)
                    current_run = []
                current_run.append(frame)
            if current_run:
                runs.append(current_run)

            for run in runs:
                last_start = len(run) - self.clip_length
                for start in range(0, last_start + 1, self.stride):
                    clip = run[start : start + self.clip_length]
                    paths = tuple(item[1] for item in clip)
                    frame_indices = tuple(item[0] for item in clip)
                    self.clips.append((branch.as_posix(), paths, frame_indices))
                    covered_files.update(paths)

        if not self.clips:
            raise ValueError(
                f"No length-{self.clip_length} clips found below {self.roots['GT']}"
            )
        self.frame_count = len(gt_files)
        self.covered_frame_count = len(covered_files)
        self.branch_count = len(branches)

        tokenized = tokenizer(
            "",
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        self.empty_input_ids = tokenized[0].detach().clone()

        resampling = getattr(Image, "Resampling", Image)
        self._rgb_resample = resampling.BILINEAR
        self._mask_resample = resampling.NEAREST

    def __len__(self) -> int:
        return len(self.clips)

    def _load_images(self, relative_paths: Tuple[Path, ...], kind: str, mode: str):
        images = []
        for relative_path in relative_paths:
            path = self.roots[kind] / relative_path
            with Image.open(path) as image:
                images.append(image.convert(mode).copy())
        return images

    def _rgb_tensor(self, image: Image.Image) -> torch.Tensor:
        resized = image.resize(
            (self.resolution, self.resolution), resample=self._rgb_resample
        )
        array = np.asarray(resized, dtype=np.float32)
        array = np.ascontiguousarray(array.transpose(2, 0, 1))
        return torch.from_numpy(array).div_(127.5).sub_(1.0)

    def _bg_mask_tensor(self, image: Image.Image) -> torch.Tensor:
        resized = image.resize(
            (self.resolution, self.resolution), resample=self._mask_resample
        )
        roi = np.asarray(resized, dtype=np.uint8) >= 128
        bg = np.ascontiguousarray((~roi).astype(np.float32)[None])
        return torch.from_numpy(bg)

    def _clip_pixels(self, images: Sequence[Image.Image]) -> torch.Tensor:
        return self.clip_image_processor(
            images=list(images), return_tensors="pt"
        ).pixel_values

    @staticmethod
    def _masked_clip_images(
        decoded_images: Sequence[Image.Image], mask_images: Sequence[Image.Image]
    ) -> Tuple[List[Image.Image], List[Image.Image]]:
        historical_fg, historical_bg = [], []
        for decoded, mask in zip(decoded_images, mask_images):
            decoded_array = np.asarray(decoded, dtype=np.uint8)
            roi = (np.asarray(mask, dtype=np.uint8) >= 128)[..., None]
            background_only = np.where(roi, 0, decoded_array).astype(np.uint8)
            roi_only = np.where(roi, decoded_array, 0).astype(np.uint8)
            # Preserve CustomDataset V4's historical checkpoint convention:
            # fg_clip_images actually receives BG-only content and vice versa.
            historical_fg.append(Image.fromarray(background_only, mode="RGB"))
            historical_bg.append(Image.fromarray(roi_only, mode="RGB"))
        return historical_fg, historical_bg

    def __getitem__(self, item: int):
        branch, relative_paths, frame_indices = self.clips[int(item)]
        gt_images = self._load_images(relative_paths, "GT", "RGB")
        decoded_images = self._load_images(relative_paths, "input", "RGB")
        mask_images = self._load_images(relative_paths, "mask", "L")
        historical_fg, historical_bg = self._masked_clip_images(
            decoded_images, mask_images
        )

        sample = {
            "pixel_values": torch.stack(
                [self._rgb_tensor(image) for image in gt_images]
            ),
            "masks": torch.stack(
                [self._bg_mask_tensor(image) for image in mask_images]
            ),
            "conditioning_pixel_values": torch.stack(
                [self._rgb_tensor(image) for image in decoded_images]
            ),
            "input_ids": self.empty_input_ids.unsqueeze(0).repeat(
                self.clip_length, 1
            ),
            "clip_images": self._clip_pixels(decoded_images),
            "fg_clip_images": self._clip_pixels(historical_fg),
            "bg_clip_images": self._clip_pixels(historical_bg),
            "drop_image_embeds": torch.zeros(
                self.clip_length, dtype=torch.long
            ),
            "video": branch,
            "frame_ids": torch.tensor(frame_indices, dtype=torch.long),
        }
        return sample


class FlatV8TestClipDataset(HierarchicalV8ClipDataset):
    """Read the legacy per-sequence BrushNet test layout.

    Expected layout::

        root/<sequence>/gt/<frame>.png
        root/<sequence>/inputs/<frame>.png
        root/<sequence>/masks/<frame>.png

    Unlike :class:`HierarchicalV8ClipDataset`, this source tree may contain
    clean GT reference frames that have no corresponding degraded input/mask
    (for example the tail of ParkScene).  Such frames cannot be sampled and
    are reported in ``ignored_gt_frame_count``; any input/mask mismatch is an
    error.  Source masks retain the shared V8 convention ``0=BG, 255=ROI``.
    """

    _SOURCE_KINDS = {"GT": "gt", "input": "inputs", "mask": "masks"}

    def __init__(
        self,
        dataset_root,
        split: str,
        tokenizer,
        clip_image_processor,
        clip_length: int = 4,
        stride: int = 1,
        resolution: int = 512,
    ):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = str(split)
        self.split_root = self.dataset_root
        self.clip_length = int(clip_length)
        self.stride = int(stride)
        self.resolution = int(resolution)
        self.clip_image_processor = clip_image_processor
        if self.clip_length < 2:
            raise ValueError("clip_length must be at least 2 for shared noise")
        if self.stride <= 0:
            raise ValueError("clip stride must be positive")
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"Flat test dataset directory not found: {self.dataset_root}"
            )

        branches: Dict[Path, List[Tuple[int, Path]]] = {}
        gt_frame_count = 0
        aligned_frame_count = 0
        for sequence_root in sorted(
            (path for path in self.dataset_root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        ):
            kind_roots = {
                kind: sequence_root / source_kind
                for kind, source_kind in self._SOURCE_KINDS.items()
            }
            missing_roots = [
                str(root) for root in kind_roots.values() if not root.is_dir()
            ]
            if missing_roots:
                # Some legacy trees also carry a ``video`` folder.  Ignore it,
                # but reject a nominal sequence that is only partially formed.
                if any(root.is_dir() for root in kind_roots.values()):
                    raise FileNotFoundError(
                        "Flat test sequence must contain gt, inputs, and masks: "
                        f"{sequence_root}; missing={missing_roots}"
                    )
                continue

            files = {
                kind: {path.name: path for path in root.glob("*.png") if path.is_file()}
                for kind, root in kind_roots.items()
            }
            gt_names = set(files["GT"])
            input_names = set(files["input"])
            mask_names = set(files["mask"])
            if input_names != mask_names:
                missing_input = sorted(mask_names - input_names)
                missing_mask = sorted(input_names - mask_names)
                raise ValueError(
                    f"inputs/masks are not aligned in {sequence_root}: "
                    f"missing_input={len(missing_input)}, "
                    f"missing_mask={len(missing_mask)}, "
                    f"first_missing_input={missing_input[:1]}, "
                    f"first_missing_mask={missing_mask[:1]}"
                )
            input_without_gt = sorted(input_names - gt_names)
            if input_without_gt:
                raise ValueError(
                    f"inputs/masks contain frames without GT in {sequence_root}: "
                    f"count={len(input_without_gt)}, "
                    f"first={input_without_gt[:1]}"
                )

            usable_names = sorted(gt_names & input_names)
            if not usable_names:
                raise ValueError(
                    f"No aligned gt/input/mask PNG frames found in {sequence_root}"
                )
            gt_frame_count += len(gt_names)
            aligned_frame_count += len(usable_names)
            branch = Path(sequence_root.name)
            indexed_paths = []
            for filename in usable_names:
                relative_path = branch / filename
                try:
                    frame_index = int(relative_path.stem)
                except ValueError as error:
                    raise ValueError(
                        f"Frame filename must have a numeric stem: {relative_path}"
                    ) from error
                indexed_paths.append((frame_index, relative_path))
            if len({frame for frame, _ in indexed_paths}) != len(indexed_paths):
                raise ValueError(f"Duplicate numeric frame id in branch {branch}")
            branches[branch] = indexed_paths

        if not branches:
            raise ValueError(
                f"No valid flat test sequences found below {self.dataset_root}"
            )

        self.clips: List[Tuple[str, Tuple[Path, ...], Tuple[int, ...]]] = []
        covered_files = set()
        for branch in sorted(branches, key=lambda path: path.as_posix()):
            frames = sorted(branches[branch], key=lambda item: item[0])
            runs: List[List[Tuple[int, Path]]] = []
            current_run: List[Tuple[int, Path]] = []
            for frame in frames:
                if current_run and frame[0] != current_run[-1][0] + 1:
                    runs.append(current_run)
                    current_run = []
                current_run.append(frame)
            if current_run:
                runs.append(current_run)

            for run in runs:
                last_start = len(run) - self.clip_length
                starts = list(range(0, last_start + 1, self.stride))
                # Evaluation should score every source frame. Retain the
                # requested stride, then append the final window when stride
                # does not land exactly on it; this only adds a tail overlap.
                if starts[-1] != last_start:
                    starts.append(last_start)
                for start in starts:
                    clip = run[start : start + self.clip_length]
                    paths = tuple(item[1] for item in clip)
                    frame_indices = tuple(item[0] for item in clip)
                    self.clips.append((branch.as_posix(), paths, frame_indices))
                    covered_files.update(paths)

        if not self.clips:
            raise ValueError(
                f"No length-{self.clip_length} clips found below {self.dataset_root}"
            )
        self.frame_count = aligned_frame_count
        self.total_gt_frame_count = gt_frame_count
        self.ignored_gt_frame_count = gt_frame_count - aligned_frame_count
        self.covered_frame_count = len(covered_files)
        self.branch_count = len(branches)
        self.roots = {}

        tokenized = tokenizer(
            "",
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        self.empty_input_ids = tokenized[0].detach().clone()

        resampling = getattr(Image, "Resampling", Image)
        self._rgb_resample = resampling.BILINEAR
        self._mask_resample = resampling.NEAREST

    def _load_images(self, relative_paths: Tuple[Path, ...], kind: str, mode: str):
        images = []
        source_kind = self._SOURCE_KINDS[kind]
        for relative_path in relative_paths:
            path = self.dataset_root / relative_path.parent / source_kind / relative_path.name
            with Image.open(path) as image:
                images.append(image.convert(mode).copy())
        return images


class SharedNoiseClipDataset(Dataset):
    """Group a frame dataset into clips without crossing manifest videos.

    The wrapped dataset must expose ``image_ids`` and return the tensor fields
    consumed by the V8 BrushNet trainer. The manifest uses inclusive ``start``
    and ``end`` frame indices.
    """

    def __init__(
        self,
        frame_dataset: Dataset,
        manifest_path,
        split: str = "train",
        clip_length: int = 4,
        stride: int = 1,
    ):
        if not hasattr(frame_dataset, "image_ids"):
            raise ValueError("frame_dataset must expose an image_ids sequence")
        self.frame_dataset = frame_dataset
        self.manifest_path = Path(manifest_path)
        self.split = str(split)
        self.clip_length = int(clip_length)
        self.stride = int(stride)
        if self.clip_length < 2:
            raise ValueError("clip_length must be at least 2 for shared noise")
        if self.stride <= 0:
            raise ValueError("clip stride must be positive")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)

        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        videos = manifest.values() if isinstance(manifest, Mapping) else manifest

        index_by_frame: Dict[int, int] = {}
        for dataset_index, image_id in enumerate(frame_dataset.image_ids):
            frame_index = int(image_id)
            if frame_index in index_by_frame:
                raise ValueError(f"Duplicate numeric frame id: {frame_index}")
            index_by_frame[frame_index] = dataset_index

        self.clips: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
        selected_videos = 0
        for video in videos:
            if str(video.get("split")) != self.split:
                continue
            selected_videos += 1
            name = str(video.get("name", f"video_{selected_videos}"))
            start, end = int(video["start"]), int(video["end"])
            if end < start:
                raise ValueError(f"Invalid frame range for {name}: {start}..{end}")
            last_start = end - self.clip_length + 1
            for clip_start in range(start, last_start + 1, self.stride):
                frame_numbers = tuple(
                    range(clip_start, clip_start + self.clip_length)
                )
                missing = [
                    frame for frame in frame_numbers if frame not in index_by_frame
                ]
                if missing:
                    raise FileNotFoundError(
                        f"Manifest clip {name}:{clip_start} references missing "
                        f"frames {missing}"
                    )
                dataset_indices = tuple(
                    index_by_frame[frame] for frame in frame_numbers
                )
                self.clips.append((name, dataset_indices, frame_numbers))

        if selected_videos == 0:
            raise ValueError(
                f"No videos with split={self.split!r} in {self.manifest_path}"
            )
        if not self.clips:
            raise ValueError(
                f"No length-{self.clip_length} clips for split={self.split!r}"
            )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, item: int):
        video, dataset_indices, frame_numbers = self.clips[item]
        frames = [self.frame_dataset[index] for index in dataset_indices]
        sample = {
            key: torch.stack([frame[key] for frame in frames], dim=0)
            for key in CLIP_TENSOR_KEYS
        }
        sample["drop_image_embeds"] = torch.tensor(
            [frame["drop_image_embed"] for frame in frames], dtype=torch.long
        )
        sample["video"] = video
        sample["frame_ids"] = torch.tensor(frame_numbers, dtype=torch.long)
        return sample


def collate_shared_noise_clips(examples: Sequence[Mapping]) -> Dict[str, object]:
    """Collate ``[T,...]`` samples and flatten them to the V8 ``[B*T,...]`` API."""
    if not examples:
        raise ValueError("Cannot collate an empty clip batch")
    clip_length = int(examples[0]["pixel_values"].shape[0])
    if clip_length <= 0:
        raise ValueError("Clip length must be positive")
    for example in examples:
        if int(example["pixel_values"].shape[0]) != clip_length:
            raise ValueError("All clips in a batch must have the same length")

    batch: Dict[str, object] = {}
    for key in CLIP_TENSOR_KEYS + ("drop_image_embeds",):
        stacked = torch.stack([example[key] for example in examples], dim=0)
        batch[key] = stacked.flatten(0, 1)
    batch["clip_batch_size"] = len(examples)
    batch["num_frames"] = clip_length
    batch["videos"] = [str(example["video"]) for example in examples]
    batch["frame_ids"] = torch.stack(
        [example["frame_ids"] for example in examples], dim=0
    )
    return batch
