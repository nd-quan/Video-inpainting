#!/usr/bin/env python
"""Compare STC feature alignment for V5, V6-light-flow DCN, and V8-RAFT DCN.

The script intentionally visualizes only the exact adjacent frame pairs already
selected by the V7 RAFT-student visualizer.  The reference CSV is therefore the
selection authority; it avoids silently cherry-picking a different subset.

Raw feature-channel colors are *not* compared across models.  V6 joint was
trained from another V5 lineage, so corresponding channels are not guaranteed
to retain the same meaning.  Instead, each case gets a case-local PCA color
projection and all cross-case conclusions are based on within-model
correspondence maps (cosine/L1), flow, and DCN diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from diffusers.models.stc_flow_training import resize_flow_sequence  # noqa: E402
from STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (  # noqa: E402
    backward_warp_feature,
)
from STC_encoder_v5_relative_crossclip.cross_clip_data import (  # noqa: E402
    CrossClipTeacherFlowV8Dataset,
)
from STC_encoder_v5_relative_crossclip.relative_crossclip_stc_adapter import (  # noqa: E402
    RelativeCrossClipBGSTCAdapter,
)
from STC_encoder_v6_flow_deformable.flow_guided_deformable_stc_adapter import (  # noqa: E402
    FlowGuidedDeformableBGSTCAdapter,
)
from STC_encoder_v8_raft_deformable.raft_flow_provider import (  # noqa: E402
    FrozenV7RAFTFlowProvider,
    resolve_raft_student_component,
)
from STC_encoder_v8_raft_deformable.raft_guided_deformable_stc_adapter import (  # noqa: E402
    RAFTGuidedDeformableBGSTCAdapter,
)


DEFAULT_ROOT = Path("/home/cilab/ndquan/videoInpainting/code/BrushNet")
DEFAULT_DATASET = Path("/home/cilab/ndquan/videoInpainting/SFU_STC_flow")
DEFAULT_V7_REFERENCE = (
    DEFAULT_ROOT / "experiments/visualize_v7_raft_student_best_valid"
)


@dataclass(frozen=True)
class SelectedPair:
    sequence: str
    frame_t: int
    frame_t1: int
    reference_dataset_index: int
    reference_montage: Path


@dataclass(frozen=True)
class ClipPair:
    selected: SelectedPair
    dataset_index: int
    local_pair_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference_dir",
        type=Path,
        default=DEFAULT_V7_REFERENCE,
        help="V7 visualizer output containing montages/ and per_pair_metrics.csv.",
    )
    parser.add_argument(
        "--selection_csv",
        type=Path,
        default=None,
        help="Optional replacement for reference_dir/per_pair_metrics.csv.",
    )
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--teacher_flow_root",
        type=Path,
        default=DEFAULT_DATASET / "teacher_flows_512x512",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "examples/brushnet/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5"
        ),
        help="Only its tokenizer is required by the existing hierarchical loader.",
    )
    parser.add_argument(
        "--v5_checkpoint",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "experiments/train_stc_v5_relative_crossclip_T16_S12_sharedNoise_0.95"
            / "checkpoint-5000/stc_v5_model"
        ),
    )
    parser.add_argument(
        "--v6_checkpoint",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "experiments/train_stc_v6_joint_fixedlr_T16_S12_sharedNoise_0.95"
            / "checkpoint-1750/stc_v6_model"
        ),
    )
    parser.add_argument(
        "--v8_checkpoint",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "experiments/train_stc_v8_raft_deform_only_T16_S12_sharedNoise_0.95"
            / "checkpoint-1750/stc_v8_model"
        ),
        help="Default matches V6's step. Override to compare a later V8 checkpoint.",
    )
    parser.add_argument(
        "--raft_student_path",
        type=Path,
        default=(
            DEFAULT_ROOT
            / "experiments/train_v7_raft_student_flow/checkpoint-0004750/raft_student"
        ),
    )
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--clip_length", type=int, default=16)
    parser.add_argument("--clip_stride", type=int, default=12)
    parser.add_argument(
        "--use_cross_clip",
        action="store_true",
        help="Construct the same detached predecessor memory for all three cases.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="Optional prefix of the reference pair list; useful for a smoke test.",
    )
    parser.add_argument("--raft_pair_batch_size", type=int, default=1)
    parser.add_argument("--deformable_alignment_scale", type=float, default=1.0)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument(
        "--tile_gap",
        type=int,
        default=0,
        help="Dark-gray spacing in pixels between montage tiles.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def resolve_component(path_value: Path, component_name: str) -> Path:
    path = path_value.expanduser().resolve()
    candidates = (path / component_name, path)
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (
            candidate / "diffusion_pytorch_model.safetensors"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot find {component_name}/config.json below {path}"
    )


def json_dump(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def rgb_to_bgr(frame: torch.Tensor) -> np.ndarray:
    value = (
        ((frame.detach().float().cpu() + 1.0) * 127.5)
        .clamp(0, 255)
        .permute(1, 2, 0)
        .byte()
        .numpy()
    )
    return cv2.cvtColor(value, cv2.COLOR_RGB2BGR)


def mask_to_bgr(mask: torch.Tensor) -> np.ndarray:
    value = (
        mask.detach().float().cpu().squeeze().clamp(0, 1).mul(255).byte().numpy()
    )
    return cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)


def flow_to_bgr(flow: torch.Tensor, scale: float) -> np.ndarray:
    value = flow.detach().float().cpu().permute(1, 2, 0).numpy()
    magnitude, angle = cv2.cartToPolar(value[..., 0], value[..., 1])
    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((angle * 90.0 / np.pi) % 180.0).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(
        magnitude / max(float(scale), 1e-6) * 255.0, 0, 255
    ).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def scalar_to_bgr(value: torch.Tensor, scale: float, *, colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    image = value.detach().float().cpu().squeeze().numpy()
    image = np.nan_to_num(image, nan=0.0, posinf=float(scale), neginf=0.0)
    image = np.clip(image / max(float(scale), 1e-6) * 255.0, 0, 255).astype(
        np.uint8
    )
    return cv2.applyColorMap(image, colormap)


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    height = min(30, result.shape[0])
    cv2.rectangle(result, (0, 0), (result.shape[1], height), (0, 0, 0), -1)
    cv2.putText(
        result,
        text,
        (6, min(21, height - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def montage(
    tiles: Sequence[Tuple[str, np.ndarray]],
    tile_size: int,
    columns: int = 4,
    gap: int = 0,
) -> np.ndarray:
    gap = int(gap)
    if gap < 0:
        raise ValueError("montage gap must be non-negative")
    rendered = [
        label(
            cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA),
            title,
        )
        for title, image in tiles
    ]
    blank = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    while len(rendered) % columns:
        rendered.append(blank.copy())
    vertical_gap = np.full((tile_size, gap, 3), 32, dtype=np.uint8)
    rows = []
    for index in range(0, len(rendered), columns):
        row_tiles = rendered[index : index + columns]
        row_parts = []
        for tile_index, tile in enumerate(row_tiles):
            if tile_index and gap:
                row_parts.append(vertical_gap)
            row_parts.append(tile)
        rows.append(np.concatenate(row_parts, axis=1))
    if not gap:
        return np.concatenate(rows, axis=0)
    horizontal_gap = np.full((gap, rows[0].shape[1], 3), 32, dtype=np.uint8)
    parts = []
    for row_index, row in enumerate(rows):
        if row_index:
            parts.append(horizontal_gap)
        parts.append(row)
    return np.concatenate(parts, axis=0)


def percentile_scale(values: Sequence[torch.Tensor], percentile: float = 99.0) -> float:
    flat = torch.cat(
        [value.detach().float().flatten().cpu() for value in values if value.numel()]
    )
    if not flat.numel():
        return 1.0
    return max(float(torch.quantile(flat, float(percentile) / 100.0)), 1e-6)


def resize_feature_flow_to_rgb(flow: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    return resize_flow_sequence(flow[None, None].float(), image_size)[0, 0]


def pca_feature_bgr_maps(features: Sequence[torch.Tensor]) -> List[np.ndarray]:
    """One PCA projection shared within *one model and one selected pair*."""
    if not features:
        return []
    channels = int(features[0].shape[0])
    height, width = features[0].shape[-2:]
    if any(tuple(feature.shape) != (channels, height, width) for feature in features):
        raise ValueError("PCA features must share C/H/W")
    flattened = torch.cat(
        [feature.detach().float().cpu().permute(1, 2, 0).reshape(-1, channels) for feature in features],
        dim=0,
    )
    mean = flattened.mean(dim=0, keepdim=True)
    centered = flattened - mean
    # 64 channels x at most a few 64x64 maps: full SVD is small and deterministic.
    _, _, vectors = torch.linalg.svd(centered, full_matrices=False)
    basis = vectors[: min(3, channels)].T
    projected = centered @ basis
    if projected.shape[1] < 3:
        projected = F.pad(projected, (0, 3 - projected.shape[1]))
    low = projected.amin(dim=0, keepdim=True)
    high = projected.amax(dim=0, keepdim=True)
    normalized = (projected - low) / (high - low).clamp_min(1e-6)
    maps = []
    offset = 0
    pixels = height * width
    for _ in features:
        rgb = normalized[offset : offset + pixels].reshape(height, width, 3)
        offset += pixels
        image = (rgb.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
        maps.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return maps


def cosine_distance(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    candidate = F.normalize(candidate.float(), dim=0, eps=1e-6)
    target = F.normalize(target.float(), dim=0, eps=1e-6)
    return (1.0 - (candidate * target).sum(dim=0, keepdim=True)).clamp_(0.0, 2.0)


def mean_abs_channel(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (candidate.float() - target.float()).abs().mean(dim=0, keepdim=True)


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> float:
    value = value.float()
    weight = weight.float()
    return float((value * weight).sum() / weight.sum().clamp_min(1e-8))


def offset_magnitude(offset: torch.Tensor, groups: int, kernel_size: int) -> torch.Tensor:
    """Mean residual DCN displacement over groups/kernel points at each pixel."""
    expected = 2 * int(groups) * int(kernel_size) * int(kernel_size)
    if offset.ndim != 3 or offset.shape[0] != expected:
        raise ValueError(
            f"Expected residual offsets [{expected},H,W], got {tuple(offset.shape)}"
        )
    height, width = offset.shape[-2:]
    value = offset.float().reshape(
        int(groups), int(kernel_size) * int(kernel_size), 2, height, width
    )
    return value.square().sum(dim=2).sqrt().mean(dim=(0, 1), keepdim=True)


def extract_mask_support(
    source_bg: torch.Tensor, target_bg: torch.Tensor, backward_flow: torch.Tensor
) -> torch.Tensor:
    warped_source_bg, valid = backward_warp_feature(
        source_bg[None].float(), backward_flow[None].float(), fallback=target_bg[None]
    )
    return (
        (warped_source_bg >= 0.5).to(target_bg.dtype)
        * (target_bg >= 0.5).to(target_bg.dtype)
        * valid.to(target_bg.dtype)
    )[0]


def read_selected_pairs(args: argparse.Namespace) -> List[SelectedPair]:
    reference_dir = args.reference_dir.expanduser().resolve()
    csv_path = (
        args.selection_csv.expanduser().resolve()
        if args.selection_csv is not None
        else reference_dir / "per_pair_metrics.csv"
    )
    montage_dir = reference_dir / "montages"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not montage_dir.is_dir():
        raise FileNotFoundError(montage_dir)
    selected: List[SelectedPair] = []
    seen = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sequence = str(row["sequence"])
            frame_t, frame_t1 = int(row["frame_t"]), int(row["frame_t1"])
            key = (sequence, frame_t, frame_t1)
            if key in seen:
                continue
            seen.add(key)
            name = f"{sanitize(sequence)}_f{frame_t:06d}_{frame_t1:06d}.png"
            reference_montage = montage_dir / name
            if not reference_montage.is_file():
                raise FileNotFoundError(
                    f"Reference CSV row {key} has no montage: {reference_montage}"
                )
            selected.append(
                SelectedPair(
                    sequence=sequence,
                    frame_t=frame_t,
                    frame_t1=frame_t1,
                    reference_dataset_index=int(row.get("dataset_index", -1)),
                    reference_montage=reference_montage,
                )
            )
    if args.max_pairs is not None:
        if args.max_pairs < 1:
            raise ValueError("--max_pairs must be positive when supplied")
        selected = selected[: int(args.max_pairs)]
    if not selected:
        raise ValueError("The requested V7 reference selection is empty")
    return selected


def build_dataset(
    args: argparse.Namespace, selected: Sequence[SelectedPair]
) -> CrossClipTeacherFlowV8Dataset:
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.pretrained_model_name_or_path.expanduser().resolve()),
        subfolder="tokenizer",
        use_fast=False,
    )
    branches = sorted({item.sequence for item in selected})
    return CrossClipTeacherFlowV8Dataset(
        dataset_root=str(args.dataset_root.expanduser().resolve()),
        split=args.split,
        tokenizer=tokenizer,
        clip_image_processor=CLIPImageProcessor(),
        clip_length=int(args.clip_length),
        stride=int(args.clip_stride),
        resolution=int(args.resolution),
        include_branches=branches,
        teacher_flow_root=str(args.teacher_flow_root.expanduser().resolve()),
    )


def append_exact_pair_windows(
    dataset: CrossClipTeacherFlowV8Dataset,
    selected: Sequence[SelectedPair],
) -> int:
    """Add deterministic tail/shifted windows needed by the V7 pair selection.

    The training dataset emits only windows whose starts lie on ``clip_stride``.
    The V7 visualizer instead selects adjacent pairs uniformly across each
    sequence, so a selected pair can legitimately lie in a tail that no
    stride-aligned T=16 window covers.  We append an exact contiguous window
    only when every RGB/mask frame and every cached teacher pair exists.  Thus
    this cannot cross a manifest source segment accidentally.
    """
    existing = {
        (str(branch), tuple(int(value) for value in frame_ids))
        for branch, _, frame_ids in dataset.clips
    }
    added = 0
    length = int(dataset.clip_length)
    center = (length - 2) / 2.0
    for item in selected:
        branch = str(item.sequence)
        gt_dir = dataset.roots["GT"] / Path(branch)
        if not gt_dir.is_dir():
            raise FileNotFoundError(gt_dir)
        by_id: Dict[int, Path] = {}
        for path in gt_dir.glob("*.png"):
            try:
                frame_id = int(path.stem)
            except ValueError:
                continue
            if frame_id in by_id:
                raise ValueError(f"Duplicate numeric frame ID in {gt_dir}: {frame_id}")
            by_id[frame_id] = path.relative_to(dataset.roots["GT"])

        starts = range(item.frame_t - length + 2, item.frame_t + 1)
        candidates = []
        for start in starts:
            ids = tuple(range(int(start), int(start) + length))
            if item.frame_t not in ids or item.frame_t1 not in ids:
                continue
            if any(frame_id not in by_id for frame_id in ids):
                continue
            paths = tuple(by_id[frame_id] for frame_id in ids)
            if any(
                not (dataset.roots[kind] / path).is_file()
                for kind in ("input", "mask")
                for path in paths
            ):
                continue
            if any(
                not dataset._teacher_path(branch, frame0, frame1).is_file()
                for frame0, frame1 in zip(ids[:-1], ids[1:])
            ):
                continue
            local_index = ids.index(item.frame_t)
            candidates.append((abs(local_index - center), int(start), paths, ids))
        if not candidates:
            raise KeyError(
                "No contiguous manifest-safe T={} window contains selected pair {}"
                .format(length, (branch, item.frame_t, item.frame_t1))
            )
        _, _, paths, ids = min(candidates, key=lambda value: (value[0], value[1]))
        key = (branch, ids)
        if key not in existing:
            dataset.clips.append((branch, paths, ids))
            existing.add(key)
            added += 1

    if added:
        # Cross-clip indices have to remain aligned with the expanded clip list
        # even if the caller ultimately requests the local-only visualizer.
        dataset.rebuild_predecessors()
    return added


def locate_clip_pairs(
    dataset: CrossClipTeacherFlowV8Dataset,
    selected: Sequence[SelectedPair],
    use_cross_clip: bool,
) -> List[ClipPair]:
    candidates: Dict[Tuple[str, int, int], List[Tuple[int, int]]] = {}
    for index, (sequence, _, frame_ids) in enumerate(dataset.clips):
        for local_index, (frame_t, frame_t1) in enumerate(
            zip(frame_ids[:-1], frame_ids[1:])
        ):
            candidates.setdefault(
                (str(sequence), int(frame_t), int(frame_t1)), []
            ).append((index, local_index))

    resolved: List[ClipPair] = []
    center = (int(dataset.clip_length) - 2) / 2.0
    for item in selected:
        key = (item.sequence, item.frame_t, item.frame_t1)
        choices = candidates.get(key, [])
        if not choices:
            raise KeyError(
                "Could not place V7 reference pair into a T={} clip: {}".format(
                    dataset.clip_length, key
                )
            )

        def rank(value: Tuple[int, int]) -> Tuple[int, float, int, int]:
            clip_index, local_index = value
            has_predecessor = dataset.predecessor_indices[clip_index] is not None
            predecessor_penalty = 0 if (not use_cross_clip or has_predecessor) else 1
            start = int(dataset.clips[clip_index][2][0])
            return predecessor_penalty, abs(local_index - center), start, clip_index

        clip_index, local_index = min(choices, key=rank)
        resolved.append(
            ClipPair(
                selected=item,
                dataset_index=int(clip_index),
                local_pair_index=int(local_index),
            )
        )
    return resolved


def autocast_context(device: torch.device, no_amp: bool):
    if device.type == "cuda" and not no_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def v5_or_v6_forward(
    model,
    rgb: torch.Tensor,
    bg: torch.Tensor,
    frame_ids: torch.Tensor,
    *,
    previous_rgb: torch.Tensor,
    previous_bg: torch.Tensor,
    previous_ids: torch.Tensor,
    previous_valid: torch.Tensor,
    use_cross_clip: bool,
    deformable_alignment_scale: float,
):
    common = {
        "output_size": (int(rgb.shape[-2] // 8), int(rgb.shape[-1] // 8)),
        "predict_flow": True,
        "return_dict": True,
        "frame_ids": frame_ids,
    }
    if hasattr(model, "deformable_alignment"):
        common["deformable_alignment_scale"] = float(deformable_alignment_scale)
    memory = None
    if use_cross_clip and bool(previous_valid.any()):
        previous_common = dict(common)
        previous_common["frame_ids"] = previous_ids
        previous_common["frame_valid_mask"] = previous_valid
        previous = model(previous_rgb, previous_bg, **previous_common)
        memory = previous.temporal_memory.detach()
    return model(rgb, bg, temporal_memory=memory, **common)


def v8_forward(
    model: RAFTGuidedDeformableBGSTCAdapter,
    provider: FrozenV7RAFTFlowProvider,
    rgb: torch.Tensor,
    bg: torch.Tensor,
    frame_ids: torch.Tensor,
    *,
    previous_rgb: torch.Tensor,
    previous_bg: torch.Tensor,
    previous_ids: torch.Tensor,
    previous_valid: torch.Tensor,
    use_cross_clip: bool,
    deformable_alignment_scale: float,
):
    common = {
        "output_size": (int(rgb.shape[-2] // 8), int(rgb.shape[-1] // 8)),
        "predict_flow": True,
        "return_dict": True,
        "deformable_alignment_scale": float(deformable_alignment_scale),
    }
    memory = None
    if use_cross_clip and bool(previous_valid.any()):
        previous_flow = provider.predict_sequence(previous_rgb)
        previous = model(
            previous_rgb,
            previous_bg,
            frame_ids=previous_ids,
            frame_valid_mask=previous_valid,
            raft_flow_forward_rgb=previous_flow.forward,
            raft_flow_backward_rgb=previous_flow.backward,
            **common,
        )
        memory = previous.temporal_memory.detach()
    current_flow = provider.predict_sequence(rgb)
    return model(
        rgb,
        bg,
        frame_ids=frame_ids,
        temporal_memory=memory,
        raft_flow_forward_rgb=current_flow.forward,
        raft_flow_backward_rgb=current_flow.backward,
        **common,
    )


def prepare_case(
    name: str,
    output,
    local_pair: int,
    bg_feature: torch.Tensor,
    *,
    deform_groups: Optional[int] = None,
    deform_kernel_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Extract comparable pair tensors on one case's own feature space."""
    spatial = output.spatial_features[0].float()
    target = spatial[local_pair + 1]
    source = spatial[local_pair]
    backward = output.predicted_flow_backward[0, local_pair].float()
    base_candidate, flow_valid = backward_warp_feature(
        source[None], backward[None], fallback=target[None]
    )
    base_candidate = base_candidate[0].float()
    support = extract_mask_support(
        bg_feature[local_pair], bg_feature[local_pair + 1], backward
    )
    case: Dict[str, torch.Tensor] = {
        "target": target,
        "source": source,
        "base_candidate": base_candidate,
        "base_valid": flow_valid[0].float(),
        "support": support,
        "aligned_target": output.aligned_spatial_features[0, local_pair + 1].float(),
        "temporal_target": output.features[0, local_pair + 1].float(),
        "delta_bg": output.delta_bg[0, local_pair + 1].float(),
        "flow_backward": backward,
    }
    if name == "v5":
        case["candidate"] = base_candidate
        case["candidate_reliability"] = support
        return case

    candidate = output.deformed_previous_features[0, local_pair].float()
    reliability = output.deform_reliability_backward[0, local_pair].float()
    residual = output.residual_offset_backward[0, local_pair].float()
    modulation = output.modulation_mask_backward[0, local_pair].float()
    if deform_groups is None or deform_kernel_size is None:
        raise ValueError("DCN metadata is required for V6/V8")
    case.update(
        {
            "candidate": candidate,
            "candidate_reliability": reliability,
            "dcn_minus_base": mean_abs_channel(candidate, base_candidate),
            "residual_offset_magnitude": offset_magnitude(
                residual, deform_groups, deform_kernel_size
            ),
            "modulation_mask_mean": modulation.mean(dim=0, keepdim=True),
            "base_aligned_target": output.base_aligned_spatial_features[
                0, local_pair + 1
            ].float(),
        }
    )
    if name == "v8":
        case["legacy_flow_backward"] = output.legacy_flow_backward[0, local_pair].float()
        case["raft_flow_backward_rgb"] = output.raft_flow_backward_rgb[
            0, local_pair
        ].float()
    return case


def teacher_feature_pair(sample: Mapping[str, torch.Tensor], local_pair: int, size: Tuple[int, int]):
    teacher = resize_flow_sequence(
        sample["teacher_flow_backward"][local_pair : local_pair + 1]
        .unsqueeze(0)
        .float(),
        size,
    )[0, 0]
    valid = F.interpolate(
        sample["teacher_valid_backward"][local_pair : local_pair + 1].float(),
        size=size,
        mode="nearest",
    )[0]
    return teacher, valid


def case_metrics(
    name: str,
    case: Mapping[str, torch.Tensor],
    teacher_flow: torch.Tensor,
    teacher_valid: torch.Tensor,
) -> Dict[str, object]:
    support = case["support"]
    candidate = case["candidate"]
    target = case["target"]
    base_candidate = case["base_candidate"]
    candidate_cos = cosine_distance(candidate, target)
    base_cos = cosine_distance(base_candidate, target)
    candidate_l1 = mean_abs_channel(candidate, target)
    flow_epe = (case["flow_backward"] - teacher_flow).square().sum(
        dim=0, keepdim=True
    ).sqrt()
    teacher_weight = teacher_valid * support
    result: Dict[str, object] = {
        "case": name,
        "bg_support_ratio": float(support.mean()),
        "base_cosine_distance_bg": weighted_mean(base_cos, support),
        "candidate_cosine_distance_bg": weighted_mean(candidate_cos, support),
        "candidate_l1_bg": weighted_mean(candidate_l1, support),
        "alignment_change_abs_bg": weighted_mean(
            mean_abs_channel(case["aligned_target"], target), support
        ),
        "temporal_change_abs_bg": weighted_mean(
            mean_abs_channel(case["temporal_target"], case["aligned_target"]), support
        ),
        "delta_bg_l2_mean": float(case["delta_bg"].square().sum(dim=0).sqrt().mean()),
        "flow_teacher_epe_bg": weighted_mean(flow_epe, teacher_weight),
        "flow_teacher_valid_bg_ratio": float(teacher_weight.mean()),
    }
    if name != "v5":
        residual = case["residual_offset_magnitude"]
        result.update(
            {
                "dcn_minus_base_abs_bg": weighted_mean(case["dcn_minus_base"], support),
                "residual_offset_magnitude_mean": float(residual.mean()),
                "residual_offset_magnitude_p95": float(
                    torch.quantile(residual.flatten(), 0.95)
                ),
                "modulation_mask_mean": float(case["modulation_mask_mean"].mean()),
                "reliability_mean": float(case["candidate_reliability"].mean()),
                "dcn_candidate_cosine_gain_vs_base": (
                    weighted_mean(base_cos, support)
                    - weighted_mean(candidate_cos, support)
                ),
            }
        )
    return result


def finite_average(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    fieldnames: List[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def make_pair_montage(
    *,
    sample: Mapping[str, torch.Tensor],
    local_pair: int,
    cases: Mapping[str, Mapping[str, torch.Tensor]],
    teacher_flow: torch.Tensor,
    tile_size: int,
    v6_groups: int,
    v6_kernel: int,
    v8_groups: int,
    v8_kernel: int,
) -> np.ndarray:
    del v6_groups, v6_kernel, v8_groups, v8_kernel  # validated during extraction
    rgb_t = sample["conditioning_pixel_values"][local_pair]
    rgb_t1 = sample["conditioning_pixel_values"][local_pair + 1]
    bg_t = sample["masks"][local_pair]
    bg_t1 = sample["masks"][local_pair + 1]
    rgb_size = tuple(int(value) for value in rgb_t.shape[-2:])

    v5, v6, v8 = cases["v5"], cases["v6"], cases["v8"]
    v5_flow = resize_feature_flow_to_rgb(v5["flow_backward"], rgb_size)
    v6_flow = resize_feature_flow_to_rgb(v6["flow_backward"], rgb_size)
    v8_flow = v8["raft_flow_backward_rgb"]
    flow_scale = percentile_scale((teacher_flow, v5_flow, v6_flow, v8_flow))
    delta_scale = percentile_scale(
        (
            v5["delta_bg"].square().sum(dim=0, keepdim=True).sqrt(),
            v6["delta_bg"].square().sum(dim=0, keepdim=True).sqrt(),
            v8["delta_bg"].square().sum(dim=0, keepdim=True).sqrt(),
        )
    )
    effect_scale = percentile_scale((v6["dcn_minus_base"], v8["dcn_minus_base"]))
    offset_scale = percentile_scale(
        (v6["residual_offset_magnitude"], v8["residual_offset_magnitude"])
    )

    # Colors are intentionally shared only within each case.  The respective
    # target/candidate/aligned/temporal features retain comparable colors.
    v5_pca = pca_feature_bgr_maps(
        (v5["base_candidate"], v5["target"], v5["aligned_target"], v5["temporal_target"])
    )
    v6_pca = pca_feature_bgr_maps(
        (v6["base_candidate"], v6["candidate"], v6["target"], v6["temporal_target"])
    )
    v8_pca = pca_feature_bgr_maps(
        (v8["base_candidate"], v8["candidate"], v8["target"], v8["temporal_target"])
    )

    v5_error = cosine_distance(v5["candidate"], v5["target"])
    v6_base_error = cosine_distance(v6["base_candidate"], v6["target"])
    v6_dcn_error = cosine_distance(v6["candidate"], v6["target"])
    v8_base_error = cosine_distance(v8["base_candidate"], v8["target"])
    v8_dcn_error = cosine_distance(v8["candidate"], v8["target"])
    zero = np.zeros_like(rgb_to_bgr(rgb_t))

    tiles: List[Tuple[str, np.ndarray]] = [
        ("Degraded t", rgb_to_bgr(rgb_t)),
        ("Degraded t+1", rgb_to_bgr(rgb_t1)),
        ("M_BG t (white=restore)", mask_to_bgr(bg_t)),
        ("M_BG t+1 (white=restore)", mask_to_bgr(bg_t1)),
        ("Teacher flow t+1->t", flow_to_bgr(teacher_flow, flow_scale)),
        ("V5 light flow", flow_to_bgr(v5_flow, flow_scale)),
        ("V6 light flow", flow_to_bgr(v6_flow, flow_scale)),
        ("V8 RAFT-student flow", flow_to_bgr(v8_flow, flow_scale)),
        ("V5 warped prev PCA*", v5_pca[0]),
        ("V5 target spatial PCA*", v5_pca[1]),
        ("V5 cosine error", scalar_to_bgr(v5_error, 1.0)),
        ("V5 post-temporal PCA*", v5_pca[3]),
        ("V6 light-base PCA*", v6_pca[0]),
        ("V6 DCN prev->current PCA*", v6_pca[1]),
        ("V6 target spatial PCA*", v6_pca[2]),
        ("V6 DCN cosine error", scalar_to_bgr(v6_dcn_error, 1.0)),
        ("V8 RAFT-base PCA*", v8_pca[0]),
        ("V8 DCN prev->current PCA*", v8_pca[1]),
        ("V8 target spatial PCA*", v8_pca[2]),
        ("V8 DCN cosine error", scalar_to_bgr(v8_dcn_error, 1.0)),
        ("V6 base cosine error", scalar_to_bgr(v6_base_error, 1.0)),
        ("V8 RAFT-base cosine error", scalar_to_bgr(v8_base_error, 1.0)),
        ("V6 |DCN-base| mean_C", scalar_to_bgr(v6["dcn_minus_base"], effect_scale)),
        ("V8 |DCN-RAFT-base| mean_C", scalar_to_bgr(v8["dcn_minus_base"], effect_scale)),
        ("V6 residual offset magnitude", scalar_to_bgr(v6["residual_offset_magnitude"], offset_scale)),
        ("V8 residual offset magnitude", scalar_to_bgr(v8["residual_offset_magnitude"], offset_scale)),
        ("V6 modulation mask mean", scalar_to_bgr(v6["modulation_mask_mean"], 1.0)),
        ("V8 modulation mask mean", scalar_to_bgr(v8["modulation_mask_mean"], 1.0)),
        ("V6 BG reliability", scalar_to_bgr(v6["candidate_reliability"], 1.0)),
        ("V8 BG reliability", scalar_to_bgr(v8["candidate_reliability"], 1.0)),
        ("V5 delta_BG L2", scalar_to_bgr(v5["delta_bg"].square().sum(dim=0, keepdim=True).sqrt(), delta_scale)),
        ("V6 delta_BG L2", scalar_to_bgr(v6["delta_bg"].square().sum(dim=0, keepdim=True).sqrt(), delta_scale)),
        ("V8 delta_BG L2", scalar_to_bgr(v8["delta_bg"].square().sum(dim=0, keepdim=True).sqrt(), delta_scale)),
        ("PCA* = case-local basis", zero),
    ]
    return montage(tiles, tile_size=tile_size, columns=4)


def main() -> None:
    args = parse_args()
    if args.resolution < 8 or args.resolution % 8:
        raise ValueError("--resolution must be a positive multiple of eight")
    if args.clip_length < 2 or args.clip_stride < 1:
        raise ValueError("clip length must be >=2 and stride must be >=1")
    if args.raft_pair_batch_size < 1:
        raise ValueError("--raft_pair_batch_size must be positive")
    if args.deformable_alignment_scale < 0:
        raise ValueError("--deformable_alignment_scale must be non-negative")
    if args.tile_size < 64:
        raise ValueError("--tile_size must be at least 64")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This comparison requires CUDA because V8 requires frozen V7 RAFT")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} is non-empty; pass --overwrite")
        shutil.rmtree(output_dir)
    montage_dir = output_dir / "montages"
    montage_dir.mkdir(parents=True, exist_ok=True)

    selected = read_selected_pairs(args)
    dataset = build_dataset(args, selected)
    exact_windows_added = append_exact_pair_windows(dataset, selected)
    clip_pairs = locate_clip_pairs(dataset, selected, bool(args.use_cross_clip))

    v5_component = resolve_component(args.v5_checkpoint, "stc_v5_model")
    v6_component = resolve_component(args.v6_checkpoint, "stc_v6_model")
    v8_component = resolve_component(args.v8_checkpoint, "stc_v8_model")
    v5 = RelativeCrossClipBGSTCAdapter.from_pretrained(str(v5_component)).to(
        device=device, dtype=torch.float32
    ).eval()
    v6 = FlowGuidedDeformableBGSTCAdapter.from_pretrained(str(v6_component)).to(
        device=device, dtype=torch.float32
    ).eval()
    v8 = RAFTGuidedDeformableBGSTCAdapter.from_pretrained(str(v8_component)).to(
        device=device, dtype=torch.float32
    ).eval()
    for model in (v5, v6, v8):
        model.requires_grad_(False)
    provider = FrozenV7RAFTFlowProvider(
        resolve_raft_student_component(args.raft_student_path),
        device=device,
        pair_batch_size=int(args.raft_pair_batch_size),
        mixed_precision=not bool(args.no_amp),
    )

    # All V6/V8 DCN diagnostics have the same public geometry, but check the
    # checkpoint rather than assume the defaults in a future experiment.
    for model, label_name in ((v6, "V6"), (v8, "V8")):
        if int(model.config.deform_kernel_size) < 1 or int(model.config.deform_groups) < 1:
            raise ValueError(f"{label_name} has invalid DCN config")

    rows: List[Dict[str, object]] = []
    amp = autocast_context(device, bool(args.no_amp))
    with torch.inference_mode():
        for ordinal, item in enumerate(clip_pairs, start=1):
            sample = dataset[item.dataset_index]
            frame_ids_cpu = sample["frame_ids"]
            pair_index = int(item.local_pair_index)
            if (
                int(frame_ids_cpu[pair_index]) != item.selected.frame_t
                or int(frame_ids_cpu[pair_index + 1]) != item.selected.frame_t1
            ):
                raise RuntimeError("Selected pair no longer matches its deterministic clip")

            rgb = sample["conditioning_pixel_values"].unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            bg = sample["masks"].unsqueeze(0).to(device=device, dtype=torch.float32)
            frame_ids = frame_ids_cpu.unsqueeze(0).to(device=device)
            previous_rgb = sample["previous_conditioning_pixel_values"].unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            previous_bg = sample["previous_masks"].unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            previous_ids = sample["previous_frame_ids"].unsqueeze(0).to(device=device)
            previous_valid = sample["previous_valid_mask"].unsqueeze(0).to(device=device)

            with amp:
                out_v5 = v5_or_v6_forward(
                    v5, rgb, bg, frame_ids,
                    previous_rgb=previous_rgb, previous_bg=previous_bg,
                    previous_ids=previous_ids, previous_valid=previous_valid,
                    use_cross_clip=bool(args.use_cross_clip),
                    deformable_alignment_scale=float(args.deformable_alignment_scale),
                )
                out_v6 = v5_or_v6_forward(
                    v6, rgb, bg, frame_ids,
                    previous_rgb=previous_rgb, previous_bg=previous_bg,
                    previous_ids=previous_ids, previous_valid=previous_valid,
                    use_cross_clip=bool(args.use_cross_clip),
                    deformable_alignment_scale=float(args.deformable_alignment_scale),
                )
                out_v8 = v8_forward(
                    v8, provider, rgb, bg, frame_ids,
                    previous_rgb=previous_rgb, previous_bg=previous_bg,
                    previous_ids=previous_ids, previous_valid=previous_valid,
                    use_cross_clip=bool(args.use_cross_clip),
                    deformable_alignment_scale=float(args.deformable_alignment_scale),
                )

            feature_size = tuple(int(value) for value in out_v5.spatial_features.shape[-2:])
            if any(
                tuple(output.spatial_features.shape[-2:]) != feature_size
                for output in (out_v6, out_v8)
            ):
                raise RuntimeError("V5/V6/V8 feature resolutions differ")
            bg_feature = F.interpolate(
                sample["masks"].float().to(device=device), size=feature_size, mode="nearest"
            )
            teacher_flow, teacher_valid = teacher_feature_pair(sample, pair_index, feature_size)
            teacher_flow = teacher_flow.to(device=device)
            teacher_valid = teacher_valid.to(device=device)
            cases = {
                "v5": prepare_case("v5", out_v5, pair_index, bg_feature),
                "v6": prepare_case(
                    "v6", out_v6, pair_index, bg_feature,
                    deform_groups=int(v6.config.deform_groups),
                    deform_kernel_size=int(v6.config.deform_kernel_size),
                ),
                "v8": prepare_case(
                    "v8", out_v8, pair_index, bg_feature,
                    deform_groups=int(v8.config.deform_groups),
                    deform_kernel_size=int(v8.config.deform_kernel_size),
                ),
            }
            image = make_pair_montage(
                sample=sample,
                local_pair=pair_index,
                cases=cases,
                teacher_flow=resize_feature_flow_to_rgb(
                    teacher_flow, tuple(sample["conditioning_pixel_values"].shape[-2:])
                ),
                tile_size=int(args.tile_size),
                v6_groups=int(v6.config.deform_groups),
                v6_kernel=int(v6.config.deform_kernel_size),
                v8_groups=int(v8.config.deform_groups),
                v8_kernel=int(v8.config.deform_kernel_size),
            )
            name = "{}_f{:06d}_{:06d}.png".format(
                sanitize(item.selected.sequence),
                item.selected.frame_t,
                item.selected.frame_t1,
            )
            target = montage_dir / name
            if not cv2.imwrite(str(target), image):
                raise OSError(f"Could not write {target}")

            common = {
                "sequence": item.selected.sequence,
                "frame_t": item.selected.frame_t,
                "frame_t1": item.selected.frame_t1,
                "reference_dataset_index": item.selected.reference_dataset_index,
                "visualization_dataset_index": item.dataset_index,
                "clip_frame_ids": " ".join(str(int(value)) for value in frame_ids_cpu.tolist()),
                "local_pair_index": pair_index,
                "use_cross_clip": int(bool(args.use_cross_clip)),
                "has_predecessor": int(bool(previous_valid.any())),
                "predecessor_overlap": int(sample["predecessor_overlap"]),
                "reference_v7_montage": str(item.selected.reference_montage),
                "output_montage": str(target),
            }
            for case_name, case in cases.items():
                row = dict(common)
                row.update(case_metrics(case_name, case, teacher_flow, teacher_valid))
                if case_name == "v8":
                    row["legacy_light_flow_magnitude_feature_px"] = float(
                        case["legacy_flow_backward"].square().sum(dim=0).sqrt().mean()
                    )
                    row["raft_flow_magnitude_rgb_px"] = float(
                        case["raft_flow_backward_rgb"].square().sum(dim=0).sqrt().mean()
                    )
                rows.append(row)
            print(
                "[{}/{}] {} {}->{} written {}".format(
                    ordinal,
                    len(clip_pairs),
                    item.selected.sequence,
                    item.selected.frame_t,
                    item.selected.frame_t1,
                    target.name,
                ),
                flush=True,
            )

    write_csv(output_dir / "per_case_pair_metrics.csv", rows)
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and key not in {"frame_t", "frame_t1", "reference_dataset_index", "visualization_dataset_index", "local_pair_index"}
        }
    )
    case_summary = {
        case: {
            key: finite_average([row for row in rows if row["case"] == case], key)
            for key in numeric_keys
        }
        for case in ("v5", "v6", "v8")
    }
    summary = {
        "experiment": "v5_v6_v8_feature_alignment_visualization",
        "selection_policy": "exact V7 reference CSV/montage pair list",
        "selected_pair_count": len(clip_pairs),
        "selection_tail_windows_added": int(exact_windows_added),
        "split": args.split,
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "teacher_flow_root": str(args.teacher_flow_root.expanduser().resolve()),
        "reference_dir": str(args.reference_dir.expanduser().resolve()),
        "v5_component": str(v5_component),
        "v6_component": str(v6_component),
        "v8_component": str(v8_component),
        "v7_raft_student": str(provider.raft_student_path),
        "v7_raft_metadata": provider.metadata(),
        "clip_length": int(args.clip_length),
        "clip_stride": int(args.clip_stride),
        "resolution": int(args.resolution),
        "use_cross_clip": bool(args.use_cross_clip),
        "deformable_alignment_scale": float(args.deformable_alignment_scale),
        "mask_semantics": "white M_BG=1 is degraded/restore region",
        "feature_comparison_policy": (
            "PCA colors are case-local; cross-case conclusions use within-case "
            "candidate-to-target cosine/L1 correspondence metrics."
        ),
        "fairness_note": (
            "V5-5000 and V8 inherit the same V5 trunk; V6-joint-1750 comes "
            "from a different V5 lineage and is qualitative rather than a "
            "single-variable V5-to-V6 ablation."
        ),
        "mean_per_case": case_summary,
    }
    json_dump(output_dir / "summary.json", summary)
    json_dump(output_dir / "run_config.json", vars(args))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
