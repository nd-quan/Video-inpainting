#!/usr/bin/env python
"""Visualize a V7 degraded-RGB RAFT student against cached clean RAFT flow.

The student consumes only degraded RGB.  Clean RGB is used offline by the
frozen RAFT teacher whose bidirectional flow is loaded from the cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from diffusers.models.stc_flow_training import prepare_teacher_flow  # noqa: E402
from raft_student import RAFTStudentFlowPredictor  # noqa: E402
from raft_teacher_pair_data import RAFTTeacherFlowPairDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="V7 experiment root, checkpoint-N, best/latest JSON, or raft_student directory.",
    )
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--teacher_flow_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--pairs_per_sequence", type=int, default=2)
    parser.add_argument("--include_branches", nargs="*", default=None)
    parser.add_argument("--flow_scale_percentile", type=float, default=99.0)
    parser.add_argument("--epe_scale_percentile", type=float, default=99.0)
    parser.add_argument("--propainter_root", type=Path, default=None)
    parser.add_argument("--raft_checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_dump(path: Path, value: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def resolve_checkpoint(value: Path) -> Tuple[Path, Path]:
    """Resolve all supported checkpoint forms to (checkpoint_root, model_dir)."""
    value = value.expanduser().resolve()
    if value.name in {"best.json", "latest.json"}:
        selected = Path(json.loads(value.read_text(encoding="utf-8"))["checkpoint"])
        return resolve_checkpoint(selected if selected.is_absolute() else value.parent / selected)
    if (value / "raft_student" / "config.json").is_file():
        return value, value / "raft_student"
    if (value / "config.json").is_file() and (value / "pytorch_model.bin").is_file():
        return value.parent, value
    pointer = value / "best.json"
    if value.is_dir() and pointer.is_file():
        return resolve_checkpoint(pointer)
    raise FileNotFoundError("Cannot find raft_student/config.json below " + str(value))


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def select_pair_indices(dataset: RAFTTeacherFlowPairDataset, count: int) -> List[int]:
    if count < 1:
        raise ValueError("--pairs_per_sequence must be positive")
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, entry in enumerate(dataset.entries):
        grouped[str(entry[0])].append(index)
    selected = []
    for sequence in sorted(grouped):
        indices = grouped[sequence]
        sample_count = min(count, len(indices))
        positions = np.linspace(0, len(indices) - 1, sample_count).round().astype(int)
        selected.extend(indices[int(position)] for position in positions)
    return selected


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
    value = mask.detach().float().cpu().squeeze().clamp(0, 1).mul(255).byte().numpy()
    return cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)


def flow_to_bgr(flow: torch.Tensor, scale: float) -> np.ndarray:
    value = flow.detach().float().cpu().permute(1, 2, 0).numpy()
    magnitude, angle = cv2.cartToPolar(value[..., 0], value[..., 1])
    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((angle * 90.0 / np.pi) % 180.0).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude / max(scale, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def heatmap(value: torch.Tensor, scale: float) -> np.ndarray:
    image = value.detach().float().cpu().squeeze().numpy()
    image = np.clip(image / max(scale, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(image, cv2.COLORMAP_TURBO)


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(result, text, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def make_montage(tiles: Sequence[Tuple[str, np.ndarray]]) -> np.ndarray:
    tile_size = (320, 320)
    rendered = [label(cv2.resize(image, tile_size, interpolation=cv2.INTER_AREA), text) for text, image in tiles]
    blank = np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)
    while len(rendered) % 4:
        rendered.append(blank.copy())
    rows = [np.concatenate(rendered[index:index + 4], axis=1) for index in range(0, len(rendered), 4)]
    return np.concatenate(rows, axis=0)


def percentile_scale(values: Sequence[torch.Tensor], percentile: float) -> float:
    if not 0.0 < percentile <= 100.0:
        raise ValueError("scale percentile must be in (0, 100]")
    flattened = torch.cat([value.detach().float().flatten() for value in values])
    return max(float(torch.quantile(flattened, percentile / 100.0)), 1e-6)


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> float:
    return float((value.float() * weight.float()).sum() / weight.float().sum().clamp_min(1e-8))


def direction_metrics(predicted: torch.Tensor, teacher: torch.Tensor, valid: torch.Tensor, bg: torch.Tensor) -> Dict[str, float]:
    epe = (predicted.float() - teacher.float()).square().sum(0, keepdim=True).sqrt()
    teacher_mag = teacher.float().square().sum(0, keepdim=True).sqrt()
    student_mag = predicted.float().square().sum(0, keepdim=True).sqrt()
    all_weight, bg_weight = valid.float(), valid.float() * bg.float()
    epe_all, epe_bg = weighted_mean(epe, all_weight), weighted_mean(epe, bg_weight)
    zero_all, zero_bg = weighted_mean(teacher_mag, all_weight), weighted_mean(teacher_mag, bg_weight)
    return {
        "epe": epe_all,
        "zero_epe": zero_all,
        "zero_gain": 1.0 - epe_all / max(zero_all, 1e-8),
        "bg_epe": epe_bg,
        "bg_zero_epe": zero_bg,
        "bg_zero_gain": 1.0 - epe_bg / max(zero_bg, 1e-8),
        "teacher_magnitude": weighted_mean(teacher_mag, all_weight),
        "student_magnitude": weighted_mean(student_mag, all_weight),
        "valid_ratio": float(all_weight.mean()),
        "valid_bg_ratio": float(bg_weight.mean()),
    }


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_average(rows: Sequence[Dict[str, object]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def main() -> None:
    args = parse_args()
    if args.resolution < 8 or args.resolution % 8:
        raise ValueError("--resolution must be a positive multiple of 8")
    output_dir = args.output_dir.expanduser().resolve()
    meaningful_entries = [entry for entry in output_dir.iterdir() if entry.name != "terminal_logs"] if output_dir.exists() else []
    if meaningful_entries and not args.overwrite:
        raise FileExistsError("{} is non-empty; pass --overwrite".format(output_dir))
    montage_dir = output_dir / "montages"
    montage_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint_root, student_dir = resolve_checkpoint(args.checkpoint)
    model = RAFTStudentFlowPredictor.from_pretrained(
        student_dir,
        propainter_root=args.propainter_root,
        raft_checkpoint=args.raft_checkpoint,
        mixed_precision=device.type == "cuda" and not args.no_amp,
    ).to(device=device, dtype=torch.float32).eval()
    model.requires_grad_(False)
    dataset = RAFTTeacherFlowPairDataset(
        args.dataset_root, args.teacher_flow_root, args.split, args.resolution, args.include_branches
    )
    selected = select_pair_indices(dataset, args.pairs_per_sequence)
    amp = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" and not args.no_amp else nullcontext()

    rows: List[Dict[str, object]] = []
    for dataset_index in selected:
        sample = dataset[dataset_index]
        degraded0 = sample["degraded0"].unsqueeze(0).to(device=device, dtype=torch.float32)
        degraded1 = sample["degraded1"].unsqueeze(0).to(device=device, dtype=torch.float32)
        bg0, bg1 = sample["bg0"].to(device), sample["bg1"].to(device)
        with torch.inference_mode(), amp:
            pred_forward, pred_backward = model.predict_bidirectional(degraded0, degraded1, pair_batch_size=1)
        size = tuple(pred_forward.shape[-2:])
        teacher_forward, valid_forward = prepare_teacher_flow(
            sample["teacher_forward"].unsqueeze(0).unsqueeze(0).to(device), size,
            sample["valid_forward"].unsqueeze(0).unsqueeze(0).to(device),
        )
        teacher_backward, valid_backward = prepare_teacher_flow(
            sample["teacher_backward"].unsqueeze(0).unsqueeze(0).to(device), size,
            sample["valid_backward"].unsqueeze(0).unsqueeze(0).to(device),
        )
        pred_forward, pred_backward = pred_forward[0].float(), pred_backward[0].float()
        teacher_forward, valid_forward = teacher_forward[0, 0], valid_forward[0, 0]
        teacher_backward, valid_backward = teacher_backward[0, 0], valid_backward[0, 0]
        bg_forward = F.interpolate(bg0.unsqueeze(0), size=size, mode="nearest")[0]
        bg_backward = F.interpolate(bg1.unsqueeze(0), size=size, mode="nearest")[0]
        residual_forward, residual_backward = pred_forward - teacher_forward, pred_backward - teacher_backward
        epe_forward = residual_forward.square().sum(0, keepdim=True).sqrt()
        epe_backward = residual_backward.square().sum(0, keepdim=True).sqrt()
        flow_scale = percentile_scale((pred_forward.norm(dim=0), pred_backward.norm(dim=0), teacher_forward.norm(dim=0), teacher_backward.norm(dim=0)), args.flow_scale_percentile)
        residual_scale = percentile_scale((residual_forward.norm(dim=0), residual_backward.norm(dim=0)), args.flow_scale_percentile)
        epe_scale = percentile_scale((epe_forward, epe_backward), args.epe_scale_percentile)
        forward = direction_metrics(pred_forward, teacher_forward, valid_forward, bg_forward)
        backward = direction_metrics(pred_backward, teacher_backward, valid_backward, bg_backward)
        row: Dict[str, object] = {
            "sequence": str(sample["video"]), "dataset_index": int(dataset_index),
            "frame_t": int(sample["frame0"]), "frame_t1": int(sample["frame1"]),
            "flow_scale_px": flow_scale, "residual_scale_px": residual_scale, "epe_scale_px": epe_scale,
        }
        row.update({"forward_" + key: value for key, value in forward.items()})
        row.update({"backward_" + key: value for key, value in backward.items()})
        zero = np.zeros_like(rgb_to_bgr(degraded0[0]))
        tiles = [
            ("Degraded t", rgb_to_bgr(degraded0[0])), ("Degraded t+1", rgb_to_bgr(degraded1[0])),
            ("BG mask t (white=BG)", mask_to_bgr(bg0)), ("BG mask t+1 (white=BG)", mask_to_bgr(bg1)),
            ("Teacher forward t->t+1", flow_to_bgr(teacher_forward, flow_scale)), ("Student forward t->t+1", flow_to_bgr(pred_forward, flow_scale)),
            ("Forward residual (student-teacher)", flow_to_bgr(residual_forward, residual_scale)), ("Forward EPE", heatmap(epe_forward, epe_scale)),
            ("Teacher backward t+1->t", flow_to_bgr(teacher_backward, flow_scale)), ("Student backward t+1->t", flow_to_bgr(pred_backward, flow_scale)),
            ("Backward residual (student-teacher)", flow_to_bgr(residual_backward, residual_scale)), ("Backward EPE", heatmap(epe_backward, epe_scale)),
            ("Teacher valid forward", mask_to_bgr(valid_forward)), ("Teacher valid backward", mask_to_bgr(valid_backward)),
            ("Forward BG gain={:.3f}".format(forward["bg_zero_gain"]), zero), ("Backward BG gain={:.3f}".format(backward["bg_zero_gain"]), zero),
        ]
        name = "{}_f{:06d}_{:06d}.png".format(sanitize(str(sample["video"])), int(sample["frame0"]), int(sample["frame1"]))
        if not cv2.imwrite(str(montage_dir / name), make_montage(tiles)):
            raise OSError("Could not write " + str(montage_dir / name))
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    write_csv(output_dir / "per_pair_metrics.csv", rows)
    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and key not in {"dataset_index", "frame_t", "frame_t1"}] if rows else []
    summary = {
        "experiment": "v7_raft_student_flow_visualization",
        "checkpoint_root": str(checkpoint_root), "raft_student": str(student_dir),
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "teacher_flow_root": str(args.teacher_flow_root.expanduser().resolve()),
        "split": args.split, "resolution": args.resolution, "pairs_per_sequence": args.pairs_per_sequence,
        "selected_pair_count": len(selected),
        "flow_convention": "forward=t->t+1 on t; backward=t+1->t on t+1; [dx,dy] in RGB pixels",
        "mask_semantics": "white BG means degraded/restore region (M_BG=1)",
        "flow_color_policy": "teacher/student share a per-montage magnitude scale; residual uses its own scale",
        "mean_per_pair": {key: finite_average(rows, key) for key in numeric_keys},
    }
    json_dump(output_dir / "summary.json", summary)
    json_dump(output_dir / "run_config.json", vars(args))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
