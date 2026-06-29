# """Evaluate temporal metrics for generated frame folders.

# Warping Error is computed on adjacent frames by estimating backward optical
# flow with Farneback, warping the previous generated frame to the current frame,
# and averaging photometric error on valid pixels. Frame Similarity is the SSIM
# score between adjacent generated frames.
# """

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class SequenceMetrics:
    sequence: str
    method: str
    frame_count: int
    pair_count: int
    warping_error_l1: float
    warping_error_l2: float
    frame_similarity_ssim: float


def natural_key(path: Path) -> List[object]:
    parts: List[object] = []
    current = ""
    is_digit = False
    for char in path.stem:
        char_is_digit = char.isdigit()
        if current and char_is_digit != is_digit:
            parts.append(int(current) if is_digit else current.lower())
            current = char
        else:
            current += char
        is_digit = char_is_digit
    if current:
        parts.append(int(current) if is_digit else current.lower())
    return parts


def ensure_cv2() -> None:
    if cv2 is None:
        raise SystemExit(
            "OpenCV is required for Warping Error because it uses optical flow. "
            "Install it with `pip install opencv-python` or run this script in an environment that already has cv2."
        )


def list_image_files(folder: Path) -> List[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )


def find_sequence_folders(root: Path) -> List[Path]:
    folders = []
    for folder in sorted([root] + [p for p in root.rglob("*") if p.is_dir()]):
        if list_image_files(folder):
            child_has_images = any(child.is_dir() and list_image_files(child) for child in folder.iterdir())
            if not child_has_images:
                folders.append(folder)
    return folders


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def rgb_to_gray_uint8(image: np.ndarray) -> np.ndarray:
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)


def resize_like(image: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if image.shape[:2] == (target_h, target_w):
        return image
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)


def compute_flow(prev_rgb: np.ndarray, next_rgb: np.ndarray) -> np.ndarray:
    prev_gray = rgb_to_gray_uint8(prev_rgb)
    next_gray = rgb_to_gray_uint8(next_rgb)
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def warp_with_flow(image: np.ndarray, flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)

    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    return warped, valid


def warping_error(prev_rgb: np.ndarray, next_rgb: np.ndarray) -> Tuple[float, float]:
    next_rgb = resize_like(next_rgb, prev_rgb.shape[:2])
    backward_flow = compute_flow(next_rgb, prev_rgb)
    warped_prev, valid = warp_with_flow(prev_rgb, backward_flow)

    if not np.any(valid):
        return float("nan"), float("nan")

    diff = warped_prev[valid] - next_rgb[valid]
    l1 = float(np.mean(np.abs(diff)))
    l2 = float(np.mean(diff * diff))
    return l1, l2


def gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> np.ndarray:
    kernel_1d = cv2.getGaussianKernel(window_size, sigma)
    return kernel_1d @ kernel_1d.T


def frame_ssim(image_a: np.ndarray, image_b: np.ndarray) -> float:
    image_b = resize_like(image_b, image_a.shape[:2])
    kernel = gaussian_kernel()
    c1 = 0.01**2
    c2 = 0.03**2

    scores = []
    for channel in range(image_a.shape[2]):
        a = image_a[..., channel]
        b = image_b[..., channel]
        mu_a = cv2.filter2D(a, -1, kernel)
        mu_b = cv2.filter2D(b, -1, kernel)

        mu_a_sq = mu_a * mu_a
        mu_b_sq = mu_b * mu_b
        mu_ab = mu_a * mu_b

        sigma_a_sq = cv2.filter2D(a * a, -1, kernel) - mu_a_sq
        sigma_b_sq = cv2.filter2D(b * b, -1, kernel) - mu_b_sq
        sigma_ab = cv2.filter2D(a * b, -1, kernel) - mu_ab

        ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
            (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
        )
        scores.append(float(np.mean(ssim_map)))

    return float(np.mean(scores))


def relative_sequence_and_method(root: Path, folder: Path) -> Tuple[str, str]:
    relative = folder.relative_to(root)
    parts = relative.parts
    if len(parts) == 0:
        return ".", folder.name
    if len(parts) >= 2:
        return "/".join(parts[:-1]), parts[-1]
    return ".", parts[0]


def mean_ignore_nan(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanmean(array))


def evaluate_folder(root: Path, folder: Path) -> Optional[SequenceMetrics]:
    frame_paths = list_image_files(folder)
    if len(frame_paths) < 2:
        print(f"[skip] {folder}: need at least 2 frames, found {len(frame_paths)}", flush=True)
        return None

    l1_values = []
    l2_values = []
    ssim_values = []

    prev = read_rgb(frame_paths[0])
    for next_path in frame_paths[1:]:
        current = read_rgb(next_path)
        l1, l2 = warping_error(prev, current)
        l1_values.append(l1)
        l2_values.append(l2)
        ssim_values.append(frame_ssim(prev, current))
        prev = current

    sequence, method = relative_sequence_and_method(root, folder)
    return SequenceMetrics(
        sequence=sequence,
        method=method,
        frame_count=len(frame_paths),
        pair_count=len(frame_paths) - 1,
        warping_error_l1=mean_ignore_nan(l1_values),
        warping_error_l2=mean_ignore_nan(l2_values),
        frame_similarity_ssim=mean_ignore_nan(ssim_values),
    )


def aggregate_by_method(metrics: Iterable[SequenceMetrics]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[SequenceMetrics]] = {}
    for item in metrics:
        grouped.setdefault(item.method, []).append(item)

    aggregate = {}
    for method, items in grouped.items():
        weights = np.asarray([item.pair_count for item in items], dtype=np.float64)
        weight_sum = float(np.sum(weights))
        aggregate[method] = {
            "sequence_count": float(len(items)),
            "frame_count": float(sum(item.frame_count for item in items)),
            "pair_count": float(sum(item.pair_count for item in items)),
            "warping_error_l1": weighted_mean([item.warping_error_l1 for item in items], weights, weight_sum),
            "warping_error_l2": weighted_mean([item.warping_error_l2 for item in items], weights, weight_sum),
            "frame_similarity_ssim": weighted_mean([item.frame_similarity_ssim for item in items], weights, weight_sum),
        }
    return aggregate


def weighted_mean(values: Sequence[float], weights: np.ndarray, weight_sum: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    valid = ~np.isnan(array)
    if not np.any(valid):
        return float("nan")
    valid_weights = weights[valid]
    valid_weight_sum = float(np.sum(valid_weights))
    if valid_weight_sum <= 0:
        return float(np.nanmean(array))
    return float(np.sum(array[valid] * valid_weights) / valid_weight_sum)


def write_csv(path: Path, rows: List[SequenceMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(path: Path, rows: List[SequenceMetrics], aggregate: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "per_sequence": [asdict(row) for row in rows],
        "aggregate_by_method": aggregate,
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate temporal consistency with Warping Error and Frame Similarity."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/media/ssd1/ndquan/model_naeun/paper/BrushNet/Quan_test/results/Generated_image/PartyScene_long/new/sharedNoise_fixedBG_PGD_v0"),
        help="Root folder containing generated frames as <sequence>/<method>/*.png.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV output path. Default: <root>/temporal_metrics_fixedBG_PGD_v0_comparison.csv",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="JSON output path. Default: <root>/temporal_metrics_fixedBG_PGD_v0_comparison.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_cv2()
    root = args.root.resolve()
    csv_path = args.csv or root / "temporal_metrics_fixedBG_PGD_v0_comparison.csv"
    json_path = args.json or root / "temporal_metrics_fixedBG_PGD_v0_comparison.json"

    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")

    folders = find_sequence_folders(root)
    if not folders:
        raise RuntimeError(f"No image sequence folders found under: {root}")

    print(f"Found {len(folders)} generated sequence folders.", flush=True)
    rows = []
    for folder in folders:
        print(f"Evaluating {folder.relative_to(root)}", flush=True)
        result = evaluate_folder(root, folder)
        if result is not None:
            rows.append(result)

    if not rows:
        raise RuntimeError("No valid sequences were evaluated.")

    aggregate = aggregate_by_method(rows)
    write_csv(csv_path, rows)
    write_json(json_path, rows, aggregate)

    print("\nAggregate by method:", flush=True)
    for method, values in aggregate.items():
        print(
            f"{method}: "
            f"WE-L1={values['warping_error_l1']:.6f}, "
            f"WE-L2={values['warping_error_l2']:.6f}, "
            f"FrameSSIM={values['frame_similarity_ssim']:.6f}, "
            f"pairs={int(values['pair_count'])}",
            flush=True,
        )
    print(f"\nSaved CSV: {csv_path}", flush=True)
    print(f"Saved JSON: {json_path}", flush=True)


if __name__ == "__main__":
    main()



# import pandas as pd

# # Đọc file kết quả evaluate
# csv_path = "/media/ssd1/ndquan/model_naeun/paper/BrushNet/Quan_test/results/Generated_image/temporal_metrics_v0.csv"
# df = pd.read_csv(csv_path)

# # Tính trung bình theo từng method
# avg_df = (
#     df.groupby("method")
#     .agg(
#         num_sequences=("sequence", "count"),
#         total_frame_count=("frame_count", "sum"),
#         total_pair_count=("pair_count", "sum"),
#         avg_warping_error_l1=("warping_error_l1", "mean"),
#         avg_warping_error_l2=("warping_error_l2", "mean"),
#         avg_frame_similarity_ssim=("frame_similarity_ssim", "mean"),
#     )
#     .reset_index()
# )

# print(avg_df)

# # Lưu ra file mới
# avg_df.to_csv("/media/ssd1/ndquan/model_naeun/paper/BrushNet/Quan_test/results/Generated_image/temporal_results_average_by_method.csv", index=False)




# import re
# from pathlib import Path


# def read_metric_txt(txt_path):
#     txt_path = Path(txt_path)

#     # Try multiple encodings
#     encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]

#     content = None

#     for enc in encodings:
#         try:
#             with open(txt_path, "r", encoding=enc) as f:
#                 content = f.read()
#             break
#         except UnicodeDecodeError:
#             continue

#     # Final fallback: ignore invalid characters
#     if content is None:
#         with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
#             content = f.read()

#     metrics = {}

#     patterns = {
#         "Frames": r"Frames:\s*([\d.]+)",
#         "PSNR": r"PSNR:\s*([\d.]+)",
#         "SSIM": r"SSIM:\s*([\d.]+)",
#         "MS-SSIM": r"MS-SSIM:\s*([\d.]+)",
#         "LPIPS": r"LPIPS:\s*([\d.]+)",
#         "FID": r"FID:\s*([\d.]+)",
#         "FVD": r"FVD:\s*([\d.]+)",
#     }

#     for key, pattern in patterns.items():
#         match = re.search(pattern, content)
#         if match:
#             metrics[key] = float(match.group(1))
#         else:
#             metrics[key] = None
#             print(f"Warning: Cannot find {key} in {txt_path}")

#     return metrics


# def average_two_metric_files(txt_path_1, txt_path_2, save_path="average_metrics.txt"):
#     metrics_1 = read_metric_txt(txt_path_1)
#     metrics_2 = read_metric_txt(txt_path_2)

#     avg_metrics = {}

#     for key in metrics_1.keys():
#         if metrics_1[key] is not None and metrics_2[key] is not None:
#             avg_metrics[key] = (metrics_1[key] + metrics_2[key]) / 2
#         else:
#             avg_metrics[key] = None

#     with open(save_path, "w", encoding="utf-8") as f:
#         f.write(f"Frames:   {avg_metrics['Frames']:.0f}\n")
#         f.write(f"PSNR:     {avg_metrics['PSNR']:.4f} dB\n")
#         f.write(f"SSIM:     {avg_metrics['SSIM']:.6f}\n")
#         f.write(f"MS-SSIM:  {avg_metrics['MS-SSIM']:.6f}\n")
#         f.write(f"LPIPS:    {avg_metrics['LPIPS']:.6f}\n")
#         f.write(f"FID:      {avg_metrics['FID']:.4f}\n")
#         f.write(f"FVD:      {avg_metrics['FVD']:.4f}\n")

#     return avg_metrics


# if __name__ == "__main__":
#     txt_path_1 = "/media/ssd1/ndquan/model_naeun/paper/BrushNet/Quan_test/results/BasketballPass_sharedNoise_v0.txt"
#     txt_path_2 = "/media/ssd1/ndquan/model_naeun/paper/BrushNet/Quan_test/results/party_scene_sharedNoise_v0.txt"

#     avg_metrics = average_two_metric_files(
#         txt_path_1,
#         txt_path_2,
#         save_path="/media/ssd1/ndquan/model_naeun/paper/BrushNet/Quan_test/results/average_metrics_shareNoise_v0.txt"
#     )

#     print("Average Metrics")
#     print("----------------")
#     print(f"Frames:   {avg_metrics['Frames']:.0f}")
#     print(f"PSNR:     {avg_metrics['PSNR']:.4f} dB")
#     print(f"SSIM:     {avg_metrics['SSIM']:.6f}")
#     print(f"MS-SSIM:  {avg_metrics['MS-SSIM']:.6f}")
#     print(f"LPIPS:    {avg_metrics['LPIPS']:.6f}")
#     print(f"FID:      {avg_metrics['FID']:.4f}")
#     print(f"FVD:      {avg_metrics['FVD']:.4f}")