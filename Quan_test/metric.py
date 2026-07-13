#!/usr/bin/env python3

import subprocess
import tempfile
import shutil
from pathlib import Path
import sys
import re
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from torchmetrics.functional import (
    structural_similarity_index_measure as ssim_fn,
    multiscale_structural_similarity_index_measure as ms_ssim_fn,
)

import lpips


FFMPEG_EXE = "/usr/bin/ffmpeg"
FVD_METRIC_ROOT = (
    Path(__file__).resolve().parents[0] / "PyTorch-Frechet-Video-Distance"
)


# ============================================================
# Basic video reading by ffmpeg
# ============================================================

def ffmpeg_rgb_stream(path: str):
    """
    Open a video with ffmpeg and stream raw RGB frames.
    """
    cmd = [
        FFMPEG_EXE,
        "-hide_banner",
        "-loglevel", "error",
        "-i", path,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE)


def get_video_info_cv2(path: str):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps and fps > 0 else 0.0

    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
    }


def read_rgb_frame(proc, width: int, height: int):
    frame_bytes = width * height * 3
    buf = proc.stdout.read(frame_bytes)

    if not buf or len(buf) != frame_bytes:
        return None

    return np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)


def skip_frames(proc, width: int, height: int, num_frames: int):
    skipped = 0

    for _ in range(num_frames):
        frame = read_rgb_frame(proc, width, height)

        if frame is None:
            break

        skipped += 1

    return skipped


# ============================================================
# Image metrics
# ============================================================

def psnr_np(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

    if mse == 0:
        return float("inf")

    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def to_torch_01(img: np.ndarray) -> torch.Tensor:
    """
    RGB uint8 image [H, W, C] -> torch tensor [1, C, H, W] in [0, 1]
    """
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def to_torch_m11(img: np.ndarray) -> torch.Tensor:
    """
    RGB uint8 image [H, W, C] -> torch tensor [1, C, H, W] in [-1, 1]
    Required by LPIPS.
    """
    t = to_torch_01(img)
    return t * 2.0 - 1.0


def avg(values):
    return float(np.mean(np.array(values))) if len(values) > 0 else float("nan")


# ============================================================
# Frame-level FID
# ============================================================

def compute_fid(
    ref_dir: Path,
    dist_dir: Path,
    device: str = "cpu",
    dims: int = 2048,
    batch_size: int = 8,
    num_workers: int = 4,
) -> float:
    """
    Compute frame-level FID using pytorch-fid.
    This is not video-level FID. It treats extracted frames as images.
    """
    cmd = [
        sys.executable, "-m", "pytorch_fid",
        str(ref_dir), str(dist_dir),
        "--device", device,
        "--dims", str(dims),
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
    ]

    try:
        out = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()

    except subprocess.CalledProcessError as e:
        print("\n[pytorch-fid FAILED]")
        print("Command:", " ".join(cmd))
        print("Output:\n", e.output)
        raise

    nums = re.findall(r"[-+]?\d*\.\d+|\d+", out)

    if not nums:
        raise RuntimeError(f"Cannot parse pytorch_fid output:\n{out}")

    return float(nums[-1])


# ============================================================
# FVD helper
# ============================================================

def read_video_as_fvd_clips(
    video_path: str,
    frame_skip: int = 0,
    max_frames: int = 150,
    clip_len: int = 16,
    resize_size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """
    Read a video and convert it to FVD input clips.

    Output shape:
        [N, C, T, H, W]

    where:
        N = number of clips
        C = 3
        T = clip_len
        H, W = resize_size
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    frame_id = 0

    while True:
        ret, frame_bgr = cap.read()

        if not ret:
            break

        if frame_id < frame_skip:
            frame_id += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if resize_size is not None:
            # resize_size is (W, H) for cv2.resize
            frame_rgb = cv2.resize(
                frame_rgb,
                resize_size,
                interpolation=cv2.INTER_AREA,
            )

        frames.append(frame_rgb)

        if len(frames) >= max_frames:
            break

        frame_id += 1

    cap.release()

    if len(frames) < clip_len:
        raise RuntimeError(
            f"Not enough frames for FVD. "
            f"Got {len(frames)} frames, but clip_len={clip_len}."
        )

    num_clips = len(frames) // clip_len
    clips = []

    for i in range(num_clips):
        clip_frames = frames[i * clip_len : (i + 1) * clip_len]

        clip = np.stack(clip_frames, axis=0)          # [T, H, W, C]
        clip = torch.from_numpy(clip).float() / 255.0 # [T, H, W, C]
        clip = clip.permute(3, 0, 1, 2).contiguous()  # [C, T, H, W]

        clips.append(clip)

    clips = torch.stack(clips, dim=0)                 # [N, C, T, H, W]

    return clips


def compute_fvd_score(
    ref_video: str,
    dist_video: str,
    ref_frame_skip: int = 0,
    dist_frame_skip: int = 0,
    max_frames: int = 150,
    clip_len: int = 16,
    resize_size: Tuple[int, int] = (224, 224),
    batch_size: int = 4,
) -> float:
    """
    Compute FVD using fvd_metric.compute_fvd.

    Required:
        PyTorch-Frechet-Video-Distance/fvd_metric must exist in the project.
    """
    fvd_metric_root = str(FVD_METRIC_ROOT)

    if fvd_metric_root not in sys.path:
        sys.path.insert(0, fvd_metric_root)

    try:
        from fvd_metric import compute_fvd

    except ImportError as e:
        raise ImportError(
            "\nCannot import fvd_metric from:\n"
            f"{FVD_METRIC_ROOT}\n\n"
            f"Original import error: {e}\n"
        ) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_clips = read_video_as_fvd_clips(
        video_path=ref_video,
        frame_skip=ref_frame_skip,
        max_frames=max_frames,
        clip_len=clip_len,
        resize_size=resize_size,
    )

    dist_clips = read_video_as_fvd_clips(
        video_path=dist_video,
        frame_skip=dist_frame_skip,
        max_frames=max_frames,
        clip_len=clip_len,
        resize_size=resize_size,
    )

    num_samples = min(ref_clips.shape[0], dist_clips.shape[0])

    if num_samples < 1:
        raise RuntimeError("No valid video clips for FVD.")

    ref_clips = ref_clips[:num_samples].to(device)
    dist_clips = dist_clips[:num_samples].to(device)

    print(f"\nFVD input shape:")
    print(f"REF : {tuple(ref_clips.shape)}")
    print(f"DIST: {tuple(dist_clips.shape)}")
    print(f"Number of FVD clips: {num_samples}")

    with torch.no_grad():
        fvd_score = compute_fvd(
            ref_clips,
            dist_clips,
            num_samples,
            device,
            batch_size=batch_size,
        )

    if isinstance(fvd_score, torch.Tensor):
        fvd_score = fvd_score.item()

    return float(fvd_score)


# ============================================================
# Main metric pipeline
# ============================================================

def compute_video_metrics_mp4(
    ref_video: str,
    dist_video: str,
    ref_frame_skip: int = 0,
    dist_frame_skip: int = 0,
    max_frames: int = 150,

    # FID config
    compute_frame_fid: bool = True,
    fid_sample_every: int = 1,
    fid_dims: int = 2048,
    fid_batch_size: int = 8,
    fid_num_workers: int = 4,

    # FVD config
    compute_fvd: bool = True,
    fvd_clip_len: int = 16,
    fvd_resize_size: Tuple[int, int] = (224, 224),
    fvd_batch_size: int = 4,

    # Output
    save_txt_path: Optional[str] = None,
    keep_temp_frames: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ref_info = get_video_info_cv2(ref_video)
    dist_info = get_video_info_cv2(dist_video)

    print("\n=== REF VIDEO INFO ===")
    for k, v in ref_info.items():
        print(f"{k}: {v}")

    print("\n=== DIST VIDEO INFO ===")
    for k, v in dist_info.items():
        print(f"{k}: {v}")

    if ref_info["width"] != dist_info["width"] or ref_info["height"] != dist_info["height"]:
        raise RuntimeError(
            f"Video size mismatch: "
            f"REF={ref_info['width']}x{ref_info['height']}, "
            f"DIST={dist_info['width']}x{dist_info['height']}"
        )

    width = ref_info["width"]
    height = ref_info["height"]

    proc_ref = ffmpeg_rgb_stream(ref_video)
    proc_dist = ffmpeg_rgb_stream(dist_video)

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    if ref_frame_skip > 0:
        skipped_ref = skip_frames(proc_ref, width, height, ref_frame_skip)
        print(f"\nSkipped {skipped_ref} frames in REF")

        if skipped_ref < ref_frame_skip:
            raise RuntimeError("REF does not have enough frames to skip.")

    if dist_frame_skip > 0:
        skipped_dist = skip_frames(proc_dist, width, height, dist_frame_skip)
        print(f"Skipped {skipped_dist} frames in DIST")

        if skipped_dist < dist_frame_skip:
            raise RuntimeError("DIST does not have enough frames to skip.")

    psnr_list = []
    ssim_list = []
    ms_ssim_list = []
    lpips_list = []

    tmp_root = Path(tempfile.mkdtemp(prefix="video_metrics_"))
    ref_dir = tmp_root / "ref"
    dist_dir = tmp_root / "dist"

    ref_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    pbar = tqdm(total=max_frames, desc="Processing frames")

    while frame_idx < max_frames:
        fr = read_rgb_frame(proc_ref, width, height)
        fd = read_rgb_frame(proc_dist, width, height)

        if fr is None or fd is None:
            print(
                f"\nStopped early at frame_idx={frame_idx} "
                f"(fr is None={fr is None}, fd is None={fd is None})"
            )
            break

        # PSNR
        psnr_list.append(psnr_np(fr, fd))

        # SSIM / MS-SSIM
        tr01 = to_torch_01(fr).to(device)
        td01 = to_torch_01(fd).to(device)

        with torch.no_grad():
            ssim_value = ssim_fn(td01, tr01, data_range=1.0).item()
            ms_ssim_value = ms_ssim_fn(td01, tr01, data_range=1.0).item()

        ssim_list.append(ssim_value)
        ms_ssim_list.append(ms_ssim_value)

        # LPIPS
        trm11 = to_torch_m11(fr).to(device)
        tdm11 = to_torch_m11(fd).to(device)

        with torch.no_grad():
            lpips_value = lpips_model(tdm11, trm11).item()

        lpips_list.append(lpips_value)

        # Save frames for frame-level FID
        if compute_frame_fid and fid_sample_every > 0 and frame_idx % fid_sample_every == 0:
            cv2.imwrite(
                str(ref_dir / f"{frame_idx:06d}.png"),
                cv2.cvtColor(fr, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                str(dist_dir / f"{frame_idx:06d}.png"),
                cv2.cvtColor(fd, cv2.COLOR_RGB2BGR),
            )

        frame_idx += 1
        pbar.update(1)

    pbar.close()


    if len(psnr_list) != 0:
        print(f"\nProcessed {len(psnr_list)} frames for metrics.")

    if proc_ref.poll() is None:
        proc_ref.kill()

    if proc_dist.poll() is None:
        proc_dist.kill()

    if len(psnr_list) == 0:
        raise RuntimeError("No frames were compared.")

    # Frame-level FID
    fid_score = float("nan")

    if compute_frame_fid:
        fid_score = compute_fid(
            ref_dir=ref_dir,
            dist_dir=dist_dir,
            device=fid_device,
            dims=fid_dims,
            batch_size=fid_batch_size,
            num_workers=fid_num_workers,
        )

    # FVD
    fvd_score = float("nan")

    if compute_fvd:
        fvd_score = compute_fvd_score(
            ref_video=ref_video,
            dist_video=dist_video,
            ref_frame_skip=ref_frame_skip,
            dist_frame_skip=dist_frame_skip,
            max_frames=max_frames,
            clip_len=fvd_clip_len,
            resize_size=fvd_resize_size,
            batch_size=fvd_batch_size,
        )

    results = {
        "frames": len(psnr_list),
        "ref_frame_skip": ref_frame_skip,
        "dist_frame_skip": dist_frame_skip,
        "PSNR": avg(psnr_list),
        "SSIM": avg(ssim_list),
        "MS-SSIM": avg(ms_ssim_list),
        "LPIPS": avg(lpips_list),
        "Frame-level FID": fid_score,
        "FVD": fvd_score,
        "fid_temp_dir": str(tmp_root),
    }

    result_text = f"""
Frames:   {results['frames']}
PSNR:     {results['PSNR']:.4f} dB
SSIM:     {results['SSIM']:.6f}
MS-SSIM:  {results['MS-SSIM']:.6f}
LPIPS:    {results['LPIPS']:.6f}
FID:      {results['Frame-level FID']:.4f}
FVD:      {results['FVD']:.4f}
"""

    print("\n" + result_text)

    if save_txt_path is not None:
        save_path = Path(save_txt_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(result_text)

        print(f"Saved metrics to: {save_txt_path}")

    if keep_temp_frames:
        print(f"Temporary FID frames are kept at: {tmp_root}")
    else:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return results


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":

    # REF_VIDEO = "/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/test/BasketballPass_512_backup/video/BasketballPass_gt.mp4"
    REF_VIDEO = "/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/test/BasketballPass_512_backup/video/BasketballPass_gt.mp4"

    DIST_VIDEO = "/media/ssd1/ndquan/model_naeun/paper/BrushNet/Quan_test/results/video/BasketballPass/fixedBG_nulltext_checkpoint3500_09_corr.mp4"

    OUTPUT_TXT = "/media/ssd1/ndquan/model_naeun/paper/BrushNet/Quan_test/results/metric/BasketballPass_fixedBG_nulltext_checkpoint3500_09_corr.txt"

    compute_video_metrics_mp4(
        ref_video=REF_VIDEO,
        dist_video=DIST_VIDEO,

        ref_frame_skip=0,
        dist_frame_skip=0,
        max_frames=150,

        # Frame-level FID
        compute_frame_fid=True,
        fid_sample_every=1,
        fid_dims=2048,
        fid_batch_size=8,
        fid_num_workers=4,

        # FVD
        compute_fvd=True,
        fvd_clip_len=16,
        fvd_resize_size=(224, 224),
        fvd_batch_size=4,

        save_txt_path=OUTPUT_TXT,
        keep_temp_frames=False,
    )
