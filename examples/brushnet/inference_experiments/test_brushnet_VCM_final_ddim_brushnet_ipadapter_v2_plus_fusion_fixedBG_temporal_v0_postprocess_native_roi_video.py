#!/usr/bin/env python3
"""Run temporal-v0 eval, restore native size/ROI, and encode native-size videos.

The inference implementation remains in the original temporal-v0 script. This
wrapper passes its environment through unchanged except for dataset/output
selection, then applies the BG52 native-resize and native-ROI composition rules
to exactly the sequences selected for this run. Use --postprocess-only to start
from already-generated 512x512 PNGs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

import numpy as np
from PIL import Image

from sfu_long_test_loader import load_sequences, parse_sequence_names, print_preflight


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
INFERENCE_SCRIPT = SCRIPT_DIR / "test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion_fixedBG_temporal_v0.py"
FPS = {
    "PeopleOnStreet": 30, "BQTerrace": 60, "BasketballDrive": 50,
    "Cactus": 50, "Kimono": 24, "BQMall": 60,
    "BasketballDrill": 50, "RaceHorsesC": 30, "BQSquare": 60,
    "BlowingBubbles": 50, "RaceHorsesD": 30, "FourPeople": 60,
    "Johnny": 60, "KristenAndSara": 60, "BasketballPass": 50,
    "ParkScene": 24, "PartyScene": 50, "RaceHorses": 30, "Traffic": 30,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("LONG_TEST_ROOT"),
                        help="Original dataset; hierarchical input/mask[/GT] or legacy flat layout")
    parser.add_argument("--output-root", type=Path, default=os.environ.get("OUTPUT_ROOT"),
                        help="Dedicated evaluation output directory")
    parser.add_argument("--sequences", nargs="+", default=None,
                        help="Sequence names; otherwise use SEQUENCES or all available sequences")
    parser.add_argument("--postprocess-only", action="store_true",
                        help="Use existing 512x512 evaluator PNGs; do not run diffusion")
    parser.add_argument("--preflight", action="store_true",
                        help="List the planned sequences and paths without writing anything")
    parser.add_argument("--skip-video", action="store_true", help="Debug only: stop after PNG composition")
    parser.add_argument("--fps", type=int, default=None,
                        help="Override FPS for every video; otherwise use the SFU sequence FPS map")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of concurrent postprocessing/video jobs")
    args = parser.parse_args()
    if args.dataset_root is None or args.output_root is None:
        parser.error("Set --dataset-root and --output-root (or LONG_TEST_ROOT and OUTPUT_ROOT)")
    if args.workers < 1 or (args.fps is not None and args.fps < 1):
        parser.error("--workers and --fps must be positive")
    return args


def frame_names(sequence):
    return tuple(path.name for path in sequence.image_paths)


def verify_source_frames(sequence):
    expected = set(frame_names(sequence))
    actual = {path.name for path in sequence.output_dir.glob("*.png")}
    if actual != expected:
        raise ValueError(
            f"Incomplete evaluator output {sequence.output_dir}: "
            f"missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}"
        )


def save_png_atomic(image: Image.Image, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp.png")
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def encode_video(frames, destination: Path, fps: int, size):
    width, height = size
    if width % 2 or height % 2:
        raise ValueError(f"yuv420p video needs even native dimensions: {size}; {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_video = destination.with_name(destination.stem + ".tmp.mp4")
    try:
        with tempfile.TemporaryDirectory(prefix="native_roi_video_") as stage_dir:
            stage = Path(stage_dir)
            for index, frame in enumerate(frames):
                (stage / f"{index:06d}.png").symlink_to(frame)
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", str(fps), "-start_number", "0",
                "-i", str(stage / "%06d.png"), "-frames:v", str(len(frames)),
                "-an", "-c:v", "mpeg4", "-q:v", "2",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary_video),
            ], check=True)
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate",
            "-of", "json", str(temporary_video),
        ], check=True, capture_output=True, text=True)
        streams = json.loads(probe.stdout).get("streams", [])
        if len(streams) != 1:
            raise ValueError(f"Expected one video stream: {temporary_video}")
        info = streams[0]
        if (int(info["width"]), int(info["height"])) != size or (
            int(info["nb_read_frames"]) != len(frames)
        ) or Fraction(info["avg_frame_rate"]) != fps:
            raise ValueError(f"Video verification failed: {temporary_video}: {info}")
        os.replace(temporary_video, destination)
    finally:
        if temporary_video.exists():
            temporary_video.unlink()


def postprocess_sequence(sequence, dataset_root: Path, output_root: Path,
                         fps: Optional[int], skip_video: bool):
    verify_source_frames(sequence)
    split_root = output_root / sequence.spec.split
    relative = sequence.output_dir.relative_to(split_root)
    resized_dir = split_root / "resized_to_original" / relative
    composite_dir = split_root / "resized_to_original_native_roi" / relative
    video_path = split_root / "videos" / "resized_to_original_native_roi" / relative.with_suffix(".mp4")
    native_sizes = set()
    composite_frames = []
    for source, input_path, mask_path in zip(
        (sequence.output_dir / name for name in frame_names(sequence)),
        sequence.image_paths, sequence.mask_paths,
    ):
        with Image.open(source) as image:
            if image.size != (512, 512):
                raise ValueError(f"Expected 512x512 evaluator output: {source}: {image.size}")
            generated = image.convert("RGB")
        with Image.open(input_path) as image:
            native_input = np.asarray(image.convert("RGB"))
        with Image.open(mask_path) as image:
            native_mask = np.asarray(image.convert("RGB"), dtype=np.uint16)
        if native_input.shape != native_mask.shape:
            raise ValueError(f"Native input/mask size mismatch: {input_path}, {mask_path}")
        native_size = (native_input.shape[1], native_input.shape[0])
        gt_path = dataset_root / "GT" / sequence.spec.class_name / sequence.spec.source_name / source.name
        if gt_path.is_file():
            with Image.open(gt_path) as image:
                if image.size != native_size:
                    raise ValueError(f"Original GT/input size mismatch: {gt_path}, {input_path}")
        native_sizes.add(native_size)
        resized = generated.resize(native_size, Image.Resampling.LANCZOS)
        save_png_atomic(resized, resized_dir / source.name)
        roi = native_mask.sum(axis=2) > 255
        composed = np.asarray(resized).copy()
        composed[roi] = native_input[roi]
        target = composite_dir / source.name
        save_png_atomic(Image.fromarray(composed), target)
        composite_frames.append(target)

    if not skip_video:
        if len(native_sizes) != 1:
            raise ValueError(f"A video requires constant frame size: {sequence.spec.label}: {native_sizes}")
        encode_video(composite_frames, video_path, fps, next(iter(native_sizes)))
    metadata = {
        "sequence": sequence.spec.label,
        "class": sequence.spec.class_name,
        "frames": len(composite_frames),
        "native_sizes_wh": sorted(native_sizes),
        "source_512": str(sequence.output_dir),
        "resized": str(resized_dir),
        "native_roi": str(composite_dir),
        "native_input": str(sequence.image_dir),
        "native_mask": str(sequence.mask_dir),
        "roi_rule": "sum(RGB mask channels) > 255; equivalent to BGR channel sum",
        "roi_composite": "native input inside ROI, Lanczos-upscaled evaluator output outside ROI",
        "roi_feather_px": 0,
        "video": None if skip_video else str(video_path),
        "video_fps": None if skip_video else fps,
        "video_codec": None if skip_video else "MPEG-4 qscale 2 yuv420p",
    }
    composite_dir.mkdir(parents=True, exist_ok=True)
    (composite_dir / "postprocess_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main():
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    selected = args.sequences if args.sequences is not None else parse_sequence_names(os.environ.get("SEQUENCES"))
    discover = (dataset_root / "input").is_dir() and (dataset_root / "mask").is_dir()
    sequences = load_sequences(dataset_root, output_root, selected,
                               discover_hierarchical=discover)
    max_images = int(os.environ.get("MAX_IMAGES", "0"))
    if max_images < 0:
        raise ValueError("MAX_IMAGES must be nonnegative")
    if max_images:
        sequences = [replace(seq, image_paths=seq.image_paths[:max_images],
                             mask_paths=seq.mask_paths[:max_images],
                             frame_ids=seq.frame_ids[:max_images]) for seq in sequences]
    print_preflight(sequences)
    if not args.skip_video:
        for program in ("ffmpeg", "ffprobe"):
            if shutil.which(program) is None:
                raise RuntimeError(f"{program} was not found in PATH")
        unknown = [seq.spec.label for seq in sequences if args.fps is None and seq.spec.source_name not in FPS]
        if unknown:
            raise ValueError(f"Unknown video FPS for {unknown}; provide --fps")
    if args.postprocess_only:
        for sequence in sequences:
            verify_source_frames(sequence)
    if args.preflight:
        print(f"[Pipeline] inference={'skip' if args.postprocess_only else INFERENCE_SCRIPT}", flush=True)
        print("[Pipeline] 512 PNG -> resized_to_original -> resized_to_original_native_roi -> videos", flush=True)
        return

    if not args.postprocess_only:
        env = os.environ.copy()
        proposed_dir = str(SCRIPT_DIR / "proposed")
        python_path = env.get("PYTHONPATH", "")
        path_parts = [part for part in python_path.split(os.pathsep) if part]
        if proposed_dir not in path_parts:
            env["PYTHONPATH"] = os.pathsep.join([proposed_dir, *path_parts])
        env.update({
            "LONG_TEST_ROOT": str(dataset_root), "OUTPUT_ROOT": str(output_root),
            "DISCOVER_HIERARCHICAL_SEQUENCES": "1" if discover else "0",
            "PREFLIGHT_ONLY": "0",
        })
        if selected is not None:
            env["SEQUENCES"] = ",".join(selected)
        else:
            env.pop("SEQUENCES", None)
        subprocess.run([sys.executable, str(INFERENCE_SCRIPT)], cwd=REPO_ROOT,
                       env=env, check=True)

    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(postprocess_sequence, sequence, dataset_root, output_root,
                        args.fps or FPS.get(sequence.spec.source_name), args.skip_video): sequence
            for sequence in sequences
        }
        for future in as_completed(pending):
            result = future.result()
            completed.append(result)
            print(f"[done] {result['sequence']}: {result['frames']} frames -> {result['video'] or result['native_roi']}",
                  flush=True)
    print(f"Completed {len(completed)} sequence(s), {sum(row['frames'] for row in completed)} frames", flush=True)


if __name__ == "__main__":
    main()
