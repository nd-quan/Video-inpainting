import argparse
import subprocess
from pathlib import Path


DEFAULT_DATASET_ROOT = Path(
    "/media/ssd1/ndquan/NAS_ndq/model_base/videoInpainting/Datasets/SFU"
)

CASES = {
    "BasketballDrill_832x480_50_roi18_bg52": 50,
    "BlowingBubbles_416x240_50_roi18_bg52": 50,
    "BQSquare_416x240_60_roi18_bg52": 60,
    "FourPeople_1280x720_60_roi18_bg52": 60,
}


def frameToVideo(img_dir, vid_output_path, fps=50):
    """Convert sequential PNG frames (000000.png, ...) to an MP4 video."""
    img_dir = Path(img_dir)
    vid_output_path = Path(vid_output_path)
    img_files = sorted(img_dir.glob("*.png"))

    if not img_files:
        raise FileNotFoundError(f"No PNG frames found in: {img_dir}")

    expected_names = [f"{index:06d}.png" for index in range(len(img_files))]
    actual_names = [path.name for path in img_files]
    if actual_names != expected_names:
        raise ValueError(
            f"Frames in {img_dir} must be contiguous from 000000.png "
            f"to {len(img_files) - 1:06d}.png"
        )

    vid_output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v", "error",
            "-y",
            "-framerate", str(fps),
            "-start_number", "0",
            "-i", str(img_dir / "%06d.png"),
            "-frames:v", str(len(img_files)),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(vid_output_path),
        ],
        check=True,
    )
    print(
        f"Saved {vid_output_path} "
        f"({len(img_files)} frames, {fps} FPS)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert GT and input frame folders to videos."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Directory containing the configured dataset cases.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=CASES,
        default=list(CASES),
        help="Cases to process (default: all four cases).",
    )
    args = parser.parse_args()

    # Validate all sources before producing partial output.
    missing = [
        args.dataset_root / case_name / frame_type
        for case_name in args.cases
        for frame_type in ("GT", "input")
        if not (args.dataset_root / case_name / frame_type).is_dir()
    ]
    if missing:
        parser.error(
            "Missing frame folder(s):\n  " + "\n  ".join(map(str, missing))
        )

    for case_name in args.cases:
        case_dir = args.dataset_root / case_name
        video_dir = case_dir / "video"
        fps = CASES[case_name]
        frameToVideo(case_dir / "GT", video_dir / "gt.mp4", fps)
        frameToVideo(case_dir / "input", video_dir / "input.mp4", fps)


if __name__ == "__main__":
    main()
