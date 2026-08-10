import argparse
import subprocess
from pathlib import Path
import os
import cv2

DEFAULT_DATASET_ROOT = Path(
    "/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/Generated_image"
)

# CASES = {
#     "BasketballDrill_832x480_50_roi18_bg52": 50,
#     "BlowingBubbles_416x240_50_roi18_bg52": 50,
#     "BQSquare_416x240_60_roi18_bg52": 60,
#     "FourPeople_1280x720_60_roi18_bg52": 60,
# }


CASES = {
    "BasketballPass": 50,
    # "PartyScene": 50,
    # "ParkScene": 24,
    # "Traffic": 30,
}

DEFAULT_FRAME_FOLDERS = ("sharedNoise_new_checkpoint2000_09_corr",)


def frameToVideo(img_dir, vid_output_path, fps=50, codec="png"):
    """Convert sequential PNG frames (000000.png, ...) to an MP4 video."""
    img_dir = Path(img_dir)
    vid_output_path = Path(vid_output_path)
    img_files = sorted(img_dir.glob("*.png"))

    first_img_path = os.path.join(img_dir, img_files[0])
    frame = cv2.imread(first_img_path)
    height, width = frame.shape[:2]

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
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4

    video = cv2.VideoWriter(str(vid_output_path), fourcc, fps, (width, height))  # fps

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()
    print(f"Video saved to {vid_output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert frame folders to videos."
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
    parser.add_argument(
        "--frame-folders",
        nargs="+",
        default=list(DEFAULT_FRAME_FOLDERS),
        help=(
            "Frame folders inside each case "
            "(default: tmpGuidance)."
        ),
    )
    parser.add_argument(
        "--codec",
        choices=("png", "mpeg4"),
        default="png",
        help=(
            "png preserves every RGB pixel but creates larger MP4 files; "
            "mpeg4 is smaller and more compatible but remains lossy."
        ),
    )
    args = parser.parse_args()

    # Validate all sources before producing partial output.
    missing = [
        args.dataset_root / case_name / frame_type
        for case_name in args.cases
        for frame_type in args.frame_folders
        if not (args.dataset_root / case_name / frame_type).is_dir()
    ]
    if missing:
        parser.error(
            "Missing frame folder(s):\n  " + "\n  ".join(map(str, missing))
        )

    for case_name in args.cases:
        case_dir = args.dataset_root / case_name
        fps = CASES[case_name]
        for frame_folder in args.frame_folders:
            frame_dir = case_dir / frame_folder
            video_dir = frame_dir / "video"
            frameToVideo(
                frame_dir,
                video_dir / "output.mp4",
                fps,
                codec=args.codec,
            )


if __name__ == "__main__":
    main()
