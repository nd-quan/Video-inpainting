"""Chuyển ROI mask thành background (BG) mask bằng cách đảo trắng/đen.

Ví dụ:
    python BG_mask.py --input roi_mask.png --output bg_mask.png
"""

from argparse import ArgumentParser
from pathlib import Path

import cv2


def roi_to_bg_mask(input_path: str, output_path: str) -> None:
    """Đọc ROI mask, nhị phân hóa và đảo vùng trắng/đen để tạo BG mask."""
    roi_mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if roi_mask is None:
        raise FileNotFoundError(f"Không thể đọc mask: {input_path}")

    # Chuẩn hóa về mask đen/trắng rồi đảo: ROI trắng -> BG đen, và ngược lại.
    _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
    bg_mask = cv2.bitwise_not(roi_mask)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(output_path, bg_mask):
        raise IOError(f"Không thể ghi BG mask: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Đảo ROI mask thành BG mask.")
    parser.add_argument("--input", "-i", required=True, help="Đường dẫn ROI mask đầu vào")
    parser.add_argument("--output", "-o", required=True, help="Đường dẫn BG mask đầu ra")
    args = parser.parse_args()

    roi_to_bg_mask(args.input, args.output)
    print(f"Đã tạo BG mask: {args.output}")
