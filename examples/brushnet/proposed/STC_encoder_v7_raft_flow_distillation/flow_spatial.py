"""Shared original-size preprocessing for cached RAFT teachers and students."""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def flow_shape(height: int, width: int, max_input_side: int = 960):
    """Match original-size eval: pad to 64, then cap RAFT's long side."""
    padded_h = ((height + 63) // 64) * 64
    padded_w = ((width + 63) // 64) * 64
    if max(padded_h, padded_w) <= max_input_side:
        return padded_h, padded_w
    scale = max_input_side / max(padded_h, padded_w)
    return (max(8, round(padded_h * scale / 8) * 8),
            max(8, round(padded_w * scale / 8) * 8))


def load_rgb(path: Path, max_input_side: int = 960) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    padded_h = ((height + 63) // 64) * 64
    padded_w = ((width + 63) // 64) * 64
    image = cv2.copyMakeBorder(image, 0, padded_h - height, 0,
                               padded_w - width, cv2.BORDER_REPLICATE)
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().unsqueeze(0)
    target = flow_shape(height, width, max_input_side)
    if target != (padded_h, padded_w):
        tensor = F.interpolate(tensor, size=target, mode="bilinear", align_corners=False)
    return tensor[0].div_(127.5).sub_(1.0)


def load_bg_mask(path: Path, max_input_side: int = 960) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape
    padded_h = ((height + 63) // 64) * 64
    padded_w = ((width + 63) // 64) * 64
    # The original-size evaluator pads ROI masks with zero (background).
    image = cv2.copyMakeBorder(image, 0, padded_h - height, 0,
                               padded_w - width, cv2.BORDER_CONSTANT, value=0)
    tensor = torch.from_numpy(image.copy()).float()[None, None]
    target = flow_shape(height, width, max_input_side)
    if target != (padded_h, padded_w):
        tensor = F.interpolate(tensor, size=target, mode="nearest")
    return (tensor[0] < 128).float()
