"""Video clips and raw/refined optical flow for temporal BrushNet training."""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BrushNetMotionClipDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        refined_flow_root,
        manifest,
        split="train",
        clip_length=4,
        height=512,
        width=512,
        stride=1,
        random_horizontal_flip=True,
        caption="",
        flow_source="refined",
    ):
        self.dataset_root = Path(dataset_root)
        self.flow_root = Path(refined_flow_root)
        self.flow_source = str(flow_source).lower()
        if self.flow_source not in {"raw", "refined"}:
            raise ValueError("flow_source must be 'raw' or 'refined'")
        self.height, self.width = int(height), int(width)
        self.clip_length = int(clip_length)
        self.random_horizontal_flip = bool(random_horizontal_flip)
        self.caption = caption
        raw_manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.clips = []
        for video in raw_manifest.values():
            if video["split"] != split:
                continue
            last_start = int(video["end"]) - self.clip_length + 1
            for start in range(int(video["start"]), last_start + 1, int(stride)):
                self.clips.append((video["name"], start))
        if not self.clips:
            raise ValueError(f"No motion clips found for split={split}")

    def __len__(self):
        return len(self.clips)

    def _rgb(self, folder, index):
        path = self.dataset_root / folder / f"{index:06d}.png"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        return torch.from_numpy(image).permute(2, 0, 1).float() / 127.5 - 1.0

    def _roi(self, index):
        path = self.dataset_root / "mask" / f"{index:06d}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        mask = cv2.resize(
            mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST
        )
        return torch.from_numpy((mask >= 128).astype(np.float32)).unsqueeze(0)

    def __getitem__(self, item):
        video, start = self.clips[item]
        indices = list(range(start, start + self.clip_length))
        gt = torch.stack([self._rgb("GT", index) for index in indices])
        decoded = torch.stack([self._rgb("input", index) for index in indices])
        roi = torch.stack([self._roi(index) for index in indices])
        flow_f, flow_b, motion_confidence, stable = [], [], [], []
        for index0, index1 in zip(indices[:-1], indices[1:]):
            path = (
                self.flow_root
                / video
                / f"{index0:06d}_{index1:06d}.npz"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path) as data:
                if self.flow_source == "raw":
                    flow_f.append(torch.from_numpy(data["raw_f"].copy()))
                    flow_b.append(torch.from_numpy(data["raw_b"].copy()))
                else:
                    flow_f.append(torch.from_numpy(data["refined_f"].copy()))
                    flow_b.append(torch.from_numpy(data["refined_b"].copy()))
                    confidence_key = (
                        "motion_confidence"
                        if "motion_confidence" in data
                        else "stable_bg"
                    )
                    motion_confidence.append(
                        torch.from_numpy(data[confidence_key].copy())
                    )
                    stable.append(torch.from_numpy(data["stable_bg"].copy()))
        sample = {
            "gt_frames": gt,
            "decoded_frames": decoded,
            "roi_masks": roi,
            "flow_forward": torch.stack(flow_f).float(),
            "flow_backward": torch.stack(flow_b).float(),
            "caption": self.caption,
            "video": video,
            "start": start,
        }
        if self.flow_source == "refined":
            sample["motion_confidence"] = torch.stack(
                motion_confidence
            ).float()
            sample["temporal_confidence"] = torch.stack(stable).float()
        if self.random_horizontal_flip and random.random() < 0.5:
            for key in ("gt_frames", "decoded_frames", "roi_masks"):
                sample[key] = torch.flip(sample[key], dims=[-1])
            flow_keys = ["flow_forward", "flow_backward"]
            if self.flow_source == "refined":
                flow_keys.extend(("motion_confidence", "temporal_confidence"))
            for key in flow_keys:
                sample[key] = torch.flip(sample[key], dims=[-1])
            sample["flow_forward"][:, 0].mul_(-1)
            sample["flow_backward"][:, 0].mul_(-1)
        return sample
