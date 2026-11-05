"""COCO BrushNet samples with source aspect ratio and bounded spatial size."""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor


class NativeResolutionDataset(Dataset):
    def __init__(
        self,
        root,
        tokenizer,
        max_width=2048,
        max_height=1280,
        pad_multiple=64,
        largest_n=0,
        benchmark_repeat_each=1,
        benchmark_image_id=None,
    ):
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.max_width = max_width
        self.max_height = max_height
        self.pad_multiple = pad_multiple
        self.clip_processor = CLIPImageProcessor()
        self.image_ids = sorted(path.stem for path in (self.root / "gt").glob("*.png"))
        if benchmark_image_id:
            if benchmark_image_id not in self.image_ids:
                raise ValueError(f"Unknown benchmark image: {benchmark_image_id}")
            self.image_ids = [benchmark_image_id]
        if largest_n:
            sizes = []
            for image_id in self.image_ids:
                with Image.open(self.root / "gt" / f"{image_id}.png") as image:
                    width, height = image.size
                scale = min(1.0, max_width / width, max_height / height)
                resized_width = max(1, round(width * scale))
                resized_height = max(1, round(height * scale))
                padded_width = math.ceil(resized_width / pad_multiple) * pad_multiple
                padded_height = math.ceil(resized_height / pad_multiple) * pad_multiple
                sizes.append((padded_width * padded_height, image_id))
            self.image_ids = [image_id for _, image_id in sorted(sizes, reverse=True)[:largest_n]]
        if benchmark_repeat_each > 1:
            self.image_ids = [image_id for image_id in self.image_ids for _ in range(benchmark_repeat_each)]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        with Image.open(self.root / "gt" / f"{image_id}.png") as image:
            gt = image.convert("RGB")
        with Image.open(self.root / "input" / f"{image_id}.png") as image:
            source = image.convert("RGB")
        with Image.open(self.root / "mask" / f"{image_id}.png") as image:
            source_mask = image.convert("L")
        if gt.size != source.size or gt.size != source_mask.size:
            raise ValueError(f"Mismatched GT/input/mask size for {image_id}")

        # Keep the same mask polarity and FG/BG CLIP inputs as custom_dataset_v04.
        source_np = np.asarray(source, dtype=np.float32)
        mask_np = 1.0 - np.asarray(source_mask, dtype=np.float32) / 255.0
        foreground = Image.fromarray((source_np * mask_np[..., None]).astype(np.uint8))
        background = Image.fromarray((source_np * (1.0 - mask_np[..., None])).astype(np.uint8))

        original_width, original_height = gt.size
        width, height = original_width, original_height
        scale = min(1.0, self.max_width / width, self.max_height / height)
        width = max(1, round(width * scale))
        height = max(1, round(height * scale))
        if (width, height) != gt.size:
            gt = gt.resize((width, height), Image.Resampling.LANCZOS)
            source = source.resize((width, height), Image.Resampling.LANCZOS)
            source_mask = source_mask.resize((width, height), Image.Resampling.NEAREST)

        gt = torch.from_numpy(np.asarray(gt, dtype=np.float32).copy()).permute(2, 0, 1) / 127.5 - 1.0
        source = torch.from_numpy(np.asarray(source, dtype=np.float32).copy()).permute(2, 0, 1) / 127.5 - 1.0
        mask = 1.0 - torch.from_numpy(np.asarray(source_mask, dtype=np.float32).copy())[None] / 255.0
        valid = torch.ones_like(mask)
        pad_right = (-width) % self.pad_multiple
        pad_bottom = (-height) % self.pad_multiple
        padding = (0, pad_right, 0, pad_bottom)
        gt = F.pad(gt, padding, mode="replicate")
        source = F.pad(source, padding, mode="replicate")
        mask = F.pad(mask, padding, value=0.0)
        valid = F.pad(valid, padding, value=0.0)

        clip = lambda image: self.clip_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        empty_ids = self.tokenizer(
            "", max_length=self.tokenizer.model_max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        ).input_ids.squeeze(0)
        return {
            "pixel_values": gt,
            "conditioning_pixel_values": source,
            "masks": mask,
            "valid_mask": valid,
            "input_ids": empty_ids,
            "clip_images": clip(Image.fromarray(source_np.astype(np.uint8))),
            "fg_clip_images": clip(foreground),
            "bg_clip_images": clip(background),
            "drop_image_embed": 0,
            "image_id": image_id,
            "original_size": (original_height, original_width),
            "prepad_size": (height, width),
        }
