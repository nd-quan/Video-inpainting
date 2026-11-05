import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import InterpolationMode
import torchvision.utils as vutils


class CustomDataset(Dataset):
    def __init__(self, captions_file, gt_folder, mask_folder, synthesis_folder, tokenizer):
        self.gt_folder = gt_folder
        self.mask_folder = mask_folder
        self.synthesis_folder = synthesis_folder
        self.tokenizer = tokenizer
        
        # RGB 이미지용 transform (resize + 정규화)
        self.image_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 마스크용 transform (resize + 정규화)
        self.mask_transform = transforms.Compose([
            # transforms.Resize((512, 512)),
            transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST), # BG/ROI 만들려고 수정
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: 2.0 * (1.0 - x) - 1.0)  # 마스크 색상 반전 후 -1 ~ 1로 정규화 기존에는 이렇게 했는데 아래께
            transforms.Lambda(lambda x: 1.0 - x)  # 마스크 색상 반전 후 0 ~ 1로 정규화
        ])

        self.captions = {}
        with open(captions_file, "r") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    image_id, caption = parts
                    self.captions[image_id] = caption

        self.image_ids = sorted([f.split(".")[0] for f in os.listdir(gt_folder) if f.endswith(".png")])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        gt_path = os.path.join(self.gt_folder, f"{image_id}.png")
        mask_path = os.path.join(self.mask_folder, f"{image_id}.png")
        synthesis_path = os.path.join(self.synthesis_folder, f"{image_id}.png")

        # 이미지 로드
        image = Image.open(gt_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 그레이스케일로 로드
        synthesis_image = Image.open(synthesis_path).convert("RGB")
        
        # debug_path = "/media/ssd2/naeun/test/debug_raw"
        # os.makedirs(debug_path, exist_ok=True)
        # image.save(os.path.join(debug_path, f"{image_id}_raw_image.png"))
        # mask.save(os.path.join(debug_path, f"{image_id}_raw_mask.png"))

        # 각각 적절한 transform 적용
        image = self.image_transform(image)  # 정규화된 GT 이미지
        mask = self.mask_transform(mask)     # 색상 반전된 마스크
              
        synthesis_image = self.image_transform(synthesis_image)  # 정규화된 synthesis 이미지

        BG= synthesis_image*mask
        ROI= synthesis_image*(1-mask)
        caption = self.captions.get(image_id, "")

        encoded_caption = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # # ── ROI/BG 저장 (아주 간단 버전) ─────────────────────────────
        # save_root = "/media/ssd2/naeun/test/debug_roi_bg_2"   # 
        # out_dir = os.path.join(save_root, image_id)
        # os.makedirs(out_dir, exist_ok=True)
        # vutils.save_image((ROI * 0.5 + 0.5).clamp(0,1), os.path.join(out_dir, "ROI.png"))
        # vutils.save_image((BG  * 0.5 + 0.5).clamp(0,1), os.path.join(out_dir, "BG.png"))
        
        return {
            "pixel_values": image,
            "masks": mask,
            "ROI_condition" : ROI,
            "BG_condition": BG,
            "input_ids": encoded_caption["input_ids"].squeeze(0)
        }




