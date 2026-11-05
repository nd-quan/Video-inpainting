import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 저장 경로 설정 (controlnet으로 수정하기)
# save_path = "/media/hdd/naeun/BrushNet/test/"

# 이미지 변환 후 저장 (0~1 범위로 변환)
def save_image(tensor, filename):
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
    img = (img * 0.5) + 0.5  # 정규화된 이미지를 [0,1] 범위로 복원
    plt.imsave(filename, img)

def save_mask(tensor, filename):
    # tensor가 2차원일 경우 (H, W) 형태로 가정
    if tensor.dim() == 2:
        img = tensor.cpu().numpy()  # (H, W)
    else:
        img = tensor.squeeze(0).cpu().numpy()  # (C, H, W) -> (H, W)

    # 마스크가 0과 1 사이의 값으로 정규화되어 있다고 가정
    img = np.clip(img, 0, 1)  # 0과 1 사이로 클리핑
    img = (img * 255).astype(np.uint8)  # 0-255 범위로 변환

    plt.imsave(filename, img, cmap='gray')  # 흑백 이미지로 저장

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
            transforms.Resize((512, 512)),
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

        # 각각 적절한 transform 적용
        image = self.image_transform(image)  # 정규화된 GT 이미지
        mask = self.mask_transform(mask)     # 색상 반전된 마스크
        synthesis_image = self.image_transform(synthesis_image)  # 정규화된 synthesis 이미지

        caption = self.captions.get(image_id, "")

        encoded_caption = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # 저장 실행
        
        # save_image(image, f"{save_path}/gt_image.png")
        # save_image(synthesis_image, f"{save_path}/synthesis_image.png")
        # save_mask(mask, f"{save_path}/mask_image.png") 
        #print("이미지가 저장되었습니다.")

        return {
            "pixel_values": image,
            "masks": mask,
            "conditioning_pixel_values": synthesis_image,
            "input_ids": encoded_caption["input_ids"].squeeze(0)
        }




