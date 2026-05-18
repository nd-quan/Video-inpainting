# CUDA_VISIBLE_DEVICES=6 python /media/ssd2/naeun/BrushNet/examples/brushnet/test_brushnet_VCM_final_ddim_just_test.py

import os
from glob import glob
from tqdm import tqdm
from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, DDIMScheduler
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

##################### 디퓨전 값 고정하기 위해서
import torch
import random

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# Deterministic behavior
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)
# torch.set_deterministic(True)  # 버전에 따라 권장되지 않음
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True

# 설정
image_dir = "/media/ssd1/daole/VCM_Proposed/data/test_in_COCO" # coco
mask_dir = "/media/ssd1/daole/VCM_Proposed/data/test_mask_COCO" # coco
caption_txt='/media/ssd2/naeun/data/caption/test/captions_test_COCO.txt' #coco(ws09)

output_dir='/media/ssd2/naeun/result/brushnet/COCO/test_newsetting/inference20_guidance3.5'
os.makedirs(output_dir, exist_ok=True)

base_model_path="stable-diffusion-v1-5/stable-diffusion-v1-5" #512
brushnet_path="/media/ssd2/naeun/NAS_NE/checkpoint/Checkpoint_brushnet_512/checkpoint-600000/brushnet"
blended = True
brushnet_conditioning_scale = 1.0

# ✅ 여기: 생성하고 싶은 orig_idx 리스트 (set으로)
target_indices = {
    4129, 1314, 4739, 3379, 185, 1215, 726 ,730, 1026, 847 ,4257,454, 2027 ,3039 ,2206 ,1366 ,2485 ,4559 ,425 ,206
}

# 모델 로드
brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False,safety_checker=None # safety 추가
)

# pipe = StableDiffusionBrushNetPipeline.from_pretrained(
#     base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False # safety 추가
# )
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

generator = torch.Generator("cuda").manual_seed(1234)

# caption 불러오기
with open(caption_txt, 'r') as f:
    captions = [line.strip() for line in f.readlines()]

# 이미지 & 마스크 경로
image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))

assert len(image_paths) == len(captions), "이미지 수와 캡션 수가 일치하지 않습니다."

# 이미 저장된 결과 basename 집합
existing_basenames = {os.path.basename(p) for p in glob(os.path.join(output_dir, "*.png"))}

# (idx, img, msk, cap) 리스트 생성
indexed_all = list(enumerate(zip(image_paths, mask_paths, captions)))

# ✅ target_indices + 아직 생성 안 된 것만 필터링
indexed_pending = [
    (idx, img, msk, cap)
    for idx, (img, msk, cap) in indexed_all
    if idx in target_indices and os.path.basename(img) not in existing_basenames
]

print(f"[Resume] 총 {len(indexed_all)}개 중 타겟 {len(target_indices)}개,"
      f" 이 중 이미 완료 {len(target_indices) - len(indexed_pending)}개,"
      f" 새로 생성 {len(indexed_pending)}개 예정.")

for orig_idx, image_path, mask_path, caption in tqdm(indexed_pending, total=len(indexed_pending)):
    init_image = cv2.imread(image_path)[:, :, ::-1]
    mask_image = 1. * (cv2.imread(mask_path).sum(-1) > 255)[:, :, np.newaxis]

    init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
    mask_image = Image.fromarray((mask_image * 255).astype(np.uint8).repeat(3, -1)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
    ])
    init_image = transform(init_image)
    mask_image = transform(mask_image)

    result = pipe(
        caption,
        init_image,
        mask_image,
        num_inference_steps=20,
        generator=generator,
        brushnet_conditioning_scale=brushnet_conditioning_scale,
        guidance_scale=3.5 # 이것도 세팅값
    )
    image = result.images[0]

    if blended:
        print(f"[{orig_idx}] blending 중...")

        image_np = np.array(image)
        init_image_np = cv2.imread(image_path)[:, :, ::-1]
        mask_np = 1. * (cv2.imread(mask_path).sum(-1) > 255)[:, :, np.newaxis]

        new_size = (512, 512)
        init_image_np = cv2.resize(init_image_np, new_size, interpolation=cv2.INTER_LINEAR)
        mask_np = cv2.resize(mask_np, new_size, interpolation=cv2.INTER_NEAREST)

        mask_np = 1 - mask_np
        mask_np = mask_np[:, :, np.newaxis]
        init_image_np = init_image_np * (1 - mask_np)

        mask_blurred = cv2.GaussianBlur(mask_np * 255, (21, 21), 0) / 255
        mask_blurred = mask_blurred[:, :, np.newaxis]
        mask_np = 1 - (1 - mask_np) * (1 - mask_blurred)

        image_pasted = init_image_np * (1 - mask_np) + image_np * mask_np
        image = Image.fromarray(image_pasted.astype(np.uint8))

    basename = os.path.basename(image_path)
    image.save(os.path.join(output_dir, basename))
