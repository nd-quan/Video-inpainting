# CUDA_VISIBLE_DEVICES=5 python /media/ssd2/naeun/BrushNet/examples/t2i_adapter/test_t2i_adapter_VCM_final.py
from glob import glob
from tqdm import tqdm
from diffusers import StableDiffusionAdapterPipeline, T2IAdapter, EulerDiscreteScheduler
import torch
import cv2
import numpy as np
from PIL import Image
import os

# 기본 경로
image_dir = "/media/ssd1/daole/VCM_Proposed/data/test_in_COCO" # coco
# mask_dir = "/media/hdd/daole/VCM_Proposed/data/test_mask_COCO"
mask_dir = "/media/ssd1/daole/VCM_Proposed/data/test_mask_COCO" # coco
# mask_dir='/media/ssd2/naeun/ws04/BrushNet/dataset/opendataset/test_mask_OID_resized' # open
caption_txt='/media/ssd2/naeun/data/caption/test/captions_test_COCO.txt' #coco(ws09)
# output_dir = "/media/hdd/naeun/save/BrushNet_200000"
# output_dir = "/media/hdd/naeun/save/test_with_originalcaption/Brushnet_200000"
output_dir='/media/ssd2/naeun/result/t2idapter/COCO/512_t2iadapter_600000'

os.makedirs(output_dir, exist_ok=True)

# 모델 로드
base_model_path="stable-diffusion-v1-5/stable-diffusion-v1-5" #512
t2iadapter_path = "/media/ssd2/naeun/save/checkpoint/Checkpoint_t2iadapter_512/checkpoint-600000/t2iadapter"

adapter = T2IAdapter.from_pretrained(t2iadapter_path, torch_dtype=torch.float16)
pipe = StableDiffusionAdapterPipeline.from_pretrained(
    base_model_path, adapter=adapter, torch_dtype=torch.float16, low_cpu_mem_usage=False
)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()


# caption 불러오기
with open(caption_txt, 'r') as f:
    captions = [line.strip() for line in f.readlines()]

# 이미지 목록 정렬
image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))

assert len(image_paths) == len(captions), "이미지 수와 캡션 수가 일치하지 않습니다."

for idx, (image_path, mask_path, caption) in tqdm(enumerate(zip(image_paths, mask_paths, captions)), total=len(captions)):
    # 원본 이미지 로딩 및 전처리
    init_image = cv2.imread(image_path)[:, :, ::-1]
    init_image = cv2.resize(init_image, (512, 512))
    init_image = torch.from_numpy(init_image.copy()).permute(2, 0, 1).float()
    init_image = (init_image / 255.0) * 2.0 - 1.0
    init_image = init_image.unsqueeze(0)
    generator = torch.Generator("cuda").manual_seed(1234 + idx)
    # 이미지 생성
    image = pipe(
        prompt=caption, 
        image=init_image,
        num_inference_steps=50, # hugging face에서는 없는 파라미터긴 함
        generator=generator,
    ).images[0]

    # Copy & Paste 적용 (VCM 방식)
    image_np = np.array(image)
    init_image_np = cv2.imread(image_path)[:, :, ::-1]
    mask_np = 1. * (cv2.imread(mask_path).sum(-1) > 255)[:, :, np.newaxis]

    init_image_np = cv2.resize(init_image_np, (512, 512), interpolation=cv2.INTER_LINEAR)
    mask_np = cv2.resize(mask_np, (512, 512), interpolation=cv2.INTER_NEAREST)

    mask_np = 1 - mask_np # 마스크 반전
    mask_np = (mask_np > 0.5).astype(np.float32)  # 마스크 이진화
    mask_np = mask_np[:, :, np.newaxis]  # 혹시 리사이즈 후 채널이 사라졌을 경우 대비
    
    init_image_np = init_image_np * (1 - mask_np)

    image_pasted = init_image_np * (1 - mask_np) + image_np * mask_np
    image_pasted = image_pasted.astype(image_np.dtype)
    image_final = Image.fromarray(image_pasted)

    # 저장
    basename = os.path.basename(image_path)
    image_final.save(os.path.join(output_dir, basename))




