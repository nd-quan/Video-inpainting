# CUDA_VISIBLE_DEVICES=0 python /home/gpu_01/nas_naeun/BrushNet/examples/brushnet/test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion.py
# proposed 알고리즘임
# conda activate brushnet_ipadapter_new
# v1은 bin으로 불러오기(허깅페이스에서) v2는 train해서 저장한거 safetensor로 불러오기
########## 그 중에서 v2는 ip_adapter 폴더에서 불러오고 v3는 diffuser안에 구현된 load_ipadapter로
# v3는 v2가 잘 나와서 구현 안함

import os
from glob import glob
from tqdm import tqdm
import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
# from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, DDIMScheduler, UNet2DConditionOutput

from diffusers import DDIMScheduler
# from diffusers.pipelines.brushnet.pipeline_brushnet import StableDiffusionBrushNetPipeline
from diffusers.pipelines.brushnet.pipeline_brushnet_sharedNoise_sameBG_v0_0 import StableDiffusionBrushNetPipeline
# from diffusers.pipelines.brushnet.pipeline_brushnet_sharedNoise_v1 import StableDiffusionBrushNetPipeline
from diffusers.models.brushnet import BrushNetModel

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from ip_adapter import FusionIPAdapter

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
device="cuda"

# 설정
# image_dir = "/home/gpu_01/nas_naeun/data/data/test_in_COCO" # coco
# image_dir='/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/test/PartyScene_512/images' # open
# image_dir='/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/test/PartyScene_512_backup/images' # open
# image_dir='/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/test/RaceHorses_512_backup/images' # open  
image_dir='/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/test/BasketballPass_512_backup/images' # open  

# image_dir="/media/ssd2/naeun/NAS_NE/data/data/New/synthesis_COCO"

# mask_dir = "/home/gpu_01/nas_naeun/data/data/test_mask_COCO" # coco
# mask_dir='/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/test/PartyScene_512_backup/masks' # open
# mask_dir='/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/test/RaceHorses_512_backup/masks' # open
mask_dir='/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/test/BasketballPass_512_backup/masks' # open

# mask_dir="/media/ssd2/naeun/NAS_NE/data/data/New/mask_COCO"

# caption_txt = "/media/ssd2/naeun/ws04/BrushNet/dataset/opendataset/captions_test_openimage.txt" #open(ws09)
# caption_txt='/home/gpu_01/nas_naeun/data/data/caption/test/captions_test_COCO.txt' #coco(ws09)
# caption_txt="/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/caption/caption_raceHorses.txt" # A100
caption_txt="/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/caption/caption_basketBallPass.txt" # A100
# caption_txt="/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/caption/caption_partyScene.txt" # A100

# output_dir = "/media/hdd/naeun/save/BrushNet_200000"
# output_dir = "/media/hdd/naeun/save/test_with_originalcaption/Brushnet_200000"
# output_dir='/media/hdd/naeun/save/Opendataset/BrushNet_300000'
output_dir='/media/ssd1/ndquan/model_naeun/paper/BrushNet/Quan_test/results/Generated_image/BasketballPass/new/fixedBG_strength_05_v0_0'
# test 15는 14에서 그냥 copy&paste
# test 16은 blending을 반대로 
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# base_model_path = "lambdalabs/miniSD-diffusers"
base_model_path="stable-diffusion-v1-5/stable-diffusion-v1-5" #512
# brushnet_path = "/media/hdd/naeun/save/checkpoint/Checkpoint_brushNet_200000"
# brushnet_path="/media/ssd2/naeun/ws04/BrushNet_previous/examples/brushnet/pretrained_brushnet/brushnet"
# brushnet_path="/media/ssd2/naeun/NAS_NE/checkpoint/Checkpoint_brushnet_512/checkpoint-600000/brushnet"
# brushnet_path="/home/gpu_01/naeun/v8/checkpoint-300000/brushnet"


brushnet_path="/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/checkpoint_naeun/checkpoint-200000/brushnet"
# brushnet_path="/media/ssd1/ndquan/model_naeun/paper/BrushNet/Quan_test/results/train_naeun/checkpoint-10/brushnet"


# 블렌딩 설정 바꾸기
blended = True


brushnet_conditioning_scale = 1.0

# 모델 로드
brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)

# print("DEBUG brushnet =", brushnet, type(brushnet))
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False,safety_checker=None
)

image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
).to(pipe.device, dtype=pipe.dtype)

# ip_ckpt="/home/gpu_01/naeun/v8/checkpoint-300000/ipadapter/model.safetensors"
ip_ckpt = "/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/checkpoint_naeun/checkpoint-200000/ipadapter/model.safetensors"
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

pipe.register_modules(
    image_encoder=image_encoder,
    feature_extractor=CLIPImageProcessor(),
)
# fusion_ckpt = "/home/gpu_01/naeun/v8/checkpoint-300000/ipadapter/fusion_module.safetensors"
fusion_ckpt = "/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/checkpoint_naeun/checkpoint-200000/ipadapter/fusion_module.safetensors"

ip_model = FusionIPAdapter(
    pipe,
    image_encoder_path,
    ip_ckpt,
    fusion_ckpt,
    device,
)

generator = torch.Generator("cuda").manual_seed(1234)

shared_bg_generator = torch.Generator(device).manual_seed(1234)
shared_bg_noise = torch.randn(
    (
        1,
        pipe.unet.config.in_channels,
        512 // pipe.vae_scale_factor,
        512 // pipe.vae_scale_factor,
    ),
    generator=shared_bg_generator,
    device=device,
    dtype=pipe.dtype,
)

# caption 불러오기
with open(caption_txt, 'r') as f:
    captions = [line.strip() for line in f.readlines()]

# 이미지 & 마스크 경로
image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))

assert len(image_paths) == len(captions), "이미지 수와 캡션 수가 일치하지 않습니다."

# 출력 폴더에 이미 생성된 결과들의 basename 집합
existing_basenames = {os.path.basename(p) for p in glob(os.path.join(output_dir, "*.png"))}

# 원본 인덱스를 보존한 채로 (idx, img, msk, cap) 리스트 생성
indexed_all = list(enumerate(zip(image_paths, mask_paths, captions)))

# 아직 생성되지 않은 샘플만 필터링
indexed_pending = [
    (idx, img, msk, cap)
    for idx, (img, msk, cap) in indexed_all
    if os.path.basename(img) not in existing_basenames
]

print(f"[Resume] 총 {len(indexed_all)}개 중 이미 {len(indexed_all) - len(indexed_pending)}개 완료, "
      f"{len(indexed_pending)}개 생성 예정.")


# """ # ----------- naeun code -----------

# ========================================================================
for orig_idx, image_path, mask_path, caption in tqdm(indexed_pending, total=len(indexed_pending)):
    init_image_np = cv2.imread(image_path)[:, :, ::-1]
    mask_np = 1. * (cv2.imread(mask_path).sum(-1) > 255)[:, :, np.newaxis]

    init_image = Image.fromarray(init_image_np.astype(np.uint8)).convert("RGB")
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8).repeat(3, -1)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
    ])
    init_image = transform(init_image)
    mask_image = transform(mask_image)
    
    # fg, bg
    fg_np = init_image_np * mask_np
    bg_np = init_image_np * (1 - mask_np)

    fg_pil = Image.fromarray(fg_np.astype(np.uint8)).convert("RGB")
    bg_pil = Image.fromarray(bg_np.astype(np.uint8)).convert("RGB")
    
    fg_pil = transform(fg_pil)
    bg_pil = transform(bg_pil)

    # # 이미지 생성
    result = ip_model.generate_fgbg(
    fg_pil_image=fg_pil,
    bg_pil_image=bg_pil,
    prompt=caption,
    image=init_image,
    mask_image=mask_image,
    num_inference_steps=50,
    generator=generator,
    use_shared_bg_noise=True,
    shared_bg_noise=shared_bg_noise,
    shared_bg_noise_strength=0.5,  # tested value: 1
    )
    
    # image = result.images[0]
    image=result[0]
    # image=init_image # 초기 이미지 보기
    # blending
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
             
    # 저장
    basename = os.path.basename(image_path)
    image.save(os.path.join(output_dir, basename))


