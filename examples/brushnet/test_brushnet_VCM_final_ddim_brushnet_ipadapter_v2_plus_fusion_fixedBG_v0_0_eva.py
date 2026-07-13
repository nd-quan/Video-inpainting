# CUDA_VISIBLE_DEVICES=0 python /home/gpu_01/nas_naeun/BrushNet/examples/brushnet/test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion.py
# proposed 알고리즘임
# conda activate brushnet_ipadapter_new
# v1은 bin으로 불러오기(허깅페이스에서) v2는 train해서 저장한거 safetensor로 불러오기
########## 그 중에서 v2는 ip_adapter 폴더에서 불러오고 v3는 diffuser안에 구현된 load_ipadapter로
# v3는 v2가 잘 나와서 구현 안함

import os
import csv
import math
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
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from ip_adapter import FusionIPAdapter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ndquan")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
output_dir='/media/ssd1/ndquan/model_naeun/paper/BrushNet/Quan_test/results/Generated_image/BasketballPass/new/fixedBG_noisy_latent_corr_eva'
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

shared_bg_generator = torch.Generator(device).manual_seed(6789)
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


def make_sampling_callback(
    output_dir,
    save_before_steps,
    bg_mask,
    decode_chunk_size=1,
):
    os.makedirs(output_dir, exist_ok=True)

    def callback(pipe, step, timestep, callback_kwargs):
        # callback_on_step_end receives the latent after `step`, which is the
        # latent used as input before the next sampling step.
        before_step = step + 1
        if before_step not in save_before_steps:
            return callback_kwargs

        latents = callback_kwargs["latents"].detach()
        timestep_value = int(timestep.item())

        latent_dir = os.path.join(output_dir, "latent_heatmaps")
        os.makedirs(latent_dir, exist_ok=True)
        save_noisy_latent(
            noise=latents,
            bg_mask=bg_mask,
            output_path=os.path.join(
                latent_dir,
                f"before_step_{before_step:02d}"
                f"_scheduler_t_{timestep_value:04d}.png",
            ),
            title=(
                f"Latent before sampling step {before_step}, "
                f"scheduler t={timestep_value}"
            ),
        )

        decoded_dir = os.path.join(output_dir, "decoded_intermediates")
        os.makedirs(decoded_dir, exist_ok=True)
        with torch.no_grad():
            for start in range(0, latents.shape[0], decode_chunk_size):
                latent_chunk = latents[start:start + decode_chunk_size]

                decoded = pipe.vae.decode(
                    latent_chunk / pipe.vae.config.scaling_factor,
                    return_dict=False,
                )[0]

                images = pipe.image_processor.postprocess(
                    decoded,
                    output_type="pil",
                )

                for local_idx, image in enumerate(images):
                    frame_idx = start + local_idx
                    image.save(
                        os.path.join(
                            decoded_dir,
                            f"before_step_{before_step:02d}"
                            f"_scheduler_t_{timestep_value:04d}"
                            f"_frame_{frame_idx:03d}.png",
                        )
                    )

                del decoded, images

        return callback_kwargs

    return callback


def save_noisy_latent(noise, bg_mask, output_path, title):
    """Save fixed-scale latent heatmaps, a BG histogram, and the raw tensor."""
    noise = noise.detach().float().cpu()
    bg_mask = bg_mask.detach().float().cpu()
    expanded_bg_mask = bg_mask.expand_as(noise) > 0.5
    bg_values = noise[expanded_bg_mask]

    figure, axes = plt.subplots(2, noise.shape[1], figsize=(16, 7))
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad(color="black")

    for channel_idx in range(noise.shape[1]):
        channel = noise[0, channel_idx].numpy()
        bg_channel = np.where(bg_mask[0, 0].numpy() > 0.5, channel, np.nan)

        image = axes[0, channel_idx].imshow(
            channel,
            cmap=cmap,
            vmin=-3.0,
            vmax=3.0,
        )
        axes[0, channel_idx].set_title(f"Full channel {channel_idx}")
        axes[0, channel_idx].axis("off")

        axes[1, channel_idx].imshow(
            bg_channel,
            cmap=cmap,
            vmin=-3.0,
            vmax=3.0,
        )
        axes[1, channel_idx].set_title(f"BG channel {channel_idx}")
        axes[1, channel_idx].axis("off")

    figure.suptitle(title)
    figure.colorbar(image, ax=axes, fraction=0.02)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    histogram_path = output_path.replace(".png", "_histogram.png")
    histogram_figure, histogram_axis = plt.subplots(figsize=(6, 4))
    histogram_axis.hist(
        bg_values.numpy(),
        bins=100,
        range=(-4.0, 4.0),
        density=True,
        alpha=0.75,
        label="BG latent",
    )
    x_values = np.linspace(-4.0, 4.0, 400)
    standard_normal = np.exp(-0.5 * x_values**2) / math.sqrt(2.0 * math.pi)
    histogram_axis.plot(x_values, standard_normal, label="N(0, 1)")
    histogram_axis.set_xlabel("Latent value")
    histogram_axis.set_ylabel("Density")
    histogram_axis.set_title(title)
    histogram_axis.legend()
    histogram_figure.savefig(histogram_path, dpi=150, bbox_inches="tight")
    plt.close(histogram_figure)

    torch.save(
        {
            "noise": noise,
            "bg_mask": bg_mask,
        },
        output_path.replace(".png", ".pt"),
    )

    return {
        "bg_mean": float(bg_values.mean()),
        "bg_std": float(bg_values.std(unbiased=False)),
        "bg_min": float(bg_values.min()),
        "bg_max": float(bg_values.max()),
    }


strengths = [0.5, 0.75, 0.9, 0.95, 1.0]
save_before_steps = {5, 10, 20, 30, 40, 49}

sampling_visualization_dir = os.path.join(
    output_dir,
    "sampling_visualization",
)


num_frames_to_visualize = 5

# True: variance-preserving mixing có căn.
# False: linear mixing cũ không có căn.
use_variance_preserving = True
run_sampling = os.environ.get("RUN_SAMPLING", "1").lower() in {"1", "true", "yes", "on"}
mixing_name = "variance_preserving" if use_variance_preserving else "linear"
noisy_latent_dir = os.path.join(output_dir, "initial_noisy_latents", mixing_name)
os.makedirs(noisy_latent_dir, exist_ok=True)

stats_path = os.path.join(noisy_latent_dir, "noise_stats.csv")
with open(stats_path, "w", newline="") as stats_file:
    stats_writer = csv.writer(stats_file)
    stats_writer.writerow(
        ["frame", "strength", "mixing", "bg_mean", "bg_std", "bg_min", "bg_max"]
    )

samples_to_visualize = indexed_all[:num_frames_to_visualize]

for orig_idx, (image_path, mask_path, caption) in tqdm(
    samples_to_visualize,
    total=len(samples_to_visualize),
):
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

    latent_shape = (
        1,
        pipe.unet.config.in_channels,
        512 // pipe.vae_scale_factor,
        512 // pipe.vae_scale_factor,
    )
    frame_seed = 1234 + orig_idx
    independent_generator = torch.Generator("cpu").manual_seed(frame_seed)
    independent_noise_cpu = torch.randn(
        latent_shape,
        generator=independent_generator,
        device="cpu",
        dtype=torch.float32,
    )
    independent_noise = independent_noise_cpu.to(device=device, dtype=pipe.dtype)

    roi_mask_latent = torch.from_numpy(mask_np).permute(2, 0, 1).unsqueeze(0)
    roi_mask_latent = F.interpolate(
        roi_mask_latent.float(),
        size=independent_noise.shape[-2:],
        mode="nearest",
    ).to(device=device, dtype=pipe.dtype)
    bg_mask_latent = 1.0 - roi_mask_latent

    for strength in strengths:
        # Every strength uses exactly the same independent and shared noise.
        if use_variance_preserving:
            mixed_bg_noise = (
                math.sqrt(1.0 - strength) * independent_noise
                + math.sqrt(strength) * shared_bg_noise
            )
        else:
            mixed_bg_noise = (
                (1.0 - strength) * independent_noise
                + strength * shared_bg_noise
            )

        final_noise = (
            independent_noise * roi_mask_latent
            + mixed_bg_noise * bg_mask_latent
        )

        latent_frame_dir = os.path.join(
            noisy_latent_dir,
            f"source_frame_{orig_idx:03d}",
        )
        os.makedirs(latent_frame_dir, exist_ok=True)
        latent_output_path = os.path.join(
            latent_frame_dir,
            f"strength_{strength:.2f}.png",
        )
        latent_stats = save_noisy_latent(
            noise=final_noise,
            bg_mask=bg_mask_latent,
            output_path=latent_output_path,
            title=(
                f"Initial BG noise ({mixing_name}), "
                f"frame={orig_idx}, strength={strength:.2f}"
            ),
        )

        with open(stats_path, "a", newline="") as stats_file:
            stats_writer = csv.writer(stats_file)
            stats_writer.writerow(
                [
                    orig_idx,
                    strength,
                    mixing_name,
                    latent_stats["bg_mean"],
                    latent_stats["bg_std"],
                    latent_stats["bg_min"],
                    latent_stats["bg_max"],
                ]
            )
        print(
            f"[Frame {orig_idx}, strength={strength:.2f}, {mixing_name}] "
            f"BG mean={latent_stats['bg_mean']:.6f}, "
            f"BG std={latent_stats['bg_std']:.6f}"
        )

        if not run_sampling:
            del mixed_bg_noise, final_noise
            continue

        torch.manual_seed(frame_seed)
        torch.cuda.manual_seed_all(frame_seed)

        frame_generator = torch.Generator(device=device).manual_seed(frame_seed)

        callback_dir = os.path.join(
            sampling_visualization_dir,
            mixing_name,
            f"source_frame_{orig_idx:03d}",
            f"strength_{strength:.2f}",
        )

        callback = make_sampling_callback(
            output_dir=callback_dir,
            save_before_steps=save_before_steps,
            bg_mask=bg_mask_latent,
            decode_chunk_size=1,
        )

        result = ip_model.generate_fgbg(
            fg_pil_image=fg_pil,
            bg_pil_image=bg_pil,
            prompt=caption,
            image=init_image,
            mask_image=mask_image,
            num_samples=1,
            num_inference_steps=50,
            generator=frame_generator,
            latents=independent_noise.clone(),
            use_shared_bg_noise=True,
            shared_bg_noise=shared_bg_noise,
            shared_bg_noise_strength=strength,
            variance_preserving_shared_noise=use_variance_preserving,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        image = result[0]

        if blended:
            image_np = np.array(image)
            original_np = cv2.imread(image_path)[:, :, ::-1]
            roi_mask = 1.0 * (
                cv2.imread(mask_path).sum(-1) > 255
            )[:, :, np.newaxis]

            original_np = cv2.resize(
                original_np,
                (512, 512),
                interpolation=cv2.INTER_LINEAR,
            )
            roi_mask = cv2.resize(
                roi_mask,
                (512, 512),
                interpolation=cv2.INTER_NEAREST,
            )

            bg_mask = (1.0 - roi_mask)[:, :, np.newaxis]
            original_roi = original_np * (1.0 - bg_mask)

            blurred_bg = cv2.GaussianBlur(
                bg_mask[:, :, 0] * 255,
                (21, 21),
                0,
            ) / 255
            blurred_bg = blurred_bg[:, :, np.newaxis]
            blend_mask = 1.0 - (1.0 - bg_mask) * (1.0 - blurred_bg)

            image_pasted = (
                original_roi * (1.0 - blend_mask)
                + image_np * blend_mask
            )
            image = Image.fromarray(image_pasted.astype(np.uint8))

        final_dir = os.path.join(
            output_dir,
            "final_outputs",
            mixing_name,
            f"strength_{strength:.2f}",
        )
        os.makedirs(final_dir, exist_ok=True)

        basename = os.path.basename(image_path)
        image.save(os.path.join(final_dir, basename))

        del result, image, mixed_bg_noise, final_noise
        torch.cuda.empty_cache()

# ========================================================================
# for orig_idx, image_path, mask_path, caption in tqdm(indexed_pending, total=len(indexed_pending)):
#     init_image_np = cv2.imread(image_path)[:, :, ::-1]
#     mask_np = 1. * (cv2.imread(mask_path).sum(-1) > 255)[:, :, np.newaxis]

#     init_image = Image.fromarray(init_image_np.astype(np.uint8)).convert("RGB")
#     mask_image = Image.fromarray((mask_np * 255).astype(np.uint8).repeat(3, -1)).convert("RGB")

#     transform = transforms.Compose([
#         transforms.Resize((512, 512)),
#     ])
#     init_image = transform(init_image)
#     mask_image = transform(mask_image)
    
#     # fg, bg
#     fg_np = init_image_np * mask_np
#     bg_np = init_image_np * (1 - mask_np)

#     fg_pil = Image.fromarray(fg_np.astype(np.uint8)).convert("RGB")
#     bg_pil = Image.fromarray(bg_np.astype(np.uint8)).convert("RGB")
    
#     fg_pil = transform(fg_pil)
#     bg_pil = transform(bg_pil)

#     # # 이미지 생성
#     result = ip_model.generate_fgbg(
#     fg_pil_image=fg_pil,
#     bg_pil_image=bg_pil,
#     prompt=caption,
#     image=init_image,
#     mask_image=mask_image,
#     num_inference_steps=50,
#     generator=generator,
#     use_shared_bg_noise=True,
#     shared_bg_noise=shared_bg_noise,
#     shared_bg_noise_strength=0.95,
#     variance_preserving_shared_noise = False,
#     )
    
#     # image = result.images[0]
#     image=result[0]
#     # image=init_image # 초기 이미지 보기
#     # blending
#     if blended:
#         print(f"[{orig_idx}] blending 중...")

#         image_np = np.array(image)
#         init_image_np = cv2.imread(image_path)[:, :, ::-1]
#         mask_np = 1. * (cv2.imread(mask_path).sum(-1) > 255)[:, :, np.newaxis]

#         new_size = (512, 512)
#         init_image_np = cv2.resize(init_image_np, new_size, interpolation=cv2.INTER_LINEAR)
#         mask_np = cv2.resize(mask_np, new_size, interpolation=cv2.INTER_NEAREST)

#         mask_np = 1 - mask_np
#         mask_np = mask_np[:, :, np.newaxis]
#         init_image_np = init_image_np * (1 - mask_np)

#         mask_blurred = cv2.GaussianBlur(mask_np * 255, (21, 21), 0) / 255
#         mask_blurred = mask_blurred[:, :, np.newaxis]
#         mask_np = 1 - (1 - mask_np) * (1 - mask_blurred)

#         image_pasted = init_image_np * (1 - mask_np) + image_np * mask_np
#         image = Image.fromarray(image_pasted.astype(np.uint8))
             
#     # 저장
#     basename = os.path.basename(image_path)
#     image.save(os.path.join(output_dir, basename))
