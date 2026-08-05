# CUDA_VISIBLE_DEVICES=0 python /home/gpu_01/nas_naeun/BrushNet/examples/brushnet/test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion.py
# proposed 알고리즘임
# conda activate brushnet_ipadapter_new
# v1은 bin으로 불러오기(허깅페이스에서) v2는 train해서 저장한거 safetensor로 불러오기
########## 그 중에서 v2는 ip_adapter 폴더에서 불러오고 v3는 diffuser안에 구현된 load_ipadapter로
# v3는 v2가 잘 나와서 구현 안함

import os
import sys
from glob import glob
from tqdm import tqdm

from diffusers.pipelines.brushnet.pipeline_brushnet_sharedNoise_sameBG_v0_0 import (
    StableDiffusionBrushNetPipeline,
)
from diffusers.models.brushnet import BrushNetModel
from diffusers.models.brushnet_motion_adapter import BrushNetFlowMotionAdapter
from diffusers.models.stc_noise_shaper import STCConditionedNoiseShaper
from diffusers.schedulers.scheduling_ddim_temporal import (
    TemporalDDIMScheduler,
    backward_warp,
    build_stable_bg_mask,
)

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from ip_adapter import FusionIPAdapter

PROPAINTER_ROOT = os.environ.get(
    "PROPAINTER_ROOT",
    "/home/cilab/ndquan/videoInpainting/pretrained/ProPainter",
)
PROPAINTER_FLOW_CKPT = os.environ.get(
    "PROPAINTER_FLOW_CKPT",
    (
        "/home/cilab/ndquan/NAS_ndq/model_base/Checkpoint/"
        "vcm_flowcomp_full_t10_b4/best.pt"
    ),
)
PROPAINTER_RAFT_CKPT = os.environ.get(
    "PROPAINTER_RAFT_CKPT",
    os.path.join(PROPAINTER_ROOT, "weights", "raft-things.pth"),
)

##################### 디퓨전 값 고정하기 위해서
import random

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# Deterministic behavior
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)
# torch.set_deterministic(True)  # 버전에 따라 권장되지 않음
torch.backends.cudnn.benchmark = False
# CUDA grid_sample backward is not deterministic, but it is required by
# temporal warping guidance. Keep deterministic checks as warnings.
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
device = "cuda"

# 설정
# image_dir = "/home/gpu_01/nas_naeun/data/data/test_in_COCO" # coco
# image_dir='/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/test/PartyScene_512/images' # open
# image_dir='/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/test/PartyScene_512_backup/images' # open
# image_dir='/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/test/RaceHorses_512_backup/images' # open  
# image_dir='/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/test/PartyScene_512_backup/inputs' # open  
# image_dir='/media/ssd1/ndquan/videoInpainting/code/BrushNet/examples/brushnet/dataset/test/Traffic/inputs'
image_dir = os.environ.get(
    "IMAGE_DIR",
    "/media/ssd1/ndquan/videoInpainting/code/BrushNet/examples/brushnet/dataset/test/ParkScene/inputs",
)


# image_dir="/media/ssd2/naeun/NAS_NE/data/data/New/synthesis_COCO"

# mask_dir = "/home/gpu_01/nas_naeun/data/data/test_mask_COCO" # coco
# mask_dir='/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/test/PartyScene_512_backup/masks' # open
# mask_dir='/media/ssd1/ndquan/model_naeun/paper/BrushNet/examples/brushnet/dataset/test/RaceHorses_512_backup/masks' # open
mask_dir = os.environ.get(
    "MASK_DIR",
    "/media/ssd1/ndquan/videoInpainting/code/BrushNet/examples/brushnet/dataset/test/ParkScene/masks",
)

# output_dir = "/media/hdd/naeun/save/BrushNet_200000"
# output_dir = "/media/hdd/naeun/save/test_with_originalcaption/Brushnet_200000"
# output_dir='/media/hdd/naeun/save/Opendataset/BrushNet_300000'
output_dir = os.environ.get(
    "OUTPUT_DIR",
    "/media/ssd1/ndquan/videoInpainting/code/BrushNet/Quan_test/results/gen_image/ParkScene/finetuning_ckpt3500/sharedNoise_fixedBG_1_temporal_propainter_v0_L2")

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# base_model_path = "lambdalabs/miniSD-diffusers"
base_model_path="stable-diffusion-v1-5/stable-diffusion-v1-5" #512
# brushnet_path = "/media/hdd/naeun/save/checkpoint/Checkpoint_brushNet_200000"
# brushnet_path="/media/ssd2/naeun/ws04/BrushNet_previous/examples/brushnet/pretrained_brushnet/brushnet"
# brushnet_path="/media/ssd2/naeun/NAS_NE/checkpoint/Checkpoint_brushnet_512/checkpoint-600000/brushnet"
# brushnet_path="/home/gpu_01/naeun/v8/checkpoint-300000/brushnet"


# brushnet_path="/media/ssd1/ndquan/videoInpainting/code/BrushNet/examples/checkpoint_naeun/checkpoint-200000/brushnet"

checkpoint_dir = os.environ.get(
    "BRUSHNET_CHECKPOINT_DIR",
    (
        "/home/cilab/ndquan/NAS_ndq/model_base/Checkpoint/"
        "fine-tuning/checkpoint-3500"
    ),
)

brushnet_path = f"{checkpoint_dir}/brushnet"


# 블렌딩 설정 바꾸기
blended = True


clip_size = int(os.environ.get("TEMPORAL_CLIP_SIZE", "4"))
num_inference_steps = int(os.environ.get("NUM_INFERENCE_STEPS", "50"))
main_noise_seed = int(os.environ.get("MAIN_NOISE_SEED", "1234"))
shared_noise_seed = int(os.environ.get("SHARED_NOISE_SEED", "6789"))
shared_bg_noise_strength = float(os.environ.get("SHARED_BG_NOISE_STRENGTH", "1"))
variance_preserving_shared_noise = (
    os.environ.get("VARIANCE_PRESERVING_SHARED_NOISE", "0").lower()
    in {"1", "true", "yes", "on"}
)

temporal_guidance_scale = float(os.environ.get("TEMPORAL_GUIDANCE_SCALE", "0.0001"))
temporal_start_step = int(os.environ.get("TEMPORAL_START_STEP", "15"))  
temporal_end_step = int(os.environ.get("TEMPORAL_END_STEP", "35"))      
temporal_every_n_steps = int(os.environ.get("TEMPORAL_EVERY_N_STEPS", "1"))
temporal_decode_chunk_size = int(os.environ.get("TEMPORAL_DECODE_CHUNK_SIZE", "1"))
temporal_loss_scale = float(os.environ.get("TEMPORAL_LOSS_SCALE", "1024"))
temporal_detach_previous = (
    os.environ.get("TEMPORAL_DETACH_PREVIOUS", "1").lower()
    in {"1", "true", "yes", "on"}       ### tested: True (v0_4)
)
flow_batch_size = int(os.environ.get("TEMPORAL_FLOW_BATCH_SIZE", "2"))
force_regenerate = os.environ.get("FORCE_REGENERATE", "1").lower() in {"1", "true", "yes", "on"}
use_propainter_flow_completion = (
    os.environ.get("USE_PROPAINTER_FLOW_COMPLETION", "1").lower()
    in {"1", "true", "yes", "on"}
)
propainter_mask_mode = os.environ.get("PROPAINTER_FLOW_MASK_MODE", "unreliable_bg").lower()
propainter_dilation_size = int(os.environ.get("PROPAINTER_DILATION_SIZE", "9"))
max_test_clips = int(os.environ.get("MAX_TEST_CLIPS", "1"))
motion_adapter_path = os.environ.get("MOTION_ADAPTER_PATH", "")
motion_adapter_scale = float(os.environ.get("MOTION_ADAPTER_SCALE", "1.0"))
use_motion_adapter = bool(motion_adapter_path)
noise_shaper_path = os.environ.get("STC_NOISE_SHAPER_PATH", "")
noise_shaper_strength = float(os.environ.get("STC_NOISE_SHAPER_STRENGTH", "1.0"))
use_stc_noise_shaper = bool(noise_shaper_path)
noise_shaper_flow_mode = "none"
noise_only_mode = (
    os.environ.get("STC_NOISE_ONLY_MODE", "0").lower()
    in {"1", "true", "yes", "on"}
)
raft_backend = os.environ.get(
    "RAFT_BACKEND",
    "propainter" if noise_only_mode else "torchvision",
).lower()
if noise_only_mode:
    if not use_stc_noise_shaper:
        raise ValueError(
            "STC_NOISE_ONLY_MODE=1 requires STC_NOISE_SHAPER_PATH"
        )
    if use_motion_adapter:
        raise ValueError(
            "STC_NOISE_ONLY_MODE=1 requires an empty MOTION_ADAPTER_PATH"
        )
    use_propainter_flow_completion = False
    temporal_guidance_scale = 0.0

brushnet_conditioning_scale = 1.0

valid_propainter_modes = {
    "unreliable_bg",
    "bg",
    "roi",
    "dilated_roi",
}

if clip_size < 2:
    raise ValueError("TEMPORAL_CLIP_SIZE must be at least 2.")
if main_noise_seed == shared_noise_seed:
    raise ValueError("MAIN_NOISE_SEED and SHARED_NOISE_SEED must be different.")
if not 0.0 <= shared_bg_noise_strength <= 1.0:
    raise ValueError("SHARED_BG_NOISE_STRENGTH must be in [0, 1].")
if not 0.0 <= noise_shaper_strength <= 1.0:
    raise ValueError("STC_NOISE_SHAPER_STRENGTH must be in [0, 1].")
if raft_backend not in {"propainter", "torchvision"}:
    raise ValueError("RAFT_BACKEND must be 'propainter' or 'torchvision'")


if propainter_mask_mode not in valid_propainter_modes:
    raise ValueError(f"PROPAINTER_FLOW_MASK_MODE must be one of {valid_propainter_modes}.")
if propainter_dilation_size <= 0 or propainter_dilation_size % 2 == 0:
    raise ValueError("PROPAINTER_DILATION_SIZE must be a positive odd integer.")
if max_test_clips < 0:
    raise ValueError("MAX_TEST_CLIPS must be non-negative; use 0 to run all clips.")

# 모델 로드
brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)

# print("DEBUG brushnet =", brushnet, type(brushnet))
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False,safety_checker=None
)
if use_motion_adapter:
    motion_adapter = BrushNetFlowMotionAdapter.from_pretrained(
        motion_adapter_path,
        torch_dtype=torch.float16,
    )
    pipe.set_motion_adapter(motion_adapter)
    print(
        f"[Motion adapter] loaded={motion_adapter_path}, "
        f"scale={motion_adapter_scale}"
    )
if use_stc_noise_shaper:
    noise_shaper = STCConditionedNoiseShaper.from_pretrained(
        noise_shaper_path,
        torch_dtype=torch.float16,
    )
    noise_shaper_flow_mode = str(
        getattr(noise_shaper.config, "flow_prediction_mode", "legacy")
    ).lower()
    pipe.set_noise_shaper(noise_shaper)
    print(
        f"[STC noise shaper] loaded={noise_shaper_path}, "
        f"strength={noise_shaper_strength}, "
        f"flow_mode={noise_shaper_flow_mode}"
    )

image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
).to(pipe.device, dtype=pipe.dtype)

# ip_ckpt="/home/gpu_01/naeun/v8/checkpoint-300000/ipadapter/model.safetensors"
ip_ckpt = f"{checkpoint_dir}/ipadapter/model.safetensors"
pipe.scheduler = TemporalDDIMScheduler.from_config(pipe.scheduler.config)
if hasattr(pipe, "enable_vae_slicing"):
    pipe.enable_vae_slicing()
for parameter in pipe.vae.parameters():
    parameter.requires_grad_(False)
pipe.enable_model_cpu_offload()

pipe.register_modules(
    image_encoder=image_encoder,
    feature_extractor=CLIPImageProcessor(),
)
# fusion_ckpt = "/home/gpu_01/naeun/v8/checkpoint-300000/ipadapter/fusion_module.safetensors"
fusion_ckpt = f"{checkpoint_dir}/ipadapter/fusion_module.safetensors"

ip_model = FusionIPAdapter(
    pipe,
    image_encoder_path,
    ip_ckpt,
    fusion_ckpt,
    device,
)

# Full-flow STC predicts motion from degraded structural latents and does not
# require RAFT. Keep RAFT only for temporal guidance, the motion adapter, or a
# legacy STC checkpoint that consumes an external flow prior.
requires_external_flow_model = (
    temporal_guidance_scale > 0
    or use_motion_adapter
    or (use_stc_noise_shaper and noise_shaper_flow_mode != "full")
)
flow_model = None
flow_transform = None
if requires_external_flow_model:
    # RAFT stays on CPU between clips and moves to GPU only for flow inference.
    if raft_backend == "propainter":
        if PROPAINTER_ROOT not in sys.path:
            sys.path.insert(0, PROPAINTER_ROOT)
        from model.modules.flow_comp_raft import initialize_RAFT

        if not os.path.isfile(PROPAINTER_RAFT_CKPT):
            raise FileNotFoundError(
                f"Missing bundled RAFT checkpoint: {PROPAINTER_RAFT_CKPT}"
            )
        flow_model = initialize_RAFT(PROPAINTER_RAFT_CKPT, device="cpu").eval()
        print(
            f"[Raw flow] backend=ProPainter RAFT "
            f"checkpoint={PROPAINTER_RAFT_CKPT}"
        )
    else:
        flow_weights = Raft_Large_Weights.DEFAULT
        flow_transform = flow_weights.transforms()
        flow_model = raft_large(weights=flow_weights, progress=True).eval()
        print("[Raw flow] backend=torchvision RAFT Large DEFAULT")
    for parameter in flow_model.parameters():
        parameter.requires_grad_(False)
else:
    print("[Raw flow] skipped: STC full-flow prediction is self-contained")
    use_propainter_flow_completion = False

# Keep the flow-completion model on CPU between clips. RAFT is also offloaded
# before this model is moved to GPU, so both models do not occupy VRAM together.
propainter_flow_complete = None
if use_propainter_flow_completion:
    if PROPAINTER_ROOT not in sys.path:
        sys.path.insert(0, PROPAINTER_ROOT)
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet

    if not os.path.isfile(PROPAINTER_FLOW_CKPT):
        raise FileNotFoundError(
            f"Missing ProPainter flow-completion checkpoint: {PROPAINTER_FLOW_CKPT}"
        )
    try:
        flow_checkpoint = torch.load(
            PROPAINTER_FLOW_CKPT,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        flow_checkpoint = torch.load(PROPAINTER_FLOW_CKPT, map_location="cpu")
    if isinstance(flow_checkpoint, dict) and "model" in flow_checkpoint:
        flow_checkpoint = flow_checkpoint["model"]
    propainter_flow_complete = RecurrentFlowCompleteNet()
    propainter_flow_complete.load_state_dict(flow_checkpoint, strict=True)
    propainter_flow_complete.eval()
    print(f"[Flow completion] loaded fine-tuned checkpoint={PROPAINTER_FLOW_CKPT}")
    for parameter in propainter_flow_complete.parameters():
        parameter.requires_grad_(False)

# 이미지 & 마스크 경로
image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))

assert len(mask_paths) == len(image_paths), "이미지 수와 마스크 수가 일치하지 않습니다."


existing_basenames = {os.path.basename(p) for p in glob(os.path.join(output_dir, "*.png"))}
if force_regenerate:
    existing_basenames = set()

indexed_all = list(enumerate(zip(image_paths, mask_paths)))

indexed_pending = [
    (idx, img, msk)
    for idx, (img, msk) in indexed_all
    if os.path.basename(img) not in existing_basenames
]

print(f"[Resume] 총 {len(indexed_all)}개 중 이미 {len(indexed_all) - len(indexed_pending)}개 완료, "
      f"{len(indexed_pending)}개 생성 예정.")

resize_transform = transforms.Compose([
    transforms.Resize((512, 512)),
])
flow_image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
mask_transform = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0.5).float()),
])


def prepare_frame_inputs(image_path, mask_path):
    init_image_np = cv2.imread(image_path)[:, :, ::-1]
    mask_np = 1.0 * (cv2.imread(mask_path).sum(-1) > 255)[:, :, np.newaxis]

    init_image = Image.fromarray(init_image_np.astype(np.uint8)).convert("RGB")
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8).repeat(3, -1)).convert("RGB")

    fg_np = init_image_np * mask_np
    bg_np = init_image_np * (1 - mask_np)
    fg_pil = Image.fromarray(fg_np.astype(np.uint8)).convert("RGB")
    bg_pil = Image.fromarray(bg_np.astype(np.uint8)).convert("RGB")

    init_image = resize_transform(init_image)
    mask_image = resize_transform(mask_image)
    fg_pil = resize_transform(fg_pil)
    bg_pil = resize_transform(bg_pil)

    flow_frame_tensor = flow_image_transform(init_image).unsqueeze(0)
    roi_mask_tensor = mask_transform(mask_image.convert("L")).unsqueeze(0)

    return (
        fg_pil,
        bg_pil,
        init_image,
        mask_image,
        flow_frame_tensor,
        roi_mask_tensor,
    )


@torch.no_grad()
def estimate_bidirectional_flow(frames):
    """Estimate adjacent forward/backward flow for a full RGB clip."""
    pair_count = frames.shape[0] - 1
    if pair_count <= 0:
        raise ValueError("At least two frames are required for temporal flow.")

    flow_model.to(device)
    forward_flows = []
    backward_flows = []

    try:
        for start in range(0, pair_count, flow_batch_size):
            end = min(start + flow_batch_size, pair_count)
            previous = frames[start:end]
            current = frames[start + 1 : end + 1]
            if raft_backend == "propainter":
                previous_input = F.interpolate(
                    previous.float(),
                    size=(240, 432),
                    mode="bilinear",
                    align_corners=False,
                )
                current_input = F.interpolate(
                    current.float(),
                    size=(240, 432),
                    mode="bilinear",
                    align_corners=False,
                )
                previous_input = (previous_input * 2.0 - 1.0).to(device)
                current_input = (current_input * 2.0 - 1.0).to(device)
                forward_flows.append(
                    flow_model(
                        previous_input,
                        current_input,
                        iters=20,
                        test_mode=True,
                    )[1]
                )
                backward_flows.append(
                    flow_model(
                        current_input,
                        previous_input,
                        iters=20,
                        test_mode=True,
                    )[1]
                )
            else:
                previous_input, current_input = flow_transform(previous, current)
                previous_input = previous_input.to(
                    device=device, dtype=torch.float32
                )
                current_input = current_input.to(
                    device=device, dtype=torch.float32
                )
                forward_flows.append(
                    flow_model(previous_input, current_input)[-1]
                )
                backward_flows.append(
                    flow_model(current_input, previous_input)[-1]
                )

            del previous_input, current_input
    finally:
        flow_model.to("cpu")

    flow_forward = torch.cat(forward_flows, dim=0)
    flow_backward = torch.cat(backward_flows, dim=0)
    torch.cuda.empty_cache()
    return flow_forward, flow_backward


@torch.no_grad()
def forward_backward_visibility(
    flow_forward,
    flow_backward,
    alpha=0.01,
    beta=0.5,
):
    """Use flow-cycle consistency to reject occluded or unreliable pixels."""
    warped_forward, in_bounds = backward_warp(flow_forward, flow_backward)
    cycle_error = (flow_backward + warped_forward).square().sum(dim=1, keepdim=True)
    flow_magnitude = (
        flow_backward.square().sum(dim=1, keepdim=True)
        + warped_forward.square().sum(dim=1, keepdim=True)
    )
    consistent = cycle_error <= alpha * flow_magnitude + beta
    return consistent.to(dtype=flow_backward.dtype) * in_bounds


@torch.no_grad()
def complete_flow_with_propainter(flow_forward, flow_backward, flow_masks):
    """Complete masked bidirectional flow with the pretrained ProPainter model."""
    if propainter_flow_complete is None:
        raise RuntimeError("ProPainter flow completion is enabled but its model is not loaded.")
    if flow_forward.ndim != 4 or flow_forward.shape[1] != 2:
        raise ValueError(
            f"flow_forward must have shape [T-1, 2, H, W], got {tuple(flow_forward.shape)}."
        )
    if flow_backward.shape != flow_forward.shape:
        raise ValueError(
            "Forward and backward flow shapes must match, "
            f"got {tuple(flow_forward.shape)} and {tuple(flow_backward.shape)}."
        )
    if flow_masks.ndim != 4 or flow_masks.shape[1] != 1:
        raise ValueError(
            f"flow_masks must have shape [T, 1, H, W], got {tuple(flow_masks.shape)}."
        )
    if flow_masks.shape[0] != flow_forward.shape[0] + 1:
        raise ValueError(
            "Flow-completion masks must contain one mask per frame, "
            f"got {flow_masks.shape[0]} masks and {flow_forward.shape[0]} flow pairs."
        )

    model = propainter_flow_complete.to(device=device, dtype=torch.float32)
    flow_forward = flow_forward.to(device=device, dtype=torch.float32)
    flow_backward = flow_backward.to(device=device, dtype=torch.float32)
    flow_masks = flow_masks.to(device=device, dtype=torch.float32)

    if flow_masks.shape[-2:] != flow_forward.shape[-2:]:
        flow_masks = F.interpolate(
            flow_masks,
            size=flow_forward.shape[-2:],
            mode="nearest",
        )

    flows_bi = (
        flow_forward.unsqueeze(0),
        flow_backward.unsqueeze(0),
    )
    masks = flow_masks.unsqueeze(0)

    try:
        predicted_flows_bi, _ = model.forward_bidirect_flow(flows_bi, masks)
        completed_flows_bi = model.combine_flow(flows_bi, predicted_flows_bi, masks)
        completed_forward = completed_flows_bi[0].squeeze(0).detach()
        completed_backward = completed_flows_bi[1].squeeze(0).detach()
    finally:
        model.to("cpu")

    forward_mask = flow_masks[:-1]
    backward_mask = flow_masks[1:]
    forward_denominator = (forward_mask.sum() * flow_forward.shape[1]).clamp_min(1.0)
    backward_denominator = (backward_mask.sum() * flow_backward.shape[1]).clamp_min(1.0)
    stats = {
        "completion_mask_ratio": float(flow_masks.mean().cpu()),
        "forward_delta": float(
            ((completed_forward - flow_forward).abs() * forward_mask).sum().cpu()
            / forward_denominator.cpu()
        ),
        "backward_delta": float(
            ((completed_backward - flow_backward).abs() * backward_mask).sum().cpu()
            / backward_denominator.cpu()
        ),
    }

    del predicted_flows_bi, completed_flows_bi, flows_bi, masks
    torch.cuda.empty_cache()
    return completed_forward, completed_backward, stats


@torch.no_grad()
def prepare_temporal_conditions(flow_frames, roi_masks):
    """Create optional completed flow and stable-BG masks for guidance."""
    # flow_forward, flow_backward = estimate_bidirectional_flow(flow_frames)
    # bg_masks = (1.0 - roi_masks).to(device=device, dtype=torch.float32)

    # completion_stats = None
    # if use_propainter_flow_completion:
    #     flow_completion_masks = roi_masks.to(device=device, dtype=torch.float32)
    #     if propainter_mask_mode == "dilated_roi":
    #         flow_completion_masks = F.max_pool2d(
    #             flow_completion_masks,
    #             kernel_size=propainter_dilation_size,
    #             stride=1,
    #             padding=propainter_dilation_size // 2,
    #         ).clamp(0.0, 1.0)

    #     flow_forward, flow_backward, completion_stats = complete_flow_with_propainter(
    #         flow_forward=flow_forward,
    #         flow_backward=flow_backward,
    #         flow_masks=flow_completion_masks,
    #     )

    

    flow_forward, flow_backward = estimate_bidirectional_flow(flow_frames)

    roi_masks = roi_masks.to(device=device, dtype=torch.float32)
    bg_masks = 1.0 - roi_masks

    completion_stats = None
    if use_propainter_flow_completion:
        raw_visibility = forward_backward_visibility(
            flow_forward,
            flow_backward,
        )

        if propainter_mask_mode == "unreliable_bg":
            unreliable_pairs = 1.0 - raw_visibility

            unreliable_frames = torch.zeros_like(bg_masks)
            unreliable_frames[:-1] = torch.maximum(
                unreliable_frames[:-1],
                unreliable_pairs,
            )
            unreliable_frames[1:] = torch.maximum(
                unreliable_frames[1:],
                unreliable_pairs,
            )

            flow_completion_masks = bg_masks * unreliable_frames

        elif propainter_mask_mode == "bg":
            flow_completion_masks = bg_masks

        elif propainter_mask_mode == "dilated_roi":
            flow_completion_masks = F.max_pool2d(
                roi_masks,
                kernel_size=propainter_dilation_size,
                stride=1,
                padding=propainter_dilation_size // 2,
            ).clamp(0.0, 1.0)

        else:
            flow_completion_masks = roi_masks

        flow_forward, flow_backward, completion_stats = (
            complete_flow_with_propainter(
                flow_forward=flow_forward,
                flow_backward=flow_backward,
                flow_masks=flow_completion_masks,
            )
        )

# Recompute reliability after co



    visibility = forward_backward_visibility(flow_forward, flow_backward)
    stable_bg = build_stable_bg_mask(
        bg_masks=bg_masks,
        flow_backward=flow_backward,
        visibility=visibility,
        threshold=0.5,
    )

    del flow_forward
    return flow_backward, visibility, stable_bg, bg_masks, completion_stats


def blend_with_input(image, image_path, mask_path):
    image_np = np.array(image)
    init_image_np = cv2.imread(image_path)[:, :, ::-1]
    mask_np = 1.0 * (cv2.imread(mask_path).sum(-1) > 255)[:, :, np.newaxis]

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
    return Image.fromarray(image_pasted.astype(np.uint8))


latent_shape = (
    pipe.unet.config.in_channels,
    512 // pipe.vae_scale_factor,
    512 // pipe.vae_scale_factor,
)

shared_generator = torch.Generator(device="cpu").manual_seed(shared_noise_seed)
shared_bg_noise = torch.randn(
    (1, *latent_shape),
    generator=shared_generator,
    device="cpu",
    dtype=torch.float32,
)

clips = [indexed_all[i : i + clip_size] for i in range(0, len(indexed_all), clip_size)]
clips_to_run = [
    (clip_idx, clip)
    for clip_idx, clip in enumerate(clips)
    if any(os.path.basename(image_path) not in existing_basenames for _, (image_path, _) in clip)
]
if max_test_clips > 0:
    clips_to_run = clips_to_run[:max_test_clips]

print(
    f"[Temporal BrushNet] clip_size={clip_size}, "
    f"clips_to_run={len(clips_to_run)}, shared_strength={shared_bg_noise_strength}, "
    f"variance_preserving={variance_preserving_shared_noise}, "
    f"flow_completion={use_propainter_flow_completion}, "
    f"raft_backend={raft_backend}, noise_only={noise_only_mode}, "
    f"flow_mask_mode={propainter_mask_mode}, "
    f"motion_adapter={use_motion_adapter}, motion_scale={motion_adapter_scale}, "
    f"temporal_window=[{temporal_start_step}, {temporal_end_step}), "
    f"temporal_scale={temporal_guidance_scale}, every_n_steps={temporal_every_n_steps}"
)

for clip_idx, clip in tqdm(clips_to_run, total=len(clips_to_run)):
    fg_list = []
    bg_list = []
    init_list = []
    mask_list = []
    prompt_list = []
    flow_frame_tensors = []
    roi_mask_tensors = []

    # for _, (image_path, mask_path, caption) in clip:
    for _, (image_path, mask_path) in clip:
        (
            fg_pil,
            bg_pil,
            init_image,
            mask_image,
            flow_frame_tensor,
            roi_mask_tensor,
        ) = prepare_frame_inputs(image_path, mask_path)
        fg_list.append(fg_pil)
        bg_list.append(bg_pil)
        init_list.append(init_image)
        mask_list.append(mask_image)
        # prompt_list.append(caption)
        prompt_list.append("")
        flow_frame_tensors.append(flow_frame_tensor)
        roi_mask_tensors.append(roi_mask_tensor)

    flow_frames = torch.cat(flow_frame_tensors, dim=0)
    roi_masks = torch.cat(roi_mask_tensors, dim=0)

    clip_noise_generator = torch.Generator(device="cpu").manual_seed(main_noise_seed + clip_idx)
    clip_noise = torch.randn(
        (len(clip), *latent_shape),
        generator=clip_noise_generator,
        device="cpu",
        dtype=torch.float32,
    )

    clip_use_motion_adapter = use_motion_adapter and len(clip) > 1
    clip_use_stc_noise_shaper = use_stc_noise_shaper and len(clip) > 1
    needs_motion_conditions = (
        len(clip) > 1
        and (
            temporal_guidance_scale > 0
            or clip_use_motion_adapter
            or (
                clip_use_stc_noise_shaper
                and noise_shaper_flow_mode != "full"
            )
        )
    )
    if needs_motion_conditions:
        (
            flow_backward,
            motion_confidence,
            stable_bg,
            bg_masks,
            completion_stats,
        ) = prepare_temporal_conditions(
                flow_frames=flow_frames,
                roi_masks=roi_masks,
            )
    if len(clip) > 1 and temporal_guidance_scale > 0:
        pipe.scheduler.set_temporal_guidance(
            decoder=pipe.vae.decode,
            flow_backward=flow_backward,
            stable_bg=stable_bg,
            # bg_masks=bg_masks,   # not used in the current implementation
            guidance_scale=temporal_guidance_scale,
            start_step=temporal_start_step,
            end_step=temporal_end_step,
            every_n_steps=temporal_every_n_steps,
            decode_chunk_size=temporal_decode_chunk_size,
            vae_scaling_factor=pipe.vae.config.scaling_factor,
            loss_scale=temporal_loss_scale,
            detach_previous=temporal_detach_previous,
            loss_type="l2",
            enabled=True,
        )
        stable_bg_ratio = float(stable_bg.mean().detach().cpu())
        print(
            f"[Clip {clip_idx}] temporal pairs={len(clip) - 1}, "
            f"stable_bg_ratio={stable_bg_ratio:.4f}"
        )
        if completion_stats is not None:
            print(
                f"[Clip {clip_idx}] ProPainter completion: "
                f"mask_ratio={completion_stats['completion_mask_ratio']:.4f}, "
                f"forward_delta={completion_stats['forward_delta']:.6f}, "
                f"backward_delta={completion_stats['backward_delta']:.6f}"
            )
    else:
        pipe.scheduler.clear_temporal_guidance()

    sampling_generator = torch.Generator(device=device).manual_seed(main_noise_seed + clip_idx)
    torch.manual_seed(main_noise_seed + clip_idx)
    torch.cuda.manual_seed_all(main_noise_seed + clip_idx)

    result = ip_model.generate_fgbg(
        fg_pil_image=fg_list,
        bg_pil_image=bg_list,
        prompt=prompt_list,
        image=init_list,
        mask_image=mask_list,
        num_samples=1,
        num_inference_steps=num_inference_steps,
        negative_prompt=[""] * len(prompt_list),
        generator=sampling_generator,
        latents=clip_noise.to(device=device, dtype=pipe.dtype),
        use_shared_bg_noise=not use_stc_noise_shaper,
        shared_bg_noise=(
            None
            if use_stc_noise_shaper
            else shared_bg_noise.to(device=device, dtype=pipe.dtype)
        ),
        shared_bg_noise_strength=shared_bg_noise_strength,
        variance_preserving_shared_noise=variance_preserving_shared_noise,
        brushnet_conditioning_scale=brushnet_conditioning_scale,
        motion_backward_flow=(
            flow_backward
            if clip_use_motion_adapter
            or (
                clip_use_stc_noise_shaper
                and noise_shaper_flow_mode != "full"
            )
            else None
        ),
        motion_confidence=(
            motion_confidence
            if clip_use_motion_adapter
            or (
                clip_use_stc_noise_shaper
                and noise_shaper_flow_mode != "full"
            )
            else None
        ),
        motion_stable_bg=(
            stable_bg
            if clip_use_stc_noise_shaper
            and noise_shaper_flow_mode != "full"
            else None
        ),
        motion_adapter_scale=motion_adapter_scale,
        use_stc_noise_shaper=clip_use_stc_noise_shaper,
        noise_shaper_strength=noise_shaper_strength,
        noise_shaper_clip_length=len(clip),
    )

    if clip_use_stc_noise_shaper:
        print(
            f"[Clip {clip_idx}] STC noise stats: "
            f"{pipe._last_noise_shaper_stats}"
        )

    if len(clip) > 1 and temporal_guidance_scale > 0:
        print(
            f"[Clip {clip_idx}] final temporal_loss={pipe.scheduler.last_temporal_loss}, "
            f"raw_grad_norm={pipe.scheduler.last_temporal_raw_grad_norm}, "
            f"masked_grad_norm={pipe.scheduler.last_temporal_masked_grad_norm}, "
            f"update_norm={pipe.scheduler.last_temporal_update_norm}, "
            f"active_frames={pipe.scheduler.last_temporal_active_frames}/{len(clip)}, "
            f"calls={pipe.scheduler.temporal_guidance_calls}, "
            f"applied={pipe.scheduler.temporal_guidance_applied_steps}, "
            f"skipped={pipe.scheduler.temporal_guidance_skipped_steps}, "
            f"skipped_reason={pipe.scheduler.last_temporal_skipped_reason}"
        )

    pipe.scheduler.clear_temporal_guidance()
    if needs_motion_conditions:
        del flow_backward, motion_confidence, stable_bg, bg_masks, completion_stats
    del flow_frames, roi_masks, flow_frame_tensors, roi_mask_tensors
    torch.cuda.empty_cache()

    for image, (orig_idx, (image_path, mask_path)) in zip(result, clip):
        basename = os.path.basename(image_path)
        if basename in existing_basenames:
            continue

        if blended:
            print(f"[{orig_idx}] blending...")
            image = blend_with_input(image, image_path, mask_path)

        image.save(os.path.join(output_dir, basename))
        existing_basenames.add(basename)
