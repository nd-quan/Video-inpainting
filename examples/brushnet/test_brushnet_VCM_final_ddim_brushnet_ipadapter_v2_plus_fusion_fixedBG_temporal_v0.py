# CUDA_VISIBLE_DEVICES=0 python /home/gpu_01/nas_naeun/BrushNet/examples/brushnet/test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion.py
# proposed 알고리즘임
# conda activate brushnet_ipadapter_new
# v1은 bin으로 불러오기(허깅페이스에서) v2는 train해서 저장한거 safetensor로 불러오기
########## 그 중에서 v2는 ip_adapter 폴더에서 불러오고 v3는 diffuser안에 구현된 load_ipadapter로
# v3는 v2가 잘 나와서 구현 안함

import json
import os
from pathlib import Path
from tqdm import tqdm

from sfu_long_test_loader import (
    DEFAULT_LONG_TEST_ROOT,
    load_sequences,
    parse_sequence_names,
    print_preflight,
)


long_test_root = Path(os.environ.get("LONG_TEST_ROOT", str(DEFAULT_LONG_TEST_ROOT)))
output_root = Path(
    os.environ.get(
        "OUTPUT_ROOT",
        "/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/"
        "eval_finetune_sharednoise/long_test_fixedBG_temporal_v0",
    )
)
selected_names = parse_sequence_names(os.environ.get("SEQUENCES"))
preflight_only = os.environ.get("PREFLIGHT_ONLY", "0").lower() in {
    "1", "true", "yes", "on"
}
sequences = load_sequences(long_test_root, output_root, selected_names)
print_preflight(sequences)
if preflight_only:
    raise SystemExit(0)

from diffusers.pipelines.brushnet.pipeline_brushnet_sharedNoise_sameBG_v0_0 import (
    StableDiffusionBrushNetPipeline,
)
from diffusers.models.brushnet import BrushNetModel
from diffusers.schedulers.scheduling_ddim_temporal import (
    TemporalDDIMScheduler,
    backward_warp,
    build_stable_bg_mask,
)
from diffusers.schedulers.scheduling_ddim_cge_temporal import (
    CGETemporalDDIMScheduler,
)
from diffusers.schedulers.scheduling_ddim_CGE import cond_fn as cge_cond_fn
from vcmrs_codec_adapter import VCMRSBackgroundOnlyCodec, VCMRSDualRegionCodec

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from ip_adapter import FusionIPAdapter

# The V7 student is imported lazily below.  The default temporal-v0 path must
# remain usable without its checkpoint or ProPainter dependency.

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

# base_model_path = "lambdalabs/miniSD-diffusers"
base_model_path = os.environ.get(
    "BASE_MODEL_PATH",
    "/home/cilab/ndquan/videoInpainting/code/BrushNet/examples/brushnet/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5",
)


checkpoint_dir = os.environ.get(
    "CHECKPOINT_DIR",
    "/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/"
    "train_sharedNoise_sameBG_0.95_T8/checkpoint-2250",
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
temporal_start_step = int(os.environ.get("TEMPORAL_START_STEP", "25"))  
temporal_end_step = int(os.environ.get("TEMPORAL_END_STEP", "35"))      
temporal_every_n_steps = int(os.environ.get("TEMPORAL_EVERY_N_STEPS", "1"))
temporal_decode_chunk_size = int(os.environ.get("TEMPORAL_DECODE_CHUNK_SIZE", "1"))
temporal_loss_scale = float(os.environ.get("TEMPORAL_LOSS_SCALE", "1024"))
temporal_guidance_space = os.environ.get("TEMPORAL_GUIDANCE_SPACE", "rgb").lower()
temporal_detach_previous = (
    os.environ.get("TEMPORAL_DETACH_PREVIOUS", "1").lower()
    in {"1", "true", "yes", "on"}       ### tested: True (v0_4)
)
flow_batch_size = int(os.environ.get("TEMPORAL_FLOW_BATCH_SIZE", "2"))
temporal_flow_backend = os.environ.get("TEMPORAL_FLOW_BACKEND", "torchvision").lower()
v7_raft_student_path = os.environ.get("V7_RAFT_STUDENT_PATH", "").strip()
temporal_bg_mask_mode = os.environ.get(
    "TEMPORAL_BG_MASK_MODE", "full_stable"
).lower()
guidance_mode = os.environ.get("GUIDANCE_MODE", "temporal").lower()
cge_guidance_scale = float(os.environ.get("CGE_GUIDANCE_SCALE", "0.0001"))
cge_start_step = int(os.environ.get("CGE_START_STEP", "25"))
_cge_end_raw = os.environ.get("CGE_END_STEP", "35").strip().lower()
cge_end_step = None if _cge_end_raw in {"", "none"} else int(_cge_end_raw)
cge_every_n_steps = int(os.environ.get("CGE_EVERY_N_STEPS", "1"))
cge_max_evals = int(os.environ.get("CGE_MAX_EVALS", "-1"))
cge_codec_mode = os.environ.get("CGE_CODEC_MODE", "bg_only").lower()
cge_per_frame = os.environ.get("CGE_PER_FRAME", "0").lower() in {
    "1", "true", "yes", "on"
}
force_regenerate = os.environ.get("FORCE_REGENERATE", "0").lower() in {"1", "true", "yes", "on"}

temporal_mode_enabled = guidance_mode in {"temporal", "combined"}
cge_mode_enabled = guidance_mode in {"cge", "combined"}

brushnet_conditioning_scale = 1.0

if clip_size < 2:
    raise ValueError("TEMPORAL_CLIP_SIZE must be at least 2.")
if main_noise_seed == shared_noise_seed:
    raise ValueError("MAIN_NOISE_SEED and SHARED_NOISE_SEED must be different.")
if not 0.0 <= shared_bg_noise_strength <= 1.0:
    raise ValueError("SHARED_BG_NOISE_STRENGTH must be in [0, 1].")
if temporal_every_n_steps <= 0:
    raise ValueError("TEMPORAL_EVERY_N_STEPS must be positive.")
if flow_batch_size <= 0:
    raise ValueError("TEMPORAL_FLOW_BATCH_SIZE must be positive.")
if temporal_flow_backend not in {"torchvision", "v7_student"}:
    raise ValueError("TEMPORAL_FLOW_BACKEND must be 'torchvision' or 'v7_student'.")
if temporal_mode_enabled and temporal_flow_backend == "v7_student" and not v7_raft_student_path:
    raise ValueError(
        "TEMPORAL_FLOW_BACKEND=v7_student requires V7_RAFT_STUDENT_PATH."
    )
if temporal_bg_mask_mode not in {"full_stable", "pair_intersection"}:
    raise ValueError(
        "TEMPORAL_BG_MASK_MODE must be 'full_stable' or 'pair_intersection'."
    )
if temporal_guidance_space not in {"rgb", "latent"}:
    raise ValueError("TEMPORAL_GUIDANCE_SPACE must be 'rgb' or 'latent'.")
if guidance_mode not in {"temporal", "cge", "combined"}:
    raise ValueError("GUIDANCE_MODE must be 'temporal', 'cge', or 'combined'.")
if cge_guidance_scale < 0.0:
    raise ValueError("CGE_GUIDANCE_SCALE must be non-negative.")
if cge_start_step < 0 or (cge_end_step is not None and cge_end_step <= cge_start_step):
    raise ValueError("Invalid [CGE_START_STEP, CGE_END_STEP) window.")
if cge_every_n_steps < 1:
    raise ValueError("CGE_EVERY_N_STEPS must be positive.")
if cge_codec_mode not in {"bg_only", "dual_region"}:
    raise ValueError("CGE_CODEC_MODE must be 'bg_only' or 'dual_region'.")
if guidance_mode == "combined" and temporal_guidance_space != "latent":
    raise ValueError(
        "GUIDANCE_MODE=combined currently requires "
        "TEMPORAL_GUIDANCE_SPACE=latent to avoid a second VAE decode."
    )
for model_dir_name, model_dir_path in (
    ("BASE_MODEL_PATH", base_model_path),
    ("CHECKPOINT_DIR", checkpoint_dir),
):
    if not os.path.isdir(model_dir_path):
        raise FileNotFoundError(f"{model_dir_name} does not exist: {model_dir_path}")

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
ip_ckpt = f"{checkpoint_dir}/ipadapter/model.safetensors"
pipe.scheduler = (
    CGETemporalDDIMScheduler.from_config(pipe.scheduler.config)
    if cge_mode_enabled
    else TemporalDDIMScheduler.from_config(pipe.scheduler.config)
)
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

# The legacy RAFT-Large stays on CPU between clips.  The V7 student follows
# the same offload policy so it does not reserve memory during diffusion.
flow_model = None
flow_transform = None
v7_flow_provider = None
if not temporal_mode_enabled:
    print("[Temporal flow] disabled because GUIDANCE_MODE=cge")
elif temporal_flow_backend == "torchvision":
    flow_weights = Raft_Large_Weights.DEFAULT
    flow_transform = flow_weights.transforms()
    flow_model = raft_large(weights=flow_weights, progress=True).eval()
    for parameter in flow_model.parameters():
        parameter.requires_grad_(False)
else:
    from STC_encoder_v8_raft_deformable.raft_flow_provider import (
        FrozenV7RAFTFlowProvider,
        resolve_raft_student_component,
    )

    v7_raft_student_path = str(resolve_raft_student_component(v7_raft_student_path))
    v7_flow_provider = FrozenV7RAFTFlowProvider(
        v7_raft_student_path,
        device=device,
        pair_batch_size=flow_batch_size,
        mixed_precision=True,
    )
    # ``FrozenV7RAFTFlowProvider`` loads on CUDA by design for V8.  Move its
    # frozen student back to CPU until a temporal clip actually needs flow.
    v7_flow_provider.student.to("cpu")
    print(
        "[Temporal flow] backend=v7_student, "
        f"checkpoint={v7_raft_student_path}, pair_batch_size={flow_batch_size}"
    )

cge_codec = None
if cge_mode_enabled:
    cge_codec = (
        VCMRSBackgroundOnlyCodec.from_env()
        if cge_codec_mode == "bg_only"
        else VCMRSDualRegionCodec.from_env()
    )
    pipe.scheduler.cge_codec = cge_codec
    pipe.scheduler.guidance_scale_cge = cge_guidance_scale
    pipe.scheduler.cge_start_step = cge_start_step
    pipe.scheduler.cge_end_step = cge_end_step
    pipe.scheduler.cge_every_n_steps = cge_every_n_steps
    pipe.scheduler.cge_max_evals = cge_max_evals
    pipe.scheduler.per_frame_cge = cge_per_frame
    pipe.scheduler.decode_chunk_size = temporal_decode_chunk_size
    pipe.scheduler.vae_scaling_factor = float(pipe.vae.config.scaling_factor)
    pipe.scheduler.direct_cge_guidance = True
    print(
        "[CGE] enabled, "
        f"codec={cge_codec_mode}, scale={cge_guidance_scale}, "
        f"window=[{cge_start_step}, {cge_end_step}), every={cge_every_n_steps}, "
        f"per_frame={cge_per_frame}"
    )

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
    """Estimate adjacent flow from the configured degraded-RGB flow backend.

    Both backends return RGB-pixel flow in the scheduler convention:
    forward is defined on frame t and samples t+1; backward is defined on
    frame t+1 and samples t.  ``frames`` contains the already degraded input
    RGB clip in [0,1], never clean GT or generated frames.
    """
    pair_count = frames.shape[0] - 1
    if pair_count <= 0:
        raise ValueError("At least two frames are required for temporal flow.")

    if temporal_flow_backend == "v7_student":
        if v7_flow_provider is None:
            raise RuntimeError("V7 temporal flow provider was not initialized.")
        # V7 was trained on degraded RGB normalized to [-1,1] and expects
        # [B,T,3,H,W].  Keep it on GPU only for this pre-sampling flow pass.
        v7_flow_provider.student.to(device)
        try:
            prediction = v7_flow_provider.predict_sequence(
                frames.to(device=device, dtype=torch.float32).mul(2.0).sub(1.0).unsqueeze(0)
            )
            flow_forward = prediction.forward.squeeze(0)
            flow_backward = prediction.backward.squeeze(0)
        finally:
            v7_flow_provider.student.to("cpu")
        torch.cuda.empty_cache()
        return flow_forward, flow_backward

    if flow_model is None or flow_transform is None:
        raise RuntimeError("Torchvision temporal flow model was not initialized.")
    flow_model.to(device)
    forward_flows = []
    backward_flows = []

    try:
        for start in range(0, pair_count, flow_batch_size):
            end = min(start + flow_batch_size, pair_count)
            previous = frames[start:end]
            current = frames[start + 1 : end + 1]
            previous_input, current_input = flow_transform(previous, current)
            previous_input = previous_input.to(device=device, dtype=torch.float32)
            current_input = current_input.to(device=device, dtype=torch.float32)

            forward_flows.append(flow_model(previous_input, current_input)[-1])
            backward_flows.append(flow_model(current_input, previous_input)[-1])

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
def prepare_temporal_conditions(flow_frames, roi_masks):
    """Create the fixed backward-flow condition for one denoising clip.

    ``pair_intersection`` is deliberately the light temporal-mask variant:
    both correspondence endpoints must be BG, and the source coordinate must
    be in bounds.  Unlike ``full_stable``, it does not require forward flow or
    a forward-backward visibility/cycle-consistency test.
    """
    flow_forward, flow_backward = estimate_bidirectional_flow(flow_frames)

    bg_masks = (1.0 - roi_masks).to(device=device, dtype=torch.float32)
    if temporal_bg_mask_mode == "pair_intersection":
        stable_bg = build_stable_bg_mask(
            bg_masks=bg_masks,
            flow_backward=flow_backward,
            visibility=None,
            threshold=0.5,
        )
        visibility = None
    else:
        visibility = forward_backward_visibility(flow_forward, flow_backward)
        stable_bg = build_stable_bg_mask(
            bg_masks=bg_masks,
            flow_backward=flow_backward,
            visibility=visibility,
            threshold=0.5,
        )

    del flow_forward
    if visibility is not None:
        del visibility
    return flow_backward, stable_bg, bg_masks


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

def run_sequence(sequence):
    output_dir = sequence.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [str(path) for path in sequence.image_paths]
    mask_paths = [str(path) for path in sequence.mask_paths]
    existing_basenames = {path.name for path in output_dir.glob("*.png")}
    if force_regenerate:
        existing_basenames = set()
    indexed_all = list(enumerate(zip(image_paths, mask_paths)))
    clips = [indexed_all[i : i + clip_size] for i in range(0, len(indexed_all), clip_size)]
    clips_to_run = [
        (clip_idx, clip)
        for clip_idx, clip in enumerate(clips)
        if any(os.path.basename(image_path) not in existing_basenames for _, (image_path, _) in clip)
    ]
    metadata = {
        "split": sequence.spec.split,
        "label": sequence.spec.label,
        "class": sequence.spec.class_name,
        "source_sequence": sequence.spec.source_name,
        "image_dir": str(sequence.image_dir),
        "mask_dir": str(sequence.mask_dir),
        "output_dir": str(output_dir),
        "frame_count": len(sequence.frame_ids),
        "frame_range": [sequence.frame_ids[0], sequence.frame_ids[-1]],
        "temporal_clip_size": clip_size,
        "temporal_flow_backend": temporal_flow_backend,
        "v7_raft_student_path": (
            v7_raft_student_path
            if temporal_mode_enabled and temporal_flow_backend == "v7_student"
            else None
        ),
        "temporal_bg_mask_mode": temporal_bg_mask_mode,
        "temporal_guidance_space": temporal_guidance_space,
        "guidance_mode": guidance_mode,
        "cge": None
        if not cge_mode_enabled
        else {
            "codec_mode": cge_codec_mode,
            "guidance_scale": cge_guidance_scale,
            "window": [cge_start_step, cge_end_step],
            "every_n_steps": cge_every_n_steps,
            "max_evals": cge_max_evals,
            "per_frame": cge_per_frame,
        },
    }
    (output_dir / "source_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[Fixed shared BG guidance={guidance_mode}:{sequence.spec.label}] "
        f"clip_size={clip_size}, clips_to_run={len(clips_to_run)}, "
        f"shared_strength={shared_bg_noise_strength}, "
        f"variance_preserving={variance_preserving_shared_noise}, "
        f"temporal_window=[{temporal_start_step}, {temporal_end_step}), "
        f"temporal_scale={temporal_guidance_scale}, every_n_steps={temporal_every_n_steps}, "
        f"flow_backend={temporal_flow_backend}, bg_mask={temporal_bg_mask_mode}, "
        f"guidance_space={temporal_guidance_space}, "
        f"cge_window=[{cge_start_step}, {cge_end_step})"
    )

    for clip_idx, clip in tqdm(clips_to_run, total=len(clips_to_run), desc=sequence.spec.label):
        fg_list, bg_list, init_list, mask_list, prompt_list = [], [], [], [], []
        flow_frame_tensors, roi_mask_tensors = [], []
        for _, (image_path, mask_path) in clip:
            prepared = prepare_frame_inputs(image_path, mask_path)
            fg_pil, bg_pil, init_image, mask_image, flow_frame_tensor, roi_mask_tensor = prepared
            fg_list.append(fg_pil)
            bg_list.append(bg_pil)
            init_list.append(init_image)
            mask_list.append(mask_image)
            prompt_list.append("")
            flow_frame_tensors.append(flow_frame_tensor)
            roi_mask_tensors.append(roi_mask_tensor)

        flow_frames = torch.cat(flow_frame_tensors, dim=0)
        roi_masks = torch.cat(roi_mask_tensors, dim=0)
        sequence_seed = main_noise_seed + clip_idx
        clip_noise_generator = torch.Generator(device="cpu").manual_seed(sequence_seed)
        clip_noise = torch.randn(
            (len(clip), *latent_shape), generator=clip_noise_generator,
            device="cpu", dtype=torch.float32,
        )
        temporal_enabled = (
            temporal_mode_enabled and len(clip) > 1 and temporal_guidance_scale > 0
        )
        cge_enabled = cge_mode_enabled and cge_guidance_scale > 0

        if cge_enabled:
            if cge_codec is None:
                raise RuntimeError("CGE was enabled but its VCM-RS codec is missing.")
            # CGE observes the exact degraded input/mask batch used by this
            # diffusion clip.  ROI is 1 here, matching cond_fn's contract.
            pipe.scheduler.x_lr = flow_frames.to(device=device, dtype=torch.float32).mul(2.0).sub(1.0)
            pipe.scheduler.mask = roi_masks.to(device=device, dtype=torch.float32)
            # With train-match descriptors, freeze/cache each frame's binary
            # mask before the finite-difference codec calls start.
            for local_index in range(len(clip)):
                cge_codec.prepare_region_mask(
                    pipe.scheduler.mask[local_index],
                    pipe.scheduler.x_lr[local_index],
                )
            pipe.scheduler.decoder = pipe.vae.decode
            pipe.scheduler.cond_fn = cge_cond_fn
            pipe.scheduler.cge_codec_eval_count = 0
            pipe.scheduler.cge_denoise_step_count = 0
        if temporal_enabled:
            flow_backward, stable_bg, bg_masks = prepare_temporal_conditions(flow_frames, roi_masks)
            if cge_enabled:
                pipe.scheduler.set_temporal_guidance(
                    flow_backward=flow_backward,
                    stable_bg=stable_bg,
                    guidance_scale=temporal_guidance_scale,
                    start_step=temporal_start_step,
                    end_step=temporal_end_step,
                    every_n_steps=temporal_every_n_steps,
                    detach_previous=temporal_detach_previous,
                    loss_type="l2",
                    enabled=True,
                )
            else:
                pipe.scheduler.set_temporal_guidance(
                    decoder=pipe.vae.decode, flow_backward=flow_backward, stable_bg=stable_bg,
                    guidance_scale=temporal_guidance_scale, start_step=temporal_start_step,
                    end_step=temporal_end_step, every_n_steps=temporal_every_n_steps,
                    decode_chunk_size=temporal_decode_chunk_size,
                    vae_scaling_factor=pipe.vae.config.scaling_factor,
                    loss_scale=temporal_loss_scale, detach_previous=temporal_detach_previous,
                    loss_type="l2", guidance_space=temporal_guidance_space, enabled=True,
                )
            print(
                f"[{sequence.spec.label} clip {clip_idx}] temporal pairs={len(clip) - 1}, "
                f"stable_bg_ratio={float(stable_bg.mean().detach().cpu()):.4f}"
            )
        elif temporal_mode_enabled:
            pipe.scheduler.clear_temporal_guidance()

        sampling_generator = torch.Generator(device=device).manual_seed(sequence_seed)
        torch.manual_seed(sequence_seed)
        torch.cuda.manual_seed_all(sequence_seed)
        result = ip_model.generate_fgbg(
            fg_pil_image=fg_list, bg_pil_image=bg_list, prompt=prompt_list,
            image=init_list, mask_image=mask_list, num_samples=1,
            num_inference_steps=num_inference_steps,
            negative_prompt=[""] * len(prompt_list), generator=sampling_generator,
            latents=clip_noise.to(device=device, dtype=pipe.dtype),
            use_shared_bg_noise=True,
            shared_bg_noise=shared_bg_noise.to(device=device, dtype=pipe.dtype),
            shared_bg_noise_strength=shared_bg_noise_strength,
            variance_preserving_shared_noise=variance_preserving_shared_noise,
            brushnet_conditioning_scale=brushnet_conditioning_scale,
        )
        if temporal_enabled:
            print(
                f"[{sequence.spec.label} clip {clip_idx}] "
                f"final temporal_loss={pipe.scheduler.last_temporal_loss}, "
                f"update_norm={pipe.scheduler.last_temporal_update_norm}, "
                f"calls={pipe.scheduler.temporal_guidance_calls}, "
                f"applied={pipe.scheduler.temporal_guidance_applied_steps}, "
                f"skipped_reason={pipe.scheduler.last_temporal_skipped_reason}"
            )
        pipe.scheduler.clear_temporal_guidance()
        if temporal_enabled:
            del flow_backward, stable_bg, bg_masks
        del flow_frames, roi_masks, flow_frame_tensors, roi_mask_tensors
        torch.cuda.empty_cache()

        for image, (orig_idx, (image_path, mask_path)) in zip(result, clip):
            basename = os.path.basename(image_path)
            if basename in existing_basenames:
                continue
            if blended:
                print(f"[{sequence.spec.label}:{orig_idx}] blending...")
                image = blend_with_input(image, image_path, mask_path)
            image.save(output_dir / basename)
            existing_basenames.add(basename)


for sequence in sequences:
    run_sequence(sequence)
