#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_rgb_stc_flow_shared_noise.py"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-${BRUSHNET_ROOT}/examples/brushnet/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5}"
DATASET_ROOT="${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
TEACHER_FLOW_ROOT="${TEACHER_FLOW_ROOT:-${DATASET_ROOT}/teacher_flows_512x512}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${BRUSHNET_ROOT}/experiments/train_sharedNoise_sameBG_0.9/checkpoint-2000}"
OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/train_rgb_stc_v3_flow_sharedNoise_0.9}"
IMAGE_ENCODER="${IMAGE_ENCODER:-laion/CLIP-ViT-H-14-laion2B-s32B-b79K}"
INIT_STC_ADAPTER="${INIT_STC_ADAPTER:-}"

TESTED_ENV_BIN="/home/cilab/ndquan/envs/guided_diff/bin"
if [[ -x "${TESTED_ENV_BIN}/python" && -x "${TESTED_ENV_BIN}/accelerate" ]]; then
    DEFAULT_PYTHON_BIN="${TESTED_ENV_BIN}/python"
    DEFAULT_ACCELERATE_BIN="${TESTED_ENV_BIN}/accelerate"
else
    DEFAULT_PYTHON_BIN="python"
    DEFAULT_ACCELERATE_BIN="accelerate"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${DEFAULT_ACCELERATE_BIN}}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

CLIP_LENGTH="${CLIP_LENGTH:-8}"
CLIP_STRIDE="${CLIP_STRIDE:-6}"
CLIPS_PER_DEVICE="${CLIPS_PER_DEVICE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-6}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-250}"
CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-5}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SHARED_BG_NOISE_STRENGTH="${SHARED_BG_NOISE_STRENGTH:-0.9}"
FLOW_LOSS_WEIGHT="${FLOW_LOSS_WEIGHT:-0.01}"
FLOW_REGION="${FLOW_REGION:-bg}"
FLOW_CHARBONNIER_EPS="${FLOW_CHARBONNIER_EPS:-0.001}"
FLOW_MAX_DX="${FLOW_MAX_DX:-8.0}"
FLOW_MAX_DY="${FLOW_MAX_DY:-8.0}"
SEED="${SEED:-1234}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
CONDITION_MODE="${CONDITION_MODE:-full_rgb_bg_mask}"
STC_INJECTION_SCALE="${STC_INJECTION_SCALE:-1.0}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_PROCESSES must be a positive integer: ${NUM_PROCESSES}" >&2
    exit 1
fi
if (( NUM_PROCESSES > 1 )) && ! command -v "${ACCELERATE_BIN}" >/dev/null 2>&1; then
    echo "Accelerate executable not found: ${ACCELERATE_BIN}" >&2
    exit 1
fi

REQUIRED_PATHS=(
    "${TRAIN_SCRIPT}"
    "${PRETRAINED_MODEL}"
    "${DATASET_ROOT}/train/GT"
    "${DATASET_ROOT}/train/input"
    "${DATASET_ROOT}/train/mask"
    "${TEACHER_FLOW_ROOT}/metadata.json"
    "${BASELINE_CHECKPOINT}/brushnet/config.json"
    "${BASELINE_CHECKPOINT}/brushnet/diffusion_pytorch_model.safetensors"
    "${BASELINE_CHECKPOINT}/ipadapter/model.safetensors"
    "${BASELINE_CHECKPOINT}/ipadapter/fusion_module.safetensors"
)
for required_path in "${REQUIRED_PATHS[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing required path: ${required_path}" >&2
        exit 1
    fi
done

COMMON_ARGS=(
    --pretrained_model_name_or_path "${PRETRAINED_MODEL}"
    --baseline_checkpoint "${BASELINE_CHECKPOINT}"
    --image_encoder_name_or_path "${IMAGE_ENCODER}"
    --dataset_root "${DATASET_ROOT}"
    --teacher_flow_root "${TEACHER_FLOW_ROOT}"
    --train_split train
    --output_dir "${OUTPUT_DIR}"
    --resolution 512
    --clip_length "${CLIP_LENGTH}"
    --clip_stride "${CLIP_STRIDE}"
    --shared_bg_noise_strength "${SHARED_BG_NOISE_STRENGTH}"
    --sequence_shared_noise_refresh epoch
    --condition_mode "${CONDITION_MODE}"
    --stc_injection_scale "${STC_INJECTION_SCALE}"
    --flow_loss_weight "${FLOW_LOSS_WEIGHT}"
    --flow_region "${FLOW_REGION}"
    --flow_charbonnier_eps "${FLOW_CHARBONNIER_EPS}"
    --flow_max_displacement "${FLOW_MAX_DX}" "${FLOW_MAX_DY}"
    --train_batch_size "${CLIPS_PER_DEVICE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --max_train_steps "${MAX_TRAIN_STEPS}"
    --checkpointing_steps "${CHECKPOINTING_STEPS}"
    --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT}"
    --learning_rate "${LEARNING_RATE}"
    --lr_scheduler constant
    --lr_warmup_steps 500
    --gradient_checkpointing
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --dataloader_pin_memory
    --mixed_precision "${MIXED_PRECISION}"
    --seed "${SEED}"
    --report_to tensorboard
    --tracker_project_name train_rgb_stc_v3_flow_shared_noise
)
if [[ -n "${INIT_STC_ADAPTER}" ]]; then
    COMMON_ARGS+=(--init_stc_adapter "${INIT_STC_ADAPTER}")
fi

export TOKENIZERS_PARALLELISM=false
cd "${BRUSHNET_ROOT}"

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}" --preflight_only "$@"
fi

if (( NUM_PROCESSES == 1 )); then
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}" "$@"
else
    "${ACCELERATE_BIN}" launch \
        --multi_gpu \
        --num_processes "${NUM_PROCESSES}" \
        --num_machines 1 \
        --mixed_precision "${MIXED_PRECISION}" \
        --dynamo_backend no \
        "${TRAIN_SCRIPT}" \
        "${COMMON_ARGS[@]}" \
        "$@"
fi
