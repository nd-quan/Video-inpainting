#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_noise_deformation.py"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-${BRUSHNET_ROOT}/examples/brushnet/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5}"
DATASET_ROOT="${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${BRUSHNET_ROOT}/experiments/checkpoint_sharedNoise_sameBG_0.9/checkpoint-2000}"
STC_V2_CHECKPOINT="${STC_V2_CHECKPOINT:-${BRUSHNET_ROOT}/experiments/checkpoint_rgb_stc_v2_sharedNoise_0.9/checkpoint-2000/stc_adapter}"
OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/train_stc_v2pp_noise_deformation_full_0.9}"
IMAGE_ENCODER="${IMAGE_ENCODER:-laion/CLIP-ViT-H-14-laion2B-s32B-b79K}"

ENV_BIN="/home/cilab/ndquan/envs/guided_diff/bin"
PYTHON_BIN="${PYTHON_BIN:-${ENV_BIN}/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${ENV_BIN}/accelerate}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

CLIP_LENGTH="${CLIP_LENGTH:-8}"
CLIP_STRIDE="${CLIP_STRIDE:-6}"
CLIPS_PER_DEVICE="${CLIPS_PER_DEVICE:-1}"
TARGET_EFFECTIVE_CLIPS_PER_UPDATE="${TARGET_EFFECTIVE_CLIPS_PER_UPDATE:-6}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-6}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-250}"
CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-5}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}"
TRANSPORT_ALPHA="${TRANSPORT_ALPHA:-0.9}"
WARP_SCOPE="${WARP_SCOPE:-full}"
MAX_DISPLACEMENT="${MAX_DISPLACEMENT:-8.0}"
MATCH_LOSS_WEIGHT="${MATCH_LOSS_WEIGHT:-1.0}"
SMOOTHNESS_LOSS_WEIGHT="${SMOOTHNESS_LOSS_WEIGHT:-0.01}"
SMOOTHNESS_GAMMA="${SMOOTHNESS_GAMMA:-1.0}"
SEED="${SEED:-1234}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
TERMINAL_LOG_DIR="${TERMINAL_LOG_DIR:-${OUTPUT_DIR}/terminal_logs}"
TERMINAL_LOG_FILE="${TERMINAL_LOG_FILE:-${TERMINAL_LOG_DIR}/train_${RUN_TIMESTAMP}.log}"

mkdir -p "$(dirname -- "${TERMINAL_LOG_FILE}")"
exec > >(tee -a "${TERMINAL_LOG_FILE}") 2>&1

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing Python executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_PROCESSES must be a positive integer" >&2
    exit 1
fi
if [[ ! "${CLIPS_PER_DEVICE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CLIPS_PER_DEVICE must be a positive integer" >&2
    exit 1
fi
if [[ ! "${TARGET_EFFECTIVE_CLIPS_PER_UPDATE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TARGET_EFFECTIVE_CLIPS_PER_UPDATE must be a positive integer" >&2
    exit 1
fi
if [[ -z "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
    clips_per_microstep=$((NUM_PROCESSES * CLIPS_PER_DEVICE))
    if (( TARGET_EFFECTIVE_CLIPS_PER_UPDATE % clips_per_microstep != 0 )); then
        echo "TARGET_EFFECTIVE_CLIPS_PER_UPDATE must be divisible by NUM_PROCESSES * CLIPS_PER_DEVICE" >&2
        exit 1
    fi
    GRADIENT_ACCUMULATION_STEPS=$((TARGET_EFFECTIVE_CLIPS_PER_UPDATE / clips_per_microstep))
fi
if [[ ! "${GRADIENT_ACCUMULATION_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GRADIENT_ACCUMULATION_STEPS must be a positive integer" >&2
    exit 1
fi
if (( NUM_PROCESSES > 1 )) && [[ ! -x "${ACCELERATE_BIN}" ]]; then
    echo "Missing Accelerate executable: ${ACCELERATE_BIN}" >&2
    exit 1
fi
if (( NUM_PROCESSES > 1 )); then
    visible_gpu_count="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
    if [[ ! "${visible_gpu_count}" =~ ^[0-9]+$ ]] || (( visible_gpu_count < NUM_PROCESSES )); then
        echo "Requested ${NUM_PROCESSES} processes but PyTorch sees ${visible_gpu_count} CUDA devices" >&2
        exit 1
    fi
fi

COMMON_ARGS=(
    --pretrained_model_name_or_path "${PRETRAINED_MODEL}"
    --baseline_checkpoint "${BASELINE_CHECKPOINT}"
    --init_stc_v2_adapter_path "${STC_V2_CHECKPOINT}"
    --image_encoder_name_or_path "${IMAGE_ENCODER}"
    --dataset_root "${DATASET_ROOT}"
    --train_split train
    --output_dir "${OUTPUT_DIR}"
    --resolution 512
    --clip_length "${CLIP_LENGTH}"
    --clip_stride "${CLIP_STRIDE}"
    --condition_mode full_rgb_bg_mask
    --stc_injection_scale 1.0
    --transport_alpha "${TRANSPORT_ALPHA}"
    --warp_scope "${WARP_SCOPE}"
    --max_displacement "${MAX_DISPLACEMENT}"
    --match_loss_weight "${MATCH_LOSS_WEIGHT}"
    --smoothness_loss_weight "${SMOOTHNESS_LOSS_WEIGHT}"
    --smoothness_gamma "${SMOOTHNESS_GAMMA}"
    --train_batch_size "${CLIPS_PER_DEVICE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --max_train_steps "${MAX_TRAIN_STEPS}"
    --checkpointing_steps "${CHECKPOINTING_STEPS}"
    --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT}"
    --learning_rate "${LEARNING_RATE}"
    --lr_scheduler "${LR_SCHEDULER}"
    --lr_warmup_steps "${LR_WARMUP_STEPS}"
    --gradient_checkpointing
    --dataloader_num_workers 2
    --dataloader_pin_memory
    --mixed_precision "${MIXED_PRECISION}"
    --seed "${SEED}"
    --logging_steps "${LOGGING_STEPS}"
    --terminal_log_file "${TERMINAL_LOG_FILE}"
    --report_to tensorboard
    --tracker_project_name train_stc_v2pp_noise_deformation
)

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "${BRUSHNET_ROOT}"

echo "Terminal log: ${TERMINAL_LOG_FILE}"
echo "STC-v2++ launch: GPUs=${NUM_PROCESSES}, clips/device=${CLIPS_PER_DEVICE}, accumulation=${GRADIENT_ACCUMULATION_STEPS}, effective_clips/update=$((NUM_PROCESSES * CLIPS_PER_DEVICE * GRADIENT_ACCUMULATION_STEPS))"
echo "Sequence state: clip-local recurrence only; no cross-clip/cross-rank lineage cache during D2 training"

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
