#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_v4pp_bg_feature_shared_noise.py"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-${BRUSHNET_ROOT}/examples/brushnet/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5}"
DATASET_ROOT="${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
TEACHER_FLOW_ROOT="${TEACHER_FLOW_ROOT:-${DATASET_ROOT}/teacher_flows_512x512}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${BRUSHNET_ROOT}/experiments/train_sharedNoise_sameBG_0.95_T8/checkpoint-2250}"
INIT_STC_ADAPTER="${INIT_STC_ADAPTER:-${BRUSHNET_ROOT}/experiments/train_rgb_stc_v2_sharedNoise_0.95/checkpoint-4000/stc_adapter}"
OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/train_stc_v4pp_bg_feature_T16_S12_sharedNoise_0.95}"
IMAGE_ENCODER="${IMAGE_ENCODER:-laion/CLIP-ViT-H-14-laion2B-s32B-b79K}"

TESTED_ENV_BIN="/home/cilab/ndquan/envs/guided_diff/bin"
PYTHON_BIN="${PYTHON_BIN:-${TESTED_ENV_BIN}/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${TESTED_ENV_BIN}/accelerate}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

CLIP_LENGTH="${CLIP_LENGTH:-16}"
CLIP_STRIDE="${CLIP_STRIDE:-12}"
CLIPS_PER_DEVICE="${CLIPS_PER_DEVICE:-1}"
TARGET_EFFECTIVE_CLIPS="${TARGET_EFFECTIVE_CLIPS:-6}"
if [[ -z "${GRADIENT_ACCUMULATION_STEPS+x}" ]]; then
    denominator=$((NUM_PROCESSES * CLIPS_PER_DEVICE))
    if (( TARGET_EFFECTIVE_CLIPS % denominator != 0 )); then
        echo "TARGET_EFFECTIVE_CLIPS must be divisible by NUM_PROCESSES*CLIPS_PER_DEVICE" >&2
        exit 1
    fi
    GRADIENT_ACCUMULATION_STEPS=$((TARGET_EFFECTIVE_CLIPS / denominator))
fi

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-250}"
CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}"
SHARED_BG_NOISE_STRENGTH="${SHARED_BG_NOISE_STRENGTH:-0.95}"
FLOW_LOSS_WEIGHT="${FLOW_LOSS_WEIGHT:-0.01}"
FLOW_REGION="${FLOW_REGION:-bg}"
FLOW_CHARBONNIER_EPS="${FLOW_CHARBONNIER_EPS:-0.001}"
FLOW_MAX_DX="${FLOW_MAX_DX:-8.0}"
FLOW_MAX_DY="${FLOW_MAX_DY:-8.0}"
FEATURE_ALIGNMENT_LOSS_WEIGHT="${FEATURE_ALIGNMENT_LOSS_WEIGHT:-0.01}"
FEATURE_ALIGNMENT_REGION="${FEATURE_ALIGNMENT_REGION:-bg}"
FEATURE_ALIGNMENT_EPS="${FEATURE_ALIGNMENT_EPS:-0.001}"
FEATURE_ALIGNMENT_CONFIDENCE_FLOOR="${FEATURE_ALIGNMENT_CONFIDENCE_FLOOR:-0.1}"
FEATURE_ALIGNMENT_WARMUP_STEPS="${FEATURE_ALIGNMENT_WARMUP_STEPS:-500}"
SEED="${SEED:-1234}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
CONDITION_MODE="${CONDITION_MODE:-full_rgb_bg_mask}"
STC_INJECTION_SCALE="${STC_INJECTION_SCALE:-1.0}"

for executable in "${PYTHON_BIN}" "${ACCELERATE_BIN}"; do
    [[ -x "${executable}" ]] || { echo "Missing executable: ${executable}" >&2; exit 1; }
done
for required in \
    "${TRAIN_SCRIPT}" "${PRETRAINED_MODEL}" \
    "${DATASET_ROOT}/train/GT" "${DATASET_ROOT}/train/input" \
    "${DATASET_ROOT}/train/mask" "${TEACHER_FLOW_ROOT}/metadata.json" \
    "${BASELINE_CHECKPOINT}/brushnet/config.json" \
    "${BASELINE_CHECKPOINT}/brushnet/diffusion_pytorch_model.safetensors" \
    "${BASELINE_CHECKPOINT}/ipadapter/model.safetensors" \
    "${BASELINE_CHECKPOINT}/ipadapter/fusion_module.safetensors" \
    "${INIT_STC_ADAPTER}/config.json"; do
    [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 1; }
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
    --feature_alignment_loss_weight "${FEATURE_ALIGNMENT_LOSS_WEIGHT}"
    --feature_alignment_region "${FEATURE_ALIGNMENT_REGION}"
    --feature_alignment_charbonnier_eps "${FEATURE_ALIGNMENT_EPS}"
    --feature_alignment_confidence_floor "${FEATURE_ALIGNMENT_CONFIDENCE_FLOOR}"
    --feature_alignment_warmup_steps "${FEATURE_ALIGNMENT_WARMUP_STEPS}"
    --init_stc_adapter "${INIT_STC_ADAPTER}"
    --train_batch_size "${CLIPS_PER_DEVICE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --max_train_steps "${MAX_TRAIN_STEPS}"
    --checkpointing_steps "${CHECKPOINTING_STEPS}"
    --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT}"
    --learning_rate "${LEARNING_RATE}"
    --lr_scheduler "${LR_SCHEDULER}"
    --lr_warmup_steps "${LR_WARMUP_STEPS}"
    --gradient_checkpointing
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --dataloader_pin_memory
    --mixed_precision "${MIXED_PRECISION}"
    --seed "${SEED}"
    --report_to tensorboard
    --tracker_project_name train_stc_v4pp_bg_feature_t16
)

mkdir -p "${OUTPUT_DIR}/terminal_logs"
timestamp="$(date +%Y%m%d_%H%M%S)"
terminal_log="${OUTPUT_DIR}/terminal_logs/train_${timestamp}.log"
exec > >(tee -a "${terminal_log}") 2>&1

echo "RGB-STC V4++: BG-focused alignment + feature-alignment loss"
echo "clip=${CLIP_LENGTH}, stride=${CLIP_STRIDE}, rho=${SHARED_BG_NOISE_STRENGTH}"
echo "flow_weight=${FLOW_LOSS_WEIGHT}, feature_weight=${FEATURE_ALIGNMENT_LOSS_WEIGHT}, feature_warmup=${FEATURE_ALIGNMENT_WARMUP_STEPS}"
echo "processes=${NUM_PROCESSES}, clips/device=${CLIPS_PER_DEVICE}, accumulation=${GRADIENT_ACCUMULATION_STEPS}"
echo "terminal_log=${terminal_log}"

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

