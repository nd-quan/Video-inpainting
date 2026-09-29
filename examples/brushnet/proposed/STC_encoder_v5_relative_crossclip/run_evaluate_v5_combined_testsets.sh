#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_v5_relative_crossclip.py"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"

CHECKPOINT_STEP="${CHECKPOINT_STEP:?Set CHECKPOINT_STEP, for example 2500}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${BRUSHNET_ROOT}/experiments/train_stc_v5_relative_crossclip_T16_S12_sharedNoise_0.95/checkpoint-${CHECKPOINT_STEP}}"
STC_V5_MODEL="${STC_V5_MODEL:-${CHECKPOINT_PATH}/stc_v5_model}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${BRUSHNET_ROOT}/experiments/train_sharedNoise_sameBG_0.95_T8/checkpoint-2250}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-${BRUSHNET_ROOT}/examples/brushnet/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5}"
IMAGE_ENCODER="${IMAGE_ENCODER:-laion/CLIP-ViT-H-14-laion2B-s32B-b79K}"
TEST_1_ROOT="${TEST_1_ROOT:-${BRUSHNET_ROOT}/examples/brushnet/dataset/test_1}"
TEST_2_ROOT="${TEST_2_ROOT:-${BRUSHNET_ROOT}/examples/brushnet/dataset/test_2}"
OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/eval_stc_v5_relative_crossclip/checkpoint-${CHECKPOINT_STEP}-combined-testsets}"

CLIP_LENGTH="${CLIP_LENGTH:-16}"
CLIP_STRIDE="${CLIP_STRIDE:-12}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
BRUSHNET_SCALE="${BRUSHNET_SCALE:-1.0}"
STC_INJECTION_SCALE="${STC_INJECTION_SCALE:-1.0}"
FUSION_SCALE="${FUSION_SCALE:-1.0}"
SHARED_BG_NOISE_STRENGTH="${SHARED_BG_NOISE_STRENGTH:-0.95}"
ROI_COMPOSITE="${ROI_COMPOSITE:-blurred}"
ROI_BLUR_KERNEL_SIZE="${ROI_BLUR_KERNEL_SIZE:-51}"
SEED="${SEED:-1234}"
SHARED_BG_SEED="${SHARED_BG_SEED:-6789}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
OVERWRITE="${OVERWRITE:-0}"

for required in \
    "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    "${STC_V5_MODEL}/config.json" \
    "${STC_V5_MODEL}/diffusion_pytorch_model.safetensors" \
    "${BASELINE_CHECKPOINT}/brushnet/config.json" \
    "${BASELINE_CHECKPOINT}/ipadapter/model.safetensors" \
    "${BASELINE_CHECKPOINT}/ipadapter/fusion_module.safetensors" \
    "${TEST_1_ROOT}" "${TEST_2_ROOT}"; do
    [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 1; }
done

mapfile -t duplicate_sequences < <(
    comm -12 \
        <(find "${TEST_1_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort) \
        <(find "${TEST_2_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
)
if (( ${#duplicate_sequences[@]} > 0 )); then
    echo "Duplicate sequence names prevent a shared output root: ${duplicate_sequences[*]}" >&2
    exit 2
fi

COMMON_ARGS=(
    --pretrained_model_name_or_path "${PRETRAINED_MODEL}"
    --baseline_checkpoint "${BASELINE_CHECKPOINT}"
    --stc_adapter_path "${STC_V5_MODEL}"
    --dataset_layout flat_test
    --split test
    --output_dir "${OUTPUT_DIR}"
    --image_encoder_name_or_path "${IMAGE_ENCODER}"
    --resolution 512
    --clip_length "${CLIP_LENGTH}"
    --clip_stride "${CLIP_STRIDE}"
    --num_inference_steps "${NUM_INFERENCE_STEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
    --brushnet_conditioning_scale "${BRUSHNET_SCALE}"
    --stc_injection_scale "${STC_INJECTION_SCALE}"
    --fusion_scale "${FUSION_SCALE}"
    --shared_bg_noise_strength "${SHARED_BG_NOISE_STRENGTH}"
    --roi_composite "${ROI_COMPOSITE}"
    --roi_blur_kernel_size "${ROI_BLUR_KERNEL_SIZE}"
    --seed "${SEED}"
    --shared_bg_seed "${SHARED_BG_SEED}"
    --device cuda
    --save_references
)
if [[ "${OVERWRITE}" == "1" ]]; then
    COMMON_ARGS+=(--overwrite)
fi

mkdir -p "${OUTPUT_DIR}/terminal_logs"
timestamp="$(date +%Y%m%d_%H%M%S)"
terminal_log="${OUTPUT_DIR}/terminal_logs/evaluate_combined_${timestamp}.log"
exec > >(tee -a "${terminal_log}") 2>&1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "RGB-STC V5 combined test-set evaluation"
echo "checkpoint=${CHECKPOINT_PATH}"
echo "stc_v5_model=${STC_V5_MODEL}"
echo "T=${CLIP_LENGTH}, S=${CLIP_STRIDE}, overlap=$((CLIP_LENGTH-CLIP_STRIDE))"
echo "output=${OUTPUT_DIR}"
echo "terminal_log=${terminal_log}"

cd "${BRUSHNET_ROOT}"
labels=(test_1 test_2)
roots=("${TEST_1_ROOT}" "${TEST_2_ROOT}")
for index in "${!roots[@]}"; do
    label="${labels[${index}]}"
    root="${roots[${index}]}"
    echo "Starting ${label}: ${root}"
    if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
        "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
            "${COMMON_ARGS[@]}" --dataset_root "${root}" --preflight_only "$@"
    fi
    "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
        "${COMMON_ARGS[@]}" --dataset_root "${root}" "$@"
    cp "${OUTPUT_DIR}/run_config.json" "${OUTPUT_DIR}/run_config_${label}.json"
done

echo "Combined V5 evaluation complete: ${OUTPUT_DIR}/summary.json"

