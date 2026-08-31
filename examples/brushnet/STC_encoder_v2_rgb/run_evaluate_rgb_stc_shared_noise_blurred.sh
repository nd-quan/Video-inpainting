#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
EVALUATE_SCRIPT="${SCRIPT_DIR}/evaluate_rgb_stc_shared_noise.py"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"

BASE_MODEL="${BASE_MODEL:-${BRUSHNET_ROOT}/examples/brushnet/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${BRUSHNET_ROOT}/experiments/checkpoint_sharedNoise_sameBG_0.9/checkpoint-2000}"
STC_ADAPTER_PATH="${STC_ADAPTER_PATH:-${BRUSHNET_ROOT}/experiments/checkpoint_rgb_stc_v2_sharedNoise_0.9/checkpoint-2000/stc_adapter}"
DATASET_ROOT="${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
SPLIT="${SPLIT:-long_test}"
DATASET_LAYOUT="${DATASET_LAYOUT:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/eval_rgb_stc_v2/long_test_10_sequences/checkpoint-2000-sBG09-blurred-k21}"
SEQUENCE_BRANCHES="${SEQUENCE_BRANCHES:-Class_D/BasketballPass Class_B/ParkScene Class_C/PartyScene Class_C/RaceHorsesC Class_A/Traffic Class_C/BQMall Class_D/BQSquare Class_B/BQTerrace Class_E/FourPeople Class_A/PeopleOnStreet}"
SHARED_BG_NOISE_STRENGTH="${SHARED_BG_NOISE_STRENGTH:-0.9}"
ROI_BLUR_KERNEL_SIZE="${ROI_BLUR_KERNEL_SIZE:-21}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
OVERWRITE="${OVERWRITE:-0}"
MAX_CLIPS="${MAX_CLIPS:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing Python executable: ${PYTHON_BIN}" >&2
    exit 1
fi
for path in "${BASE_MODEL}" "${BASELINE_CHECKPOINT}" "${STC_ADAPTER_PATH}" "${DATASET_ROOT}"; do
    if [[ ! -d "${path}" ]]; then
        echo "Missing required directory: ${path}" >&2
        exit 1
    fi
done
for name in RUN_PREFLIGHT OVERWRITE; do
    value="${!name}"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "${name} must be 0 or 1, got ${value}" >&2
        exit 1
    fi
done
read -r -a SEQUENCE_BRANCH_ARGS <<< "${SEQUENCE_BRANCHES}"
if [[ "${#SEQUENCE_BRANCH_ARGS[@]}" -eq 0 ]]; then
    echo "SEQUENCE_BRANCHES must contain at least one Class/sequence branch" >&2
    exit 1
fi

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
TERMINAL_LOG_FILE="${TERMINAL_LOG_FILE:-${OUTPUT_DIR}/evaluate_${RUN_TIMESTAMP}.log}"
mkdir -p "$(dirname -- "${TERMINAL_LOG_FILE}")"
exec > >(tee -a "${TERMINAL_LOG_FILE}") 2>&1

COMMON_ARGS=(
    --pretrained_model_name_or_path "${BASE_MODEL}"
    --baseline_checkpoint "${BASELINE_CHECKPOINT}"
    --stc_adapter_path "${STC_ADAPTER_PATH}"
    --dataset_root "${DATASET_ROOT}"
    --dataset_layout "${DATASET_LAYOUT}"
    --split "${SPLIT}"
    --include_branches "${SEQUENCE_BRANCH_ARGS[@]}"
    --output_dir "${OUTPUT_DIR}"
    --clip_length 8
    --clip_stride 6
    --num_inference_steps 50
    --guidance_scale 7.5
    --brushnet_conditioning_scale 1.0
    --stc_injection_scale 1.0
    --fusion_scale 1.0
    --shared_bg_noise_strength "${SHARED_BG_NOISE_STRENGTH}"
    --seed 1234
    --shared_bg_seed 6789
    --roi_composite blurred
    --roi_blur_kernel_size "${ROI_BLUR_KERNEL_SIZE}"
    --save_references
    --device cuda
)
if [[ "${OVERWRITE}" == "1" ]]; then
    COMMON_ARGS+=(--overwrite)
fi
if [[ -n "${MAX_CLIPS}" ]]; then
    COMMON_ARGS+=(--max_clips "${MAX_CLIPS}")
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "${BRUSHNET_ROOT}"

echo "Terminal log: ${TERMINAL_LOG_FILE}"
echo "Baseline checkpoint: ${BASELINE_CHECKPOINT}"
echo "STC adapter: ${STC_ADAPTER_PATH}"
echo "Dataset: ${DATASET_ROOT}"
echo "Split/layout: ${SPLIT}/${DATASET_LAYOUT}"
echo "Sequence branches: ${SEQUENCE_BRANCH_ARGS[*]}"
echo "Output: ${OUTPUT_DIR}"
echo "Composite: blurred, Gaussian kernel ${ROI_BLUR_KERNEL_SIZE}"

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
    "${PYTHON_BIN}" "${EVALUATE_SCRIPT}" "${COMMON_ARGS[@]}" --preflight_only "$@"
fi
"${PYTHON_BIN}" "${EVALUATE_SCRIPT}" "${COMMON_ARGS[@]}" "$@"
