#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_flow_aligned_stc.py"
VIDEO_SCRIPT="${BRUSHNET_ROOT}/Quan_test/imgToVideo_rgb_stc_eval.py"

PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${BRUSHNET_ROOT}/experiments/train_stc_v4_T16_flow005/checkpoint-5000}"
STC_FLOW_MODEL="${STC_FLOW_MODEL:-${CHECKPOINT_PATH}/stc_flow_model}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${BRUSHNET_ROOT}/experiments/checkpoint_sharedNoise_sameBG_0.9/checkpoint-2000}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-${BRUSHNET_ROOT}/examples/brushnet/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5}"
IMAGE_ENCODER="${IMAGE_ENCODER:-laion/CLIP-ViT-H-14-laion2B-s32B-b79K}"
DATASET_ROOT="${DATASET_ROOT:-${BRUSHNET_ROOT}/examples/brushnet/dataset/test_1}"
DATASET_LAYOUT="${DATASET_LAYOUT:-flat_test}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/eval_stc_v4_T16_flow_aligned/checkpoint-5000/test_1-blurred-k21}"

CLIP_LENGTH="${CLIP_LENGTH:-16}"
CLIP_STRIDE="${CLIP_STRIDE:-12}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
BRUSHNET_SCALE="${BRUSHNET_SCALE:-1.0}"
STC_INJECTION_SCALE="${STC_INJECTION_SCALE:-1.0}"
FUSION_SCALE="${FUSION_SCALE:-1.0}"
SHARED_BG_NOISE_STRENGTH="${SHARED_BG_NOISE_STRENGTH:-0.9}"
ROI_COMPOSITE="${ROI_COMPOSITE:-blurred}"
ROI_BLUR_KERNEL_SIZE="${ROI_BLUR_KERNEL_SIZE:-21}"
SEED="${SEED:-1234}"
SHARED_BG_SEED="${SHARED_BG_SEED:-6789}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
CONVERT_TO_VIDEO="${CONVERT_TO_VIDEO:-1}"
VIDEO_FPS="${VIDEO_FPS:-60}"
OVERWRITE="${OVERWRITE:-0}"

for executable in "${PYTHON_BIN}"; do
    [[ -x "${executable}" ]] || { echo "Missing executable: ${executable}" >&2; exit 1; }
done
for required in \
    "${EVAL_SCRIPT}" "${VIDEO_SCRIPT}" "${STC_FLOW_MODEL}/config.json" \
    "${STC_FLOW_MODEL}/diffusion_pytorch_model.safetensors" \
    "${DATASET_ROOT}" "${BASELINE_CHECKPOINT}/brushnet/config.json"; do
    [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 1; }
done

COMMON_ARGS=(
    --pretrained_model_name_or_path "${PRETRAINED_MODEL}"
    --baseline_checkpoint "${BASELINE_CHECKPOINT}"
    --stc_adapter_path "${STC_FLOW_MODEL}"
    --dataset_root "${DATASET_ROOT}"
    --dataset_layout "${DATASET_LAYOUT}"
    --split "${SPLIT}"
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

mkdir -p "${OUTPUT_DIR}"
timestamp="$(date +%Y%m%d_%H%M%S)"
terminal_log="${OUTPUT_DIR}/evaluate_${timestamp}.log"
exec > >(tee -a "${terminal_log}") 2>&1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "V4 flow-aligned evaluation"
echo "checkpoint=${CHECKPOINT_PATH}"
echo "stc_flow_model=${STC_FLOW_MODEL}"
echo "dataset=${DATASET_ROOT}"
echo "output=${OUTPUT_DIR}"
echo "clip=${CLIP_LENGTH}, stride=${CLIP_STRIDE}, composite=${ROI_COMPOSITE}, blur=${ROI_BLUR_KERNEL_SIZE}"
echo "terminal_log=${terminal_log}"

cd "${BRUSHNET_ROOT}"
if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
    "${PYTHON_BIN}" "${EVAL_SCRIPT}" "${COMMON_ARGS[@]}" --preflight_only "$@"
fi
"${PYTHON_BIN}" "${EVAL_SCRIPT}" "${COMMON_ARGS[@]}" "$@"

if [[ "${CONVERT_TO_VIDEO}" == "1" ]]; then
    video_args=(
        --eval_root "${OUTPUT_DIR}"
        --frame_kinds final
        --selection first
        --overwrite
    )
    # An empty VIDEO_FPS means per-sequence native SFU FPS; do not pass an
    # empty argparse value (and never replace it with a global default).
    if [[ -n "${VIDEO_FPS}" ]]; then
        video_args+=(--fps "${VIDEO_FPS}")
    fi
    "${PYTHON_BIN}" "${VIDEO_SCRIPT}" "${video_args[@]}"
fi
