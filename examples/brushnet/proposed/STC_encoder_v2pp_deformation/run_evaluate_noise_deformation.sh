#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
EVALUATE_SCRIPT="${SCRIPT_DIR}/evaluate_noise_deformation.py"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${BRUSHNET_ROOT}/experiments/train_stc_v2pp_T8_cosine_0.9/checkpoint-1000}"
DATASET_ROOT="${DATASET_ROOT:-${BRUSHNET_ROOT}/examples/brushnet/dataset/test}"
OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/eval_stc_v2pp_T8_cosine_0.9/test/checkpoint-1000-sequence-state}"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"

DATASET_LAYOUT="${DATASET_LAYOUT:-auto}"
SPLIT="${SPLIT:-test}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
BRUSHNET_CONDITIONING_SCALE="${BRUSHNET_CONDITIONING_SCALE:-1.0}"
FUSION_SCALE="${FUSION_SCALE:-1.0}"
ROI_COMPOSITE="${ROI_COMPOSITE:-hard}"
ROI_BLUR_KERNEL_SIZE="${ROI_BLUR_KERNEL_SIZE:-21}"
NOISE_SEED="${NOISE_SEED:-1234}"
CONDITION_SEED="${CONDITION_SEED:-2345}"
GENERATION_SEED="${GENERATION_SEED:-3456}"
DEVICE="${DEVICE:-cuda}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
SAVE_NOISE_TENSORS="${SAVE_NOISE_TENSORS:-0}"
OVERWRITE="${OVERWRITE:-0}"
MAX_CLIPS="${MAX_CLIPS:-}"
MAX_FRAMES_PER_SEQUENCE="${MAX_FRAMES_PER_SEQUENCE:-}"
TEMPORAL_GUIDANCE_SCALE="${TEMPORAL_GUIDANCE_SCALE:-0}"
TEMPORAL_START_STEP="${TEMPORAL_START_STEP:-15}"
TEMPORAL_END_STEP="${TEMPORAL_END_STEP:-35}"
TEMPORAL_EVERY_N_STEPS="${TEMPORAL_EVERY_N_STEPS:-1}"
TEMPORAL_DECODE_CHUNK_SIZE="${TEMPORAL_DECODE_CHUNK_SIZE:-1}"
TEMPORAL_LOSS_SCALE="${TEMPORAL_LOSS_SCALE:-1024}"
TEMPORAL_FLOW_BATCH_SIZE="${TEMPORAL_FLOW_BATCH_SIZE:-2}"
TEMPORAL_SAMPLING_SCOPE="${TEMPORAL_SAMPLING_SCOPE:-full_clip}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
# Keep the log as a top-level file.  Legacy image-to-video tooling interprets
# every top-level directory as a clip directory.
TERMINAL_LOG_FILE="${TERMINAL_LOG_FILE:-${OUTPUT_DIR}/evaluate_${RUN_TIMESTAMP}.log}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing Python executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
    echo "Missing checkpoint/experiment directory: ${CHECKPOINT_PATH}" >&2
    exit 1
fi
if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "Missing evaluation dataset: ${DATASET_ROOT}" >&2
    exit 1
fi
for name in RUN_PREFLIGHT SAVE_NOISE_TENSORS OVERWRITE; do
    value="${!name}"
    if [[ "${value}" != "0" && "${value}" != "1" ]]; then
        echo "${name} must be 0 or 1, got ${value}" >&2
        exit 1
    fi
done
if [[ -n "${MAX_CLIPS}" && ! "${MAX_CLIPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_CLIPS must be empty or a positive integer" >&2
    exit 1
fi
if [[ -n "${MAX_FRAMES_PER_SEQUENCE}" && ! "${MAX_FRAMES_PER_SEQUENCE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_FRAMES_PER_SEQUENCE must be empty or a positive integer" >&2
    exit 1
fi

mkdir -p "$(dirname -- "${TERMINAL_LOG_FILE}")"
exec > >(tee -a "${TERMINAL_LOG_FILE}") 2>&1

COMMON_ARGS=(
    --checkpoint_path "${CHECKPOINT_PATH}"
    --dataset_root "${DATASET_ROOT}"
    --dataset_layout "${DATASET_LAYOUT}"
    --split "${SPLIT}"
    --output_dir "${OUTPUT_DIR}"
    --num_inference_steps "${NUM_INFERENCE_STEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
    --brushnet_conditioning_scale "${BRUSHNET_CONDITIONING_SCALE}"
    --fusion_scale "${FUSION_SCALE}"
    --roi_composite "${ROI_COMPOSITE}"
    --roi_blur_kernel_size "${ROI_BLUR_KERNEL_SIZE}"
    --noise_seed "${NOISE_SEED}"
    --condition_seed "${CONDITION_SEED}"
    --generation_seed "${GENERATION_SEED}"
    --device "${DEVICE}"
    --temporal_guidance_scale "${TEMPORAL_GUIDANCE_SCALE}"
    --temporal_start_step "${TEMPORAL_START_STEP}"
    --temporal_end_step "${TEMPORAL_END_STEP}"
    --temporal_every_n_steps "${TEMPORAL_EVERY_N_STEPS}"
    --temporal_decode_chunk_size "${TEMPORAL_DECODE_CHUNK_SIZE}"
    --temporal_loss_scale "${TEMPORAL_LOSS_SCALE}"
    --temporal_flow_batch_size "${TEMPORAL_FLOW_BATCH_SIZE}"
    --temporal_sampling_scope "${TEMPORAL_SAMPLING_SCOPE}"
)
if [[ "${SAVE_NOISE_TENSORS}" == "1" ]]; then
    COMMON_ARGS+=(--save_noise_tensors)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
    COMMON_ARGS+=(--overwrite)
fi
if [[ -n "${MAX_CLIPS}" ]]; then
    COMMON_ARGS+=(--max_clips "${MAX_CLIPS}")
fi
if [[ -n "${MAX_FRAMES_PER_SEQUENCE}" ]]; then
    COMMON_ARGS+=(--max_frames_per_sequence "${MAX_FRAMES_PER_SEQUENCE}")
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "${BRUSHNET_ROOT}"

echo "Terminal log: ${TERMINAL_LOG_FILE}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Dataset: ${DATASET_ROOT}"
echo "Output: ${OUTPUT_DIR}"
echo "Evaluation policy: one ordered sequence-state owner on ${DEVICE}; no per-clip multi-GPU sharding"

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
    "${PYTHON_BIN}" "${EVALUATE_SCRIPT}" "${COMMON_ARGS[@]}" --preflight_only "$@"
fi

"${PYTHON_BIN}" "${EVALUATE_SCRIPT}" "${COMMON_ARGS[@]}" "$@"
