#!/usr/bin/env bash
# Evaluate V6 + frozen V7 RAFT temporal DDIM guidance across independent GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_v6_flow_deformable_temporal_student.py"

CHECKPOINT_STEP="${CHECKPOINT_STEP:?Set CHECKPOINT_STEP, for example 1750}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${BRUSHNET_ROOT}/experiments/train_stc_v6_deform_only_T16_S12_sharedNoise_0.95/checkpoint-${CHECKPOINT_STEP}}"
STC_V6_MODEL="${STC_V6_MODEL:-${CHECKPOINT_PATH}/stc_v6_model}"
RAFT_STUDENT_PATH="${RAFT_STUDENT_PATH:-${BRUSHNET_ROOT}/experiments/train_v7_raft_student_flow/best.json}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${BRUSHNET_ROOT}/experiments/train_sharedNoise_sameBG_0.95_T8/checkpoint-2250}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-${BRUSHNET_ROOT}/examples/brushnet/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5}"
IMAGE_ENCODER="${IMAGE_ENCODER:-laion/CLIP-ViT-H-14-laion2B-s32B-b79K}"
TEST_1_ROOT="${TEST_1_ROOT:-${BRUSHNET_ROOT}/examples/brushnet/dataset/test_1}"
TEST_2_ROOT="${TEST_2_ROOT:-${BRUSHNET_ROOT}/examples/brushnet/dataset/test_2}"
OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/eval_stc_v6_temporal_student/checkpoint-${CHECKPOINT_STEP}-combined-testsets}"

CLIP_LENGTH="${CLIP_LENGTH:-16}"
CLIP_STRIDE="${CLIP_STRIDE:-12}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
BRUSHNET_SCALE="${BRUSHNET_SCALE:-1.0}"
STC_INJECTION_SCALE="${STC_INJECTION_SCALE:-1.0}"
DEFORMABLE_ALIGNMENT_SCALE="${DEFORMABLE_ALIGNMENT_SCALE:-1.0}"
FUSION_SCALE="${FUSION_SCALE:-1.0}"
SHARED_BG_NOISE_STRENGTH="${SHARED_BG_NOISE_STRENGTH:-0.95}"
ROI_COMPOSITE="${ROI_COMPOSITE:-blurred}"
ROI_BLUR_KERNEL_SIZE="${ROI_BLUR_KERNEL_SIZE:-51}"
SEED="${SEED:-1234}"
SHARED_BG_SEED="${SHARED_BG_SEED:-6789}"
TEMPORAL_GUIDANCE_SCALE="${TEMPORAL_GUIDANCE_SCALE:-0.0001}"
TEMPORAL_START_STEP="${TEMPORAL_START_STEP:-25}"
TEMPORAL_END_STEP="${TEMPORAL_END_STEP:-35}"
TEMPORAL_FLOW_BATCH_SIZE="${TEMPORAL_FLOW_BATCH_SIZE:-1}"
RAFT_PAIR_BATCH_SIZE="${RAFT_PAIR_BATCH_SIZE:-1}"
V6_TEMPORAL_GPU_IDS="${V6_TEMPORAL_GPU_IDS:-1,3,4,5}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
OVERWRITE="${OVERWRITE:-0}"

IFS=',' read -r -a GPU_IDS <<< "${V6_TEMPORAL_GPU_IDS}"
(( ${#GPU_IDS[@]} > 0 )) || { echo "V6_TEMPORAL_GPU_IDS must not be empty" >&2; exit 2; }

for required in "${PYTHON_BIN}" "${EVAL_SCRIPT}" "${STC_V6_MODEL}/config.json" \
    "${STC_V6_MODEL}/diffusion_pytorch_model.safetensors" "${TEST_1_ROOT}" "${TEST_2_ROOT}"; do
    [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 1; }
done

COMMON_ARGS=(
    --pretrained_model_name_or_path "${PRETRAINED_MODEL}"
    --baseline_checkpoint "${BASELINE_CHECKPOINT}"
    --stc_adapter_path "${STC_V6_MODEL}"
    --raft_student_path "${RAFT_STUDENT_PATH}"
    --raft_pair_batch_size "${RAFT_PAIR_BATCH_SIZE}"
    --dataset_layout flat_test --split test
    --image_encoder_name_or_path "${IMAGE_ENCODER}"
    --resolution 512 --clip_length "${CLIP_LENGTH}" --clip_stride "${CLIP_STRIDE}"
    --num_inference_steps "${NUM_INFERENCE_STEPS}" --guidance_scale "${GUIDANCE_SCALE}"
    --brushnet_conditioning_scale "${BRUSHNET_SCALE}"
    --stc_injection_scale "${STC_INJECTION_SCALE}"
    --deformable_alignment_scale "${DEFORMABLE_ALIGNMENT_SCALE}"
    --fusion_scale "${FUSION_SCALE}"
    --shared_bg_noise_strength "${SHARED_BG_NOISE_STRENGTH}"
    --roi_composite "${ROI_COMPOSITE}" --roi_blur_kernel_size "${ROI_BLUR_KERNEL_SIZE}"
    --seed "${SEED}" --shared_bg_seed "${SHARED_BG_SEED}" --device cuda --save_references
    --temporal_guidance_scale "${TEMPORAL_GUIDANCE_SCALE}"
    --temporal_start_step "${TEMPORAL_START_STEP}" --temporal_end_step "${TEMPORAL_END_STEP}"
    --temporal_flow_batch_size "${TEMPORAL_FLOW_BATCH_SIZE}"
)
if [[ "${OVERWRITE}" == "1" ]]; then COMMON_ARGS+=(--overwrite); fi

mkdir -p "${OUTPUT_DIR}/terminal_logs"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

run_split() {
    local label="$1"
    local root="$2"
    local split_root="${OUTPUT_DIR}/${label}"
    local shard_count="${#GPU_IDS[@]}"
    local pids=()
    local status=0

    mkdir -p "${split_root}"
    echo "Starting ${label} on ${shard_count} shard(s): ${root}"
    for index in "${!GPU_IDS[@]}"; do
        local gpu="${GPU_IDS[$index]}"
        local shard_root="${split_root}/shard-${index}"
        local log_path="${OUTPUT_DIR}/terminal_logs/${label}_shard-${index}_gpu-${gpu}.log"
        local shard_args=(
            --dataset_root "${root}" --output_dir "${shard_root}"
            --num_shards "${shard_count}" --shard_index "${index}"
        )
        if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
            CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
                "${COMMON_ARGS[@]}" "${shard_args[@]}" --preflight_only > "${log_path}" 2>&1
        fi
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u "${EVAL_SCRIPT}" \
            "${COMMON_ARGS[@]}" "${shard_args[@]}" >> "${log_path}" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "${pid}" || status=1
    done
    return "${status}"
}

cd "${BRUSHNET_ROOT}"
run_split test_1 "${TEST_1_ROOT}"
run_split test_2 "${TEST_2_ROOT}"
echo "V6 temporal-student evaluation complete: ${OUTPUT_DIR}"
