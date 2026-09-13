#!/usr/bin/env bash
# Evaluate V8 with background-only CGE + latent temporal DDIM guidance.
# Every process owns whole video branches; V8 cross-clip condition state is
# therefore preserved inside each independently evaluated sequence.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_v8_raft_deformable_cge_temporal_bg_only.py"
AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_v8_sharded_metrics.py"

CHECKPOINT_STEP="${CHECKPOINT_STEP:-7500}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${BRUSHNET_ROOT}/experiments/train_stc_v8_raft_deform_only_T16_S12_sharedNoise_0.95/checkpoint-${CHECKPOINT_STEP}}"
STC_V8_MODEL="${STC_V8_MODEL:-${CHECKPOINT_PATH}/stc_v8_model}"
RAFT_STUDENT_PATH="${RAFT_STUDENT_PATH:-${BRUSHNET_ROOT}/experiments/train_v7_raft_student_flow/checkpoint-0004750/raft_student}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${BRUSHNET_ROOT}/experiments/train_sharedNoise_sameBG_0.95_T8/checkpoint-2250}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-${BRUSHNET_ROOT}/examples/brushnet/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5}"
IMAGE_ENCODER="${IMAGE_ENCODER:-laion/CLIP-ViT-H-14-laion2B-s32B-b79K}"
TEST_1_ROOT="${TEST_1_ROOT:-${BRUSHNET_ROOT}/examples/brushnet/dataset/test_1}"
TEST_2_ROOT="${TEST_2_ROOT:-${BRUSHNET_ROOT}/examples/brushnet/dataset/test_2}"
OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/eval_stc_v8_cge_temporal_bg_only/checkpoint-${CHECKPOINT_STEP}-T16-S12}"

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
RAFT_PAIR_BATCH_SIZE="${RAFT_PAIR_BATCH_SIZE:-1}"

# Both terms use the same late [25,35) DDIM window by default.  VCM-RS is
# expensive because its background codec is invoked once per frame per CGE
# evaluation, so the V6-compatible default limits each clip to two calls.
TEMPORAL_GUIDANCE_SCALE="${TEMPORAL_GUIDANCE_SCALE:-0.0001}"
TEMPORAL_START_STEP="${TEMPORAL_START_STEP:-25}"
TEMPORAL_END_STEP="${TEMPORAL_END_STEP:-35}"
TEMPORAL_EVERY_N_STEPS="${TEMPORAL_EVERY_N_STEPS:-1}"
TEMPORAL_LOSS_TYPE="${TEMPORAL_LOSS_TYPE:-l2}"
TEMPORAL_VISIBILITY_ALPHA="${TEMPORAL_VISIBILITY_ALPHA:-0.01}"
TEMPORAL_VISIBILITY_BETA="${TEMPORAL_VISIBILITY_BETA:-0.5}"
CGE_GUIDANCE_SCALE="${CGE_GUIDANCE_SCALE:-0.0001}"
CGE_SCALE_SCHEDULE="${CGE_SCALE_SCHEDULE:-fixed}"
CGE_START_STEP="${CGE_START_STEP:-25}"
CGE_END_STEP="${CGE_END_STEP:-35}"
CGE_EVERY_N_STEPS="${CGE_EVERY_N_STEPS:-1}"
CGE_MAX_EVALS="${CGE_MAX_EVALS:-2}"
CGE_DECODE_CHUNK_SIZE="${CGE_DECODE_CHUNK_SIZE:-1}"

# Use a comma-separated list, for example V8_CGE_TEMPORAL_GPU_IDS=1,3.
V8_CGE_TEMPORAL_GPU_IDS="${V8_CGE_TEMPORAL_GPU_IDS:-4,5}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
OVERWRITE="${OVERWRITE:-0}"

IFS=',' read -r -a GPU_IDS <<< "${V8_CGE_TEMPORAL_GPU_IDS}"
(( ${#GPU_IDS[@]} > 0 )) || {
    echo "V8_CGE_TEMPORAL_GPU_IDS must not be empty" >&2
    exit 2
}

for required in "${PYTHON_BIN}" "${EVAL_SCRIPT}" "${AGGREGATE_SCRIPT}" \
    "${STC_V8_MODEL}/config.json" \
    "${STC_V8_MODEL}/diffusion_pytorch_model.safetensors" \
    "${RAFT_STUDENT_PATH}/config.json" \
    "${BASELINE_CHECKPOINT}/brushnet/config.json" \
    "${BASELINE_CHECKPOINT}/ipadapter/model.safetensors" \
    "${BASELINE_CHECKPOINT}/ipadapter/fusion_module.safetensors" \
    "${TEST_1_ROOT}" "${TEST_2_ROOT}"; do
    [[ -e "${required}" ]] || {
        echo "Missing required path: ${required}" >&2
        exit 1
    }
done

COMMON_ARGS=(
    --pretrained_model_name_or_path "${PRETRAINED_MODEL}"
    --baseline_checkpoint "${BASELINE_CHECKPOINT}"
    --stc_adapter_path "${STC_V8_MODEL}"
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
    --temporal_every_n_steps "${TEMPORAL_EVERY_N_STEPS}"
    --temporal_loss_type "${TEMPORAL_LOSS_TYPE}"
    --temporal_visibility_alpha "${TEMPORAL_VISIBILITY_ALPHA}"
    --temporal_visibility_beta "${TEMPORAL_VISIBILITY_BETA}"
    --cge_guidance_scale "${CGE_GUIDANCE_SCALE}"
    --cge_scale_schedule "${CGE_SCALE_SCHEDULE}"
    --cge_start_step "${CGE_START_STEP}" --cge_end_step "${CGE_END_STEP}"
    --cge_every_n_steps "${CGE_EVERY_N_STEPS}" --cge_max_evals "${CGE_MAX_EVALS}"
    --cge_decode_chunk_size "${CGE_DECODE_CHUNK_SIZE}"
)
if [[ -n "${DEFORMABLE_ALIGNMENT_DIRECTION:-}" ]]; then
    COMMON_ARGS+=(--deformable_alignment_direction "${DEFORMABLE_ALIGNMENT_DIRECTION}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
    COMMON_ARGS+=(--overwrite)
fi

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
    echo "Starting ${label} on ${shard_count} whole-video shard(s): ${root}"
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
                "${COMMON_ARGS[@]}" "${shard_args[@]}" --preflight_only \
                > "${log_path}" 2>&1
        fi
        CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u "${EVAL_SCRIPT}" \
            "${COMMON_ARGS[@]}" "${shard_args[@]}" >> "${log_path}" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "${pid}" || status=1
    done
    if (( status != 0 )); then
        echo "${label} failed; inspect ${OUTPUT_DIR}/terminal_logs" >&2
        return "${status}"
    fi
    "${PYTHON_BIN}" "${AGGREGATE_SCRIPT}" \
        --split_root "${split_root}" --num_shards "${shard_count}" \
        > "${OUTPUT_DIR}/terminal_logs/${label}_aggregate.log" 2>&1
    echo "Completed ${label}: ${split_root}/summary.json"
}

cd "${BRUSHNET_ROOT}"
run_split test_1 "${TEST_1_ROOT}"
run_split test_2 "${TEST_2_ROOT}"
echo "V8 CGE + temporal evaluation complete: ${OUTPUT_DIR}"
