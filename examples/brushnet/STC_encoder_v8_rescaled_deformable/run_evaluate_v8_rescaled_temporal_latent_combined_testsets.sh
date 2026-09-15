#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_v8_rescaled_deformable_temporal_student.py"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"

CHECKPOINT_STEP="${CHECKPOINT_STEP:?Set CHECKPOINT_STEP, for example 2500}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${BRUSHNET_ROOT}/experiments/train_stc_v8_rescaled_x4_joint_raftbest_T16_S12_sharedNoise_0.95/checkpoint-${CHECKPOINT_STEP}}"
STC_V8R_MODEL="${STC_V8R_MODEL:-${CHECKPOINT_PATH}/stc_v8r_model}"
RAFT_STUDENT_PATH="${RAFT_STUDENT_PATH:-${BRUSHNET_ROOT}/experiments/train_v7_raft_student_flow/best.json}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${BRUSHNET_ROOT}/experiments/train_sharedNoise_sameBG_0.95_T8/checkpoint-2250}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-${BRUSHNET_ROOT}/examples/brushnet/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5}"
IMAGE_ENCODER="${IMAGE_ENCODER:-laion/CLIP-ViT-H-14-laion2B-s32B-b79K}"
TEST_1_ROOT="${TEST_1_ROOT:-${BRUSHNET_ROOT}/examples/brushnet/dataset/test_1}"
TEST_2_ROOT="${TEST_2_ROOT:-${BRUSHNET_ROOT}/examples/brushnet/dataset/test_2}"
TEST_SET="${TEST_SET:-both}"
SEQUENCE_NAME="${SEQUENCE_NAME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/eval_stc_v8_rescaled_x4_latent_tg/checkpoint-${CHECKPOINT_STEP}-combined-testsets}"

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

TEMPORAL_GUIDANCE_SCALE="${TEMPORAL_GUIDANCE_SCALE:-0.001}"
TEMPORAL_START_STEP="${TEMPORAL_START_STEP:-25}"
TEMPORAL_END_STEP="${TEMPORAL_END_STEP:-35}"
TEMPORAL_EVERY_N_STEPS="${TEMPORAL_EVERY_N_STEPS:-1}"
TEMPORAL_LOSS_SCALE="${TEMPORAL_LOSS_SCALE:-1024.0}"
TEMPORAL_LOSS_TYPE="${TEMPORAL_LOSS_TYPE:-l2}"
TEMPORAL_VISIBILITY_ALPHA="${TEMPORAL_VISIBILITY_ALPHA:-0.01}"
TEMPORAL_VISIBILITY_BETA="${TEMPORAL_VISIBILITY_BETA:-0.5}"
TEMPORAL_DETACH_PREVIOUS="${TEMPORAL_DETACH_PREVIOUS:-1}"
RAFT_PAIR_BATCH_SIZE="${RAFT_PAIR_BATCH_SIZE:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
OVERWRITE="${OVERWRITE:-0}"

case "${TEST_SET}" in
    both)
        labels=(test_1 test_2)
        roots=("${TEST_1_ROOT}" "${TEST_2_ROOT}")
        ;;
    test_1)
        labels=(test_1)
        roots=("${TEST_1_ROOT}")
        ;;
    test_2)
        labels=(test_2)
        roots=("${TEST_2_ROOT}")
        ;;
    *)
        echo "TEST_SET must be one of: both, test_1, test_2; got ${TEST_SET}" >&2
        exit 2
        ;;
esac

for required in \
    "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
    "${STC_V8R_MODEL}/config.json" \
    "${STC_V8R_MODEL}/diffusion_pytorch_model.safetensors" \
    "${BASELINE_CHECKPOINT}/brushnet/config.json" \
    "${BASELINE_CHECKPOINT}/ipadapter/model.safetensors" \
    "${BASELINE_CHECKPOINT}/ipadapter/fusion_module.safetensors" \
    "${roots[@]}"; do
    [[ -e "${required}" ]] || {
        echo "Missing required path: ${required}" >&2
        exit 1
    }
done

COMMON_ARGS=(
    --pretrained_model_name_or_path "${PRETRAINED_MODEL}"
    --baseline_checkpoint "${BASELINE_CHECKPOINT}"
    --stc_adapter_path "${STC_V8R_MODEL}"
    --raft_student_path "${RAFT_STUDENT_PATH}"
    --raft_pair_batch_size "${RAFT_PAIR_BATCH_SIZE}"
    --dataset_layout flat_test --split test
    --image_encoder_name_or_path "${IMAGE_ENCODER}"
    --resolution 512
    --clip_length "${CLIP_LENGTH}" --clip_stride "${CLIP_STRIDE}"
    --num_inference_steps "${NUM_INFERENCE_STEPS}"
    --guidance_scale "${GUIDANCE_SCALE}"
    --brushnet_conditioning_scale "${BRUSHNET_SCALE}"
    --stc_injection_scale "${STC_INJECTION_SCALE}"
    --deformable_alignment_scale "${DEFORMABLE_ALIGNMENT_SCALE}"
    --fusion_scale "${FUSION_SCALE}"
    --shared_bg_noise_strength "${SHARED_BG_NOISE_STRENGTH}"
    --roi_composite "${ROI_COMPOSITE}"
    --roi_blur_kernel_size "${ROI_BLUR_KERNEL_SIZE}"
    --seed "${SEED}" --shared_bg_seed "${SHARED_BG_SEED}"
    --device cuda --save_references
    --temporal_guidance_space latent
    --temporal_guidance_scale "${TEMPORAL_GUIDANCE_SCALE}"
    --temporal_start_step "${TEMPORAL_START_STEP}"
    --temporal_end_step "${TEMPORAL_END_STEP}"
    --temporal_every_n_steps "${TEMPORAL_EVERY_N_STEPS}"
    --temporal_decode_chunk_size 1
    --temporal_loss_scale "${TEMPORAL_LOSS_SCALE}"
    --temporal_loss_type "${TEMPORAL_LOSS_TYPE}"
    --temporal_visibility_alpha "${TEMPORAL_VISIBILITY_ALPHA}"
    --temporal_visibility_beta "${TEMPORAL_VISIBILITY_BETA}"
    --temporal_flow_batch_size "${RAFT_PAIR_BATCH_SIZE}"
    --num_shards 1 --shard_index 0
)
if [[ "${TEMPORAL_DETACH_PREVIOUS}" == "1" ]]; then
    COMMON_ARGS+=(--temporal_detach_previous)
else
    COMMON_ARGS+=(--no_temporal_detach_previous)
fi
if [[ -n "${DEFORMABLE_ALIGNMENT_DIRECTION:-}" ]]; then
    COMMON_ARGS+=(
        --deformable_alignment_direction "${DEFORMABLE_ALIGNMENT_DIRECTION}"
    )
fi
if [[ "${OVERWRITE}" == "1" ]]; then
    COMMON_ARGS+=(--overwrite)
fi
if [[ -n "${SEQUENCE_NAME}" ]]; then
    COMMON_ARGS+=(--include_branches "${SEQUENCE_NAME}")
fi

mkdir -p "${OUTPUT_DIR}/terminal_logs"
timestamp="$(date +%Y%m%d_%H%M%S)"
terminal_log="${OUTPUT_DIR}/terminal_logs/evaluate_latent_tg_${timestamp}.log"
exec > >(tee -a "${terminal_log}") 2>&1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

echo "V8-R latent temporal guidance evaluation"
echo "checkpoint=${CHECKPOINT_STEP}, model=${STC_V8R_MODEL}"
echo "temporal_space=latent, scale=${TEMPORAL_GUIDANCE_SCALE}, window=[${TEMPORAL_START_STEP},${TEMPORAL_END_STEP})"
echo "test_set=${TEST_SET}, sequence=${SEQUENCE_NAME:-all}"
echo "CGE=off, output=${OUTPUT_DIR}"

cd "${BRUSHNET_ROOT}"
for index in "${!roots[@]}"; do
    root="${roots[${index}]}"
    label="${labels[${index}]}"
    echo "Starting ${label}: ${root}"
    if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
        "${PYTHON_BIN}" "${EVAL_SCRIPT}" "${COMMON_ARGS[@]}" \
            --dataset_root "${root}" --output_dir "${OUTPUT_DIR}" \
            --preflight_only "$@"
    fi
    "${PYTHON_BIN}" "${EVAL_SCRIPT}" "${COMMON_ARGS[@]}" \
        --dataset_root "${root}" --output_dir "${OUTPUT_DIR}" "$@"
    cp "${OUTPUT_DIR}/run_config.json" \
        "${OUTPUT_DIR}/run_config_${label}.json"
done

echo "Combined V8-R latent-guidance evaluation complete: ${OUTPUT_DIR}/summary.json"
