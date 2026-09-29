#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:?Set CHECKPOINT_STEP for the standalone stage-2 checkpoint}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ROOT}/experiments/train_v8r_standalone/checkpoint-${CHECKPOINT_STEP}}"
STC_MODEL="${STC_MODEL:-${CHECKPOINT_PATH}/stc_v8r_standalone_model}"
RAFT_STUDENT_PATH="${RAFT_STUDENT_PATH:-${ROOT}/experiments/train_v7_raft_student_flow/best.json}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${ROOT}/experiments/train_sharedNoise_sameBG_0.95_T8/checkpoint-2250}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-${ROOT}/examples/brushnet/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5}"
TEST_SET="${TEST_SET:-test_1}"
case "${TEST_SET}" in
  test_1|test_2) DATASET_ROOT="${DATASET_ROOT:-${ROOT}/examples/brushnet/dataset/${TEST_SET}}" ;;
  *) echo "TEST_SET must be test_1 or test_2" >&2; exit 2 ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/experiments/eval_v8r_standalone_ckpt${CHECKPOINT_STEP}_${TEST_SET}}"
for item in "${STC_MODEL}/config.json" "${RAFT_STUDENT_PATH}" \
  "${BASELINE_CHECKPOINT}/brushnet/config.json" "${DATASET_ROOT}"; do
  [[ -e "${item}" ]] || { echo "Missing required path: ${item}" >&2; exit 1; }
done
ARGS=(--pretrained_model_name_or_path "${PRETRAINED_MODEL}"
      --baseline_checkpoint "${BASELINE_CHECKPOINT}"
      --stc_adapter_path "${STC_MODEL}"
      --raft_student_path "${RAFT_STUDENT_PATH}"
      --raft_pair_batch_size "${RAFT_PAIR_BATCH_SIZE:-1}"
      --dataset_layout flat_test --split test --dataset_root "${DATASET_ROOT}"
      --image_encoder_name_or_path "${IMAGE_ENCODER:-laion/CLIP-ViT-H-14-laion2B-s32B-b79K}"
      --output_dir "${OUTPUT_DIR}" --resolution 512
      --clip_length "${CLIP_LENGTH:-16}" --clip_stride "${CLIP_STRIDE:-12}"
      --num_inference_steps "${NUM_INFERENCE_STEPS:-50}"
      --guidance_scale "${GUIDANCE_SCALE:-7.5}"
      --brushnet_conditioning_scale "${BRUSHNET_SCALE:-1.0}"
      --stc_injection_scale "${STC_INJECTION_SCALE:-1.0}"
      --deformable_alignment_scale "${DEFORMABLE_ALIGNMENT_SCALE:-1.0}"
      --fusion_scale "${FUSION_SCALE:-1.0}"
      --shared_bg_noise_strength "${SHARED_BG_NOISE_STRENGTH:-0.95}"
      --roi_composite "${ROI_COMPOSITE:-blurred}"
      --roi_blur_kernel_size "${ROI_BLUR_KERNEL_SIZE:-51}"
      --seed "${SEED:-1234}" --shared_bg_seed "${SHARED_BG_SEED:-6789}"
      --device cuda --save_references
      --temporal_guidance_space "${TEMPORAL_GUIDANCE_SPACE:-latent}"
      --temporal_guidance_scale "${TEMPORAL_GUIDANCE_SCALE:-0}"
      --temporal_start_step "${TEMPORAL_START_STEP:-0}"
      --temporal_end_step "${TEMPORAL_END_STEP:-50}"
      --temporal_every_n_steps "${TEMPORAL_EVERY_N_STEPS:-1}"
      --temporal_loss_scale "${TEMPORAL_LOSS_SCALE:-1024}"
      --temporal_loss_type "${TEMPORAL_LOSS_TYPE:-l2}"
      --temporal_decode_chunk_size "${TEMPORAL_DECODE_CHUNK_SIZE:-1}"
      --temporal_flow_batch_size "${RAFT_PAIR_BATCH_SIZE:-1}"
      --num_shards 1 --shard_index 0)
if [[ -n "${SEQUENCE_NAME:-}" ]]; then ARGS+=(--include_branches "${SEQUENCE_NAME}"); fi
if [[ -n "${DEFORMABLE_ALIGNMENT_DIRECTION:-}" ]]; then
  ARGS+=(--deformable_alignment_direction "${DEFORMABLE_ALIGNMENT_DIRECTION}")
fi
if [[ "${TEMPORAL_DETACH_PREVIOUS:-1}" == 1 ]]; then
  ARGS+=(--temporal_detach_previous)
else
  ARGS+=(--no_temporal_detach_previous)
fi
if [[ "${OVERWRITE:-0}" == 1 ]]; then ARGS+=(--overwrite); fi
mkdir -p "${OUTPUT_DIR}"
cd "${ROOT}"
if [[ "${RUN_PREFLIGHT:-1}" == 1 ]]; then
  "${PYTHON_BIN}" "${HERE}/evaluate_temporal.py" "${ARGS[@]}" --preflight_only "$@"
fi
exec "${PYTHON_BIN}" "${HERE}/evaluate_temporal.py" "${ARGS[@]}" "$@"
