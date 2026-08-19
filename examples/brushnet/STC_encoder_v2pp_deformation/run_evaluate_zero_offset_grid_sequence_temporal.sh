#!/usr/bin/env bash
# Run V2++ zero-grid sequence-state evaluation with RAFT stable-BG temporal
# DDIM guidance.  Override DATASET_ROOT/OUTPUT_DIR for each independent run.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${BRUSHNET_ROOT}/experiments/train_stc_v2pp_T8_cosine_0.9/checkpoint-2000}"
export DATASET_ROOT="${DATASET_ROOT:-${BRUSHNET_ROOT}/examples/brushnet/dataset/test}"
export OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/eval_stc_v2pp_T8_cosine_0.9/test/checkpoint-2000-zero-offset-grid-sequence-state-blurred-k51-temporal-raft}"

# Retain the currently used final compositing protocol so this measures the
# effect of temporal guidance rather than changing the boundary blend as well.
export ROI_COMPOSITE="${ROI_COMPOSITE:-blurred}"
export ROI_BLUR_KERNEL_SIZE="${ROI_BLUR_KERNEL_SIZE:-51}"

# Defaults match fixedBG_temporal_v0.  Guidance is applied to full T=8 clip
# batches so it includes overlap -> new-frame pairs; output ownership remains
# the evaluator's first-owner policy.
export TEMPORAL_GUIDANCE_SCALE="${TEMPORAL_GUIDANCE_SCALE:-0.0001}"
export TEMPORAL_START_STEP="${TEMPORAL_START_STEP:-15}"
export TEMPORAL_END_STEP="${TEMPORAL_END_STEP:-35}"
export TEMPORAL_EVERY_N_STEPS="${TEMPORAL_EVERY_N_STEPS:-1}"
export TEMPORAL_DECODE_CHUNK_SIZE="${TEMPORAL_DECODE_CHUNK_SIZE:-1}"
export TEMPORAL_LOSS_SCALE="${TEMPORAL_LOSS_SCALE:-1024}"
export TEMPORAL_FLOW_BATCH_SIZE="${TEMPORAL_FLOW_BATCH_SIZE:-2}"
export TEMPORAL_SAMPLING_SCOPE="${TEMPORAL_SAMPLING_SCOPE:-full_clip}"

exec bash "${SCRIPT_DIR}/run_evaluate_noise_deformation.sh" \
    --state_scope sequence \
    --offset_mode zero_grid \
    --temporal_guidance_scale "${TEMPORAL_GUIDANCE_SCALE}" \
    --temporal_start_step "${TEMPORAL_START_STEP}" \
    --temporal_end_step "${TEMPORAL_END_STEP}" \
    --temporal_every_n_steps "${TEMPORAL_EVERY_N_STEPS}" \
    --temporal_decode_chunk_size "${TEMPORAL_DECODE_CHUNK_SIZE}" \
    --temporal_loss_scale "${TEMPORAL_LOSS_SCALE}" \
    --temporal_flow_batch_size "${TEMPORAL_FLOW_BATCH_SIZE}" \
    --temporal_sampling_scope "${TEMPORAL_SAMPLING_SCOPE}" \
    "$@"
