#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/accelerate}"
WARP_MODE="${WARP_MODE:?Set WARP_MODE=direct or dgaf, or use a version launcher}"
USE_CONFIDENCE_MASK="${USE_CONFIDENCE_MASK:-1}"
if [[ "$USE_CONFIDENCE_MASK" != 0 && "$USE_CONFIDENCE_MASK" != 1 ]]; then
    echo "USE_CONFIDENCE_MASK must be 0 or 1" >&2
    exit 2
fi
CONFIDENCE_ARG=--use_confidence_mask
if [[ "$USE_CONFIDENCE_MASK" == 0 ]]; then CONFIDENCE_ARG=--no-use_confidence_mask; fi
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/experiments/train_dgaf_${WARP_MODE}_conf${USE_CONFIDENCE_MASK}_T16_S12_095}"
ARGS=(--warp_mode "$WARP_MODE" "$CONFIDENCE_ARG" --output_dir "$OUTPUT_DIR"
      --upscale_factor "${UPSCALE_FACTOR:-4}"
      --clip_length "${CLIP_LENGTH:-16}" --clip_stride "${CLIP_STRIDE:-12}"
      --resolution "${RESOLUTION:-512}"
      --train_batch_size "${CLIPS_PER_DEVICE:-1}"
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-3}"
      --learning_rate "${LEARNING_RATE:-1e-5}" --max_train_steps "${MAX_TRAIN_STEPS:-2000}"
      --mixed_precision "${MIXED_PRECISION:-fp16}"
      --checkpointing_steps "${CHECKPOINTING_STEPS:-250}"
      --shared_bg_noise_strength "${SHARED_BG_NOISE_STRENGTH:-0.95}")
if [[ -n "${DATASET_ROOT:-}" ]]; then ARGS+=(--dataset_root "$DATASET_ROOT"); fi
if [[ -n "${BASELINE_CHECKPOINT:-}" ]]; then ARGS+=(--baseline_checkpoint "$BASELINE_CHECKPOINT"); fi
if [[ -n "${RAFT_STUDENT_PATH:-}" ]]; then ARGS+=(--raft_student_path "$RAFT_STUDENT_PATH"); fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"); fi
cd "$REPO_ROOT"
if [[ "${RUN_PREFLIGHT:-1}" == 1 ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/train_dgaf.py" "${ARGS[@]}" "$@" --preflight_only
fi
NUM_PROCESSES="${NUM_PROCESSES:-1}"
if [[ "$NUM_PROCESSES" == 1 ]]; then
    exec "$PYTHON_BIN" "$SCRIPT_DIR/train_dgaf.py" "${ARGS[@]}" "$@"
else
    exec "$ACCELERATE_BIN" launch --multi_gpu --num_processes "$NUM_PROCESSES" \
        --main_process_port "${MAIN_PROCESS_PORT:-29680}" \
        "$SCRIPT_DIR/train_dgaf.py" "${ARGS[@]}" "$@"
fi
