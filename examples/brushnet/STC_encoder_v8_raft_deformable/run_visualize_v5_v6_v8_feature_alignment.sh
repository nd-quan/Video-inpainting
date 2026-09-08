#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# BrushNet's training/evaluation stack lives in guided_diff on this workspace.
# Keep PYTHON_BIN overridable for another host, while avoiding an accidental
# system Python that lacks transformers/diffusers.
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

REFERENCE_DIR="${REFERENCE_DIR:-${BRUSHNET_ROOT}/experiments/visualize_v7_raft_student_best_valid}"
DATASET_ROOT="${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
TEACHER_FLOW_ROOT="${TEACHER_FLOW_ROOT:-${DATASET_ROOT}/teacher_flows_512x512}"
V5_CHECKPOINT="${V5_CHECKPOINT:-${BRUSHNET_ROOT}/experiments/train_stc_v5_relative_crossclip_T16_S12_sharedNoise_0.95/checkpoint-5000/stc_v5_model}"
V6_CHECKPOINT="${V6_CHECKPOINT:-${BRUSHNET_ROOT}/experiments/train_stc_v6_joint_fixedlr_T16_S12_sharedNoise_0.95/checkpoint-1750/stc_v6_model}"
# Match V6's step first.  Override this path to compare a later V8 checkpoint.
V8_CHECKPOINT="${V8_CHECKPOINT:-${BRUSHNET_ROOT}/experiments/train_stc_v8_raft_deform_only_T16_S12_sharedNoise_0.95/checkpoint-1750/stc_v8_model}"
RAFT_STUDENT_PATH="${RAFT_STUDENT_PATH:-${BRUSHNET_ROOT}/experiments/train_v7_raft_student_flow/checkpoint-0004750/raft_student}"
OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/visualize_v5_v6_v8_feature_alignment_step1750}"
DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-valid}"
CLIP_LENGTH="${CLIP_LENGTH:-16}"
CLIP_STRIDE="${CLIP_STRIDE:-12}"
RESOLUTION="${RESOLUTION:-512}"
RAFT_PAIR_BATCH_SIZE="${RAFT_PAIR_BATCH_SIZE:-1}"
TILE_SIZE="${TILE_SIZE:-256}"
DEFORMABLE_ALIGNMENT_SCALE="${DEFORMABLE_ALIGNMENT_SCALE:-1.0}"

EXTRA_ARGS=()
if [[ "${USE_CROSS_CLIP:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--use_cross_clip)
fi
if [[ "${NO_AMP:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no_amp)
fi
if [[ -n "${MAX_PAIRS:-}" ]]; then
  EXTRA_ARGS+=(--max_pairs "${MAX_PAIRS}")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/visualize_v5_v6_v8_feature_alignment.py" \
  --reference_dir "${REFERENCE_DIR}" \
  --dataset_root "${DATASET_ROOT}" \
  --teacher_flow_root "${TEACHER_FLOW_ROOT}" \
  --v5_checkpoint "${V5_CHECKPOINT}" \
  --v6_checkpoint "${V6_CHECKPOINT}" \
  --v8_checkpoint "${V8_CHECKPOINT}" \
  --raft_student_path "${RAFT_STUDENT_PATH}" \
  --split "${SPLIT}" \
  --resolution "${RESOLUTION}" \
  --clip_length "${CLIP_LENGTH}" \
  --clip_stride "${CLIP_STRIDE}" \
  --raft_pair_batch_size "${RAFT_PAIR_BATCH_SIZE}" \
  --deformable_alignment_scale "${DEFORMABLE_ALIGNMENT_SCALE}" \
  --tile_size "${TILE_SIZE}" \
  --device "${DEVICE}" \
  --output_dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
