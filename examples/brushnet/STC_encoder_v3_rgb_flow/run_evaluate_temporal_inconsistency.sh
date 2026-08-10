#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
EVAL_ROOT="${EVAL_ROOT:-${BRUSHNET_DIR}/experiments/eval_rgb_stc_v2/valid/checkpoint-2000-sBG-09}"
DATASET_ROOT="${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
TEACHER_FLOW_ROOT="${TEACHER_FLOW_ROOT:-${DATASET_ROOT}/teacher_flows_512x512}"
SPLIT="${SPLIT:-valid}"
FRAME_KIND="${FRAME_KIND:-final}"
OVERLAP_MODE="${OVERLAP_MODE:-per_clip}"
SCALES="${SCALES:-8,16,32}"
PRIMARY_SCALE="${PRIMARY_SCALE:-16}"
DEVICE="${DEVICE:-cpu}"
VISUALIZATIONS_PER_UNIT="${VISUALIZATIONS_PER_UNIT:-1}"

ARGS=(
  --eval_root "${EVAL_ROOT}"
  --dataset_root "${DATASET_ROOT}"
  --teacher_flow_root "${TEACHER_FLOW_ROOT}"
  --split "${SPLIT}"
  --frame_kind "${FRAME_KIND}"
  --overlap_mode "${OVERLAP_MODE}"
  --scales "${SCALES}"
  --primary_scale "${PRIMARY_SCALE}"
  --device "${DEVICE}"
  --visualizations_per_unit "${VISUALIZATIONS_PER_UNIT}"
)

if [[ -n "${OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output_dir "${OUTPUT_DIR}")
fi
if [[ -n "${VIDEO_FILTER:-}" ]]; then
  ARGS+=(--video_filter "${VIDEO_FILTER}")
fi
if [[ -n "${MAX_UNITS:-}" ]]; then
  ARGS+=(--max_units "${MAX_UNITS}")
fi
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  ARGS+=(--preflight_only)
fi
if [[ "${ALLOW_MISSING:-0}" == "1" ]]; then
  ARGS+=(--allow_missing)
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

cd "${BRUSHNET_DIR}"
exec "${PYTHON_BIN}" \
  examples/brushnet/STC_encoder_v3_rgb_flow/evaluate_temporal_inconsistency.py \
  "${ARGS[@]}"
