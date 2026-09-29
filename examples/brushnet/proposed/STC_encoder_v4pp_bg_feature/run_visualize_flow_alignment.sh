#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"

CHECKPOINT="${CHECKPOINT:-${BRUSHNET_ROOT}/experiments/train_stc_v4pp_sflow01_sfeat01/checkpoint-5000}"
DATASET_ROOT="${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
TEACHER_FLOW_ROOT="${TEACHER_FLOW_ROOT:-${DATASET_ROOT}/teacher_flows_512x512}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-${BRUSHNET_ROOT}/examples/brushnet/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5}"
SPLIT="${SPLIT:-valid}"
CLIP_LENGTH="${CLIP_LENGTH:-16}"
CLIP_STRIDE="${CLIP_STRIDE:-12}"
CLIPS_PER_SEQUENCE="${CLIPS_PER_SEQUENCE:-1}"
PAIRS_PER_CLIP="${PAIRS_PER_CLIP:-2}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/visualize_v4pp_flow_alignment/$(basename "${CHECKPOINT}")-${SPLIT}}"

cd "${BRUSHNET_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/visualize_flow_alignment.py" \
  --checkpoint "${CHECKPOINT}" \
  --dataset_root "${DATASET_ROOT}" \
  --teacher_flow_root "${TEACHER_FLOW_ROOT}" \
  --pretrained_model_name_or_path "${PRETRAINED_MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --clip_length "${CLIP_LENGTH}" \
  --clip_stride "${CLIP_STRIDE}" \
  --clips_per_sequence "${CLIPS_PER_SEQUENCE}" \
  --pairs_per_clip "${PAIRS_PER_CLIP}" \
  --device "${DEVICE}" \
  "$@"
