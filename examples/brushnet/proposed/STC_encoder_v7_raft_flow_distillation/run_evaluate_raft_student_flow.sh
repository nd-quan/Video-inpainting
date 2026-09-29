#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
TEACHER_FLOW_ROOT="${TEACHER_FLOW_ROOT:-${DATASET_ROOT}/teacher_flows_512x512}"
PROPAINTER_ROOT="${PROPAINTER_ROOT:-/home/cilab/ndquan/videoInpainting/pretrained/ProPainter}"
RAFT_CHECKPOINT="${RAFT_CHECKPOINT:-${PROPAINTER_ROOT}/weights/raft-things.pth}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the training experiment directory}"
CHECKPOINT="${CHECKPOINT:-best}"
SPLIT="${SPLIT:-valid}"
RESOLUTION="${RESOLUTION:-512}"
RAFT_PAIR_BATCH_SIZE="${RAFT_PAIR_BATCH_SIZE:-1}"

cd "${BRUSHNET_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/train_raft_student_flow.py" \
  --dataset_root "${DATASET_ROOT}" \
  --teacher_flow_root "${TEACHER_FLOW_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --propainter_root "${PROPAINTER_ROOT}" \
  --raft_checkpoint "${RAFT_CHECKPOINT}" \
  --train_split train \
  --valid_split "${SPLIT}" \
  --resolution "${RESOLUTION}" \
  --raft_pair_batch_size "${RAFT_PAIR_BATCH_SIZE}" \
  --mixed_precision fp16 \
  --evaluate_only "${CHECKPOINT}" \
  "$@"
