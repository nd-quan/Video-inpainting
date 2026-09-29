#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the stage-1 RAFT training directory}"
cd "${ROOT}"
exec "${PYTHON_BIN}" "${HERE}/evaluate_flow.py" \
  --dataset_root "${DATASET_ROOT}" \
  --teacher_flow_root "${TEACHER_FLOW_ROOT:-${DATASET_ROOT}/teacher_flows_512x512}" \
  --output_dir "${OUTPUT_DIR}" \
  --propainter_root "${PROPAINTER_ROOT:-/home/cilab/ndquan/videoInpainting/pretrained/ProPainter}" \
  --raft_checkpoint "${RAFT_CHECKPOINT:-/home/cilab/ndquan/videoInpainting/pretrained/ProPainter/weights/raft-things.pth}" \
  --resolution "${RESOLUTION:-512}" --valid_split "${SPLIT:-valid}" \
  --raft_pair_batch_size "${RAFT_PAIR_BATCH_SIZE:-1}" --mixed_precision fp16 \
  --evaluate_only "${CHECKPOINT:-best}" "$@"
