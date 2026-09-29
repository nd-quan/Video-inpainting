#!/usr/bin/env bash
# Evaluate a saved SEA-RAFT V7 student against its cached clean teacher flow.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
TEACHER_FLOW_ROOT="${TEACHER_FLOW_ROOT:-${DATASET_ROOT}/teacher_flows_sea_raft_512x512}"
SEA_RAFT_ROOT="${SEA_RAFT_ROOT:-/home/cilab/ndquan/videoInpainting/pretrained/SEA-RAFT}"
SEA_RAFT_CFG="${SEA_RAFT_CFG:-${SEA_RAFT_ROOT}/config/eval/spring-M.json}"
SEA_RAFT_CHECKPOINT="${SEA_RAFT_CHECKPOINT:?Set SEA_RAFT_CHECKPOINT to the SEA-RAFT initialization checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the SEA-RAFT student experiment directory}"
CHECKPOINT="${CHECKPOINT:-best}"
SPLIT="${SPLIT:-valid}"
RESOLUTION="${RESOLUTION:-512}"
RAFT_PAIR_BATCH_SIZE="${RAFT_PAIR_BATCH_SIZE:-1}"

cd "${BRUSHNET_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/train_sea_raft_student_flow.py" \
  --dataset_root "${DATASET_ROOT}" \
  --teacher_flow_root "${TEACHER_FLOW_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --sea_raft_root "${SEA_RAFT_ROOT}" \
  --sea_raft_cfg "${SEA_RAFT_CFG}" \
  --sea_raft_checkpoint "${SEA_RAFT_CHECKPOINT}" \
  --train_split train \
  --valid_split "${SPLIT}" \
  --resolution "${RESOLUTION}" \
  --raft_pair_batch_size "${RAFT_PAIR_BATCH_SIZE}" \
  --mixed_precision fp16 \
  --evaluate_only "${CHECKPOINT}" \
  "$@"
