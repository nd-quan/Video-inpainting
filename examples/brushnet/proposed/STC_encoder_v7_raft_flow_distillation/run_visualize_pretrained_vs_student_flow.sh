#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
CHECKPOINT="${CHECKPOINT:-${BRUSHNET_ROOT}/experiments/train_v7_raft_student_flow/checkpoint-0004750/raft_student}"
DATASET_ROOT="${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
TEACHER_FLOW_ROOT="${TEACHER_FLOW_ROOT:-${DATASET_ROOT}/teacher_flows_512x512}"
PAIR_CSV="${PAIR_CSV:-${BRUSHNET_ROOT}/experiments/visualize_v7_raft_student_best_valid/per_pair_metrics.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/visualize_raft_pretrained_vs_v7_student4750_valid}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/visualize_pretrained_vs_student_flow.py" \
    --checkpoint "${CHECKPOINT}" --dataset_root "${DATASET_ROOT}" \
    --teacher_flow_root "${TEACHER_FLOW_ROOT}" --pair_csv "${PAIR_CSV}" \
    --output_dir "${OUTPUT_DIR}" --split valid --resolution 512 "$@"
