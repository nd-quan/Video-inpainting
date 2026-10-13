#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${BRUSHNET_ROOT}/examples/brushnet:${BRUSHNET_ROOT}/examples/brushnet/proposed${PYTHONPATH:+:${PYTHONPATH}}"
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29731}"
export OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/train_stc_v8_rescaled_x4_joint_teacherflow_T16_S12_sharedNoise_0.95}"
export RAFT_STUDENT_PATH="${RAFT_STUDENT_PATH:-${BRUSHNET_ROOT}/experiments/train_v7_raft_student_flow/checkpoint-0004750/raft_student}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-4000}"

exec "${SCRIPT_DIR}/run_train_v8_rescaled_joint_t16.sh" \
    --condition_flow_source teacher "$@"
