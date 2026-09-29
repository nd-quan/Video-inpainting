#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
"${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}" \
  "${SCRIPT_DIR}/visualize_student_flow_before_after.py" \
  --v8_checkpoint "${V8_CHECKPOINT:-${ROOT}/experiments/train_stc_v8_raft_joint_from7500_T16_S12_sharedNoise_0.95_constant5e6/checkpoint-4000/stc_v8_model}" \
  --output_dir "${OUTPUT_DIR:-${ROOT}/experiments/visualize_student_flow_before_after_joint4000}" \
  --tile_gap "${TILE_GAP:-6}" "$@"
