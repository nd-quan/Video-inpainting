#!/usr/bin/env bash
# Evaluate fixed-BG temporal DDIM guidance with the frozen V7 RAFT student.
# The temporal-v0 Python entry point remains the single implementation; these
# defaults select only the V7-flow / pair-BG-intersection ablation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

V7_RAFT_STUDENT_PATH="${V7_RAFT_STUDENT_PATH:-${BRUSHNET_ROOT}/experiments/train_v7_raft_student_flow/best.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BRUSHNET_ROOT}/experiments/eval_fixedbg_temporal_v7_pair_bg}"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"

export TEMPORAL_FLOW_BACKEND="v7_student"
export TEMPORAL_BG_MASK_MODE="pair_intersection"
export V7_RAFT_STUDENT_PATH
export OUTPUT_ROOT

mkdir -p "${OUTPUT_ROOT}/terminal_logs"
LOG_PATH="${OUTPUT_ROOT}/terminal_logs/evaluate_$(date +%Y%m%d_%H%M%S).log"

cd "${BRUSHNET_ROOT}"
"${PYTHON_BIN}" -u \
  examples/brushnet/test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion_fixedBG_temporal_v0.py \
  2>&1 | tee "${LOG_PATH}"

