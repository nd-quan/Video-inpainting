#!/usr/bin/env bash
# Ablation launcher for spatial CGE, latent temporal guidance, or their sum.
#
# GUIDANCE_MODE=temporal  -> g_temp only
# GUIDANCE_MODE=cge       -> g_CGE only
# GUIDANCE_MODE=combined  -> g_CGE + g_temp in one DDIM transition

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

GUIDANCE_MODE="${GUIDANCE_MODE:-combined}"
V7_RAFT_STUDENT_PATH="${V7_RAFT_STUDENT_PATH:-${BRUSHNET_ROOT}/experiments/train_v7_raft_student_flow/best.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BRUSHNET_ROOT}/experiments/eval_fixedbg_cge_temporal_v7_pair_bg_latent_${GUIDANCE_MODE}}"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"

export GUIDANCE_MODE
export TEMPORAL_FLOW_BACKEND="v7_student"
export TEMPORAL_BG_MASK_MODE="pair_intersection"
export TEMPORAL_GUIDANCE_SPACE="latent"
export V7_RAFT_STUDENT_PATH
export OUTPUT_ROOT

# Defaults make spatial and temporal terms act on the same late DDIM window.
export TEMPORAL_START_STEP="${TEMPORAL_START_STEP:-25}"
export TEMPORAL_END_STEP="${TEMPORAL_END_STEP:-35}"
export TEMPORAL_GUIDANCE_SCALE="${TEMPORAL_GUIDANCE_SCALE:-0.0001}"
export CGE_START_STEP="${CGE_START_STEP:-25}"
export CGE_END_STEP="${CGE_END_STEP:-35}"
export CGE_EVERY_N_STEPS="${CGE_EVERY_N_STEPS:-1}"
export CGE_GUIDANCE_SCALE="${CGE_GUIDANCE_SCALE:-0.0001}"
export CGE_CODEC_MODE="${CGE_CODEC_MODE:-bg_only}"

mkdir -p "${OUTPUT_ROOT}/terminal_logs"
LOG_PATH="${OUTPUT_ROOT}/terminal_logs/evaluate_$(date +%Y%m%d_%H%M%S).log"

cd "${BRUSHNET_ROOT}"
"${PYTHON_BIN}" -u \
  examples/brushnet/test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion_fixedBG_temporal_v0.py \
  2>&1 | tee "${LOG_PATH}"

