#!/usr/bin/env bash
# Render a V7 RAFT student against cached clean-RAFT teacher flow.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a V7 experiment root, checkpoint-N, or best.json}"
DATASET_ROOT="${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
TEACHER_FLOW_ROOT="${TEACHER_FLOW_ROOT:-${DATASET_ROOT}/teacher_flows_512x512}"
SPLIT="${SPLIT:-valid}"
RESOLUTION="${RESOLUTION:-512}"
PAIRS_PER_SEQUENCE="${PAIRS_PER_SEQUENCE:-2}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/visualize_v7_raft_student_flow/$(basename "${CHECKPOINT}")-${SPLIT}}"
OVERWRITE="${OVERWRITE:-0}"

for required in "${PYTHON_BIN}" "${SCRIPT_DIR}/visualize_raft_student_flow.py" "${CHECKPOINT}" "${DATASET_ROOT}/manifest.json" "${TEACHER_FLOW_ROOT}/metadata.json"; do
  [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 1; }
done

args=(--checkpoint "${CHECKPOINT}" --dataset_root "${DATASET_ROOT}" --teacher_flow_root "${TEACHER_FLOW_ROOT}" --output_dir "${OUTPUT_DIR}" --split "${SPLIT}" --resolution "${RESOLUTION}" --pairs_per_sequence "${PAIRS_PER_SEQUENCE}" --device "${DEVICE}")
if [[ "${OVERWRITE}" == "1" ]]; then args+=(--overwrite); fi
mkdir -p "${OUTPUT_DIR}/terminal_logs"
timestamp="$(date +%Y%m%d_%H%M%S)"
terminal_log="${OUTPUT_DIR}/terminal_logs/visualize_${timestamp}.log"
exec > >(tee -a "${terminal_log}") 2>&1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
echo "V7 RAFT student flow visualization"
echo "checkpoint=${CHECKPOINT}; dataset=${DATASET_ROOT}; split=${SPLIT}; pairs=${PAIRS_PER_SEQUENCE}"
echo "output=${OUTPUT_DIR}; terminal_log=${terminal_log}"
cd "${BRUSHNET_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/visualize_raft_student_flow.py" "${args[@]}" "$@"
