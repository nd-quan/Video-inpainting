#!/usr/bin/env bash
# Cache frozen SEA-RAFT pseudo flow on clean VCM/STC frames for V7.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"

DATASET_ROOT="${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
SEA_RAFT_ROOT="${SEA_RAFT_ROOT:-/home/cilab/ndquan/videoInpainting/pretrained/SEA-RAFT}"
SEA_RAFT_CFG="${SEA_RAFT_CFG:-${SEA_RAFT_ROOT}/config/eval/spring-M.json}"
SEA_RAFT_CHECKPOINT="${SEA_RAFT_CHECKPOINT:?Set SEA_RAFT_CHECKPOINT to a SEA-RAFT checkpoint (.safetensors or .pth)}"
TEACHER_FLOW_ROOT="${TEACHER_FLOW_ROOT:-${DATASET_ROOT}/teacher_flows_sea_raft_512x512}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
SEA_RAFT_ITERS="${SEA_RAFT_ITERS:-4}"
DEVICE="${DEVICE:-cuda}"
SPLITS="${SPLITS:-train valid}"
RESUME="${RESUME:-1}"

for required in \
  "${PYTHON_BIN}" \
  "${SCRIPT_DIR}/precompute_sfu_stc_sea_raft_teacher_flows.py" \
  "${DATASET_ROOT}/manifest.json" \
  "${SEA_RAFT_ROOT}/core/raft.py" \
  "${SEA_RAFT_CFG}" \
  "${SEA_RAFT_CHECKPOINT}"; do
  [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 1; }
done
if (( HEIGHT <= 0 || WIDTH <= 0 || HEIGHT % 8 || WIDTH % 8 )); then
  echo "HEIGHT and WIDTH must be positive multiples of 8" >&2
  exit 2
fi
if (( SEA_RAFT_ITERS < 1 )); then
  echo "SEA_RAFT_ITERS must be positive" >&2
  exit 2
fi

ARGS=(
  --dataset-root "${DATASET_ROOT}"
  --output-root "${TEACHER_FLOW_ROOT}"
  --sea-raft-root "${SEA_RAFT_ROOT}"
  --cfg "${SEA_RAFT_CFG}"
  --checkpoint "${SEA_RAFT_CHECKPOINT}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --iters "${SEA_RAFT_ITERS}"
  --device "${DEVICE}"
  --splits ${SPLITS}
)
if [[ "${RESUME}" == "1" ]]; then
  ARGS+=(--resume)
fi

cd "${BRUSHNET_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/precompute_sfu_stc_sea_raft_teacher_flows.py" "${ARGS[@]}" "$@"
