#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
CHECKPOINT_STEP="${CHECKPOINT_STEP:?Set CHECKPOINT_STEP, for example 2500}"
OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/eval_stc_v8_rescaled_x4/checkpoint-${CHECKPOINT_STEP}-original-size-testsets}"
export OUTPUT_DIR
export PYTHONPATH="${BRUSHNET_ROOT}/examples/brushnet${PYTHONPATH:+:${PYTHONPATH}}"

exec bash "${SCRIPT_DIR}/run_evaluate_v8_rescaled_combined_testsets.sh" --original_size "$@"
