#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${BRUSHNET_ROOT}/experiments/train_stc_v2pp_T8_cosine_0.9/checkpoint-2000}"
export DATASET_ROOT="${DATASET_ROOT:-${BRUSHNET_ROOT}/examples/brushnet/dataset/test}"
export OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/eval_stc_v2pp_T8_cosine_0.9/test/checkpoint-2000-sequence-state-blurred-k21}"
export ROI_COMPOSITE="${ROI_COMPOSITE:-blurred}"
export ROI_BLUR_KERNEL_SIZE="${ROI_BLUR_KERNEL_SIZE:-21}"

exec bash "${SCRIPT_DIR}/run_evaluate_noise_deformation.sh" \
    --state_scope sequence \
    --offset_mode learned \
    "$@"
