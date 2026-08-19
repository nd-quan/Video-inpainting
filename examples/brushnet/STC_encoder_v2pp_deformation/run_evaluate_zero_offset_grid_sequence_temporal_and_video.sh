#!/usr/bin/env bash
# Evaluate zero-grid sequence-state V2++ with RAFT temporal guidance, then
# stitch the authoritative first-owner ``final`` images into per-sequence MP4s.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"

export OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/eval_stc_v2pp_T8_cosine_0.9/test/checkpoint-2000-zero-offset-grid-sequence-state-blurred-k51-temporal-raft}"

bash "${SCRIPT_DIR}/run_evaluate_zero_offset_grid_sequence_temporal.sh" "$@"

exec "${PYTHON_BIN}" "${BRUSHNET_ROOT}/Quan_test/imgToVideo_rgb_stc_eval.py" \
    --eval_root "${OUTPUT_DIR}" \
    --frame_kinds final \
    --selection first
