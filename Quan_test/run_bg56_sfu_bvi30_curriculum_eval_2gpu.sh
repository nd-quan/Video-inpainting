#!/usr/bin/env bash
# Full BG56 evaluation for the SFU+BVI independent/shared-noise curriculum.
# Use PREPARE_ONLY=1 for CPU preflight; default candidate is checkpoint-6000.
set -euo pipefail

BRUSHNET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_RUN_ROOT="${MODEL_RUN_ROOT:-/user/Cilab/Quan/experiment/finetune_sfu_bvi30_bg56_512_from_coco8000_T8S4_ind5000_shared3000_ddp4}"
export BG_QP=56
export CHECKPOINT_STEP="${CHECKPOINT_STEP:-6000}"
export CHECKPOINT_DIR="${MODEL_RUN_ROOT}/checkpoint-${CHECKPOINT_STEP}"
export LONG_TEST_ROOT=/home/cilab/ndquan/vcm-rs/output/sfu_qp0_bg_table_window_test/bg56/test
export OUTPUT_ROOT="${OUTPUT_ROOT:-${BRUSHNET_ROOT}/experiments/EVAL_FOR_FINAL/eval_bg56_sfu_bvi30_ind5000_shared3000_ckpt${CHECKPOINT_STEP}_latent_tg0p1_v7student_T8S2_512}"
export EVAL_GPU_0="${EVAL_GPU_0:-0}"
export EVAL_GPU_1="${EVAL_GPU_1:-1}"

exec bash "${BRUSHNET_ROOT}/Quan_test/run_bg56_sfu_ft_latent_tg_v7_2gpu.sh"
