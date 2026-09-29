#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
GPU_ID="${GPU_ID:-5}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export LATENT_ANALYSIS_GPU_ID="${GPU_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/experiments/v8_latent_warp_bg_only_test1}"
MODEL="${STC_V8_MODEL:-${ROOT}/experiments/train_stc_v8_raft_joint_from7500_T16_S12_sharedNoise_0.95_constant5e6/checkpoint-4000/stc_v8_model}"
mkdir -p "${OUTPUT_DIR}/terminal_logs"
cd "${ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_v8_latent_warp_bg_only_analysis.py" \
  --dataset_root "${ROOT}/examples/brushnet/dataset/test_1" --dataset_layout flat_test --split test \
  --baseline_checkpoint "${ROOT}/experiments/train_sharedNoise_sameBG_0.95_T8/checkpoint-2250" \
  --stc_adapter_path "${MODEL}" \
  --raft_student_path "${ROOT}/experiments/train_v7_raft_student_flow/checkpoint-0004750/raft_student" \
  --output_dir "${OUTPUT_DIR}" --clip_length 16 --clip_stride 12 \
  --num_inference_steps 50 --shared_bg_noise_strength 0.95 --seed 1234 --shared_bg_seed 6789 \
  --roi_composite blurred --roi_blur_kernel_size 51 --save_references \
  --latent_capture_steps "${LATENT_CAPTURE_STEPS:-0,10,25,40,49}" \
  --latent_warp_scale "${LATENT_WARP_SCALE:-4}" "$@" \
  2>&1 | tee "${OUTPUT_DIR}/terminal_logs/run_$(date +%Y%m%d_%H%M%S).log"
