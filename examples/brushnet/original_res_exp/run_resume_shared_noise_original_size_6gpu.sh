#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(realpath "${SCRIPT_DIR}/../../experiments/finetune_shared_noise_original_size")"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NUM_PROCESSES=6
export CLIP_LENGTH=6
export CLIP_STRIDE=4
export GRADIENT_ACCUMULATION_STEPS=4
export MAX_TRAIN_STEPS=1000
export CHECKPOINTING_STEPS=250
export OUTPUT_DIR="${EXPERIMENT_DIR}"

bash "${SCRIPT_DIR}/run_finetune_shared_noise_original_size.sh" \
    --resume_from_checkpoint "${EXPERIMENT_DIR}/checkpoint-500" \
    2>&1 | tee -a "${EXPERIMENT_DIR}/train_original_size_6gpu.log"
