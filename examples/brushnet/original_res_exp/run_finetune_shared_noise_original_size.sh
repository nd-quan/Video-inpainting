#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
ARGS=(
    --dataset_root "${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
    --train_split train
    --initial_checkpoint "${INITIAL_CHECKPOINT:-${SCRIPT_DIR}/../../experiments/train_sharedNoise_sameBG_0.95_T8/checkpoint-2250}"
    --output_dir "${OUTPUT_DIR:-${SCRIPT_DIR}/../../experiments/finetune_shared_noise_original_size}"
    --clip_length "${CLIP_LENGTH:-4}"
    --clip_stride "${CLIP_STRIDE:-1}"
    --frame_microbatch "${FRAME_MICROBATCH:-1}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}"
    --max_train_steps "${MAX_TRAIN_STEPS:-1000}"
    --checkpointing_steps "${CHECKPOINTING_STEPS:-250}"
    --learning_rate "${LEARNING_RATE:-1e-6}"
    --shared_bg_noise_strength "${SHARED_BG_NOISE_STRENGTH:-0.95}"
    --mixed_precision "${MIXED_PRECISION}"
)
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "${SCRIPT_DIR}"
# Preflight stays on CPU even if NUM_PROCESSES requests multiple GPUs.
for arg in "$@"; do
    if [[ "$arg" == "--preflight_only" ]]; then
        exec "${PYTHON_BIN}" finetune_shared_noise_original_size.py "${ARGS[@]}" "$@"
    fi
done
if (( NUM_PROCESSES == 1 )); then
    exec "${PYTHON_BIN}" finetune_shared_noise_original_size.py "${ARGS[@]}" "$@"
fi
exec "${PYTHON_BIN}" -m accelerate.commands.launch \
    --multi_gpu --num_machines 1 --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PROCESS_PORT:-29640}" \
    --mixed_precision "${MIXED_PRECISION}" --dynamo_backend no \
    finetune_shared_noise_original_size.py "${ARGS[@]}" "$@"
