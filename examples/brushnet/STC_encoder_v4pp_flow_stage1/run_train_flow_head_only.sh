#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/accelerate}"

INIT_CHECKPOINT="${INIT_CHECKPOINT:-${BRUSHNET_ROOT}/experiments/train_stc_v4pp_sflow01_sfeat01/checkpoint-5000}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-${BRUSHNET_ROOT}/examples/brushnet/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5}"
DATASET_ROOT="${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
TEACHER_FLOW_ROOT="${TEACHER_FLOW_ROOT:-${DATASET_ROOT}/teacher_flows_512x512}"
OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/train_v4pp_flow_head_stage1_all}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
CLIP_LENGTH="${CLIP_LENGTH:-16}"
CLIP_STRIDE="${CLIP_STRIDE:-12}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-250}"
VALIDATION_STEPS="${VALIDATION_STEPS:-250}"
VALID_CLIPS_PER_SEQUENCE="${VALID_CLIPS_PER_SEQUENCE:-1}"
CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-10}"

mkdir -p "${OUTPUT_DIR}/terminal_logs"
timestamp="$(date +%Y%m%d_%H%M%S)"
terminal_log="${OUTPUT_DIR}/terminal_logs/train_${timestamp}.log"
exec > >(tee -a "${terminal_log}") 2>&1

common_args=(
  --init_checkpoint "${INIT_CHECKPOINT}"
  --pretrained_model_name_or_path "${PRETRAINED_MODEL}"
  --dataset_root "${DATASET_ROOT}"
  --teacher_flow_root "${TEACHER_FLOW_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --clip_length "${CLIP_LENGTH}"
  --clip_stride "${CLIP_STRIDE}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --lr_scheduler "${LR_SCHEDULER}"
  --lr_warmup_steps "${LR_WARMUP_STEPS}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --checkpointing_steps "${CHECKPOINTING_STEPS}"
  --validation_steps "${VALIDATION_STEPS}"
  --valid_clips_per_sequence "${VALID_CLIPS_PER_SEQUENCE}"
  --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT}"
  --mixed_precision "${MIXED_PRECISION}"
)

echo "V4++ flow-head-only Stage-1"
echo "init=${INIT_CHECKPOINT}"
echo "objective=teacher Charbonnier only, flow_region=all"
echo "T=${CLIP_LENGTH}, stride=${CLIP_STRIDE}, lr=${LEARNING_RATE}, steps=${MAX_TRAIN_STEPS}"
echo "terminal_log=${terminal_log}"

cd "${BRUSHNET_ROOT}"
if (( NUM_PROCESSES == 1 )); then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/train_flow_head_only.py" "${common_args[@]}" "$@"
else
  "${ACCELERATE_BIN}" launch \
    --multi_gpu \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines 1 \
    --main_process_port "${MAIN_PROCESS_PORT:-29501}" \
    --mixed_precision "${MIXED_PRECISION}" \
    --dynamo_backend no \
    "${SCRIPT_DIR}/train_flow_head_only.py" \
    "${common_args[@]}" \
    "$@"
fi
