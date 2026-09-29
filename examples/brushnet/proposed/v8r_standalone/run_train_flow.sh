#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/accelerate}"
DATASET_ROOT="${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
TEACHER_FLOW_ROOT="${TEACHER_FLOW_ROOT:-${DATASET_ROOT}/teacher_flows_512x512}"
PROPAINTER_ROOT="${PROPAINTER_ROOT:-/home/cilab/ndquan/videoInpainting/pretrained/ProPainter}"
RAFT_CHECKPOINT="${RAFT_CHECKPOINT:-${PROPAINTER_ROOT}/weights/raft-things.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/experiments/train_v8r_standalone_flow}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
ARGS=(--dataset_root "${DATASET_ROOT}" --teacher_flow_root "${TEACHER_FLOW_ROOT}"
      --propainter_root "${PROPAINTER_ROOT}" --raft_checkpoint "${RAFT_CHECKPOINT}"
      --output_dir "${OUTPUT_DIR}" --resolution "${RESOLUTION:-512}"
      --raft_iterations "${RAFT_ITERATIONS:-20}" --raft_pair_batch_size "${RAFT_PAIR_BATCH_SIZE:-1}"
      --train_batch_size "${TRAIN_BATCH_SIZE:-1}" --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
      --learning_rate "${LEARNING_RATE:-1e-5}" --max_train_steps "${MAX_TRAIN_STEPS:-5000}"
      --checkpointing_steps "${CHECKPOINTING_STEPS:-250}" --validation_steps "${VALIDATION_STEPS:-250}"
      --teacher_weight "${TEACHER_WEIGHT:-1.0}" --fb_weight "${FB_WEIGHT:-0.1}"
      --smoothness_weight "${SMOOTHNESS_WEIGHT:-0.01}" --mixed_precision "${MIXED_PRECISION}"
      --freeze_batchnorm)
[[ -f "${RAFT_CHECKPOINT}" && -f "${TEACHER_FLOW_ROOT}/metadata.json" ]] || {
  echo "Missing RAFT checkpoint or teacher-flow cache" >&2; exit 1;
}
mkdir -p "${OUTPUT_DIR}"
cd "${ROOT}"
if [[ "${RUN_PREFLIGHT:-1}" == 1 ]]; then
  "${PYTHON_BIN}" "${HERE}/train_flow.py" "${ARGS[@]}" --preflight_only "$@"
fi
if (( NUM_PROCESSES == 1 )); then
  exec "${PYTHON_BIN}" "${HERE}/train_flow.py" "${ARGS[@]}" "$@"
fi
exec "${ACCELERATE_BIN}" launch --multi_gpu --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT:-29620}" --mixed_precision "${MIXED_PRECISION}" \
  "${HERE}/train_flow.py" "${ARGS[@]}" "$@"
