#!/usr/bin/env bash
# Phase 1: frozen clean SEA-RAFT teacher -> trainable degraded-RGB SEA-RAFT student.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/accelerate}"

DATASET_ROOT="${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
TEACHER_FLOW_ROOT="${TEACHER_FLOW_ROOT:-${DATASET_ROOT}/teacher_flows_sea_raft_512x512}"
SEA_RAFT_ROOT="${SEA_RAFT_ROOT:-/home/cilab/ndquan/videoInpainting/pretrained/SEA-RAFT}"
SEA_RAFT_CFG="${SEA_RAFT_CFG:-${SEA_RAFT_ROOT}/config/eval/spring-M.json}"
SEA_RAFT_CHECKPOINT="${SEA_RAFT_CHECKPOINT:?Set SEA_RAFT_CHECKPOINT to the SEA-RAFT .safetensors or .pth initialization checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/train_v7_sea_raft_student_flow_512}"

NUM_PROCESSES="${NUM_PROCESSES:-2}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29631}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
RESOLUTION="${RESOLUTION:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
RAFT_PAIR_BATCH_SIZE="${RAFT_PAIR_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"

# SEA-RAFT-M was pretrained with four iterative refinements.  The teacher cache
# also records iters=4, but student/teacher iteration counts are independent.
RAFT_ITERATIONS="${RAFT_ITERATIONS:-4}"
FINAL_FLOW_ONLY="${FINAL_FLOW_ONLY:-0}"
RAFT_SEQUENCE_GAMMA="${RAFT_SEQUENCE_GAMMA:-0.8}"
FLOW_LOSS_REGION="${FLOW_LOSS_REGION:-all}"
TEACHER_WEIGHT="${TEACHER_WEIGHT:-1.0}"
FB_WEIGHT="${FB_WEIGHT:-0.1}"
SMOOTHNESS_WEIGHT="${SMOOTHNESS_WEIGHT:-0.01}"
SMOOTHNESS_EDGE_WEIGHT="${SMOOTHNESS_EDGE_WEIGHT:-10.0}"

LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-250}"
VALIDATION_STEPS="${VALIDATION_STEPS:-250}"
VALID_PAIRS_PER_SEQUENCE="${VALID_PAIRS_PER_SEQUENCE:-8}"
CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-20}"
DEFORM_FEATURE_SIZE="${DEFORM_FEATURE_SIZE:-64}"
DEFORM_RESIDUAL_RANGE="${DEFORM_RESIDUAL_RANGE:-2.0}"
BEST_METRIC="${BEST_METRIC:-bg_epe}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
INIT_FROM_CHECKPOINT="${INIT_FROM_CHECKPOINT:-}"

for required in \
  "${PYTHON_BIN}" "${ACCELERATE_BIN}" \
  "${SCRIPT_DIR}/train_sea_raft_student_flow.py" \
  "${SEA_RAFT_ROOT}/core/raft.py" "${SEA_RAFT_CFG}" "${SEA_RAFT_CHECKPOINT}" \
  "${DATASET_ROOT}/manifest.json" "${DATASET_ROOT}/train/input" \
  "${DATASET_ROOT}/valid/input" "${TEACHER_FLOW_ROOT}/metadata.json"; do
  [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 1; }
done
if (( RESOLUTION % 8 )); then
  echo "RESOLUTION must be divisible by 8" >&2
  exit 2
fi
if (( RAFT_ITERATIONS < 1 || RAFT_PAIR_BATCH_SIZE < 1 )); then
  echo "RAFT_ITERATIONS and RAFT_PAIR_BATCH_SIZE must be positive" >&2
  exit 2
fi

COMMON_ARGS=(
  --student_backend sea_raft
  --dataset_root "${DATASET_ROOT}"
  --teacher_flow_root "${TEACHER_FLOW_ROOT}"
  --output_dir "${OUTPUT_DIR}"
  --sea_raft_root "${SEA_RAFT_ROOT}"
  --sea_raft_cfg "${SEA_RAFT_CFG}"
  --sea_raft_checkpoint "${SEA_RAFT_CHECKPOINT}"
  --resolution "${RESOLUTION}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --raft_pair_batch_size "${RAFT_PAIR_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --raft_iterations "${RAFT_ITERATIONS}"
  --raft_sequence_gamma "${RAFT_SEQUENCE_GAMMA}"
  --flow_loss_region "${FLOW_LOSS_REGION}"
  --teacher_weight "${TEACHER_WEIGHT}"
  --fb_weight "${FB_WEIGHT}"
  --smoothness_weight "${SMOOTHNESS_WEIGHT}"
  --smoothness_edge_weight "${SMOOTHNESS_EDGE_WEIGHT}"
  --learning_rate "${LEARNING_RATE}"
  --lr_scheduler "${LR_SCHEDULER}"
  --lr_warmup_steps "${LR_WARMUP_STEPS}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --checkpointing_steps "${CHECKPOINTING_STEPS}"
  --validation_steps "${VALIDATION_STEPS}"
  --valid_pairs_per_sequence "${VALID_PAIRS_PER_SEQUENCE}"
  --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT}"
  --deform_feature_size "${DEFORM_FEATURE_SIZE}"
  --deform_residual_range "${DEFORM_RESIDUAL_RANGE}"
  --best_metric "${BEST_METRIC}"
  --mixed_precision "${MIXED_PRECISION}"
  --freeze_batchnorm
)
if [[ "${FINAL_FLOW_ONLY}" == "1" ]]; then
  COMMON_ARGS+=(--final_flow_only)
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  COMMON_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ -n "${INIT_FROM_CHECKPOINT}" ]]; then
  COMMON_ARGS+=(--init_from_checkpoint "${INIT_FROM_CHECKPOINT}")
fi

mkdir -p "${OUTPUT_DIR}/terminal_logs"
timestamp="$(date +%Y%m%d_%H%M%S)"
terminal_log="${OUTPUT_DIR}/terminal_logs/train_${timestamp}.log"
exec > >(tee -a "${terminal_log}") 2>&1

echo "SEA-RAFT clean-to-degraded flow distillation (standalone; no V6 deformation)"
echo "output=${OUTPUT_DIR}; processes=${NUM_PROCESSES}; image=${RESOLUTION}; pair_batch=${RAFT_PAIR_BATCH_SIZE}"
echo "SEA-RAFT iterations=${RAFT_ITERATIONS}; iterative_supervision=$((1 - FINAL_FLOW_ONLY)); gamma=${RAFT_SEQUENCE_GAMMA}"
echo "loss=${TEACHER_WEIGHT}*L_teacher + ${FB_WEIGHT}*L_FB + ${SMOOTHNESS_WEIGHT}*L_smooth; region=${FLOW_LOSS_REGION}"
if [[ -n "${INIT_FROM_CHECKPOINT}" ]]; then
  echo "warm_start=${INIT_FROM_CHECKPOINT}; optimizer/scheduler/global_step are fresh"
fi
echo "terminal_log=${terminal_log}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "${BRUSHNET_ROOT}"
if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/train_sea_raft_student_flow.py" "${COMMON_ARGS[@]}" --preflight_only "$@"
fi
if (( NUM_PROCESSES == 1 )); then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/train_sea_raft_student_flow.py" "${COMMON_ARGS[@]}" "$@"
else
  "${ACCELERATE_BIN}" launch \
    --multi_gpu \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines 1 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    --mixed_precision "${MIXED_PRECISION}" \
    --dynamo_backend no \
    "${SCRIPT_DIR}/train_sea_raft_student_flow.py" \
    "${COMMON_ARGS[@]}" \
    "$@"
fi
