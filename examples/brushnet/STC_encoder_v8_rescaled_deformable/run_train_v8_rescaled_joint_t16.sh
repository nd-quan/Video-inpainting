#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_v8_rescaled_joint.py"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/accelerate}"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-${BRUSHNET_ROOT}/examples/brushnet/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5}"
DATASET_ROOT="${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
TEACHER_FLOW_ROOT="${TEACHER_FLOW_ROOT:-${DATASET_ROOT}/teacher_flows_512x512}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${BRUSHNET_ROOT}/experiments/train_sharedNoise_sameBG_0.95_T8/checkpoint-2250}"
IMAGE_ENCODER="${IMAGE_ENCODER:-laion/CLIP-ViT-H-14-laion2B-s32B-b79K}"
INIT_V8_MODEL="${INIT_V8_MODEL:-${BRUSHNET_ROOT}/experiments/train_stc_v8_raft_joint_from7500_T16_S12_sharedNoise_0.95_constant5e6/checkpoint-4000/stc_v8_model}"
RAFT_STUDENT_PATH="${RAFT_STUDENT_PATH:-${BRUSHNET_ROOT}/experiments/train_v7_raft_student_flow/best.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/train_stc_v8_rescaled_joint_T16_S12_sharedNoise_0.95}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29652}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
CLIP_LENGTH="${CLIP_LENGTH:-16}"
CLIP_STRIDE="${CLIP_STRIDE:-12}"
CROSS_CLIP_MEMORY_FRAMES="${CROSS_CLIP_MEMORY_FRAMES:-4}"
RELATIVE_POSITION_MAX_DISTANCE="${RELATIVE_POSITION_MAX_DISTANCE:-32}"
CLIPS_PER_DEVICE="${CLIPS_PER_DEVICE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-3}"
RAFT_PAIR_BATCH_SIZE="${RAFT_PAIR_BATCH_SIZE:-1}"
RAFT_MIXED_PRECISION="${RAFT_MIXED_PRECISION:-1}"

SHARED_BG_NOISE_STRENGTH="${SHARED_BG_NOISE_STRENGTH:-0.95}"
RESCALED_WARP_SCALE="${RESCALED_WARP_SCALE:-4}"
DEFORMABLE_ALIGNMENT_DIRECTION="${DEFORMABLE_ALIGNMENT_DIRECTION:-bidirectional}"
JOINT_UNFREEZE_STEP="${JOINT_UNFREEZE_STEP:-500}"
DEFORM_LEARNING_RATE="${DEFORM_LEARNING_RATE:-2e-5}"
TEMPORAL_LEARNING_RATE="${TEMPORAL_LEARNING_RATE:-5e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}"
TEMPORAL_LR_WARMUP_STEPS="${TEMPORAL_LR_WARMUP_STEPS:-100}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-4000}"

DEFORM_ALIGNMENT_LOSS_WEIGHT="${DEFORM_ALIGNMENT_LOSS_WEIGHT:-0.1}"
DEFORM_ALIGNMENT_WARMUP_STEPS="${DEFORM_ALIGNMENT_WARMUP_STEPS:-500}"
DEFORM_OFFSET_LOSS_WEIGHT="${DEFORM_OFFSET_LOSS_WEIGHT:-0.0005}"
DEFORM_HIDDEN_CHANNELS="${DEFORM_HIDDEN_CHANNELS:-128}"
DEFORM_KERNEL_SIZE="${DEFORM_KERNEL_SIZE:-3}"
DEFORM_GROUPS="${DEFORM_GROUPS:-4}"
DEFORM_RESIDUAL_MAX_DISPLACEMENT="${DEFORM_RESIDUAL_MAX_DISPLACEMENT:-2.0}"
FLOW_MAX_DX="${FLOW_MAX_DX:-8.0}"
FLOW_MAX_DY="${FLOW_MAX_DY:-8.0}"

CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-250}"
CHECKPOINTS_TOTAL_LIMIT="${CHECKPOINTS_TOTAL_LIMIT:-20}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
SEED="${SEED:-1234}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

for required in \
    "${PYTHON_BIN}" "${ACCELERATE_BIN}" "${TRAIN_SCRIPT}" \
    "${PRETRAINED_MODEL}" "${INIT_V8_MODEL}/config.json" \
    "${DATASET_ROOT}/train/GT" "${DATASET_ROOT}/train/input" \
    "${DATASET_ROOT}/train/mask" "${TEACHER_FLOW_ROOT}/metadata.json" \
    "${BASELINE_CHECKPOINT}/brushnet/config.json" \
    "${BASELINE_CHECKPOINT}/ipadapter/model.safetensors" \
    "${BASELINE_CHECKPOINT}/ipadapter/fusion_module.safetensors"; do
    [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 1; }
done
if [[ -f "${RAFT_STUDENT_PATH}" ]]; then
    grep -q '"checkpoint"' "${RAFT_STUDENT_PATH}" || {
        echo "Not a RAFT best/latest pointer: ${RAFT_STUDENT_PATH}" >&2
        exit 1
    }
elif [[ ! -f "${RAFT_STUDENT_PATH}/pytorch_model.bin" && ! -f "${RAFT_STUDENT_PATH}/diffusion_pytorch_model.safetensors" ]]; then
    echo "Missing RAFT student weights below ${RAFT_STUDENT_PATH}" >&2
    exit 1
fi
[[ "${MIXED_PRECISION}" != "bf16" ]] || {
    echo "torchvision deform_conv2d does not support BF16 here; use fp16" >&2
    exit 2
}

COMMON_ARGS=(
    --pretrained_model_name_or_path "${PRETRAINED_MODEL}"
    --baseline_checkpoint "${BASELINE_CHECKPOINT}"
    --image_encoder_name_or_path "${IMAGE_ENCODER}"
    --dataset_root "${DATASET_ROOT}" --teacher_flow_root "${TEACHER_FLOW_ROOT}"
    --train_split train --output_dir "${OUTPUT_DIR}"
    --training_stage rescaled_joint --init_v8_model "${INIT_V8_MODEL}"
    --raft_student_path "${RAFT_STUDENT_PATH}" --raft_pair_batch_size "${RAFT_PAIR_BATCH_SIZE}"
    --resolution 512 --clip_length "${CLIP_LENGTH}" --clip_stride "${CLIP_STRIDE}"
    --relative_position_max_distance "${RELATIVE_POSITION_MAX_DISTANCE}"
    --cross_clip_memory_frames "${CROSS_CLIP_MEMORY_FRAMES}" --detach_cross_clip_memory
    --shared_bg_noise_strength "${SHARED_BG_NOISE_STRENGTH}" --sequence_shared_noise_refresh epoch
    --condition_mode full_rgb_bg_mask --stc_injection_scale 1.0
    --flow_loss_weight 0 --flow_region bg --flow_charbonnier_eps 0.001
    --flow_max_displacement "${FLOW_MAX_DX}" "${FLOW_MAX_DY}"
    --feature_alignment_loss_weight 0 --feature_alignment_region bg
    --feature_alignment_charbonnier_eps 0.001 --feature_alignment_confidence_floor 0.1
    --feature_alignment_warmup_steps 500
    --deform_hidden_channels "${DEFORM_HIDDEN_CHANNELS}"
    --deform_kernel_size "${DEFORM_KERNEL_SIZE}" --deform_groups "${DEFORM_GROUPS}"
    --deform_residual_max_displacement "${DEFORM_RESIDUAL_MAX_DISPLACEMENT}"
    --deform_alignment_loss_weight "${DEFORM_ALIGNMENT_LOSS_WEIGHT}"
    --deform_alignment_charbonnier_eps 0.001
    --deform_alignment_warmup_steps "${DEFORM_ALIGNMENT_WARMUP_STEPS}"
    --deform_offset_loss_weight "${DEFORM_OFFSET_LOSS_WEIGHT}"
    --deformable_alignment_direction "${DEFORMABLE_ALIGNMENT_DIRECTION}"
    --rescaled_warp_scale "${RESCALED_WARP_SCALE}" --reset_deformable_branch
    --joint_unfreeze_step "${JOINT_UNFREEZE_STEP}"
    --deform_learning_rate "${DEFORM_LEARNING_RATE}"
    --temporal_learning_rate "${TEMPORAL_LEARNING_RATE}"
    --temporal_lr_warmup_steps "${TEMPORAL_LR_WARMUP_STEPS}"
    --train_batch_size "${CLIPS_PER_DEVICE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning_rate "${DEFORM_LEARNING_RATE}" --lr_scheduler constant
    --lr_warmup_steps "${LR_WARMUP_STEPS}" --max_train_steps "${MAX_TRAIN_STEPS}"
    --checkpointing_steps "${CHECKPOINTING_STEPS}"
    --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT}" --gradient_checkpointing
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" --dataloader_pin_memory
    --mixed_precision "${MIXED_PRECISION}" --seed "${SEED}"
    --report_to tensorboard --tracker_project_name train_stc_v8_rescaled_joint
)
if [[ "${RAFT_MIXED_PRECISION}" != "1" ]]; then
    COMMON_ARGS+=(--raft_no_mixed_precision)
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    COMMON_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

mkdir -p "${OUTPUT_DIR}/terminal_logs"
timestamp="$(date +%Y%m%d_%H%M%S)"
terminal_log="${OUTPUT_DIR}/terminal_logs/train_${timestamp}.log"
exec > >(tee -a "${terminal_log}") 2>&1

echo "V8-R joint: frozen RAFT + x${RESCALED_WARP_SCALE} pre-warp + native residual DCN"
echo "init_v8=${INIT_V8_MODEL}"
echo "raft_student=${RAFT_STUDENT_PATH}"
echo "reliability=geometric_only; FB=diagnostic_only; direction=${DEFORMABLE_ALIGNMENT_DIRECTION}"
echo "steps=${MAX_TRAIN_STEPS}; joint_unfreeze=${JOINT_UNFREEZE_STEP}; deform_lr=${DEFORM_LEARNING_RATE}; temporal_lr=${TEMPORAL_LEARNING_RATE}"
echo "loss=L_diff + ramp(${DEFORM_ALIGNMENT_LOSS_WEIGHT})*L_deform + ${DEFORM_OFFSET_LOSS_WEIGHT}*L_offset"
echo "processes=${NUM_PROCESSES}; clips/device=${CLIPS_PER_DEVICE}; accumulation=${GRADIENT_ACCUMULATION_STEPS}"
echo "terminal_log=${terminal_log}"

export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "${BRUSHNET_ROOT}"
if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}" --preflight_only "$@"
fi
if (( NUM_PROCESSES == 1 )); then
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}" "$@"
else
    "${ACCELERATE_BIN}" launch --multi_gpu --num_processes "${NUM_PROCESSES}" \
        --num_machines 1 --main_process_port "${MAIN_PROCESS_PORT}" \
        --mixed_precision "${MIXED_PRECISION}" --dynamo_backend no \
        "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}" "$@"
fi
