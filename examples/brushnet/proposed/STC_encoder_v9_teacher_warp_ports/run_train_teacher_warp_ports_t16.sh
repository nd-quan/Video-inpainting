#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/accelerate}"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-${BRUSHNET_ROOT}/examples/brushnet/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5}"
DATASET_ROOT="${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
TEACHER_FLOW_ROOT="${TEACHER_FLOW_ROOT:-${DATASET_ROOT}/teacher_flows_512x512}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${BRUSHNET_ROOT}/experiments/train_sharedNoise_sameBG_0.95_T8/checkpoint-2250}"
INIT_V8R_MODEL="${INIT_V8R_MODEL:-${BRUSHNET_ROOT}/experiments/train_stc_v8_rescaled_x4_joint_raftbest_T16_S12_sharedNoise_0.95/checkpoint-4000/stc_v8r_model}"
IMAGE_ENCODER="${IMAGE_ENCODER:-laion/CLIP-ViT-H-14-laion2B-s32B-b79K}"
OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/train_stc_v9_teacher_x4_ports_T16_S12}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29659}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-4000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-250}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-3}"
PORT_LEARNING_RATE="${PORT_LEARNING_RATE:-2e-5}"
PORT_BOTTLENECK_CHANNELS="${PORT_BOTTLENECK_CHANNELS:-64}"
WARP_STRENGTH="${WARP_STRENGTH:-1.0}"
STC_INJECTION_SCALE="${STC_INJECTION_SCALE:-1.0}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

for required in \
    "${PYTHON_BIN}" "${ACCELERATE_BIN}" \
    "${PRETRAINED_MODEL}" "${INIT_V8R_MODEL}/config.json" \
    "${BASELINE_CHECKPOINT}/brushnet/config.json" \
    "${BASELINE_CHECKPOINT}/ipadapter/model.safetensors" \
    "${BASELINE_CHECKPOINT}/ipadapter/fusion_module.safetensors" \
    "${DATASET_ROOT}/train/GT" "${DATASET_ROOT}/train/input" \
    "${DATASET_ROOT}/train/mask" "${TEACHER_FLOW_ROOT}/metadata.json"; do
    [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 1; }
done

ARGS=(
    --pretrained_model_name_or_path "${PRETRAINED_MODEL}"
    --baseline_checkpoint "${BASELINE_CHECKPOINT}"
    --image_encoder_name_or_path "${IMAGE_ENCODER}"
    --dataset_root "${DATASET_ROOT}"
    --teacher_flow_root "${TEACHER_FLOW_ROOT}"
    --train_split train
    --output_dir "${OUTPUT_DIR}"
    --init_v8r_model "${INIT_V8R_MODEL}"
    --resolution 512
    --clip_length 16
    --clip_stride 12
    --condition_mode full_rgb_bg_mask
    --shared_bg_noise_strength 0.95
    --sequence_shared_noise_refresh epoch
    --stc_injection_scale "${STC_INJECTION_SCALE}"
    --warp_strength "${WARP_STRENGTH}"
    --flow_loss_weight 0
    --feature_alignment_loss_weight 0
    --train_batch_size 1
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning_rate "${PORT_LEARNING_RATE}"
    --port_learning_rate "${PORT_LEARNING_RATE}"
    --port_bottleneck_channels "${PORT_BOTTLENECK_CHANNELS}"
    --lr_scheduler constant
    --max_train_steps "${MAX_TRAIN_STEPS}"
    --checkpointing_steps "${CHECKPOINTING_STEPS}"
    --checkpoints_total_limit 20
    --gradient_checkpointing
    --mixed_precision "${MIXED_PRECISION}"
    --dataloader_num_workers 2
    --dataloader_pin_memory
    --seed 1234
    --report_to tensorboard
    --tracker_project_name train_stc_v9_teacher_x4_ports
)
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

cd "${BRUSHNET_ROOT}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/train_teacher_warp_ports.py" "${ARGS[@]}" --preflight_only "$@"
fi
if (( NUM_PROCESSES == 1 )); then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/train_teacher_warp_ports.py" "${ARGS[@]}" "$@"
else
    "${ACCELERATE_BIN}" launch --multi_gpu --num_processes "${NUM_PROCESSES}" \
        --num_machines 1 --main_process_port "${MAIN_PROCESS_PORT}" \
        --mixed_precision "${MIXED_PRECISION}" --dynamo_backend no \
        "${SCRIPT_DIR}/train_teacher_warp_ports.py" "${ARGS[@]}" "$@"
fi
