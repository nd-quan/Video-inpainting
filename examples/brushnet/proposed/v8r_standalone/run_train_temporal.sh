#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/accelerate}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-${ROOT}/examples/brushnet/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${ROOT}/experiments/train_sharedNoise_sameBG_0.95_T8/checkpoint-2250}"
DATASET_ROOT="${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
TEACHER_FLOW_ROOT="${TEACHER_FLOW_ROOT:-${DATASET_ROOT}/teacher_flows_512x512}"
INIT_V8_MODEL="${INIT_V8_MODEL:-${ROOT}/experiments/train_stc_v8_rescaled_x4_joint_raftbest_T16_S12_sharedNoise_0.95/checkpoint-4000/stc_v8r_model}"
RAFT_STUDENT_PATH="${RAFT_STUDENT_PATH:-${ROOT}/experiments/train_v7_raft_student_flow/best.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/experiments/train_v8r_standalone}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
[[ "${MIXED_PRECISION}" != bf16 ]] || { echo "Use fp16 or no mixed precision for DCNv2" >&2; exit 2; }
for item in "${PRETRAINED_MODEL}" "${BASELINE_CHECKPOINT}/brushnet/config.json" \
  "${DATASET_ROOT}/train/input" "${TEACHER_FLOW_ROOT}/metadata.json" \
  "${INIT_V8_MODEL}/config.json" "${RAFT_STUDENT_PATH}"; do
  [[ -e "${item}" ]] || { echo "Missing required path: ${item}" >&2; exit 1; }
done
ARGS=(--pretrained_model_name_or_path "${PRETRAINED_MODEL}"
      --baseline_checkpoint "${BASELINE_CHECKPOINT}"
      --image_encoder_name_or_path "${IMAGE_ENCODER:-laion/CLIP-ViT-H-14-laion2B-s32B-b79K}"
      --dataset_root "${DATASET_ROOT}" --teacher_flow_root "${TEACHER_FLOW_ROOT}"
      --train_split train --output_dir "${OUTPUT_DIR}"
      --training_stage rescaled_joint --init_v8_model "${INIT_V8_MODEL}"
      --raft_student_path "${RAFT_STUDENT_PATH}"
      --raft_pair_batch_size "${RAFT_PAIR_BATCH_SIZE:-1}"
      --resolution 512 --clip_length "${CLIP_LENGTH:-16}" --clip_stride "${CLIP_STRIDE:-12}"
      --relative_position_max_distance "${RELATIVE_POSITION_MAX_DISTANCE:-32}"
      --cross_clip_memory_frames "${CROSS_CLIP_MEMORY_FRAMES:-4}" --detach_cross_clip_memory
      --shared_bg_noise_strength "${SHARED_BG_NOISE_STRENGTH:-0.95}"
      --sequence_shared_noise_refresh epoch --condition_mode full_rgb_bg_mask
      --stc_injection_scale "${STC_INJECTION_SCALE:-1.0}"
      --flow_loss_weight 0 --flow_region bg --flow_charbonnier_eps 0.001
      --flow_max_displacement "${FLOW_MAX_DX:-8.0}" "${FLOW_MAX_DY:-8.0}"
      --feature_alignment_loss_weight 0 --feature_alignment_region bg
      --feature_alignment_charbonnier_eps 0.001 --feature_alignment_confidence_floor 0.1
      --deform_hidden_channels "${DEFORM_HIDDEN_CHANNELS:-128}"
      --deform_kernel_size "${DEFORM_KERNEL_SIZE:-3}" --deform_groups "${DEFORM_GROUPS:-4}"
      --deform_residual_max_displacement "${DEFORM_RESIDUAL_MAX_DISPLACEMENT:-2.0}"
      --deform_alignment_loss_weight "${DEFORM_ALIGNMENT_LOSS_WEIGHT:-0.1}"
      --deform_alignment_warmup_steps "${DEFORM_ALIGNMENT_WARMUP_STEPS:-500}"
      --deform_offset_loss_weight "${DEFORM_OFFSET_LOSS_WEIGHT:-0.0005}"
      --deformable_alignment_direction "${DEFORMABLE_ALIGNMENT_DIRECTION:-bidirectional}"
      --rescaled_warp_scale "${RESCALED_WARP_SCALE:-4}" --reset_deformable_branch
      --joint_unfreeze_step "${JOINT_UNFREEZE_STEP:-500}"
      --deform_learning_rate "${DEFORM_LEARNING_RATE:-2e-5}"
      --temporal_learning_rate "${TEMPORAL_LEARNING_RATE:-5e-6}"
      --temporal_lr_warmup_steps "${TEMPORAL_LR_WARMUP_STEPS:-100}"
      --temporal_train_loss_weight "${TEMPORAL_TRAIN_LOSS_WEIGHT:-0}"
      --train_batch_size "${CLIPS_PER_DEVICE:-1}"
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-3}"
      --learning_rate "${DEFORM_LEARNING_RATE:-2e-5}" --lr_scheduler constant
      --lr_warmup_steps "${LR_WARMUP_STEPS:-100}" --max_train_steps "${MAX_TRAIN_STEPS:-4000}"
      --checkpointing_steps "${CHECKPOINTING_STEPS:-250}"
      --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT:-20}" --gradient_checkpointing
      --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-2}" --dataloader_pin_memory
      --mixed_precision "${MIXED_PRECISION}" --seed "${SEED:-1234}"
      --report_to tensorboard --tracker_project_name train_v8r_standalone)
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
mkdir -p "${OUTPUT_DIR}"
cd "${ROOT}"
if [[ "${RUN_PREFLIGHT:-1}" == 1 ]]; then
  "${PYTHON_BIN}" "${HERE}/train_temporal.py" "${ARGS[@]}" --preflight_only "$@"
fi
if (( NUM_PROCESSES == 1 )); then
  exec "${PYTHON_BIN}" "${HERE}/train_temporal.py" "${ARGS[@]}" "$@"
fi
exec "${ACCELERATE_BIN}" launch --multi_gpu --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT:-29652}" --mixed_precision "${MIXED_PRECISION}" \
  "${HERE}/train_temporal.py" "${ARGS[@]}" "$@"
