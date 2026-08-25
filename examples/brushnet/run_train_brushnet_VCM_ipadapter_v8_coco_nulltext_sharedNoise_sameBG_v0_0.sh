#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_brushnet_VCM_ipadapter_v8_coco_nulltext_sharedNoise_sameBG_v0_0.py"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-${SCRIPT_DIR}/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5}"
DATASET_ROOT="${DATASET_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
DATASET_LAYOUT="${DATASET_LAYOUT:-hierarchical}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
CAPTION_FILE="${CAPTION_FILE:-${DATASET_ROOT}/caption_empty.txt}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${DATASET_ROOT}/split.json}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/baselines/sharedNoise_sameBG_0.7/checkpoint-2000}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/train_sharedNoise_sameBG_0.95}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
CLIP_LENGTH="${CLIP_LENGTH:-16}"
CLIP_STRIDE="${CLIP_STRIDE:-12}"
CLIPS_PER_DEVICE="${CLIPS_PER_DEVICE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-6}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-4170}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-250}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SHARED_BG_NOISE_STRENGTH="${SHARED_BG_NOISE_STRENGTH:-0.95}"
SEQUENCE_SHARED_NOISE_REFRESH="${SEQUENCE_SHARED_NOISE_REFRESH:-epoch}"
SEED="${SEED:-1234}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"

DATASET_ARGS=(
    --dataset_root "${DATASET_ROOT}"
    --dataset_layout "${DATASET_LAYOUT}"
    --null_caption_file "${CAPTION_FILE}"
    --train_split "${TRAIN_SPLIT}"
)
REQUIRED_DATA_PATHS=("${CAPTION_FILE}")

if [[ "${DATASET_LAYOUT}" == "hierarchical" ]]; then
    REQUIRED_DATA_PATHS+=(
        "${DATASET_ROOT}/${TRAIN_SPLIT}/GT"
        "${DATASET_ROOT}/${TRAIN_SPLIT}/input"
        "${DATASET_ROOT}/${TRAIN_SPLIT}/mask"
    )
elif [[ "${DATASET_LAYOUT}" == "flat_manifest" ]]; then
    DATASET_ARGS+=(--split_manifest "${SPLIT_MANIFEST}")
    REQUIRED_DATA_PATHS+=(
        "${DATASET_ROOT}/GT"
        "${DATASET_ROOT}/input"
        "${DATASET_ROOT}/mask"
        "${SPLIT_MANIFEST}"
    )
else
    echo "DATASET_LAYOUT must be hierarchical or flat_manifest" >&2
    exit 1
fi

REQUIRED_PATHS=(
    "${TRAIN_SCRIPT}"
    "${PRETRAINED_MODEL}"
    "${DATASET_ROOT}"
    "${INITIAL_CHECKPOINT}/brushnet"
    "${INITIAL_CHECKPOINT}/ipadapter/model.safetensors"
    "${INITIAL_CHECKPOINT}/ipadapter/fusion_module.safetensors"
    "${REQUIRED_DATA_PATHS[@]}"
)
for required_path in "${REQUIRED_PATHS[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing required path: ${required_path}" >&2
        exit 1
    fi
done

cd "${SCRIPT_DIR}"

accelerate launch \
    --num_processes "${NUM_PROCESSES}" \
    --mixed_precision "${MIXED_PRECISION}" \
    "${TRAIN_SCRIPT}" \
    --pretrained_model_name_or_path "${PRETRAINED_MODEL}" \
    "${DATASET_ARGS[@]}" \
    --brushnet_model_name_or_path "${INITIAL_CHECKPOINT}/brushnet" \
    --pretrained_ip_adapter_path "${INITIAL_CHECKPOINT}/ipadapter/model.safetensors" \
    --fusion_module_path "${INITIAL_CHECKPOINT}/ipadapter/fusion_module.safetensors" \
    --output_dir "${OUTPUT_DIR}" \
    --resolution 512 \
    --clip_length "${CLIP_LENGTH}" \
    --clip_stride "${CLIP_STRIDE}" \
    --shared_bg_noise_strength "${SHARED_BG_NOISE_STRENGTH}" \
    --sequence_shared_noise_refresh "${SEQUENCE_SHARED_NOISE_REFRESH}" \
    --train_batch_size "${CLIPS_PER_DEVICE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --checkpointing_steps "${CHECKPOINTING_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --lr_scheduler constant \
    --lr_warmup_steps 500 \
    --gradient_checkpointing \
    --set_grads_to_none \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --seed "${SEED}" \
    --report_to tensorboard \
    --tracker_project_name train_brushnet_shared_noise_same_bg \
    "$@"
