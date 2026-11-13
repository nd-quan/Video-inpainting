#!/usr/bin/env bash
# latent_warp_v1 / run 1: stable-BG latent warp with oracle (clean-GT RAFT) flow.
#
# Prerequisites (both produced by scripts in this folder):
#   1. Flow cache  : precompute_bg52_teacher_flows_512.py -> ${TEACHER_FLOW_ROOT}
#   2. 14-ch init  : prepare_brushnet_14ch_init.py        -> ${INITIAL_CHECKPOINT}
#
# GPU safety: CUDA_VISIBLE_DEVICES must be given explicitly, and the script
# refuses to launch if any selected GPU already runs another process
# (override only with ALLOW_SHARED_GPU=1).  PRINT_ONLY=1 prints the command
# and exits without launching.
#
# Example (after checking the GPUs are free):
#   CUDA_VISIBLE_DEVICES=0 bash run_train_latent_warp_v1_run1.sh
# Run-0 baseline with identical data/steps:
#   LATENT_WARP_DROP_PROB=1.0 OUTPUT_DIR=/media/ssd1/ndquan/video-inpainting/code/experiments/latent_warp_run0/train \
#   CUDA_VISIBLE_DEVICES=0 bash run_train_latent_warp_v1_run1.sh
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_DIR="$(cd -- "${HERE}/../.." && pwd)"
PROJECT_ROOT="$(cd -- "${BRUSHNET_DIR}/../../.." && pwd)"
export HF_HOME="${PROJECT_ROOT}/code/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
TRAIN_SCRIPT="${HERE}/train_brushnet_latent_warp_v1.py"
ACCELERATE_BIN="${ACCELERATE_BIN:-/media/ssd1/ndquan/anaconda3/envs/guided_diff/bin/accelerate}"

PRETRAINED_MODEL="${PRETRAINED_MODEL:-${BRUSHNET_DIR}/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5}"
DATASET_ROOT="${DATASET_ROOT:-${BRUSHNET_DIR}/dataset/bg52}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
TEACHER_FLOW_ROOT="${TEACHER_FLOW_ROOT:-${PROJECT_ROOT}/flow-training/flows/bvi_dvc_bg52_512}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-${PROJECT_ROOT}/code/checkpoints/init_T8_ckpt2250_14ch}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/code/experiments/latent_warp_run1/train}"

MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
RESOLUTION="${RESOLUTION:-512}"
CLIP_LENGTH="${CLIP_LENGTH:-8}"
CLIP_STRIDE="${CLIP_STRIDE:-6}"
CLIPS_PER_DEVICE="${CLIPS_PER_DEVICE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-3}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-250}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LR_SCHEDULER="${LR_SCHEDULER:-constant}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}"  # unused by the constant scheduler (kept = T8)
SHARED_BG_NOISE_STRENGTH="${SHARED_BG_NOISE_STRENGTH:-0.95}"
SEQUENCE_SHARED_NOISE_REFRESH="${SEQUENCE_SHARED_NOISE_REFRESH:-epoch}"
LATENT_WARP_DROP_PROB="${LATENT_WARP_DROP_PROB:-0.0}"
LATENT_WARP_X0_CLAMP="${LATENT_WARP_X0_CLAMP:-0}"
SEED="${SEED:-1234}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-0}"
PRINT_ONLY="${PRINT_ONLY:-0}"

# ---- inputs ---------------------------------------------------------------
REQUIRED_PATHS=(
    "${TRAIN_SCRIPT}"
    "${ACCELERATE_BIN}"
    "${PRETRAINED_MODEL}"
    "${DATASET_ROOT}/${TRAIN_SPLIT}/GT"
    "${DATASET_ROOT}/${TRAIN_SPLIT}/input"
    "${DATASET_ROOT}/${TRAIN_SPLIT}/mask"
    "${TEACHER_FLOW_ROOT}/metadata.json"
    "${TEACHER_FLOW_ROOT}/manifest.json"
    "${INITIAL_CHECKPOINT}/brushnet/config.json"
    "${INITIAL_CHECKPOINT}/ipadapter/model.safetensors"
    "${INITIAL_CHECKPOINT}/ipadapter/fusion_module.safetensors"
)
for required_path in "${REQUIRED_PATHS[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing required path: ${required_path}" >&2
        exit 1
    fi
done
if ! grep -q '"conditioning_channels": 10' "${INITIAL_CHECKPOINT}/brushnet/config.json"; then
    echo "${INITIAL_CHECKPOINT}/brushnet is not the 14-input init (conditioning_channels != 10)" >&2
    exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]] && [[ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]] && [[ "${PRINT_ONLY}" != "1" ]]; then
    echo "OUTPUT_DIR is not empty: ${OUTPUT_DIR} (choose a new OUTPUT_DIR; resume is not wired for run 1)" >&2
    exit 1
fi

# ---- GPU guard ----------------------------------------------------------------
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]] || [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    echo "Set CUDA_VISIBLE_DEVICES explicitly (e.g. CUDA_VISIBLE_DEVICES=0)." >&2
    exit 1
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_PROCESSES="${#GPU_IDS[@]}"
BUSY=0
for gpu in "${GPU_IDS[@]}"; do
    uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="${gpu}" '$1==g || $2==g {print $2}')"
    if [[ -z "${uuid}" ]]; then
        echo "GPU ${gpu} not found by nvidia-smi" >&2
        exit 1
    fi
    procs="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | awk -F', ' -v u="${uuid}" '$1==u {print "pid " $2 " (" $3 ")"}')"
    if [[ -n "${procs}" ]]; then
        owners=""
        for pid in $(echo "${procs}" | awk '{print $2}'); do
            owners+=" ${pid}:$(ps -o user= -p "${pid}" 2>/dev/null || echo '?')"
        done
        echo "GPU ${gpu} is in use by:${owners}" >&2
        BUSY=1
    fi
done
if [[ "${BUSY}" == "1" ]] && [[ "${ALLOW_SHARED_GPU}" != "1" ]]; then
    echo "Refusing to launch: selected GPU(s) are busy. Other users' training must not be disturbed." >&2
    [[ "${PRINT_ONLY}" == "1" ]] || exit 1
fi

# ---- launch -----------------------------------------------------------------------
LAUNCH_ARGS=(--num_processes "${NUM_PROCESSES}" --mixed_precision "${MIXED_PRECISION}")
if (( NUM_PROCESSES > 1 )); then
    LAUNCH_ARGS+=(--multi_gpu)
fi

CMD=(
    "${ACCELERATE_BIN}" launch "${LAUNCH_ARGS[@]}"
    "${TRAIN_SCRIPT}"
    --pretrained_model_name_or_path "${PRETRAINED_MODEL}"
    --dataset_root "${DATASET_ROOT}"
    --dataset_layout hierarchical
    --train_split "${TRAIN_SPLIT}"
    --teacher_flow_root "${TEACHER_FLOW_ROOT}"
    --brushnet_model_name_or_path "${INITIAL_CHECKPOINT}/brushnet"
    --pretrained_ip_adapter_path "${INITIAL_CHECKPOINT}/ipadapter/model.safetensors"
    --fusion_module_path "${INITIAL_CHECKPOINT}/ipadapter/fusion_module.safetensors"
    --output_dir "${OUTPUT_DIR}"
    --resolution "${RESOLUTION}"
    --clip_length "${CLIP_LENGTH}"
    --clip_stride "${CLIP_STRIDE}"
    --shared_bg_noise_strength "${SHARED_BG_NOISE_STRENGTH}"
    --sequence_shared_noise_refresh "${SEQUENCE_SHARED_NOISE_REFRESH}"
    --latent_warp_drop_prob "${LATENT_WARP_DROP_PROB}"
    --latent_warp_x0_clamp "${LATENT_WARP_X0_CLAMP}"
    --train_batch_size "${CLIPS_PER_DEVICE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --max_train_steps "${MAX_TRAIN_STEPS}"
    --checkpointing_steps "${CHECKPOINTING_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --lr_scheduler "${LR_SCHEDULER}"
    --lr_warmup_steps "${LR_WARMUP_STEPS}"
    --gradient_checkpointing
    --set_grads_to_none
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --seed "${SEED}"
    --report_to tensorboard
    --tracker_project_name latent_warp_v1_run1
    "$@"
)

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (num_processes=${NUM_PROCESSES})"
printf '%q ' "${CMD[@]}"; echo
if [[ "${PRINT_ONLY}" == "1" ]]; then
    exit 0
fi
cd "${BRUSHNET_DIR}"
mkdir -p "${OUTPUT_DIR}"
"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
