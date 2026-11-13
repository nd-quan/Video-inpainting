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
# Run-0 baseline: use --config /absolute/path/to/configs/run0.yaml
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_DIR="$(cd -- "${HERE}/../.." && pwd)"
REPO_DIR="$(cd -- "${BRUSHNET_DIR}/../.." && pwd)"
TRAIN_SCRIPT="${HERE}/train_brushnet_latent_warp_v1.py"
# One resolver is shared with the trainer: CLI > environment > YAML > defaults.
PYTHON_BIN="${PYTHON_BIN:-/media/ssd1/ndquan/envs/guided_diff/bin/python}"
SETTINGS="$("${PYTHON_BIN}" "${HERE}/training_config.py" "$@")"
eval "${SETTINGS}"  # values are shell-quoted by shlex.quote
export CUDA_VISIBLE_DEVICES
PRINT_ONLY="${PRINT_ONLY:-0}"
CHECK_PATHS_ONLY="${CHECK_PATHS_ONLY:-0}"

# ---- inputs ---------------------------------------------------------------
REQUIRED_PATHS=(
    "${TRAIN_SCRIPT}"
    "${ACCELERATE_BIN}"
    "${PRETRAINED_MODEL}/scheduler/scheduler_config.json"
    "${PRETRAINED_MODEL}/tokenizer/vocab.json"
    "${PRETRAINED_MODEL}/tokenizer/merges.txt"
    "${PRETRAINED_MODEL}/text_encoder/config.json"
    "${PRETRAINED_MODEL}/text_encoder/model.safetensors"
    "${PRETRAINED_MODEL}/vae/config.json"
    "${PRETRAINED_MODEL}/vae/diffusion_pytorch_model.safetensors"
    "${PRETRAINED_MODEL}/unet/config.json"
    "${PRETRAINED_MODEL}/unet/diffusion_pytorch_model.safetensors"
    "${DATASET_ROOT}/${TRAIN_SPLIT}/GT"
    "${DATASET_ROOT}/${TRAIN_SPLIT}/input"
    "${DATASET_ROOT}/${TRAIN_SPLIT}/mask"
    "${TEACHER_FLOW_ROOT}/metadata.json"
    "${TEACHER_FLOW_ROOT}/manifest.json"
    "${BRUSHNET_CHECKPOINT}/config.json"
    "${BRUSHNET_CHECKPOINT}/diffusion_pytorch_model.safetensors"
    "${IPADAPTER_CHECKPOINT}"
    "${FUSION_CHECKPOINT}"
)
MISSING=0
for required_path in "${REQUIRED_PATHS[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing required path: ${required_path}" >&2
        MISSING=1
    fi
done
if [[ "${MISSING}" == "1" ]]; then
    echo "Restore the missing files; the init must be the T8 checkpoint-2250 widened to 14 inputs." >&2
    exit 1
fi
if ! grep -Eq '"conditioning_channels"[[:space:]]*:[[:space:]]*10([[:space:]]*[,}]|[[:space:]]*$)' "${BRUSHNET_CHECKPOINT}/config.json"; then
    echo "${BRUSHNET_CHECKPOINT} is not the 14-input init (conditioning_channels != 10)" >&2
    exit 1
fi
if [[ "${CHECK_PATHS_ONLY}" == "1" ]]; then
    echo "Required paths exist and BrushNet config declares conditioning_channels=10."
    exit 0
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
    "$@"
    --config "${CONFIG}"
)

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (num_processes=${NUM_PROCESSES})"
printf '%q ' "${CMD[@]}"; echo
if [[ "${PRINT_ONLY}" == "1" ]]; then
    exit 0
fi
cd "${BRUSHNET_DIR}"
mkdir -p "${OUTPUT_DIR}"
"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
