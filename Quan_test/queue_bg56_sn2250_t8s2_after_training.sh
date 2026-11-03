#!/usr/bin/env bash
# Start the fixed BG56 evaluation only after its three GPU training run exits.
set -euo pipefail

TRAIN_PANE_PID="${1:?Pass the current training tmux pane PID}"
TRAIN_OUT=/user/Cilab/Quan/experiment/finetune_sfu_video_bg56_512_from_coco8000_T8S4_3000steps_ddp3
BRUSHNET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${BRUSHNET_ROOT}/experiments/eval_bg56_sn2250_latent_tg0p1_v7student_T8S2_512"
RUNNER="${BRUSHNET_ROOT}/Quan_test/run_bg56_sn2250_latent_tg_v7_3gpu.sh"
STATUS_FILE="${OUTPUT_ROOT}/queue_status.txt"
mkdir -p "${OUTPUT_ROOT}/terminal_logs"

printf '%s queued; waiting for training pane PID %s\n' "$(date -Is)" "${TRAIN_PANE_PID}" | tee "${STATUS_FILE}"
while kill -0 "${TRAIN_PANE_PID}" 2>/dev/null; do
    sleep 30
done

FINAL_CKPT="${TRAIN_OUT}/checkpoint-3000"
for required in \
    "${FINAL_CKPT}/brushnet/config.json" \
    "${FINAL_CKPT}/brushnet/diffusion_pytorch_model.safetensors" \
    "${FINAL_CKPT}/ipadapter/model.safetensors" \
    "${FINAL_CKPT}/ipadapter/fusion_module.safetensors"; do
    if [[ ! -s "${required}" ]]; then
        printf '%s training stopped without complete checkpoint-3000: %s\n' \
            "$(date -Is)" "${required}" | tee "${STATUS_FILE}" >&2
        exit 1
    fi
done
if ! grep -q '3000/3000' "${TRAIN_OUT}/launcher.log"; then
    printf '%s training stopped before step 3000\n' "$(date -Is)" | tee "${STATUS_FILE}" >&2
    exit 1
fi

printf '%s training complete; starting BG56 T8/S2 evaluation\n' "$(date -Is)" | tee "${STATUS_FILE}"
if bash "${RUNNER}"; then
    printf '%s evaluation complete\n' "$(date -Is)" | tee "${STATUS_FILE}"
else
    result=$?
    printf '%s evaluation failed with exit code %s\n' "$(date -Is)" "${result}" | tee "${STATUS_FILE}" >&2
    exit "${result}"
fi
