#!/usr/bin/env bash
# Two sequential phases; phase 2 restores the full phase-1 training state.
set -euo pipefail

BRUSH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/cilab/ndquan/envs/guided_diff/bin/python
TRAINER="$BRUSH/train_brushnet_VCM_ipadapter_v8_coco_nulltext_sharedNoise_sameBG_v0_0.py"
DATASET_ROOT="$BRUSH/dataset/sfu_bvi30_bg56"
INITIAL_CHECKPOINT=/user/Cilab/Quan/experiment/finetune_coco_bg56_512_from_ckpt200000_batch8_10000steps/checkpoint-8000
OUTPUT_DIR=/user/Cilab/Quan/experiment/finetune_sfu_bvi30_bg56_512_from_coco8000_T8S4_ind5000_shared3000_ddp4

DRY_RUN=0
if [[ "${1:-}" == --dry-run && $# == 1 ]]; then
    DRY_RUN=1
elif (( $# != 0 )); then
    echo "Usage: bash $0 [--dry-run]" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$BRUSH${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

COMMON=(
    "$PY" -m accelerate.commands.launch
    --multi_gpu --num_machines 1 --num_processes 4 --main_process_port 29686
    --mixed_precision fp16 --dynamo_backend no "$TRAINER"
    --pretrained_model_name_or_path "$BRUSH/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5"
    --brushnet_model_name_or_path "$INITIAL_CHECKPOINT/brushnet"
    --pretrained_ip_adapter_path "$INITIAL_CHECKPOINT/ipadapter/model.safetensors"
    --fusion_module_path "$INITIAL_CHECKPOINT/ipadapter/fusion_module.safetensors"
    --dataset_root "$DATASET_ROOT" --dataset_layout hierarchical --train_split train
    --null_caption_file /home/cilab/ndquan/videoInpainting/SFU_STC_flow/caption_empty.txt
    --output_dir "$OUTPUT_DIR" --resolution 512
    --clip_length 8 --clip_stride 4 --train_batch_size 1
    --gradient_accumulation_steps 3 --checkpointing_steps 500
    --learning_rate 1e-6 --lr_scheduler constant_with_warmup --lr_warmup_steps 50
    --variance_preserving_shared_noise --sequence_shared_noise_refresh epoch
    --mixed_precision fp16 --gradient_checkpointing --set_grads_to_none
    --dataloader_num_workers 2 --seed 1234
    --report_to tensorboard --logging_dir logs
    --tracker_project_name train_sfu_bvi30_bg56_noise_curriculum
)
PHASE1=(--max_train_steps 5000 --shared_bg_noise_strength 0.0)
PHASE2=(--max_train_steps 8000 --shared_bg_noise_strength 0.95
        --resume_from_checkpoint checkpoint-5000)

for required in "$PY" "$TRAINER" "$DATASET_ROOT/manifest.json" \
    "$INITIAL_CHECKPOINT/brushnet/config.json" \
    "$INITIAL_CHECKPOINT/ipadapter/model.safetensors" \
    "$INITIAL_CHECKPOINT/ipadapter/fusion_module.safetensors"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required path: $required" >&2
        exit 1
    fi
done

if (( DRY_RUN )); then
    printf 'GPU=%s; 1 clip/GPU x 8 frames x 4 GPUs x accumulation 3 = 96 frames/update\n' "$CUDA_VISIBLE_DEVICES"
    printf 'Phase 1: '
    printf '%q ' "${COMMON[@]}" "${PHASE1[@]}"
    printf '\nPhase 2: '
    printf '%q ' "${COMMON[@]}" "${PHASE2[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$OUTPUT_DIR"
exec 200>"$OUTPUT_DIR/launch.lock"
flock -n 200 || { echo "This training run already has an active launcher." >&2; exit 1; }
cd "$BRUSH"

if [[ -d "$OUTPUT_DIR/checkpoint-8000" ]]; then
    echo "checkpoint-8000 already exists; inspect it before starting another run." >&2
    exit 1
fi

if [[ ! -d "$OUTPUT_DIR/checkpoint-5000" ]]; then
    if compgen -G "$OUTPUT_DIR/checkpoint-*" > /dev/null; then
        echo "A partial training run exists; resume it explicitly before using this launcher." >&2
        exit 1
    fi
    printf 'Phase 1 (independent noise) started: %s\n' "$(date -Is)"
    "${COMMON[@]}" "${PHASE1[@]}" >> "$OUTPUT_DIR/phase1_independent.log" 2>&1
fi

# Exported weights are written after save_state; all must exist before phase 2.
for required in optimizer.bin scheduler.bin scaler.pt \
    pytorch_model.bin pytorch_model_1.bin pytorch_model_2.bin \
    random_states_0.pkl random_states_1.pkl random_states_2.pkl random_states_3.pkl \
    brushnet/config.json brushnet/diffusion_pytorch_model.safetensors \
    ipadapter/model.safetensors ipadapter/fusion_module.safetensors; do
    if [[ ! -s "$OUTPUT_DIR/checkpoint-5000/$required" ]]; then
        echo "checkpoint-5000 is incomplete: missing $required" >&2
        exit 1
    fi
done

printf 'Phase 2 (shared BG noise, rho=0.95) started: %s\n' "$(date -Is)"
"${COMMON[@]}" "${PHASE2[@]}" >> "$OUTPUT_DIR/phase2_shared.log" 2>&1
test -s "$OUTPUT_DIR/checkpoint-8000/ipadapter/fusion_module.safetensors"
printf 'Both phases completed: %s\n' "$(date -Is)"
