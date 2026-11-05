#!/usr/bin/env bash
set -euo pipefail

root=/home/cilab/ndquan/videoInpainting/code/BrushNet
script_dir="$root/examples/brushnet"
source="$root/experiments/finetune_coco_bg46_native_area_bucket_ckpt200000_3000steps/checkpoint-2000"
output="$root/experiments/finetune_coco_bg48_from_bg46_step2000_native_area_bucket_3000steps"

if [[ -e "$output" ]]; then
  echo "Output directory already exists: $output" >&2
  exit 2
fi
for path in "$source/brushnet/config.json" "$source/brushnet/diffusion_pytorch_model.safetensors" \
    "$source/ipadapter/model.safetensors" "$source/ipadapter/fusion_module.safetensors"; do
  if [[ ! -s "$path" ]]; then
    echo "Source checkpoint file is missing or empty: $path" >&2
    exit 2
  fi
done

mkdir -p "$output"
cd "$script_dir"
export CUDA_VISIBLE_DEVICES=2,3
export PYTHONUNBUFFERED=1

exec /home/cilab/ndquan/envs/guided_diff/bin/python -m accelerate.commands.launch \
  --multi_gpu --num_machines 1 --num_processes 2 --main_process_port 29655 \
  --mixed_precision fp16 --dynamo_backend no \
  training/train_brushnet_coco_native_resolution.py \
  --pretrained_model_name_or_path "$script_dir/base_model/stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --coco_train_root "$script_dir/dataset/COCO_train_bg48" \
  --null_caption_file /home/cilab/ndquan/videoInpainting/SFU_STC_flow/caption_empty.txt \
  --brushnet_model_name_or_path "$source/brushnet" \
  --pretrained_ip_adapter_path "$source/ipadapter/model.safetensors" \
  --fusion_module_path "$source/ipadapter/fusion_module.safetensors" \
  --output_dir "$output" \
  --native_resolution --area_bucket_batches \
  --max_width 1920 --max_height 1152 --pad_multiple 64 \
  --train_batch_size 1 --gradient_accumulation_steps 1 \
  --max_train_steps 3000 --checkpointing_steps 500 \
  --learning_rate 1e-6 --lr_scheduler constant \
  --gradient_checkpointing --set_grads_to_none \
  --dataloader_num_workers 0 --seed 1234 \
  --report_to tensorboard --logging_dir tensorboard \
  --tracker_project_name coco_bg48_from_bg46_step2000 \
  --benchmark_log
