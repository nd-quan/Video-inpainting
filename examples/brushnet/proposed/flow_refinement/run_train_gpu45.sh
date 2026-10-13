#!/usr/bin/env bash
set -euo pipefail

root=/home/cilab/ndquan/videoInpainting/code/BrushNet
output_dir="$root/experiments/train_flow_refinement_v8r_b29_gpu45"
mkdir -p "$output_dir"
cd "$root"

export CUDA_VISIBLE_DEVICES=4,5
export PYTHONUNBUFFERED=1

exec /home/cilab/ndquan/envs/guided_diff/bin/accelerate launch \
  --multi_gpu --num_processes 2 --mixed_precision fp16 --main_process_port 29640 \
  examples/brushnet/proposed/flow_refinement/train.py \
  --dataset_root /home/cilab/ndquan/videoInpainting/SFU_STC_flow \
  --teacher_flow_root /home/cilab/ndquan/videoInpainting/SFU_STC_flow/teacher_flows_512x512 \
  --raft_student_path experiments/train_v7_raft_student_flow/best.json \
  --output_dir "$output_dir" \
  --resolution 512 --batch_size 29 --pair_batch_size 1 \
  --gradient_accumulation_steps 1 --max_steps 10000 \
  --validation_every 250 --checkpoint_every 500 \
  --num_workers 2 --mixed_precision fp16 >> "$output_dir/train.log" 2>&1
