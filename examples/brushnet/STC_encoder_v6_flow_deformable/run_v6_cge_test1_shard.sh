#!/usr/bin/env bash
# Launch one independent V6-CGE clip shard after its assigned GPU is free.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: V6_CGE_GPU_IDS=1,3,4,5 $0 <shard-index>" >&2
    exit 2
fi

shard_index="$1"
IFS=',' read -r -a gpus <<< "${V6_CGE_GPU_IDS:-1,3,4,5}"
if [[ ! "$shard_index" =~ ^[0-9]+$ ]] || (( shard_index >= ${#gpus[@]} )); then
    echo "shard index must be in [0, ${#gpus[@]}); got: $shard_index" >&2
    exit 2
fi

gpu="${gpus[$shard_index]}"
brushnet_dir="/home/cilab/ndquan/videoInpainting/code/BrushNet/examples/brushnet"
python_bin="/home/cilab/miniconda3/envs/aeic/bin/python"
script_path="$brushnet_dir/STC_encoder_v6_flow_deformable/evaluate_v6_flow_deformable_cge.py"
output_root="/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/eval_stc_v6_deformable_cge/basketballpass_sharded"
log_path="$output_root/shard-${shard_index}.log"

mkdir -p "$output_root"

printf '[%s] shard=%s starts on physical GPU %s\n' \
    "$(date '+%F %T')" "$shard_index" "$gpu"

cd "$brushnet_dir"
set +e
CUDA_VISIBLE_DEVICES="$gpu" \
CGE_VCMRS_CUDA_VISIBLE_DEVICES="$gpu" \
CGE_VCMRS_MAX_PARALLEL=1 \
"$python_bin" "$script_path" \
    --baseline_checkpoint /home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/train_sharedNoise_sameBG_0.95_T8/checkpoint-2250 \
    --stc_adapter_path /home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/train_stc_v6_deform_only_T16_S12_sharedNoise_0.95/checkpoint-1000/stc_v6_model \
    --dataset_root /home/cilab/ndquan/videoInpainting/code/BrushNet/examples/brushnet/dataset/test_1 \
    --split test \
    --dataset_layout flat_test \
    --include_sequences BasketballPass \
    --clip_length 16 \
    --clip_stride 12 \
    --cge_start_step 35 \
    --cge_end_step 50 \
    --cge_every_n_steps 1 \
    --cge_max_evals 2 \
    --output_dir "$output_root" \
    --num_shards "${#gpus[@]}" \
    --shard_index "$shard_index" \
    2>&1 | tee -a "$log_path"
status="${PIPESTATUS[0]}"
set -e

printf '[%s] shard=%s exited with status=%s\n' \
    "$(date '+%F %T')" "$shard_index" "$status" | tee -a "$log_path"
exit "$status"
