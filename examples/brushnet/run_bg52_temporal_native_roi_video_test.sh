#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/cilab/ndquan/videoInpainting/code/BrushNet
cd "$repo_root"

output_root="$repo_root/experiments/eval_bg52_temporal_native_roi_video"
dataset_root=/home/cilab/ndquan/vcm-rs/output/sfu_qp0_bg_table_window_test/bg52/test
pipeline=examples/brushnet/test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion_fixedBG_temporal_v0_postprocess_native_roi_video.py
python_bin=/home/cilab/ndquan/envs/guided_diff/bin/python

mkdir -p "$output_root/terminal_logs"
exec > >(tee -a "$output_root/terminal_logs/run.log") 2>&1

export CUDA_VISIBLE_DEVICES=3
export CHECKPOINT_DIR="$repo_root/experiments/train_sharedNoise_sameBG_0.95_T8/checkpoint-2250"
export V7_RAFT_STUDENT_PATH="$repo_root/experiments/train_v7_raft_student_flow/checkpoint-0004750/raft_student"
export GUIDANCE_MODE=temporal
export TEMPORAL_FLOW_BACKEND=v7_student
export TEMPORAL_GUIDANCE_SPACE=latent
export TEMPORAL_GUIDANCE_SCALE=0.1
export TEMPORAL_START_STEP=0
export TEMPORAL_END_STEP=50
export TEMPORAL_EVERY_N_STEPS=1
export TEMPORAL_DETACH_PREVIOUS=1
export TEMPORAL_BG_MASK_MODE=pair_intersection
export TEMPORAL_CLIP_SIZE=4
export TEMPORAL_CLIP_STRIDE=4
export NUM_INFERENCE_STEPS=50
export SHARED_BG_NOISE_STRENGTH=1.0
export VARIANCE_PRESERVING_SHARED_NOISE=1
export MAIN_NOISE_SEED=1234
export SHARED_NOISE_SEED=6789
export MAX_IMAGES=0
export FORCE_REGENERATE=0

echo "[stage] preflight ParkScene"
"$python_bin" "$pipeline" \
    --dataset-root "$dataset_root" --output-root "$output_root" \
    --sequences ParkScene --preflight

echo "[stage] smoke ParkScene: inference -> resize -> native ROI -> MP4"
"$python_bin" "$pipeline" \
    --dataset-root "$dataset_root" --output-root "$output_root" \
    --sequences ParkScene
test -s "$output_root/test/videos/resized_to_original_native_roi/Class_B/ParkScene.mp4"

echo "[stage] full BG52 test"
"$python_bin" "$pipeline" \
    --dataset-root "$dataset_root" --output-root "$output_root"

video_count=$(find "$output_root/test/videos/resized_to_original_native_roi" -type f -name '*.mp4' | wc -l)
if [[ "$video_count" -ne 14 ]]; then
    echo "Expected 14 MP4 videos; found $video_count" >&2
    exit 1
fi
echo "[done] 14 BG52 videos: $output_root/test/videos/resized_to_original_native_roi"
