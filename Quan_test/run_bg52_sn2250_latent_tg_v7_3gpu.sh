#!/usr/bin/env bash
# Fixed 512x512 shared-noise + latent temporal-guidance evaluation of BG52/test.
set -euo pipefail

BRUSHNET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
export LONG_TEST_ROOT="${LONG_TEST_ROOT:-/home/cilab/ndquan/vcm-rs/output/sfu_qp0_bg_table_window_test/bg52/test}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${BRUSHNET_ROOT}/experiments/eval_bg52_sn2250_latent_tg0p1_v7student_512}"
export CHECKPOINT_DIR="${BRUSHNET_ROOT}/experiments/train_sharedNoise_sameBG_0.95_T8/checkpoint-2250"
export V7_RAFT_STUDENT_PATH="${BRUSHNET_ROOT}/experiments/train_v7_raft_student_flow/checkpoint-0004750/raft_student"
export PYTHONPATH="${BRUSHNET_ROOT}/examples/brushnet/proposed:${PYTHONPATH:-}"
export DISCOVER_HIERARCHICAL_SEQUENCES=1
export MAX_IMAGES=0 FORCE_REGENERATE=0
export GUIDANCE_MODE=temporal TEMPORAL_FLOW_BACKEND=v7_student
export TEMPORAL_GUIDANCE_SPACE=latent TEMPORAL_GUIDANCE_SCALE=0.1
export TEMPORAL_START_STEP=0 TEMPORAL_END_STEP=50 TEMPORAL_EVERY_N_STEPS=1
export TEMPORAL_CLIP_SIZE=4 TEMPORAL_BG_MASK_MODE=pair_intersection
export TEMPORAL_DETACH_PREVIOUS=1 NUM_INFERENCE_STEPS=50
export SHARED_BG_NOISE_STRENGTH=1 VARIANCE_PRESERVING_SHARED_NOISE=1
export MAIN_NOISE_SEED=1234 SHARED_NOISE_SEED=6789
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1

cd "${BRUSHNET_ROOT}"
mkdir -p "${OUTPUT_ROOT}/terminal_logs"
evaluator=examples/brushnet/inference_experiments/test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion_fixedBG_temporal_v0.py
PREFLIGHT_ONLY=1 "${PYTHON_BIN}" -u "${evaluator}" \
  > "${OUTPUT_ROOT}/terminal_logs/preflight.log" 2>&1

# Assign each sequence to exactly one GPU; all workers write disjoint folders.
CUDA_VISIBLE_DEVICES=3 SEQUENCES=Traffic,BQTerrace,BQMall,BQSquare \
  "${PYTHON_BIN}" -u "${evaluator}" > "${OUTPUT_ROOT}/terminal_logs/gpu3.log" 2>&1 &
pid3=$!
CUDA_VISIBLE_DEVICES=4 SEQUENCES=BasketballDrive,Cactus,BasketballDrill,PartyScene,Kimono \
  "${PYTHON_BIN}" -u "${evaluator}" > "${OUTPUT_ROOT}/terminal_logs/gpu4.log" 2>&1 &
pid4=$!
CUDA_VISIBLE_DEVICES=5 SEQUENCES=BasketballPass,BlowingBubbles,RaceHorsesC,RaceHorsesD,ParkScene \
  "${PYTHON_BIN}" -u "${evaluator}" > "${OUTPUT_ROOT}/terminal_logs/gpu5.log" 2>&1 &
pid5=$!

failed=0
wait "${pid3}" || failed=1
wait "${pid4}" || failed=1
wait "${pid5}" || failed=1
if [[ "${failed}" -ne 0 ]]; then
  echo "At least one GPU worker failed; inspect ${OUTPUT_ROOT}/terminal_logs" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=5 "${PYTHON_BIN}" -u Quan_test/evaluate_flat_sequence_metrics.py \
  --eval-root "${OUTPUT_ROOT}/test" --gt-root "${LONG_TEST_ROOT}" --hierarchical \
  > "${OUTPUT_ROOT}/terminal_logs/metrics.log" 2>&1
echo "Inference and metrics complete: ${OUTPUT_ROOT}"
