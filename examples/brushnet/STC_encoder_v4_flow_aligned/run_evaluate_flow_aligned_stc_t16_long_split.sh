#!/usr/bin/env bash
# Evaluate all 17 SFU long-test sequences with V4, split by clip count.
#
# PART=1 has 161 T16/S12 windows; PART=2 has 166.  Each part owns a separate
# output root, so the two evaluators never race on summary.json or clip files.
# Leave VIDEO_FPS unset: imgToVideo_rgb_stc_eval.py then uses the per-sequence
# SFU FPS table (24/30/50/60) rather than one incorrect global FPS.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
BASE_LAUNCHER="${SCRIPT_DIR}/run_evaluate_flow_aligned_stc_t16.sh"

PART="${PART:?Set PART=1 or PART=2}"
case "${PART}" in
    1)
        # 161 clips / 1,920 frames.
        BRANCHES=(
            Class_B/BQTerrace Class_D/BQSquare Class_E/Johnny
            Class_B/BasketballDrive Class_C/PartyScene Class_D/BlowingBubbles
            Class_C/RaceHorsesC Class_B/Kimono
        )
        ;;
    2)
        # 166 clips / 1,970 frames.
        BRANCHES=(
            Class_C/BQMall Class_E/FourPeople Class_E/KristenAndSara
            Class_B/Cactus Class_D/BasketballPass Class_A/PeopleOnStreet
            Class_A/Traffic Class_D/RaceHorsesD Class_B/ParkScene
        )
        ;;
    *)
        echo "PART must be 1 or 2, got: ${PART}" >&2
        exit 2
        ;;
esac

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${BRUSHNET_ROOT}/experiments/train_stc_v4_T16_flow005/checkpoint-4000}"
LONG_TEST_ROOT="${LONG_TEST_ROOT:-/home/cilab/ndquan/videoInpainting/SFU_STC_flow}"
OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/eval_stc_v4_T16_flow_aligned/checkpoint-4000/long_test-blurred-k51/part${PART}}"

echo "V4 long-test part ${PART}: ${#BRANCHES[@]} sequences"
echo "branches=${BRANCHES[*]}"

CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
DATASET_ROOT="${LONG_TEST_ROOT}" \
DATASET_LAYOUT=hierarchical \
SPLIT=long_test \
OUTPUT_DIR="${OUTPUT_DIR}" \
ROI_COMPOSITE=blurred \
ROI_BLUR_KERNEL_SIZE=51 \
bash "${BASE_LAUNCHER}" --include_branches "${BRANCHES[@]}" "$@"
