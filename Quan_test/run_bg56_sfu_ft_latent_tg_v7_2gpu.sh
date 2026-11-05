#!/usr/bin/env bash
# Retest an SFU-finetuned checkpoint with the previous BG56 T8/S2 settings.
# PREPARE_ONLY=1 validates paths and the dataset without starting GPU inference.
set -euo pipefail

BRUSHNET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
BG_QP="${BG_QP:-56}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-3000}"
case "${BG_QP}" in
    46|48|56) ;;
    *) echo "Unsupported BG_QP: ${BG_QP}; expected 46, 48 or 56" >&2; exit 1 ;;
esac
export LONG_TEST_ROOT="${LONG_TEST_ROOT:-/home/cilab/ndquan/vcm-rs/output/sfu_qp0_bg_table_window_test/bg${BG_QP}/test}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/user/Cilab/Quan/experiment/finetune_sfu_video_bg${BG_QP}_512_from_coco8000_T8S4_3000steps_ddp3/checkpoint-${CHECKPOINT_STEP}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${BRUSHNET_ROOT}/experiments/eval_bg${BG_QP}_sfu_ft_ckpt${CHECKPOINT_STEP}_latent_tg0p1_v7student_T8S2_512}"
if [[ -f "${OUTPUT_ROOT}/parallel_session.txt" ]]; then
    parallel_session="$(<"${OUTPUT_ROOT}/parallel_session.txt")"
    current_session=""
    if [[ -n "${TMUX_PANE:-}" ]]; then
        current_session="$(tmux display-message -p -t "${TMUX_PANE}" '#S' 2>/dev/null || true)"
    fi
    if [[ "${parallel_session}" != "${current_session}" ]]; then
        echo "BG${BG_QP} already assigned to ${parallel_session}; observing its completion without launching workers."
        while tmux has-session -t "${parallel_session}" 2>/dev/null; do sleep 30; done
        if [[ -s "${OUTPUT_ROOT}/test/metrics_full_reference/metrics.csv" ]]; then
            echo "BG${BG_QP} parallel evaluation is complete: ${OUTPUT_ROOT}"
            exit 0
        fi
        echo "BG${BG_QP} parallel session ended before metrics completed; inspect ${OUTPUT_ROOT}/terminal_logs" >&2
        exit 1
    fi
fi
export V7_RAFT_STUDENT_PATH="${BRUSHNET_ROOT}/experiments/train_v7_raft_student_flow/checkpoint-0004750/raft_student"
export PYTHONPATH="${BRUSHNET_ROOT}/examples/brushnet/proposed:${PYTHONPATH:-}"
export DISCOVER_HIERARCHICAL_SEQUENCES=1
export MAX_IMAGES=0 FORCE_REGENERATE=0 PREFLIGHT_ONLY=0
export GUIDANCE_MODE=temporal TEMPORAL_FLOW_BACKEND=v7_student
export TEMPORAL_GUIDANCE_SPACE=latent TEMPORAL_GUIDANCE_SCALE=0.1
export TEMPORAL_START_STEP=0 TEMPORAL_END_STEP=50 TEMPORAL_EVERY_N_STEPS=1
export TEMPORAL_CLIP_SIZE=8 TEMPORAL_CLIP_STRIDE=2 TEMPORAL_BG_MASK_MODE=pair_intersection
export TEMPORAL_DETACH_PREVIOUS=1 NUM_INFERENCE_STEPS=50
export TEMPORAL_FLOW_BATCH_SIZE=2 TEMPORAL_LOSS_SCALE=1024 TEMPORAL_DECODE_CHUNK_SIZE=1
export SHARED_BG_NOISE_STRENGTH=0.95 VARIANCE_PRESERVING_SHARED_NOISE=1
export MAIN_NOISE_SEED=1234 SHARED_NOISE_SEED=6789
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
unset SEQUENCES

cd "${BRUSHNET_ROOT}"
pipeline=examples/brushnet/inference_experiments/test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion_fixedBG_temporal_v0_postprocess_native_roi_video.py
for required in \
    "${CHECKPOINT_DIR}/brushnet/config.json" \
    "${CHECKPOINT_DIR}/brushnet/diffusion_pytorch_model.safetensors" \
    "${CHECKPOINT_DIR}/ipadapter/model.safetensors" \
    "${CHECKPOINT_DIR}/ipadapter/fusion_module.safetensors"; do
    if [[ ! -s "${required}" ]]; then
        echo "Missing or empty checkpoint component: ${required}" >&2
        exit 1
    fi
done
printf 'BG_QP=%s\nCHECKPOINT_DIR=%s\nOUTPUT_ROOT=%s\nGPUs=%s,%s\n' "${BG_QP}" "${CHECKPOINT_DIR}" "${OUTPUT_ROOT}" "${EVAL_GPU_0:-0}" "${EVAL_GPU_1:-1}"
"${PYTHON_BIN}" -u "${pipeline}" \
    --dataset-root "${LONG_TEST_ROOT}" --output-root "${OUTPUT_ROOT}" --preflight
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then
    echo "Preparation complete; GPU inference was not started."
    exit 0
fi

mkdir -p "${OUTPUT_ROOT}/terminal_logs"
# T8/S2 gives 278 windows on the first GPU and 286 on the second GPU.
(
    printf 'CHECKPOINT_DIR=%s\nCUDA_VISIBLE_DEVICES=%s\n' "${CHECKPOINT_DIR}" "${EVAL_GPU_0:-0}"
    CUDA_VISIBLE_DEVICES="${EVAL_GPU_0:-0}" "${PYTHON_BIN}" -u "${pipeline}" \
        --dataset-root "${LONG_TEST_ROOT}" --output-root "${OUTPUT_ROOT}" \
        --sequences BQTerrace BQMall BQSquare BasketballDrive Cactus --workers 2
) > "${OUTPUT_ROOT}/terminal_logs/gpu${EVAL_GPU_0:-0}.log" 2>&1 &
pid0=$!
(
    printf 'CHECKPOINT_DIR=%s\nCUDA_VISIBLE_DEVICES=%s\n' "${CHECKPOINT_DIR}" "${EVAL_GPU_1:-1}"
    CUDA_VISIBLE_DEVICES="${EVAL_GPU_1:-1}" "${PYTHON_BIN}" -u "${pipeline}" \
        --dataset-root "${LONG_TEST_ROOT}" --output-root "${OUTPUT_ROOT}" \
        --sequences BasketballDrill BasketballPass BlowingBubbles Kimono ParkScene \
                    PartyScene RaceHorsesC RaceHorsesD Traffic --workers 2
) > "${OUTPUT_ROOT}/terminal_logs/gpu${EVAL_GPU_1:-1}.log" 2>&1 &
pid1=$!

failed=0
wait "${pid0}" || failed=1
wait "${pid1}" || failed=1
if [[ "${failed}" -ne 0 ]]; then
    echo "A worker failed; inspect ${OUTPUT_ROOT}/terminal_logs/gpu*.log" >&2
    exit 1
fi

# Match the previous BG56 metric domain: 512x512 PNGs with GT resized to 512.
if [[ ! -f "${OUTPUT_ROOT}/test/metrics_full_reference/metrics.csv" ]]; then
    CUDA_VISIBLE_DEVICES="${EVAL_GPU_0:-0}" "${PYTHON_BIN}" -u Quan_test/evaluate_flat_sequence_metrics.py \
        --eval-root "${OUTPUT_ROOT}/test" --gt-root "${LONG_TEST_ROOT}" --hierarchical \
        > "${OUTPUT_ROOT}/terminal_logs/metrics.log" 2>&1
fi
echo "BG${BG_QP} inference, native-ROI videos and 512x512 metrics complete: ${OUTPUT_ROOT}"
