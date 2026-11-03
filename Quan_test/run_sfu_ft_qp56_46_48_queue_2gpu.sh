#!/usr/bin/env bash
# Run QP56 -> QP46 -> QP48 sequentially on GPUs 0 and 1.
set -euo pipefail
export TZ=Asia/Seoul

BRUSHNET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
queue_root="${BRUSHNET_ROOT}/experiments/eval_sfu_ft_qp56_46_48_T8S2_queue_20261023"
mkdir -p "${queue_root}"
exec > >(tee -a "${queue_root}/queue.log") 2>&1
cd "${BRUSHNET_ROOT}"

active_qp=none
trap 'result=$?; printf "%s FAILED qp=%s exit=%s\n" "$(date -Is)" "${active_qp}" "${result}" > "${queue_root}/status.txt"; exit "${result}"' ERR
for job in 56:3000 46:2000 48:2000; do
    active_qp="${job%:*}"
    ckpt_step="${job#*:}"
    ckpt_dir="/user/Cilab/Quan/experiment/finetune_sfu_video_bg${active_qp}_512_from_coco8000_T8S4_3000steps_ddp3/checkpoint-${ckpt_step}"
    dataset_dir="/home/cilab/ndquan/vcm-rs/output/sfu_qp0_bg_table_window_test/bg${active_qp}/test"
    eval_dir="${BRUSHNET_ROOT}/experiments/eval_bg${active_qp}_sfu_ft_ckpt${ckpt_step}_latent_tg0p1_v7student_T8S2_512"
    printf '%s RUNNING qp=%s checkpoint=%s GPUs=0,1\n' "$(date -Is)" "${active_qp}" "${ckpt_step}" | tee "${queue_root}/status.txt"
    BG_QP="${active_qp}" CHECKPOINT_STEP="${ckpt_step}" \
        CHECKPOINT_DIR="${ckpt_dir}" LONG_TEST_ROOT="${dataset_dir}" \
        OUTPUT_ROOT="${eval_dir}" PREPARE_ONLY=0 \
        bash Quan_test/run_bg56_sfu_ft_latent_tg_v7_2gpu.sh
    printf '%s COMPLETE qp=%s checkpoint=%s\n' "$(date -Is)" "${active_qp}" "${ckpt_step}"
done
printf '%s ALL COMPLETE: QP56/3000, QP46/2000, QP48/2000\n' "$(date -Is)" | tee "${queue_root}/status.txt"
