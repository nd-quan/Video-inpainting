#!/usr/bin/env bash
# CPU-only smoke test of train_brushnet_latent_warp_v1.py (never uses a GPU).
#
# Runs ONE optimisation step twice from the Phase-1 init with the same seed:
#   A) latent warp ON  (drop_prob 0.0)  -> extra x0 pass + 14-ch condition
#   B) latent warp OFF (drop_prob 1.0)  -> zero warp channels (run-0 path)
# Because the new conv slices are zero at init, the step-0 losses must match.
#
# Needs a flow cache that covers BRANCH (e.g. made with
#   precompute_bg52_teacher_flows_512.py --device cpu --include-branches BRANCH).
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_DIR="$(cd -- "${HERE}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
TEST_ROOT="${TEST_ROOT:-/user/Cilab/Quan/experiment/latent_warp_run1/_cpu_test}"
FLOW_ROOT="${FLOW_ROOT:-${TEST_ROOT}/flows}"
BRANCH="${BRANCH:-Class_D/DAmericanFootballS2Harmonics_480x272_60fps_10bit_420}"
INIT="${INIT:-/user/Cilab/Quan/experiment/latent_warp_run1/init_T8_ckpt2250_14ch}"
CLIP_LENGTH="${CLIP_LENGTH:-3}"
THREADS="${THREADS:-8}"

export CUDA_VISIBLE_DEVICES=""   # CPU only
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}"

cd "${BRUSHNET_DIR}"
for mode in on off; do
    drop=0.0; [[ "${mode}" == "off" ]] && drop=1.0
    out="${TEST_ROOT}/dryrun_warp_${mode}"
    rm -rf "${out}"
    echo "=== warp ${mode} (drop_prob=${drop}) -> ${out}"
    nice -n 19 "${PYTHON_BIN}" "${HERE}/train_brushnet_latent_warp_v1.py" \
        --dataset_root /home/cilab/ndquan/vcm-rs/output/bvi_dvc_part1_qp0_bg_train/bg52 \
        --dataset_layout hierarchical \
        --include_branches "${BRANCH}" \
        --teacher_flow_root "${FLOW_ROOT}" \
        --brushnet_model_name_or_path "${INIT}/brushnet" \
        --pretrained_ip_adapter_path "${INIT}/ipadapter/model.safetensors" \
        --fusion_module_path "${INIT}/ipadapter/fusion_module.safetensors" \
        --output_dir "${out}" \
        --resolution 512 --clip_length "${CLIP_LENGTH}" --clip_stride 30 \
        --shared_bg_noise_strength 0.95 --sequence_shared_noise_refresh epoch \
        --latent_warp_drop_prob "${drop}" \
        --train_batch_size 1 --gradient_accumulation_steps 1 --max_train_steps 1 \
        --checkpointing_steps 100000 --learning_rate 1e-5 --lr_scheduler constant \
        --gradient_checkpointing --set_grads_to_none --dataloader_num_workers 0 \
        --seed 1234 --report_to tensorboard --mixed_precision no \
        2>&1 | tr '\r' '\n' | grep -E "Latent-warp first batch|Sequence-shared first batch|conv_in_condition|Teacher-flow cache|loss=|Error|error|Traceback" | tail -8
done
