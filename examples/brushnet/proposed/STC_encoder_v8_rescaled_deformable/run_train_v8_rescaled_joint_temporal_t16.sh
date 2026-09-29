#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRUSHNET_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
BASELINE_LAUNCHER="${SCRIPT_DIR}/run_train_v8_rescaled_joint_t16.sh"

# Keep the baseline launcher and output intact.  All of its environment
# overrides remain available to this temporal experiment.
export OUTPUT_DIR="${OUTPUT_DIR:-${BRUSHNET_ROOT}/experiments/train_stc_v8_rescaled_x4_joint_raftbest_T16_S12_sharedNoise_0.95_temporal_predclean_0.1}"

echo "temporal_loss=predicted-clean latent Charbonnier; weight=0.1; warmup=500; SNR_gamma=5.0; previous=detached"
exec "${BASELINE_LAUNCHER}" \
    --temporal_train_loss_weight 0.1 \
    --temporal_train_loss_warmup_steps 500 \
    --temporal_train_charbonnier_eps 0.001 \
    --temporal_train_detach_previous \
    --temporal_train_snr_gamma 5.0 \
    "$@"
