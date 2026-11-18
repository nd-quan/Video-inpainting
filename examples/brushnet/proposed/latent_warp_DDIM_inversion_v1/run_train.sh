#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LATENT_WARP_TRAIN_SCRIPT="$here/train_ddim_inversion.py"
export INVERSION_TEACHER="${INVERSION_TEACHER:-/media/ssd1/ndquan/NAS_Quan/experiments/latent-warping/run1/train/checkpoint-6750}"
export INVERSION_STEPS="${INVERSION_STEPS:-50}"
exec bash "$here/../latent_warp_v1/run_train_latent_warp_v1_run1.sh" --config "$here/configs/train.yaml" "$@"
