#!/usr/bin/env bash
set -euo pipefail
export WARP_MODE=direct
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_train.sh" "$@"
