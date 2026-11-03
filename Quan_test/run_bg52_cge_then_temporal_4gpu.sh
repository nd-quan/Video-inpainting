#!/usr/bin/env bash
# New isolated experiment; all arguments are forwarded to the Python launcher.
set -euo pipefail
BRUSHNET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
cd "${BRUSHNET_ROOT}"
exec "${PYTHON_BIN}" -u Quan_test/run_bg52_cge_then_temporal_4gpu.py "$@"
