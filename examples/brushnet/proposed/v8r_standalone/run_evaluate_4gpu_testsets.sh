#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${HERE}/../../../.." && pwd)"
CHECKPOINT_STEP="${CHECKPOINT_STEP:?Set CHECKPOINT_STEP}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the standalone training checkpoint}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/eval_v8r_standalone_ckpt${CHECKPOINT_STEP}_testsets_tg0}"
PYTHON_BIN="${PYTHON_BIN:-/home/cilab/ndquan/envs/guided_diff/bin/python}"
IFS=, read -r -a gpu_ids <<< "${GPU_IDS:-0,1,2,3}"
if (( ${#gpu_ids[@]} != 4 )); then
    echo "GPU_IDS must contain exactly four comma-separated GPU IDs" >&2
    exit 2
fi
for gpu in "${gpu_ids[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU ID: ${gpu}" >&2; exit 2; }
done
[[ -f "${CHECKPOINT_PATH}/stc_v8r_standalone_model/config.json" ]] || {
    echo "Missing standalone adapter checkpoint under ${CHECKPOINT_PATH}" >&2
    exit 1
}
mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT}"

pids=()
for shard in 0 1 2 3; do
    gpu="${gpu_ids[${shard}]}"
    (
        for split in test_1 test_2; do
            output="${OUTPUT_ROOT}/${split}/shard-${shard}"
            mkdir -p "${output}/terminal_logs"
            log="${output}/terminal_logs/evaluate.log"
            printf 'START split=%s shard=%s gpu=%s log=%s\n' "${split}" "${shard}" "${gpu}" "${log}"
            if ! env CUDA_VISIBLE_DEVICES="${gpu}" CHECKPOINT_STEP="${CHECKPOINT_STEP}" \
                CHECKPOINT_PATH="${CHECKPOINT_PATH}" TEST_SET="${split}" \
                OUTPUT_DIR="${output}" RUN_PREFLIGHT=0 \
                bash "${HERE}/run_evaluate.sh" --num_shards 4 --shard_index "${shard}" \
                > "${log}" 2>&1; then
                printf 'FAILED split=%s shard=%s gpu=%s log=%s\n' "${split}" "${shard}" "${gpu}" "${log}" >&2
                exit 1
            fi
            printf 'DONE split=%s shard=%s gpu=%s\n' "${split}" "${shard}" "${gpu}"
        done
    ) &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then failed=1; fi
done
if (( failed )); then
    echo "At least one shard failed; inspect shard logs. No combined summary was written." >&2
    exit 1
fi
for split in test_1 test_2; do
    "${PYTHON_BIN}" "${HERE}/merge_eval_shards.py" "${OUTPUT_ROOT}/${split}" --num_shards 4
done
printf 'COMPLETE output=%s\n' "${OUTPUT_ROOT}"
