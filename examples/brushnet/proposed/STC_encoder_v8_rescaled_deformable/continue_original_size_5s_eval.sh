#!/usr/bin/env bash
set -euo pipefail

# Wait for the five-sequence small-resolution batch before reusing its GPUs.
# Invoke from a detached tmux session.
subset_pid="${1:?small-resolution runner PID required}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="/home/cilab/ndquan/envs/guided_diff/bin/python"

while kill -0 "${subset_pid}" 2>/dev/null; do
    sleep 30
done

exec "${python}" "${script_dir}/run_original_size_5s_eval.py" \
    --skip-preflight --sequences PeopleOnStreet FourPeople KristenAndSara \
    BasketballPass BQMall BQSquare PartyScene RaceHorses --gpus 3,4,5
