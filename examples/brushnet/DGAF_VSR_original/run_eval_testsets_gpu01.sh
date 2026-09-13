#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PY=/home/cilab/ndquan/envs/guided_diff/bin/python
OUT="$ROOT/experiments/eval_dgaf_original_dgaf_conf0_T3_S1/checkpoint-2000"
if [[ "${1:-}" == worker ]]; then
    label="$2"
    export CUDA_VISIBLE_DEVICES="$3" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
    cd "$ROOT"
    exec > >(tee -a "$OUT/logs/$label.log") 2>&1
    trap 'rc=$?; if ((rc)); then echo "FAILED exit=$rc"; echo "$rc" > "$OUT/logs/$label.failed"; fi' EXIT
    "$PY" "$SCRIPT_DIR/evaluate_dgaf.py" \
        --checkpoint "$ROOT/experiments/train_dgaf_original_dgaf_conf0_T3_S1/checkpoint-2000" \
        --dataset_root "$ROOT/examples/brushnet/dataset/$label" --dataset_layout flat_test \
        --output_dir "$OUT/$label" --clip_length 3 --clip_stride 1 --no-save_auxiliary
    "$PY" "$ROOT/Quan_test/imgToVideo_rgb_stc_eval.py" --eval_root "$OUT/$label" --frame_kinds final
    "$PY" "$SCRIPT_DIR/postprocess_eval.py" --eval_root "$OUT/$label"
    date -Iseconds > "$OUT/logs/$label.done"
    echo "DONE $label"
else
    mkdir -p "$OUT/logs"
    tmux new-session -d -s eval_dgaf_original_ckpt2000 -n test_1 "bash '$SCRIPT_DIR/run_eval_testsets_gpu01.sh' worker test_1 0"
    tmux set-option -t eval_dgaf_original_ckpt2000 remain-on-exit on
    tmux new-window -t eval_dgaf_original_ckpt2000 -n test_2 "bash '$SCRIPT_DIR/run_eval_testsets_gpu01.sh' worker test_2 1"
    echo "Started tmux session eval_dgaf_original_ckpt2000: test_1 GPU 0, test_2 GPU 1"
fi
