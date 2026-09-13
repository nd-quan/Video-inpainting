#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/cilab/ndquan/videoInpainting/code/BrushNet
PY=/home/cilab/ndquan/envs/guided_diff/bin/python
OUT="$ROOT/experiments/v8_video_metric_cleanup_20260911"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=2 "$PY" "$ROOT/Quan_test/complete_v8_evaluations.py" --inventory "$OUT/inventory.json" --worker 0 > "$OUT/worker_0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=3 "$PY" "$ROOT/Quan_test/complete_v8_evaluations.py" --inventory "$OUT/inventory.json" --worker 1 > "$OUT/worker_1.log" 2>&1 &
pid1=$!
wait "$pid0"; rc0=$?
wait "$pid1"; rc1=$?
"$PY" - "$OUT" <<'PY'
import csv,json,sys
from pathlib import Path
out=Path(sys.argv[1]); roots=json.loads((out/'inventory.json').read_text())['roots']; reports=[]; rows=[]
for r in roots:
 root=Path(r); done=root/'postprocess_v8/complete.json'
 if done.exists(): reports.append(json.loads(done.read_text()))
 metrics=root/'metrics_full_reference/metrics.json'
 if metrics.exists():
  for row in json.loads(metrics.read_text())['results']:rows.append({'eval_root':r,**row})
if rows:
 with (out/'all_metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
summary={'expected_runs':len(roots),'completed_runs':len(reports),'deleted_pngs':sum(r['deleted_pngs'] for r in reports),'freed_bytes':sum(r['freed_bytes'] for r in reports),'videos_verified':sum(r['videos_verified'] for r in reports),'reports':reports}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
PY
printf 'Worker exit codes: %s %s\n' "$rc0" "$rc1"
(( rc0 == 0 && rc1 == 0 ))
