# V7: clean-to-degraded RAFT flow distillation

This folder is intentionally a standalone flow-learning stage.  A frozen
ProPainter RAFT teacher has already generated clean-GT flow in
`teacher_flows_512x512`.  A second, trainable RAFT receives the corresponding
degraded RGB pairs and predicts the same bidirectional motion:

```text
clean GT pair -> frozen RAFT -> cached F_teacher
degraded RGB pair -> trainable RAFT -> F_student
                                      |
                         L_teacher + L_FB + L_smooth
```

This is not a V6/DCN training run.  It makes one question measurable before
deformation is introduced: does a degradation-adapted RAFT predict the clean
teacher correspondence better than the current lightweight STC flow head?

The student input is full degraded RGB in `[-1,1]`; `M_BG` is used only for
optional BG-only supervision and metrics.  RAFT produces full-resolution flow
in `[dx,dy]`, with the same forward/backward convention as V4--V6.  Later,
the selected student flow must be resized to the STC feature grid with both
spatial resize and displacement scaling.

## Training

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet

CUDA_VISIBLE_DEVICES=1,3 \
NUM_PROCESSES=2 \
TRAIN_BATCH_SIZE=1 \
RAFT_PAIR_BATCH_SIZE=1 \
LEARNING_RATE=1e-5 \
RAFT_ITERATIONS=20 \
MAX_TRAIN_STEPS=5000 \
OUTPUT_DIR=/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/train_v7_raft_student_flow \
bash examples/brushnet/STC_encoder_v7_raft_flow_distillation/run_train_raft_student_flow.sh
```

`TRAIN_BATCH_SIZE` is the number of adjacent image pairs loaded by one rank;
`RAFT_PAIR_BATCH_SIZE` chunks those pairs inside a RAFT direction.  Keep both
at one for the first 512x512 run.  Unlike T=16 STC training, RAFT is pairwise,
so a clip batch offers no temporal receptive-field benefit and only increases
the correlation-volume memory cost.

The default objective is:

\[
L = L_{teacher}^{iterative} + 0.1 L_{FB} + 0.01 L_{smooth}.
\]

All RAFT iterations are supervised with normalized geometric weights
`gamma=0.8`, retaining the configured overall teacher-loss scale.  Set
`FINAL_FLOW_ONLY=1` only for the final-flow ablation.

## Outputs and selection

Each checkpoint contains:

```text
checkpoint-0000250/
  raft_student/          # portable RAFT student state + config
  accelerator_state/     # exact optimizer/scheduler resume state
  metadata.json
```

`best.json` is selected by validation BG EPE by default.  Validation also
logs all/BG EPE, zero-flow gain, residual quantiles, motion-bin EPE, and the
fraction whose error at the 64x64 feature grid exceeds V6's residual range of
two feature pixels.

Evaluate the selected student without training:

```bash
OUTPUT_DIR=/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/train_v7_raft_student_flow \
CHECKPOINT=best \
SPLIT=valid \
bash examples/brushnet/STC_encoder_v7_raft_flow_distillation/run_evaluate_raft_student_flow.sh
```

Only after the student outperforms the lightweight baseline on those flow
metrics should it replace the V6 base-flow predictor.  Initially RAFT should
remain frozen during V6 `deform_only` training.

## Flow visualization

`visualize_raft_student_flow.py` is separate from the V4++ visualizer because
a V7 checkpoint contains `raft_student`, not an STC feature-flow model.  It
renders the degraded pair, cached clean-RAFT teacher flow, RAFT-student flow,
signed residual flow, EPE maps, cached-valid masks, and white-is-BG masks.

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet

CUDA_VISIBLE_DEVICES=1 \
CHECKPOINT=/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/train_v7_raft_student_flow/best.json \
SPLIT=valid \
PAIRS_PER_SEQUENCE=2 \
OUTPUT_DIR=/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/visualize_v7_raft_student_flow/best-valid \
bash examples/brushnet/STC_encoder_v7_raft_flow_distillation/run_visualize_raft_student_flow.sh
```

The default selects two adjacent pairs distributed over every sequence.  Add
`--include_branches Class_C/BQMall` after the shell command to limit the run,
or raise `PAIRS_PER_SEQUENCE` for denser coverage.
