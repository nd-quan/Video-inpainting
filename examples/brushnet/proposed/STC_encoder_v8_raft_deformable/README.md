# STC encoder V8

V8 uses the selected frozen V7 RAFT student as the DCN base-flow prior while
retaining V5 as the zero-init identity stream.  See
[V8_DESCRIPTOR.md](V8_DESCRIPTOR.md) for the exact model and loss contract.

## Validate code

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet

/home/cilab/ndquan/envs/guided_diff/bin/python \
  examples/brushnet/STC_encoder_v8_raft_deformable/test_v8_raft_deformable.py -v
```

The tests verify V5 identity at warm start, RGB-to-feature flow-vector scaling,
external-flow requirement, and gradients into DCN parameters.

## Stage A: frozen V7 flow, DCN only

Use V7 `checkpoint-0004750` (the best validation BG-EPE checkpoint), not the
near-zero-LR checkpoint 5000.

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet

CUDA_VISIBLE_DEVICES=1,3 \
NUM_PROCESSES=2 \
RAFT_STUDENT_PATH=/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/train_v7_raft_student_flow/checkpoint-0004750 \
INIT_V5_MODEL=/path/to/selected_v5/checkpoint-N/stc_v5_model \
OUTPUT_DIR=/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/train_stc_v8_raft_deform_only_T16_S12_sharedNoise_0.95 \
MAX_TRAIN_STEPS=1000 \
LEARNING_RATE=2e-5 \
bash examples/brushnet/STC_encoder_v8_raft_deformable/run_train_v8_raft_deformable_t16.sh
```

`RAFT_PAIR_BATCH_SIZE=1` is recommended first.  Each V8 rank holds the frozen
RAFT model in addition to the diffusion stack, so this run is materially more
memory-intensive and slower than V6.  The launcher records a terminal log in
`OUTPUT_DIR/terminal_logs/`.

## Evaluate into one root for test_1 + test_2

```bash
CUDA_VISIBLE_DEVICES=1 \
CHECKPOINT_STEP=1000 \
CHECKPOINT_PATH=/path/to/v8/checkpoint-1000 \
RAFT_STUDENT_PATH=/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/train_v7_raft_student_flow/checkpoint-0004750 \
DEFORMABLE_ALIGNMENT_SCALE=1.0 \
bash examples/brushnet/STC_encoder_v8_raft_deformable/run_evaluate_v8_combined_testsets.sh
```

Use the same seeds and `DEFORMABLE_ALIGNMENT_SCALE=0` as the exact V5-path
ablation.  This does not turn off V5's original light-flow alignment; it turns
off only V8's learned DCN residual fusion.
