# STC Encoder V5

V5 adds learned relative temporal bias and DDP-safe paired cross-clip overlap
memory to the repaired V4++ model. See `V5_DESCRIPTOR.md` for the exact model,
data, loss, and checkpoint contracts.

## Files

- `relative_crossclip_stc_adapter.py`: V5 model and explicit temporal memory.
- `cross_clip_data.py`: predecessor/current datasets and training collation.
- `train_v5_relative_crossclip.py`: V5 integration with the frozen V8 trainer.
- `run_train_v5_relative_crossclip_t16.sh`: audited T16/S12 launcher.
- `evaluate_v5_relative_crossclip.py`: paired-context evaluator.
- `run_evaluate_v5_combined_testsets.sh`: combined test_1/test_2 evaluation.
- `test_v5_relative_crossclip.py`: identity, memory, gradient, ID, and save/load tests.

## Train on GPU 1 and 3

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet

CUDA_VISIBLE_DEVICES=1,3 \
NUM_PROCESSES=2 \
RUN_PREFLIGHT=1 \
bash examples/brushnet/STC_encoder_v5_relative_crossclip/run_train_v5_relative_crossclip_t16.sh
```

For a one-update smoke test, use a new output directory:

```bash
CUDA_VISIBLE_DEVICES=1,3 \
NUM_PROCESSES=2 \
MAX_TRAIN_STEPS=1 \
CHECKPOINTING_STEPS=1 \
OUTPUT_DIR=/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/smoke_stc_v5 \
bash examples/brushnet/STC_encoder_v5_relative_crossclip/run_train_v5_relative_crossclip_t16.sh
```

Exact resume:

```bash
CUDA_VISIBLE_DEVICES=1,3 \
NUM_PROCESSES=2 \
RESUME_FROM_CHECKPOINT=latest \
bash examples/brushnet/STC_encoder_v5_relative_crossclip/run_train_v5_relative_crossclip_t16.sh
```

## Evaluate a trained checkpoint

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet

CUDA_VISIBLE_DEVICES=1 \
CHECKPOINT_STEP=2500 \
bash examples/brushnet/STC_encoder_v5_relative_crossclip/run_evaluate_v5_combined_testsets.sh
```

Evaluation recomputes the predecessor inside each sample. It therefore remains
correct when an existing clip is skipped during resume and does not depend on
hidden sequence state.

## Unit tests

```bash
/home/cilab/ndquan/envs/guided_diff/bin/python \
  examples/brushnet/STC_encoder_v5_relative_crossclip/test_v5_relative_crossclip.py -v
```

