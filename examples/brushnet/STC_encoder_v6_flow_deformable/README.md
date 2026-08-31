# STC encoder V6

V6 is V5 plus first-order bidirectional flow-guided modulated deformable
alignment. See [V6_DESCRIPTOR.md](V6_DESCRIPTOR.md) for the mathematical and
training contract.

## Files

- `flow_guided_deformable_stc_adapter.py`: V6 model and V5 condition injection.
- `deformable_alignment_loss.py`: differentiable DCN feature/offset losses.
- `train_v6_flow_deformable.py`: staged training wrapper.
- `run_train_v6_flow_deformable_t16.sh`: T=16/S=12 launcher with terminal log.
- `evaluate_v6_flow_deformable.py`: exact V5/V8 evaluation with V6 diagnostics.
- `run_evaluate_v6_combined_testsets.sh`: shared-root test_1 + test_2 evaluation.
- `test_v6_flow_deformable.py`: identity, geometry, masking, gradient, and
  checkpoint tests.

## Validate the implementation

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet

/home/cilab/ndquan/envs/guided_diff/bin/python \
  examples/brushnet/STC_encoder_v6_flow_deformable/test_v6_flow_deformable.py -v
```

V6 supports FP32 and FP16. The installed torchvision deformable CUDA kernel
does not support BF16, and the launcher/parser reject it.

## Stage A: deformable modules only

Select a validated V5 checkpoint rather than relying on the default latest
pointer when a V5 run is still active.

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet

CUDA_VISIBLE_DEVICES=1,3,4,5 \
NUM_PROCESSES=4 \
MAIN_PROCESS_PORT=29610 \
TRAINING_STAGE=deform_only \
INIT_V5_MODEL=/path/to/v5/checkpoint-N/stc_v5_model \
OUTPUT_DIR=/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/train_stc_v6_deform_only_T16_S12_sharedNoise_0.95 \
MAX_TRAIN_STEPS=1000 \
LEARNING_RATE=2e-5 \
bash examples/brushnet/STC_encoder_v6_flow_deformable/run_train_v6_flow_deformable_t16.sh
```

For a short multi-step smoke test, use a new output directory:

```bash
CUDA_VISIBLE_DEVICES=1,3 \
NUM_PROCESSES=2 \
TRAINING_STAGE=deform_only \
INIT_V5_MODEL=/path/to/v5/checkpoint-N/stc_v5_model \
OUTPUT_DIR=/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/smoke_stc_v6_10steps \
MAX_TRAIN_STEPS=10 \
CHECKPOINTING_STEPS=5 \
DATALOADER_NUM_WORKERS=0 \
bash examples/brushnet/STC_encoder_v6_flow_deformable/run_train_v6_flow_deformable_t16.sh
```

## Stage B: joint temporal/output adaptation

```bash
CUDA_VISIBLE_DEVICES=1,3,4,5 \
NUM_PROCESSES=4 \
MAIN_PROCESS_PORT=29610 \
TRAINING_STAGE=joint \
INIT_V6_MODEL=/path/to/stage-A/checkpoint-N/stc_v6_model \
OUTPUT_DIR=/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/train_stc_v6_joint_T16_S12_sharedNoise_0.95 \
MAX_TRAIN_STEPS=4000 \
LEARNING_RATE=5e-6 \
bash examples/brushnet/STC_encoder_v6_flow_deformable/run_train_v6_flow_deformable_t16.sh
```

## Evaluate test_1 and test_2 into one sequence root

```bash
CUDA_VISIBLE_DEVICES=1 \
CHECKPOINT_STEP=1000 \
CHECKPOINT_PATH=/path/to/v6/checkpoint-1000 \
DEFORMABLE_ALIGNMENT_SCALE=1.0 \
bash examples/brushnet/STC_encoder_v6_flow_deformable/run_evaluate_v6_combined_testsets.sh
```

Use `DEFORMABLE_ALIGNMENT_SCALE=0` with the same checkpoint and seeds for the
exact V5-path ablation.
