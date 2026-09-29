# Standalone V8-R (spatial base, distilled RAFT)

This folder is the entrypoint for the two-stage experiment. Stage 1 distills a
frozen clean-frame RAFT teacher into a RAFT student that reads degraded RGB.
Stage 2 freezes that student and the existing BrushNet/diffusion U-Net, trains
the temporal adapter, and evaluates it with optional temporal DDIM guidance.

The adapter implementation is in `model.py`. It **does not instantiate or run**
the V5 lightweight flow predictor or the V5 BG alignment module. Each RGB
frame concatenated with its BG mask is encoded independently into 64-channel
spatial features. The frozen RAFT student estimates both directions of RGB
flow. Features are nearest-upsampled by ×4, backward-warped with resized RAFT
flow, and nearest-downsampled. A DCNv2 block predicts **residual offsets only**
from target/prewarped features, their difference, normalized student flow and
BG masks; the flow is not added to DCN offsets a second time. BG-valid
previous/current/next features enter fusion, followed by relative temporal
attention and a zero-conv 64→4 BG-only latent correction. Cross-clip memory
uses overlapping absolute frame IDs. Temporal guidance at evaluation is a
separate DDIM operation on the predicted-clean latent (or RGB).

The new code reuses the repository's dataset, RAFT distillation loop,
diffusion training loop, and evaluation loop from sibling `STC_encoder_v*`
packages; those packages and the BrushNet environment remain dependencies.
"Standalone" here means the **V8-R model architecture and its launch/config
entrypoints** are colocated, not that every upstream library is vendored.

## Stage 1: RAFT distillation

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet
CUDA_VISIBLE_DEVICES=0 NUM_PROCESSES=1 \
  OUTPUT_DIR="$PWD/experiments/train_v8r_standalone_flow" \
  bash examples/brushnet/proposed/v8r_standalone/run_train_flow.sh

OUTPUT_DIR="$PWD/experiments/train_v8r_standalone_flow" \
  bash examples/brushnet/proposed/v8r_standalone/run_evaluate_flow.sh
```

`OUTPUT_DIR/best.json` is a pointer to the selected RAFT student checkpoint.
An already validated V7 RAFT student may also be used directly; there is no
need to repeat stage 1 solely to exercise this architecture.

## Stage 2: temporal adapter

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet
CUDA_VISIBLE_DEVICES=0 NUM_PROCESSES=1 \
  INIT_V8_MODEL="$PWD/experiments/train_stc_v8_rescaled_x4_joint_raftbest_T16_S12_sharedNoise_0.95/checkpoint-4000/stc_v8r_model" \
  RAFT_STUDENT_PATH="$PWD/experiments/train_v7_raft_student_flow/best.json" \
  OUTPUT_DIR="$PWD/experiments/train_v8r_standalone" \
  RESCALED_WARP_SCALE=4 JOINT_UNFREEZE_STEP=500 \
  DEFORM_LEARNING_RATE=2e-5 TEMPORAL_LEARNING_RATE=5e-6 \
  DEFORM_ALIGNMENT_LOSS_WEIGHT=0.1 DEFORM_OFFSET_LOSS_WEIGHT=0.0005 \
  CLIP_LENGTH=16 CLIP_STRIDE=12 SHARED_BG_NOISE_STRENGTH=0.95 \
  MAX_TRAIN_STEPS=4000 RUN_PREFLIGHT=1 \
  bash examples/brushnet/proposed/v8r_standalone/run_train_temporal.sh
```

Migration from an existing V8/V8-R checkpoint copies only spatial encoder,
relative temporal blocks, output norm, and 64→4 zero-conv. It deliberately
**resets DCN and fusion**, because the old V5-aligned base has different input
semantics. Migration reports copied/new keys at startup. Spatial encoder and
RAFT stay frozen. DCN/fusion start training immediately; temporal blocks and
zero-conv are held for `JOINT_UNFREEZE_STEP` optimizer updates, then train at
the lower temporal LR. BrushNet and diffusion U-Net remain frozen. The default
extra temporal training loss is zero; set `TEMPORAL_TRAIN_LOSS_WEIGHT` only for
a separate controlled experiment.

## Evaluation and temporal guidance

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet
CUDA_VISIBLE_DEVICES=0 CHECKPOINT_STEP=2500 \
  TEST_SET=test_1 SEQUENCE_NAME=BasketballPass \
  RAFT_STUDENT_PATH="$PWD/experiments/train_v7_raft_student_flow/best.json" \
  OUTPUT_DIR="$PWD/experiments/eval_v8r_standalone_basketballpass" \
  STC_INJECTION_SCALE=1.0 DEFORMABLE_ALIGNMENT_SCALE=1.0 \
  TEMPORAL_GUIDANCE_SPACE=latent TEMPORAL_GUIDANCE_SCALE=0.1 \
  TEMPORAL_START_STEP=0 TEMPORAL_END_STEP=50 \
  NUM_INFERENCE_STEPS=50 RUN_PREFLIGHT=1 \
  bash examples/brushnet/proposed/v8r_standalone/run_evaluate.sh
```

Set `TEMPORAL_GUIDANCE_SCALE=0` for no temporal DDIM guidance. Use
`STC_INJECTION_SCALE=0` as the same-checkpoint no-STC ablation. `TEST_SET` is
`test_1` or `test_2`; omit `SEQUENCE_NAME` to evaluate all sequences in that
set. The adapter checkpoint component is
`checkpoint-N/stc_v8r_standalone_model`, **not** the legacy `stc_v8r_model`.

Before a full run, compare one-sequence metrics/video against the existing
shared-noise + temporal-guidance baseline with identical masks, seeds and
sampling settings. The model and entrypoints were smoke-tested on CPU; a
complete GPU train/evaluation has not been run by this change.
