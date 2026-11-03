# BG48 then BG46 student flow at square 512x512

The queue starts BG48 from the original constant-schedule checkpoint 7500.
After BG48 finishes 4,000 new steps, it reads all 16 saved checkpoint metadata
files and selects the lowest validation background EPE. It verifies that the
selected checkpoint agrees with `best.json`, records its path and SHA256 in
`selected_bg48_checkpoint_for_bg46.json`, then initializes BG46 from those
weights with a fresh optimizer/scheduler and a new 4,000-step counter.

## Locations

All outputs are under `/user/Cilab/Quan/experiment/`:

- Queue: `train_v7_raft_student_sfu_bg48_then_bg46_512_queue/`
- BG48: `train_v7_raft_student_sfu_bg48_512_from7500/`
- BG46: `train_v7_raft_student_sfu_bg46_512_from_bg48_best/`

Each stage stores `dataset`, `teacher_flows_512x512`, `student`, `logs`,
`experiment_config.json`, `runtime_config.json`, `preflight_audit.json`,
`teacher_cache_provenance.json`, and copies of its code in `scripts`.
BG46 initially has `queued_config.json`; its full initialization contract is
written only when the best BG48 checkpoint is known.

## Data and supervision

Source datasets are `sfu_train_test_80_20/bg48` and `bg46`. For both, the
complete source `train` folder is training (4,824 frames, 4,810 adjacent pairs)
and the source `test` folder is validation (1,206 frames, 1,192 pairs).
The manifest prevents pairs from crossing videos or splits.

Training and validation RGB are Pillow bilinear square 512x512. Masks use
nearest resize and background-one convention internally. Student and teacher
use the same RGB preprocessing. Cached labels must contain forward/backward
flow `[2,512,512]` and validity masks `[1,512,512]`, float32 and finite.

Clean teacher labels can be reused from the completed BG56 **512** experiment
after checking all 12,060 GT/mask files and all split segments. Shared inodes
prove identity directly; other files are compared byte for byte. The teacher
reference's successful full-cache audit and unchanged label modification
times are checked, then the 6,002 cache files are hard-linked into the new
stage with separate dataset metadata and a reuse proof. If a clean GT/mask
differs, the queue computes a new cache from that stage's own clean GT using
the original ProPainter RAFT, 20 iterations. No max960 artifact is used.

## Optimization and validation

- Initial GPUs: physical 0, 2, 5, used sequentially by the two cases.
- Batch per GPU 4, accumulation 1: effective global batch **12** on three GPUs.
  This differs from the BG56 run's global batch 16 because the GPU count changed.
- AdamW, learning rate 1e-5 constant, fp16, frozen BatchNorm, seed 20260903.
- 20 RAFT iterations with iterative teacher supervision, gamma 0.8.
- Loss weights: teacher 1, forward/backward consistency 0.1, smoothness 0.01;
  region `all`, matching the completed BG56 experiment.
- Validation and full-state checkpoint every 250 steps; retention limit 20.
- Validation uses the same eight evenly spaced pairs per sequence, 112 total.
  Each rank evaluates disjoint indices without Accelerator's batch-padding
  duplication, so validation support counts remain the same on 2 or 3 GPUs.
- `best.json` retains the historical best on resume.

## Launch, pause and resume

From the BrushNet repository:

```bash
/home/cilab/ndquan/envs/guided_diff/bin/python \
  examples/brushnet/proposed/STC_encoder_v7_raft_flow_distillation/run_bg48_then_bg46_student_512.py \
  --gpus 0 2 5
```

Request a pause after the next complete checkpoint:

```bash
/home/cilab/ndquan/envs/guided_diff/bin/python \
  examples/brushnet/proposed/STC_encoder_v7_raft_flow_distillation/run_bg48_then_bg46_student_512.py \
  --request-pause
```

Wait for the queue status to become `paused`. The dependent stage will not
start while a pause is requested. Resume the active stage with full state:

```bash
/home/cilab/ndquan/envs/guided_diff/bin/python \
  examples/brushnet/proposed/STC_encoder_v7_raft_flow_distillation/run_bg48_then_bg46_student_512.py \
  --resume --gpus 0 2 5
```

A later resume on two idle GPUs is supported by `--resume --gpus 0 2`.
It uses batch 3/GPU and accumulation 2, keeping the effective global batch 12.
The shuffled loader position is not restored by the original trainer, so
sample order need not match an uninterrupted run bit for bit.

The queue waits if the selected GPUs are occupied. Each stage has independent
checkpoints, optimizer states, logs and validation results. The detached queue
process continues after the chat ends. Inspect queue `status.json` to find the
active stage, and that stage's `status.json` or `logs/train.log` for progress.
