# Validation — 2026-09-10

Environment: `/home/cilab/ndquan/envs/guided_diff/bin/python`, Python 3.8,
PyTorch 2.0.1+cu117, Accelerate 0.21.0. Actual frozen RAFT student checkpoint
`checkpoint-0004750` and shared-noise baseline `checkpoint-2250` were loaded.

| Check | Result |
|---|---|
| CPU unit tests (11) | Pass |
| Default T16/S12 data/config preflight | Pass, 393 train clips |
| Direct, confidence off, T2/128, FP16, one optimizer update | Pass; loss 0.00208281, new-channel gradient norm 0.00768927 |
| DGAF 4×, confidence on, T2/128, FP16, one optimizer update | Pass; loss 0.00208246, new-channel gradient norm 0.00765488 |
| Direct, confidence on, T16/512, FP16, one optimizer update | Pass; loss 0.000368322, new-channel gradient norm 0.000679296 |
| DGAF 4×, confidence on, 2 GPUs, accumulation 2 | Pass |
| Save at step 1, restore optimizer/scaler/RNG/cursor, continue to step 2 | Pass |
| Two uninterrupted DDP updates, including second backward | Pass |
| Load saved DGAF checkpoint and evaluate one real validation clip, 3 DDIM steps | Pass, raw/final/reference PNGs and metric JSONs saved |
| Same evaluator with explicit direct-warp + confidence-off override | Pass |
| Python compilation / launcher shell syntax | Pass |

Unit tests cover pixel-displacement scaling, previous/backward versus
next/forward direction, zero-flow identity, fractional-motion distinction,
confidence toggling, out-of-bounds rejection, destination BG gating with allowed
source ROI, cross-clip isolation, detached memory, bidirectional normalization,
effective residual targets after DDIM, warm-start baseline equivalence, native
model save/load, new-channel gradients, CFG duplication, and per-call cache reset.

The resumed and uninterrupted DDP runs selected the same next timestep and mask
support. Their losses were close but not bitwise equal (step 2: 2.641289 versus
2.641367); even their first updates differed slightly. Deterministic CUDA kernels
were not enabled. These runs verify working state/cursor restoration, not a
claim of bitwise replay.

Lightweight logs remain in `experiments/smoke_dgaf_*_20260910`. The three large
checkpoint directories created solely for save/load/resume testing are cleaned
up after validation (approximately 28 GiB). They are not usable trained models;
rerun the documented training commands to create persistent checkpoints.

These are implementation smoke checks, **not restoration quality comparisons**.
Three-step inference and one/two-step training cannot establish a gain over
baseline. No long training run was launched. Existing BrushNet/STC source files
and existing model checkpoints were not edited.
