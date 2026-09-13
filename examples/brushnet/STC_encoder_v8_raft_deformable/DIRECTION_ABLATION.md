# V8 DCN direction: fine-tuning and inference

`--deformable_alignment_direction` accepts `bidirectional`, `previous_only`,
or `next_only`. Train and evaluation inherit the checkpoint setting when the
argument is omitted. Old checkpoints default to `bidirectional`.
The standard train, combined-testsets, temporal-student sharded and CGE-temporal
sharded launchers expose the same option as `DEFORMABLE_ALIGNMENT_DIRECTION`.

`previous_only` retains the candidate from the previous frame using backward
flow. `next_only` retains the next-frame candidate using forward flow.
The disabled candidate is replaced by `base_aligned`, and its fusion/loss
reliability is zero. Simply zeroing confidence is insufficient because fusion
also consumes the candidate and its difference from the base.
Both flow directions are still computed for forward-backward confidence.
DCN pair computation is retained for diagnostics, so this is not a speed optimization.

This switches only the added DCN contribution. The legacy V5 base alignment,
temporal self-attention and cross-clip memory remain intact: previous-only DCN
does not make V8 causal or remove every source of future-frame information.
Inference ablations may be out of distribution for bidirectionally trained weights.

During fine-tuning, previous-only uses the backward deform/offset loss;
next-only uses the forward loss. No factor of 1/2 is applied to the single
active direction. Bidirectional retains the original two-direction average.
Inactive directional loss metrics are zero under their zero support.
Other diagnostics (e.g. mean DCN offsets) can still include both computed candidates.

The direction is saved in `stc_v8_model/config.json`, training metadata and
evaluation run configuration. Exact resume must keep the saved direction.
To change direction, initialize from V8 weights in a new output directory.

## Fine-tune from joint checkpoint 4000

From the BrushNet root (set CUDA_VISIBLE_DEVICES/NUM_PROCESSES for the desired GPUs):

```bash
TRAINING_STAGE=joint \
INIT_V8_MODEL="$PWD/experiments/train_stc_v8_raft_joint_from7500_T16_S12_sharedNoise_0.95_constant5e6/checkpoint-4000/stc_v8_model" \
DEFORMABLE_ALIGNMENT_DIRECTION=previous_only \
RESUME_FROM_CHECKPOINT="" \
OUTPUT_DIR="$PWD/experiments/train_v8_joint_previous_only" \
LEARNING_RATE=5e-6 LR_SCHEDULER=constant \
bash examples/brushnet/STC_encoder_v8_raft_deformable/run_train_v8_raft_deformable_t16.sh
```

For the other case use `DEFORMABLE_ALIGNMENT_DIRECTION=next_only` and
`OUTPUT_DIR="$PWD/experiments/train_v8_joint_next_only"`.
These commands start training; use `--preflight_only` for inspection only.

## Evaluate an existing checkpoint without training

```bash
DEFORMABLE_ALIGNMENT_DIRECTION=previous_only \
STC_V8_MODEL="$PWD/experiments/train_stc_v8_raft_joint_from7500_T16_S12_sharedNoise_0.95_constant5e6/checkpoint-4000/stc_v8_model" \
OUTPUT_DIR="$PWD/experiments/eval_v8_previous_only" \
bash examples/brushnet/STC_encoder_v8_raft_deformable/run_evaluate_v8_combined_testsets.sh
```

For next-only change the direction and output directory. The launcher retains
its existing dataset/GPU settings. The Python evaluator also accepts the CLI
flag directly. Temporal/CGE launchers accept the environment variable; keep
their other experiment settings fixed for a controlled comparison.

No jobs are automatically launched by adding this capability.
