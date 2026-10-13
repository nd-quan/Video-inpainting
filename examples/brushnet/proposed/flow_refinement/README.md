# Two-scale flow refinement for V8-R standalone

This package implements a separate flow-learning stage and an optional V8-R flow provider. It is an experiment; no trained refiner checkpoint or improvement claim is bundled.

The frozen V7 student receives degraded RGB in `[-1,1]`. The refiner takes its full-resolution bidirectional `[dx,dy]` flow and the same degraded frames. A shared image encoder produces 1/4 and 1/2 features. At 1/4, the model samples a radius-4 feature cost volume around `p + F0(p)` and predicts a residual. At 1/2, it warps features using the updated flow and predicts a second residual with a small CNN. Only the sum of both residuals is upsampled to RGB resolution:

```text
Finit_2 = resize(F0, H/2,W/2) + resize(Delta_4, H/2,W/2)
Fref   = F0 + resize(resize(Delta_4,H/2,W/2) + Delta_2, H,W)
```

Both residual heads are zero-initialized, so the initial model reproduces `F0` exactly. The same weights process forward/backward flow after swapping image order. Invalid matching candidates get a high cost and explicit validity channels.

The loss is `L_flow + edge_weight * L_edge` on final RGB-resolution flow. `L_flow` is teacher-valid, BG-weighted Charbonnier error to cached clean-RAFT pseudo-labels. `L_edge` compares flow gradients at teacher motion boundaries; both pixels of each finite difference must have valid teacher labels. The teacher and V7 student remain frozen. Clean frames/teacher flow are not passed to the inference-time refiner.

## Stage 1: train and evaluate flow alone

From `videoInpainting/code/BrushNet`, using the repository's `guided_diff` Python environment:

```bash
CUDA_VISIBLE_DEVICES=0 /home/cilab/ndquan/envs/guided_diff/bin/python \
  examples/brushnet/proposed/flow_refinement/train.py \
  --dataset_root /home/cilab/ndquan/videoInpainting/SFU_STC_flow \
  --teacher_flow_root /home/cilab/ndquan/videoInpainting/SFU_STC_flow/teacher_flows_512x512 \
  --raft_student_path experiments/train_v7_raft_student_flow/best.json \
  --output_dir experiments/train_flow_refinement_v8r \
  --preflight_only
```

Remove `--preflight_only` to train. Defaults: 512×512, radius 4 at 1/4, `edge_weight=0.05`, `bg_weight=1`, batch size 1, 10k optimizer steps, validation every 250 steps. Monitor reported `raw_bg_epe`, `bg_epe`, `raw_boundary_epe`, and `boundary_epe`; the first two use the same clean-teacher pseudo-labels. A selected checkpoint is saved at `best.pt`, with checkpoint-specific RAFT identity and architecture. The maximum quarter/half residual displacement and edge threshold are configurable; the defaults are starting values, not validated optima.

Run evaluation on a saved checkpoint with `--evaluate_only experiments/train_flow_refinement_v8r/best.pt`. This measures forward/backward averaged full-frame, BG, and motion-boundary EPE on the selected split. `best.pt` is written only when validation BG EPE is below the raw student baseline; periodic `checkpoint-N.pt` files are written regardless. `--valid_pairs_per_sequence 0` uses every pair; the default caps validation to eight evenly selected pairs per sequence. Use independent sequences where available and keep the same support for raw/refined metrics.

For the completed two-GPU run, evaluate its selected checkpoint on every test pair (the teacher-flow cache must contain the `test/` split):

```bash
CUDA_VISIBLE_DEVICES=4 /home/cilab/ndquan/envs/guided_diff/bin/python \
  examples/brushnet/proposed/flow_refinement/train.py \
  --dataset_root /home/cilab/ndquan/videoInpainting/SFU_STC_flow \
  --teacher_flow_root /home/cilab/ndquan/videoInpainting/SFU_STC_flow/teacher_flows_512x512 \
  --raft_student_path experiments/train_v7_raft_student_flow/best.json \
  --output_dir experiments/train_flow_refinement_v8r_b29_gpu45 \
  --valid_split test --valid_pairs_per_sequence 0 \
  --resolution 512 --batch_size 8 --pair_batch_size 2 \
  --evaluate_only experiments/train_flow_refinement_v8r_b29_gpu45/best.pt
```

The command prints JSON with `epe`, `bg_epe`, and `boundary_epe` for the refined flow and matching `raw_*` values for the frozen coarse flow. These are errors against the cached clean-RAFT *pseudo-target*, not ground-truth optical flow. A pair-level diagnostic similar to `experiments/visualize_pair_diagnostics` is available:

```bash
CUDA_VISIBLE_DEVICES=4 /home/cilab/ndquan/envs/guided_diff/bin/python \
  examples/brushnet/proposed/flow_refinement/visualize_pair.py \
  --dataset_root /home/cilab/ndquan/videoInpainting/SFU_STC_flow \
  --teacher_flow_root /home/cilab/ndquan/videoInpainting/SFU_STC_flow/teacher_flows_512x512 \
  --checkpoint experiments/train_flow_refinement_v8r_b29_gpu45/best.pt \
  --split test --branch Class_D/BasketballPass --frame0 440 \
  --save_dir experiments/train_flow_refinement_v8r_b29_gpu45/visualizations/test_BasketballPass_000440_000441
```

The visualization writes `montage.png`, individual flow/EPE/gain/warp maps, `flows.npz`, and `metrics.json`. All flow images share one color scale, and all EPE images share another. The gain map uses green for lower EPE after refinement and magenta for higher EPE.

For this run, `best.pt` is from step 3,250 (training completed at step 10,000). Its frozen coarse-flow input is exactly `experiments/train_v7_raft_student_flow/checkpoint-0004750/raft_student`. The `raw_*` metrics are that student checkpoint evaluated on the same pairs and masks as the refined flow. Evaluation on every pair gives:

| Split | Pairs | Measure (px) | Student step 4,750 | Refined | Relative decrease |
| --- | ---: | --- | ---: | ---: | ---: |
| Valid | 1,140 | EPE | 0.216484 | 0.213637 | 1.32% |
| Valid | 1,140 | BG EPE | 0.316078 | 0.312059 | 1.27% |
| Valid | 1,140 | Boundary EPE | 1.030589 | 1.029019 | 0.15% |
| Test | 1,449 | EPE | 0.259483 | 0.257373 | 0.81% |
| Test | 1,449 | BG EPE | 0.283697 | 0.280653 | 1.07% |
| Test | 1,449 | Boundary EPE | 1.169042 | 1.167351 | 0.14% |

Full-precision results are in `experiments/train_flow_refinement_v8r_b29_gpu45/evaluation_{valid,test}_full_best.json`. These are modest improvements relative to the clean-RAFT pseudo-target. The warp and gain maps can vary by pair, so do not infer downstream video-inpainting quality from these flow metrics alone.

`accelerate launch --multi_gpu ... train.py ...` supports multi-GPU training. Checkpoint directories contain Accelerate state for `--resume`. A `--quarter_only` ablation suppresses the 1/2 residual branch; leave it off for the main experiment.

## Stage 2: use refined flow in V8-R standalone

Set one environment variable when launching existing standalone training or evaluation:

```bash
V8R_FLOW_REFINER_PATH="$PWD/experiments/train_flow_refinement_v8r/best.pt" \
  bash examples/brushnet/proposed/v8r_standalone/run_evaluate.sh
```

The existing launchers' other required variables still apply. Both the TCE condition path and DDIM temporal guidance use the same wrapped provider. Prior clips used for cross-clip memory also pass through it. The wrapper rejects a checkpoint trained with a different V7 student component. The refiner remains frozen during TCE training/evaluation. Without `V8R_FLOW_REFINER_PATH`, the original frozen student is used.

Keep flow units, checkpoint identity, seeds, masks, and TCE settings fixed when comparing coarse and refined flow. First compare flow metrics, then test the refiner with an unchanged TCE checkpoint, then fine-tune TCE on refined flow. The current standalone DCN still receives residual offsets after pre-warp; flow is applied once.

## Verification

```bash
PYTHONPATH=examples/brushnet/proposed \
  /home/cilab/ndquan/envs/guided_diff/bin/python \
  -m unittest flow_refinement.test_flow_refinement -v
```

Tests cover displacement scaling, local matching direction/bounds, exact zero-residual identity, finite/nonzero gradients in both heads, and teacher-valid motion-edge support.
