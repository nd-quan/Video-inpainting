# RGB-STC v3: diffusion + flow supervision

This experiment isolates one change relative to `STC_encoder_v2_rgb`: a
train-only bidirectional flow head supervises the same RGB-STC features used by
BrushNet conditioning.

## Training contract

Raw dataset masks use `0=degraded BG` and `1/255=high-quality ROI`. The shared
V8 loader inverts them exactly once, so every model/loss tensor uses
`M_BG=1` and `M_ROI=0`.

The model input and restoration path are:

```text
[degraded RGB sequence, M_BG]
              |
          RGB-STC
              |-----------------> directional flow head -> L_flow (train only)
              |
          ZeroConv delta_z
              |
 M_BG gate + degraded-input VAE latent
              |
 frozen checkpoint-2000 BrushNet + U-Net -> L_diff
```

The objective is

```text
L_total = L_diff + flow_loss_weight * L_flow
```

`L_flow` is the mean bidirectional Charbonnier distance to clean-GT RAFT flow.
The default `flow_region=bg` applies forward supervision on `M_BG[t]` and
backward supervision on `M_BG[t+1]`, each combined with the cached RAFT
forward/backward confidence map. There is no temporal-delta, flow-warp,
forward/backward-cycle, smoothness, or background reconstruction penalty in
this version.

Teacher flow is cached at 512x512 and is resized to latent resolution 64x64;
both spatial interpolation and displacement scaling are handled by
`prepare_teacher_flow`. Flow never changes shared-noise construction and is not
needed at inference.

## Run

```bash
bash examples/brushnet/STC_encoder_v3_rgb_flow/run_train_rgb_stc_flow_shared_noise.sh
```

Defaults:

- baseline: `experiments/train_sharedNoise_sameBG_0.9/checkpoint-2000`
- output: `experiments/train_rgb_stc_v3_flow_sharedNoise_0.9`
- clean teacher: `SFU_STC_flow/teacher_flows_512x512`
- clip: `T=8`, stride `6`
- shared BG noise: `rho=0.9`
- `flow_loss_weight=0.01`, `flow_region=bg`
- flow bound: 8 latent pixels in x/y

Useful overrides:

```bash
FLOW_LOSS_WEIGHT=0.005 FLOW_REGION=all \
OUTPUT_DIR=/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/my_v3_run \
bash examples/brushnet/STC_encoder_v3_rgb_flow/run_train_rgb_stc_flow_shared_noise.sh
```

To initialize the deployable RGB-STC branch from a completed v2 adapter while
keeping a fresh zero-flow head:

```bash
INIT_STC_ADAPTER=/path/to/v2/checkpoint-250/stc_adapter \
bash examples/brushnet/STC_encoder_v3_rgb_flow/run_train_rgb_stc_flow_shared_noise.sh
```

The default is a fresh RGB-STC model on checkpoint-2000, as requested. A v2
warm start is optional and mutually exclusive with `--resume_from_checkpoint`.

Each v3 checkpoint contains one combined `stc_flow_model` for exact training
resume and a smaller `stc_adapter` for inference. At inference, use only
`stc_adapter`; do not load or compute teacher/predicted flow.

## Diagnose local versus global temporal inconsistency

`evaluate_temporal_inconsistency.py` evaluates an already generated V2/V3
clip tree. It does not run diffusion inference and does not use the V3 flow
head. Instead, it uses clean-GT RAFT `teacher_b` to align the previous-frame
restoration error into the current frame:

```text
e_t = prediction_t - clean_GT_t
R_t = e_t - warp(e_(t-1), teacher_b_t)
```

The effective evaluation domain is the valid degraded BG in both frames. Raw
masks remain `0=degraded BG, 255=HQ ROI`; the evaluator converts this once to
`M_BG=1`. It erodes mask boundaries by two pixels so a hard-composite seam does
not dominate the diagnosis.

Run the current V2 validation result:

```bash
bash examples/brushnet/STC_encoder_v3_rgb_flow/run_evaluate_temporal_inconsistency.sh
```

Fast preflight or a focused BasketballPass diagnosis:

```bash
PREFLIGHT_ONLY=1 \
bash examples/brushnet/STC_encoder_v3_rgb_flow/run_evaluate_temporal_inconsistency.sh

VIDEO_FILTER=BasketballPass MAX_UNITS=1 SCALES=8,16 \
bash examples/brushnet/STC_encoder_v3_rgb_flow/run_evaluate_temporal_inconsistency.sh
```

The default `OVERLAP_MODE=per_clip` measures only transitions produced in the
same diffusion call. `OVERLAP_MODE=first` or `last` diagnoses the stitched
sequence; rows whose two frames came from different calls are marked
`window_boundary=1` because those transitions can include clip-stitching
flicker, not only STC behavior.

Outputs are written below the evaluation root:

- `per_pair_metrics.csv`: every frame pair, motion, coverage, absolute error,
  multi-scale local/coarse scores and labels.
- `per_sequence_metrics.csv`: valid-pixel-weighted sequence summaries.
- `summary.json`: metric contract, preflight counts and overall diagnosis.
- `visualizations/`: worst pair per unit with prediction/GT warps, effective
  mask, raw residual, coarse residual and local residual.

Interpret the primary fields together:

- `raw_temporal_rmse` is inconsistency severity in RGB `[0,1]` units.
- `local_rmse_share` (also emitted as the compatibility alias
  `local_frequency_share`) is
  `local_RMSE / (local_RMSE + coarse_RMSE)`. It says whether error is mostly
  high-frequency local (`>=0.6`), coarse/global (`<=0.4`), or mixed at the
  selected Gaussian scale (16 pixels by default).
- `global_dc_l1` detects coherent brightness/color drift over the whole valid
  BG.
- `energy_support90_fraction` and `directional_coherence` distinguish a
  spatially localized patch from a distributed/global error.

The local/global label is a scale-dependent diagnostic. A high local share can
still describe a very small error, so it must never be reported without the
absolute RMSE.
