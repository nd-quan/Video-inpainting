# STC Full-Flow Prediction and Noise-Warping: Developer Handoff

Status: implementation and data preparation are complete. Clean-flow teacher
precomputation and model training have deliberately **not** been started.

## 1. Dataset

Prepared root:

```text
/home/cilab/ndquan/videoInpainting/SFU_STC_flow
```

Layout:

```text
<split>/{GT,input,mask}/<class>/<sequence>/<source_frame>.png
```

Mask semantics are fixed everywhere in the prepared files:

```text
0   = background
255 = ROI
```

The loader converts this to `0=background, 1=ROI`. The model derives
`background_mask = 1 - roi_mask` only as semantic context; the full degraded
latent is retained and noise is warped over the whole frame.

Validated totals:

| Split | Frames | Adjacent teacher pairs |
|---|---:|---:|
| train | 4,906 | 4,882 |
| valid | 1,158 | 1,140 |
| test | 1,466 | 1,449 |
| total | 7,530 | 7,471 |

The default per-video chronological split is 65/15/20. `BasketballDrill` is
the explicit exception: frames 0--139 are train, frames 140--199 are valid,
and test is disabled, giving the requested 70/30/0 ratio.

`postprocess_remaining/03_frames` is the preferred source. Twelve sequences
use it exclusively. Six sequences are missing an early prefix there, so those
prefixes are supplied from `SFU_train` **for train only**: `Traffic` (60
frames), `ParkScene` (100), `BQSquare` (180), `BasketballPass` (150),
`BlowingBubbles` (150), and `FourPeople` (180). This gives the following
physical source counts:

| Split | `03_frames` | `SFU_train` |
|---|---:|---:|
| train | 4,086 | 820 |
| valid | 1,158 | 0 |
| test | 1,466 | 0 |

Each SFU/03 transition is represented as two separate manifest segments even
when the frame indices are adjacent. Clips and clean-RAFT teacher pairs are
created only after splitting and only within one segment; they cannot cross a
split, video, or decoded-input source boundary. GT for `03_frames` samples is
decoded from the original clean YUV and masks are rendered from matching ROI
descriptors.

Authoritative metadata:

- `SFU_STC_flow/manifest.json`: source-frame ranges and hierarchy.
- `SFU_STC_flow/dataset_report.json`: validated counts and coverage.
- `SFU_STC_flow/BUILD_COMPLETE.json`: successful build marker.
- `SFU_STC_flow/logs/prepare.log`: preparation and validation log.

## 2. Stage-1 full-flow model

The implementation uses a resolution-adapted VideoComposer-style condition
encoder, not the broken training behavior found in the exploratory
`NAS_Ngoc/vcm` prototype.

For each degraded frame, the deterministic VAE posterior mode produces a
latent `z_t^D`. The structural condition is

```text
C_t = concat(z_t^D, 1 - M_t),
F_1...F_T = STC(C_1...C_T).
```

Here `M_t=1` denotes ROI. The mask is context only; it does not zero ROI
features. One shared ordered-pair decoder predicts both directions:

```text
B_t = f(F_{t-1}, F_t, F_t - F_{t-1})   # current -> previous
F_t = f(F_t, F_{t-1}, F_{t-1} - F_t)   # previous -> current
```

The output is bounded with `tanh` in latent-pixel units. No raw RAFT flow,
refined flow, depth, or sketch is required at inference.

Training supervision is generated offline from **clean GT frames** using the
bundled ProPainter RAFT checkpoint. The objective is

```text
L = L_teacher + 0.1 L_forward-backward + 0.01 L_smoothness.
```

`L_teacher` is a robust bidirectional Charbonnier flow loss. Reported EPE is in
latent pixels. Cached validity, finite-value checks, and geometric in-bounds
validity are combined after displacement-correct flow resizing.

## 3. Whole-frame noise warping

The trained backward flows compose a sampling field from frame `t` to the
anchor frame. With independent standard Gaussian innovations `eta_t`, noise is
shaped as

```text
epsilon_0 ~ N(0, I)
epsilon_t = beta * Warp(epsilon_0, Phi_t->0)
          + sqrt(1 - beta^2) * eta_t.
```

The current Stage-1 configs use fixed `beta=0.5`; beta is not learned during
flow pretraining. Bilinear-warp variance is corrected per pixel, and
out-of-bounds locations receive fresh noise. Warping is applied to the whole
frame (`warp_region="all"`), with no stable-region gate.

The BrushNet pipeline preserves its historical sampled conditioning latent,
while the STC flow predictor receives the deterministic VAE posterior mode.
Full-flow inference skips RAFT initialization and requires an explicit clip
length, preventing separate videos from being silently merged.

## 4. Code map

- `examples/brushnet/prepare_sfu_stc_dataset.py`: reproducible build and full
  triplet validator.
- `examples/brushnet/stc_flow_dataset.py`: manifest/segment-aware clip loader.
  Stage-1 configs set `load_gt=false` because clean GT has already been reduced
  to the teacher-flow cache; GT remains present and validated on disk.
- `examples/brushnet/precompute_sfu_stc_teacher_flows.py`: clean-RAFT teacher
  cache with forward/backward confidence.
- `examples/brushnet/diffusers/models/stc_noise_shaper.py`: STC encoder,
  bidirectional full-flow head, variance-preserving whole-frame noise warp.
- `examples/brushnet/diffusers/models/stc_flow_training.py`: Stage-1 losses,
  EPE, validity handling, and checkpoint helpers.
- `examples/brushnet/train_stc_flow_vcm.py`: DDP/AMP trainer, validation,
  TensorBoard, best/latest checkpoints, and exact resume state.
- `examples/brushnet/configs/train_vcm_stc_flow_prediction_smoke_2gpu.json`:
  20-step developer smoke configuration.
- `examples/brushnet/configs/train_vcm_stc_flow_prediction_2gpu.json`: proposed
  full configuration (`T=8`, `B=4/GPU`, two GPUs, accumulation=2).
- `examples/brushnet/tests/test_sfu_stc_data_planning.py`: split/leakage and
  pair-boundary regression tests.
- `examples/brushnet/tests/test_stc_flow_dataset.py`: aligned loader/cache tests.
- `examples/brushnet/tests/test_stc_flow_training.py`: flow loss/checkpoint tests.
- `examples/brushnet/tests/test_stc_noise_shaper.py`: full-flow and Gaussian
  warping tests.
- `examples/brushnet/tests/test_stc_pipeline_inputs.py`: pipeline layout and
  deterministic-latent integration tests.

## 5. Review results and known limitations

- The complete local unit suite passes: 62 tests.
- All new/modified Python entry points compile.
- Both Stage-1 JSON configs parse and instantiate.
- Physical dataset validation reads every GT/input/mask triplet, checks aligned
  dimensions, and enforces masks in `{0,255}`.
- No source index crosses train/valid/test, sequence, or teacher-pair boundaries.
- `BasketballDrill/test` is the only intentionally empty sequence-level split;
  this is recorded as `test_source="disabled"` in the manifest.
- The current inference experiment supports `num_images_per_prompt=1`. A generic
  multi-sample call would need an explicit `[video, frame, sample]` layout
  permutation before STC sequence shaping.
- Adjacent segments from the same source video are disjoint by frame index but
  still visually correlated. Report validation results with this protocol
  explicitly; sequence-level generalization would require additional videos.

## 6. Commands reserved for after developer approval

Teacher cache (not executed):

```bash
CUDA_VISIBLE_DEVICES=1 \
python examples/brushnet/precompute_sfu_stc_teacher_flows.py \
  --dataset-root /home/cilab/ndquan/videoInpainting/SFU_STC_flow \
  --device cuda \
  --height 256 \
  --width 256 \
  --iters 20 \
  --resume
```

Two-GPU smoke training (not executed):

```bash
CUDA_VISIBLE_DEVICES=1,2 \
torchrun --standalone --nproc_per_node=2 \
  examples/brushnet/train_stc_flow_vcm.py \
  --config examples/brushnet/configs/train_vcm_stc_flow_prediction_smoke_2gpu.json
```

Do not start the full configuration until the smoke run verifies cache
directions, latent-pixel EPE, VRAM, checkpointing, and TensorBoard output.
