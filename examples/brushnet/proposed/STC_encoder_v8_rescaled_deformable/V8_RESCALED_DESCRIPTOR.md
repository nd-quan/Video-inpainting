# V8-R: Rescaled RAFT Pre-warp with Native-grid DCN Refinement

## 1. Purpose

V8-R is an experimental V8 variant for testing whether motion alignment is
more accurate when the STC spatial feature is warped on a grid finer than its
native resolution.

The original V8 resizes full-resolution V7 RAFT flow directly to the native
STC feature grid and uses that flow as the base offset of DCNv2. V8-R instead
uses the following path:

```text
raw spatial feature
    -> nearest-neighbor upsample xS
    -> warp with V7 RAFT flow resized directly from RGB to the xS grid
    -> nearest-neighbor downsample to the native feature grid
    -> native-grid DCNv2 with learned residual offsets only
    -> BG-reliable bidirectional fusion
    -> relative temporal attention and cross-clip memory
    -> delta_z_BG
    -> BrushNet condition
```

The default rescaling factor is `S=4`.

This operation does **not** create new semantic or high-frequency feature
content. Nearest-neighbor upsampling only creates a finer sampling grid. The
hypothesis is that this finer grid preserves more spatial variation from the
RGB-resolution motion field during warping and reduces coarse-grid
interpolation error before returning to the native STC resolution.

## 2. Components retained from V8

V8-R retains the following V8 components and contracts:

- the V8 spatial encoder and inherited V5 alignment path;
- the frozen V7 ProPainter-RAFT student operating on degraded RGB frames;
- relative temporal positional bias;
- exact-overlap cross-clip memory;
- bidirectional previous/current/next feature fusion;
- BG-only latent correction;
- the frozen BrushNet and diffusion U-Net backbone; and
- the output condition

  \[
  z^{cond}_t = z^D_t + M^{BG}_t \odot \Delta z^{BG}_t.
  \]

The new behavior is isolated in the RAFT-guided deformable alignment path.

## 3. Notation and flow convention

Let:

- \(S_t \in \mathbb{R}^{C\times H\times W}\) be the raw STC spatial feature
  of frame \(t\);
- \(A^{base}_t\) be the inherited V5-aligned feature;
- \(M^{BG}_t\) be the background mask, where BG is 1 and ROI is 0;
- \(F^f_t\) be forward flow \(t\rightarrow t+1\), defined on frame \(t\);
- \(F^b_t\) be backward flow \(t+1\rightarrow t\), defined on frame \(t+1\);
- \(W(X,F)\) be backward sampling of source \(X\) with flow \(F\);
- \(U_S\) and \(D_S\) be nearest-neighbor upsampling and downsampling; and
- \(R_{h,w}(F)\) be flow resizing to spatial size \((h,w)\).

All V7 RAFT flows use `[dx,dy]` in original RGB-pixel units. Flow resizing
must change both the spatial map and the displacement units:

\[
R_{h,w}(F)=
\left(
  \operatorname{interp}(F_x)\frac{w}{W_{RGB}},
  \operatorname{interp}(F_y)\frac{h}{H_{RGB}}
\right).
\]

For example, with 512x512 RGB input and a 64x64 native STC grid:

- native V8 flow units are scaled by \(64/512=1/8\);
- V8-R x4 uses a 256x256 intermediate grid and scales flow by
  \(256/512=1/2\); and
- the warped feature is then returned to 64x64.

## 4. Rescaled bidirectional pre-warp

### 4.1 Previous frame to current frame

Backward flow is defined on the current frame and samples the previous frame.
The previous-frame candidate for target frame \(t+1\) is

\[
P^{prev}_{t+1} =
D_S\left[
W\left(
U_S(S_t),
R_{SH,SW}(F^b_t)
\right)
\right].
\]

### 4.2 Next frame to current frame

Forward flow is defined on the current frame and samples the next frame. The
next-frame candidate for target frame \(t\) is

\[
P^{next}_{t} =
D_S\left[
W\left(
U_S(S_{t+1}),
R_{SH,SW}(F^f_t)
\right)
\right].
\]

The source BG mask and the geometric in-bounds mask follow the same
upsample-warp-downsample geometry. Each pair always uses the original spatial
source feature; warped features are not recursively propagated through time.

## 5. Geometric BG support

For each direction, active V8-R reliability is

\[
r_t = M^{BG}_{target}
      \odot \widetilde{M}^{BG}_{source}
      \odot V_{in\mbox{-}bounds},
\]

where \(\widetilde{M}^{BG}_{source}\) is the source BG mask warped with the
same rescaled path.

Consequently:

- ROI target locations cannot receive a deformable residual;
- BG locations cannot sample content from the ROI;
- out-of-bounds samples cannot inject zero padding; and
- an invalid aligned candidate falls back to the target feature.

V7 forward-backward confidence remains available as a detached diagnostic,
but it is not used by the active DCN head, fusion reliability, or deformable
alignment loss. The loss uses the frozen V7 flow only to establish finite and
geometrically valid sampling support. Cached clean teacher flow is not an
optimization target because `flow_loss_weight=0`.

This geometric-only policy keeps difficult motion in the active experiment,
but it also means inaccurate V7 flow can enter the alignment path. That is an
intentional trade-off to be checked during evaluation.

## 6. Native-grid DCNv2 residual refinement

After rescaled pre-warping, DCNv2 operates at the original \(H\times W\)
feature resolution. Its prediction head receives

\[
X_t = [
S_{target},
P_{source},
S_{target}-P_{source},
\bar F,
\mathbf{1},
M^{BG}_{target},
\widetilde M^{BG}_{source}
],
\]

where \(\bar F\) is native-grid flow normalized by the configured flow bounds.
The all-one channel preserves the V8 head schema in the location previously
occupied by flow confidence; it does not reintroduce FB confidence.

The head predicts a bounded residual offset and a modulation mask:

\[
\Delta p_t = d_{max}\tanh(h_{offset}(X_t)),
\qquad
m_t = \sigma(h_{mask}(X_t)).
\]

The current default is \(d_{max}=2\) native feature pixels. DCNv2 then
refines the already motion-compensated feature:

\[
D_t = \operatorname{DCNv2}(P_{source},\Delta p_t,m_t).
\]

Critically, RAFT flow is **not** added to the DCN offset. The rescaled pre-warp
has already applied the main motion. Adding the RAFT displacement again would
apply motion twice:

```text
correct:   prewarp(F_RAFT) -> DCN(delta_offset)
incorrect: prewarp(F_RAFT) -> DCN(F_RAFT + delta_offset)
```

The residual offset is intended to correct local flow error, interpolation
error, and feature-space mismatch rather than relearn the full motion field.

## 7. Bidirectional fusion before temporal attention

For every target frame, the inherited base feature, previous candidate, next
candidate, their differences, and directional reliabilities are concatenated.
The fusion module predicts a feature residual \(Q_t\) and gate \(g_t\).

Directional support is combined as

\[
r^{union}_t = 1-(1-r^{prev}_t)(1-r^{next}_t).
\]

The aligned output is

\[
A_t = A^{base}_t
 + \lambda_{fusion}
   M^{BG}_t
   r^{union}_t
   \sigma(g_t)
   Q_t.
\]

Thus ROI locations remain identical to the base path. The fused features are
then processed by V5 relative temporal attention and cross-clip memory before
the zero-convolution predicts \(\Delta z^{BG}\).

The current experiment uses `bidirectional`. The implementation also supports
`previous_only` and `next_only` for controlled directional ablations.

## 8. Initialization strategy

The current V8-R run initializes from native V8 joint checkpoint 4000:

```text
experiments/
  train_stc_v8_raft_joint_from7500_T16_S12_sharedNoise_0.95_constant5e6/
    checkpoint-4000/stc_v8_model
```

The transfer is strict so compatible V8 weights are retained. The
geometry-dependent modules are then reset:

- `deformable_alignment`; and
- `deformable_fusion`.

Their last projections are zero-initialized. The inherited temporal blocks,
relative position parameters, cross-clip path, output normalization, and
zero-convolution are retained from V8.

The frozen V7 RAFT pointer

```text
experiments/train_v7_raft_student_flow/best.json
```

currently resolves to validated checkpoint 4750 and the
`propainter_raft_large` architecture. V8-R intentionally rejects a SEA-RAFT
checkpoint so the flow source cannot change silently inside this ablation.

## 9. Joint optimization schedule

The run uses two optimizer groups.

### Steps 0-499

- train `deformable_alignment` and `deformable_fusion`;
- use deform LR `2e-5`, with 100-step warmup;
- keep temporal-group LR at zero; and
- explicitly discard gradients for the temporal group.

### From step 500

- continue training the deformable group;
- unfreeze optimization of `alignment_fusion`, temporal attention blocks,
  `output_norm`, and `zero_conv`; and
- warm the temporal LR from zero to `5e-6` over 100 steps.

The spatial encoder, legacy lightweight flow head, V7 RAFT, BrushNet, and
diffusion U-Net remain frozen. Although BrushNet and the U-Net are frozen,
`L_diff` still propagates through them to the trainable V8-R condition path.

Both LR groups remain constant after their respective warmups.

## 10. Training objective

The current objective is

\[
L_{total}=L_{diff}
 + \lambda_{deform}(k)L_{deform}
 + \lambda_{offset}L_{offset},
\]

with

\[
\lambda_{deform}(k)=0.1\min\left(1,\frac{k+1}{500}\right),
\qquad
\lambda_{offset}=0.0005.
\]

`L_deform` compares channel-normalized DCN candidates with stop-gradient
target spatial features using a Charbonnier penalty over geometric BG-valid
support. `L_offset` regularizes the magnitude of the actual bounded residual
offset, rather than the unconstrained offset logits.

The following objectives are disabled in this first V8-R experiment:

```text
L_flow weight             = 0
L_feature-alignment weight = 0
L_temporal-output weight   = 0
```

Therefore, this run tests the effect of rescaled feature warping and joint
DCN/temporal fine-tuning without adding a new temporal-output loss.

## 11. Verified configuration of the current run

The metadata stored at checkpoint 1750 records:

| Setting | Value |
|---|---:|
| Training stage | `rescaled_joint` |
| RGB resolution | 512x512 |
| Native STC grid | 64x64 (downsample factor 8) |
| Rescaled warp grid | 256x256 (`S=4`) |
| Clip length / stride | 16 / 12 |
| Cross-clip memory | 4 frames |
| Alignment direction | bidirectional |
| Frozen flow | V7 ProPainter-RAFT checkpoint 4750 |
| Shared BG noise strength | 0.95 |
| Deform LR | `2e-5` |
| Temporal LR | `5e-6` after step 500 |
| LR scheduler | constant after warmup |
| Deform loss weight | ramp to 0.1 over 500 steps |
| Offset loss weight | 0.0005 |
| Residual offset bound | 2 native feature pixels |
| GPUs / processes | 3 |
| Clips per device | 1 |
| Gradient accumulation | 3 |
| Effective batch | 9 clips per optimizer update |
| Maximum steps | 4000 |

Current experiment directory:

```text
experiments/
  train_stc_v8_rescaled_x4_joint_raftbest_T16_S12_sharedNoise_0.95/
```

## 12. Main hypotheses

V8-R tests the following hypotheses:

1. Resizing spatially varying RGB flow directly to 64x64 removes useful local
   motion variation, even though displacement values remain floating point.
2. Warping on a 256x256 grid can reduce interpolation error for motion smaller
   than or near one native feature pixel.
3. Native-grid residual DCN can correct remaining local misalignment without
   relearning the full RAFT motion.
4. BG-to-BG geometric support and residual fusion prevent the new alignment
   from modifying the high-quality ROI.
5. Better correspondence before temporal attention should make temporal
   aggregation more useful, but this is not guaranteed to improve the final
   generated video without a downstream temporal objective.

## 13. Metrics to monitor

### Optimization

- `train/loss_diff`
- `train/loss_deform_alignment`
- `train/loss_deform_alignment_weighted`
- `train/loss_deform_offset`
- `train/loss_deform_offset_weighted`
- `train/lr_deform`
- `train/lr_temporal`
- `train/temporal_branch_active`

### Alignment behavior

- `train/deform_valid_forward/backward`
- `train/deform_reliability_forward/backward`
- `train/deform_offset_abs_mean/p95/max`
- `train/deform_mask_mean/std/saturation`
- `train/deformation_minus_base_abs_mean`
- V7 RAFT flow magnitude in RGB and feature pixels
- `train/delta_abs_mean`

A decreasing `L_deform` alone does not demonstrate better temporal output. It
must be accompanied by controlled feature-correspondence and final-video
evaluation.

## 14. Required evaluation and ablations

The principal comparison should keep seed, shared noise, checkpoint backbone,
clip layout, DDIM settings, and inference guidance fixed:

1. shared noise + temporal guidance baseline;
2. native V8 with frozen V7 RAFT;
3. V8-R x4 with the same frozen V7 RAFT;
4. V8-R with deformable alignment scale zero;
5. optional scale ablation `S in {1,2,4}`; and
6. optional direction ablation: previous-only, next-only, bidirectional.

Evaluation should inspect correspondence at the same frame pairs and BG
support:

```text
raw spatial feature
    -> rescaled prewarped feature
    -> DCN-refined feature
    -> feature after temporal attention
    -> delta_z_BG
    -> predicted-clean latent
    -> generated RGB frame
```

Recommended temporal metrics include flow-warp error, cosine correspondence,
tLPIPS, and per-sequence flicker statistics. Spatial/task metrics must be
reported alongside them to detect temporal improvement obtained by blur or
loss of detail.

## 15. Limitations and interpretation

- Nearest upsampling does not recover feature details absent from the 64x64
  spatial representation.
- The x4 pre-warp has 16 times the spatial area of native warping, although
  DCNv2 itself remains on the native grid.
- Nearest downsampling can introduce aliasing or discard part of the
  high-grid benefit.
- Geometric-only reliability does not reject V7 forward/backward disagreement.
- Bidirectional candidates can conflict near occlusion and motion boundaries.
- `L_deform` supervises pre-attention feature correspondence, not final video
  temporal consistency.
- A lower training loss does not prove that V8-R outperforms native V8 or the
  shared-noise plus temporal-guidance baseline.

These limitations are part of the experiment. The intended conclusion must
come from controlled native-V8 versus V8-R evaluation, not from training-loss
convergence alone.

## 16. Implementation map

- `rescaled_raft_deformable_stc_adapter.py`
  implements rescaled pre-warp, geometric support, residual-only DCNv2, and
  bidirectional fusion integration.
- `train_v8_rescaled_joint.py`
  implements V8 initialization, deformation reset, optimizer groups, staged
  temporal unfreezing, metadata, and resume contracts.
- `run_train_v8_rescaled_joint_t16.sh`
  defines the reproducible T16/S12 training configuration.
- `evaluate_v8_rescaled_deformable.py`
  selects the V8-R model class while reusing the established non-CGE V8
  evaluation protocol.
- `run_evaluate_v8_rescaled_combined_testsets.sh`
  evaluates one V8-R checkpoint sequentially on `test_1` and `test_2` using
  one visible GPU.
- `evaluate_v8_rescaled_deformable_temporal_student.py`
  combines the V8-R condition path with the existing frozen-V7 temporal DDIM
  scheduler.
- `run_evaluate_v8_rescaled_temporal_latent_combined_testsets.sh`
  evaluates predicted-clean latent temporal guidance without CGE.
- `test_v8_rescaled_deformable.py`
  verifies scale-1 geometry, residual-only offsets, no double application of
  flow, BG/ROI support, and argument validation.
