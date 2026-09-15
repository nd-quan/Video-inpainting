# Temporal Training Loss Proposal for V8 / V8-R

## 1. Status and purpose

This document specifies the proposed temporal loss for a future V8-R joint
fine-tuning experiment. It is a design document only: the loss described here
has **not** yet been added to the training code and no checkpoint should be
reported as having used it.

The current V8-R objective is

\[
\mathcal L_{current}
= \mathcal L_{diff}
+ \lambda_{def}(s)\mathcal L_{def}
+ \lambda_{off}\mathcal L_{off},
\]

where the current run uses a ramped deformable-alignment weight, an offset
regularizer, and no explicit loss on the temporal consistency of the final
diffusion prediction. Consequently, lower feature/DCN alignment loss does not
guarantee less flicker after BrushNet and the frozen diffusion U-Net.

The proposed extension is

\[
\mathcal L_{total}
= \mathcal L_{diff}
+ \lambda_{def}(s)\mathcal L_{def}
+ \lambda_{off}\mathcal L_{off}
+ \lambda_{temp}(s,t)\mathcal L_{temp}.
\]

The main purpose of \(\mathcal L_{temp}\) is to supervise the representation
closest to the generated result while still allowing gradients to train the
V8-R condition path.

## 2. Current trainable and frozen paths

During V8-R joint training, the following modules are trainable:

- `deformable_alignment`
- `deformable_fusion`
- `alignment_fusion`
- temporal-attention blocks
- temporal output normalization
- temporal `zero_conv`

The frozen V7 ProPainter-RAFT student supplies optical flow. The VAE,
BrushNet, diffusion U-Net, text encoder, image encoder, IP-Adapter projection,
and baseline fusion module remain frozen. Frozen does not mean detached: the
BrushNet/U-Net forward path must preserve autograd with respect to the V8-R
condition so that temporal-loss gradients can reach the trainable V8-R
modules.

The relevant path is

```text
degraded RGB + BG mask
    -> frozen V7 RAFT flow
    -> V8-R spatial feature
    -> x4 upsample / flow warp / downsample
    -> residual DCN and feature fusion
    -> temporal attention and cross-clip memory
    -> delta_z_BG
    -> BrushNet condition
    -> frozen BrushNet + frozen diffusion U-Net
    -> model prediction
    -> predicted-clean latent z0_hat
    -> temporal training loss
```

The loss therefore evaluates the cumulative effect of spatial encoding,
alignment, temporal modelling, and condition injection rather than optimizing
only an intermediate feature.

## 3. Recommended supervision representation

The primary representation should be the diffusion predicted-clean latent

\[
\hat z_0 = \text{pred\_original\_sample}.
\]

This is preferred over the following alternatives:

- **Noisy latent \(z_t\):** contains deliberately injected noise and is not a
  clean temporal target.
- **`delta_z_BG`:** diagnoses the condition adapter but does not measure what
  survives the frozen BrushNet/U-Net.
- **Spatial/aligned feature:** useful as an auxiliary loss, but earlier
  experiments show that feature correspondence alone may not reduce final
  flicker.
- **Decoded RGB:** closest to the output but substantially more expensive;
  backpropagating through the VAE decoder for 16 frames has a high memory cost.

For epsilon prediction, construct the differentiable clean estimate using the
same sampled training timestep as the diffusion objective:

\[
\hat z_0
= \frac{z_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon_θ}
        {\sqrt{\bar\alpha_t}}.
\]

For velocity prediction, use the scheduler-consistent conversion:

\[
\hat z_0
= \sqrt{\bar\alpha_t}\,z_t
- \sqrt{1-\bar\alpha_t}\,\hat v_θ.
\]

Do not detach `model_prediction` or \(\hat z_0\). Detaching either tensor would
make the temporal term a logging-only number with no training effect.

All frames in a training clip should use the same timestep. The existing
default already shares the timestep across a clip; enabling independent frame
timesteps would make adjacent \(\hat z_0\) estimates incomparable and should
be rejected for this experiment.

## 4. Backward-warp convention

For adjacent frames \(k\) and \(k+1\), the frozen student backward flow
\(F^b_k\) is defined on the coordinates of frame \(k+1\) and samples frame
\(k\). Thus

\[
\tilde z_{0,k\rightarrow k+1}
= \mathcal W(\hat z_{0,k},F^b_k),
\]

and the temporal residual is

\[
r_k(x)=\hat z_{0,k+1}(x)-
       \tilde z_{0,k\rightarrow k+1}(x).
\]

The source must always be the original predicted-clean latent for frame
\(k\). Do not recursively warp an already warped latent.

The full-resolution RGB flow must be resized to latent resolution and its
displacements rescaled. If RGB resolution is \((H_r,W_r)\) and latent
resolution is \((H_l,W_l)\), then

\[
F^b_l = \operatorname{resize}(F^b_r,H_l,W_l),\qquad
F^b_{l,x}\mathrel{*}=W_l/W_r,\quad
F^b_{l,y}\mathrel{*}=H_l/H_r.
\]

For 512-to-64 conversion, both displacement components are divided by eight.

## 5. Stable-BG support

The loss must not compare ROI content or invalid samples. With `BG=1` and
`ROI=0`, define the geometric support

\[
M^{geo}_k
=M^{BG}_{k+1}
\cdot \mathcal W(M^{BG}_{k},F^b_k)
\cdot V^{bounds}_k.
\]

The warped source mask should be thresholded consistently (recommended
threshold: 0.5). All masks and flow are constants for this objective and must
be detached.

There are two defensible reliability policies, which answer different
questions:

1. **Recommended first training run — reliable correspondence:**

   \[
   M_k=M^{geo}_k\cdot\operatorname{stopgrad}(C^{FB}_k).
   \]

   This prevents uncertain/occluded student-flow regions from providing
   destructive gradients. It is safer for optimization.

2. **Geometric-only ablation:**

   \[
   M_k=M^{geo}_k.
   \]

   This retains difficult motion and occlusion but may force the model to fit
   correspondence that is unavailable or wrong. This should be an explicit
   ablation, not silently mixed with the reliable-support result.

The existing V8-R DCN uses geometric-only reliability and treats FB
confidence as diagnostic. That architectural choice does not require the new
training loss to ignore flow reliability. The temporal loss is a target, and
bad target correspondence is more harmful than uncertain feature fusion.

Always log both geometric support and FB-reliable support. A falling loss with
collapsing valid support is not an improvement.

## 6. Temporal loss definition

### 6.1 Recommended robust normalized loss

Use a channel-wise Charbonnier residual on stable BG:

\[
\rho(r)=\sqrt{r^2+\epsilon^2},\qquad \epsilon=10^{-3}.
\]

The masked loss is

\[
\mathcal L_{temp}^{abs}
=\frac{
  \sum_{k,x,c}M_k(x)\rho(r_{k,c}(x))
}{
  C\sum_{k,x}M_k(x)+\varepsilon
}.
\]

Because latent scale can change with timestep, also compute a relative term:

\[
\mathcal L_{temp}^{rel}
=\frac{
  \sum_{k,x}M_k(x)\|r_k(x)\|_2^2
}{
  \sum_{k,x}M_k(x)\|\hat z_{0,k+1}(x)\|_2^2+arepsilon
}.
\]

Use one of these as the optimized loss and log both. The recommended initial
optimization target is Charbonnier because it is less dominated by large
flow/occlusion outliers. Relative MSE provides easier comparison with the
existing latent-warp diagnostic.

### 6.2 Detaching the previous frame

For the first experiment, use

```text
previous = z0_hat[:-1].detach()
current  = z0_hat[1:]
```

This makes the previous frame a stop-gradient reference and prevents adjacent
frames from moving toward each other symmetrically or collapsing to an
over-smoothed solution. It also reduces graph retention. A later symmetric
ablation may remove the detach, but it must be reported separately.

### 6.3 Bidirectional option

The minimum implementation needs only backward warping because it provides a
well-defined causal previous-to-current target. A bidirectional loss can later
average backward and forward terms:

\[
\mathcal L_{temp}^{bi}
=\tfrac12(\mathcal L_{prev\rightarrow current}
          +\mathcal L_{next\rightarrow current}).
\]

Do not enable this automatically. Previous-only is the recommended first
controlled experiment because it matches sequential temporal memory and
avoids conflicting supervision when one direction is occluded.

## 7. Timestep weighting

At very high noise, \(\hat z_0\) is poorly conditioned and division by
\(\sqrt{\bar\alpha_t}\) can amplify errors. Temporal supervision should
therefore be weaker at low SNR.

A safe weighting is

\[
w_{snr}(t)=\min\left(1,\frac{\operatorname{SNR}(t)}{\gamma}\right),
\quad
\operatorname{SNR}(t)=\frac{\bar\alpha_t}{1-\bar\alpha_t},
\]

with \(\gamma\) exposed as a configuration value. The effective weight is

\[
\lambda_{temp}(s,t)
=\lambda_{temp}^{max}\,ramp(s)\,w_{snr}(t).
\]

This is different from inference-step indices such as 0, 10, 25, 40, and 49.
Training samples diffusion training timesteps, not positions in a 50-step DDIM
schedule.

## 8. Weight and curriculum

Do not choose \(\lambda_{temp}\) only from its numeric magnitude. Choose it
from its **weighted contribution and gradient effect** relative to
\(\mathcal L_{diff}\).

Recommended conservative first run:

```ini
TEMPORAL_TRAIN_LOSS_TYPE=charbonnier
TEMPORAL_TRAIN_LOSS_WEIGHT=0.01
TEMPORAL_TRAIN_LOSS_WARMUP_STEPS=500
TEMPORAL_TRAIN_DETACH_PREVIOUS=1
TEMPORAL_TRAIN_DIRECTION=previous_only
TEMPORAL_TRAIN_USE_FB_CONFIDENCE=1
TEMPORAL_TRAIN_CHARBONNIER_EPS=0.001
```

Candidate weight sweep after a short validation run:

```text
0.005, 0.01, 0.02
```

Initially target a weighted temporal contribution around 5–15% of the
diffusion loss and inspect gradient norms. A larger raw weight is not
automatically stronger if the valid support is small. Do not immediately use
the inference-guidance value `0.1` as a training-loss weight: scheduler
guidance scale and objective weight have different units and effects.

Recommended curriculum:

1. Load the chosen V8-R checkpoint and frozen V7 RAFT student.
2. Preserve the existing deform and temporal parameter groups.
3. Ramp the new temporal loss from zero over 500 optimizer steps.
4. Keep the existing diffusion loss active to protect spatial quality.
5. Start with previous-only, detached-reference supervision.
6. Select checkpoints using validation temporal metrics plus spatial metrics,
   not training temporal loss alone.

## 9. Exact integration point

The current shared trainer computes `model_prediction`, then `loss_diff`, then
variant losses before calling `accelerator.backward(loss)`. The temporal term
belongs immediately after `model_prediction` and predicted-clean conversion,
before the final loss sum:

```python
model_prediction = frozen_v8_predict(...)
loss_diff = diffusion_loss(model_prediction, target)

z0_hat = differentiable_predicted_clean(
    noisy_latents,
    model_prediction,
    timesteps,
    noise_scheduler,
)

loss_temporal, temporal_stats = temporal_training_loss(
    predicted_clean=z0_hat.reshape(B, T, 4, H, W),
    flow_backward=stc_output.raft_flow_backward_rgb,
    bg_masks=bg_mask_sequence,
    fb_confidence=stc_output.flow_confidence_backward,
    detach_previous=True,
)

loss = (
    loss_diff
    + deform_weight * loss_deform
    + offset_weight * loss_offset
    + temporal_weight * loss_temporal
)
```

Names above are conceptual; the implementation must use the actual available
V8-R output fields. It must reuse the frozen V7 flow already computed during
the adapter forward and must not instantiate another flow network.

The cleanest implementation is a V8-R-specific hook in the shared trainer,
because the existing `EXTRA_TRAIN_LOSS_FN` currently receives `stc_output` but
not `model_prediction`, `noisy_latents`, or `timesteps`. Either extend the hook
contract with these tensors in a backward-compatible way or introduce a new
post-prediction loss hook. Do not hide this computation inside the inference
scheduler.

## 10. Relationship to inference temporal guidance

The existing temporal DDIM scheduler computes a temporal objective at
inference and changes `pred_original_sample`/the DDIM update using its
gradient. That is **training-free guidance**.

The proposed training loss instead:

- contributes to `loss_total` during optimization;
- changes V8-R trainable weights;
- does not modify scheduler state;
- does not use `TEMPORAL_GUIDANCE_SCALE`;
- can later be evaluated with temporal guidance disabled to isolate what was
  learned.

The first comparison should be:

```text
same checkpoint initialization, same seed, no inference temporal guidance
    A: existing V8-R joint objective
    B: existing objective + temporal training loss
```

Only after this comparison should the learned model be combined with temporal
guidance.

## 11. Required TensorBoard metrics

At minimum log:

```text
train/loss_temporal
train/loss_temporal_weighted
train/temporal_effective_weight
train/temporal_relative_mse
train/temporal_charbonnier
train/temporal_no_warp_relative_mse
train/temporal_warp_gain
train/temporal_geometric_support_ratio
train/temporal_fb_reliable_support_ratio
train/temporal_flow_magnitude_rgb_px
train/temporal_snr_weight
train/temporal_grad_norm
train/loss_diff
train/loss_total
```

Retain the existing DCN diagnostics:

```text
train/loss_deform_alignment
train/loss_deform_alignment_weighted
train/loss_deform_offset
train/loss_deform_offset_weighted
train/deform_valid_backward
train/deform_valid_forward
train/deformation_minus_base_abs_mean
train/delta_abs_mean
train/lr_deform
train/lr_temporal
```

Useful motion-bin metrics should use the original RGB-pixel flow magnitude:

```text
motion_0_4
motion_4_16
motion_16_32
motion_32_plus
```

## 12. Validation protocol and acceptance criteria

Use a fixed validation subset and fixed seed. Evaluate the same sequences and
same output support for the baseline and temporal-loss checkpoints.

Required comparisons:

1. Shared noise + inference temporal guidance baseline.
2. Existing V8-R without temporal training loss.
3. V8-R with temporal training loss and inference guidance disabled.
4. V8-R with temporal training loss plus the unchanged inference guidance.

Do not select a checkpoint solely because `loss_temporal` decreases. Accept the
change only if:

- warped temporal error decreases on validation/output frames;
- flicker metrics improve at sequence macro-average, not just pair-weighted
  average;
- improvement persists in large-motion bins;
- support ratio does not collapse;
- spatial PSNR/SSIM/LPIPS degradation remains within the agreed trade-off;
- qualitative videos do not exhibit ghosting, dragging, frozen texture, or
  boundary leakage.

A low temporal error can be misleading when output becomes over-smoothed or
copies the previous frame. Spatial metrics and motion-region videos are
therefore mandatory.

## 13. Failure modes and checks

### Zero or ineffective gradient

Check that `z0_hat.requires_grad` is true and temporal gradients reach
`zero_conv`, temporal blocks, alignment fusion, and deformable modules. Frozen
BrushNet/U-Net parameters should have no gradients, while their operations
must remain in the graph.

### Incorrect flow direction

Backward flow on current-frame coordinates must warp previous to current. A
direction error often produces a temporal loss worse than no warp, especially
at motion boundaries.

### Incorrect displacement scaling

Resizing only the flow image without scaling dx/dy yields wrong motion at
latent resolution. Add an explicit synthetic translation test.

### Mask leakage

Verify that ROI pixels, source-ROI samples, and out-of-bounds coordinates have
zero loss weight. Visualize the final pair mask for several sequences.

### Support collapse

Track geometric and FB-reliable ratios independently. If loss falls while
support falls, optimization may simply be avoiding difficult pixels.

### Temporal collapse or ghosting

Compare no-warp error, warp error, spatial metrics, and video. Detaching the
previous frame and retaining diffusion loss reduce but do not eliminate this
risk.

### High-noise instability

Monitor loss by training timestep/SNR. Apply SNR weighting or restrict the
first experiment to a well-conditioned timestep range if gradients explode at
low SNR.

## 14. Implementation phases

### Phase 0 — diagnostic confirmation

Use the existing predicted-clean latent warp analysis to confirm that V7 flow
actually reduces latent correspondence error on the selected dataset and to
identify useful motion/support bins.

### Phase 1 — minimum temporal training loss

- predicted-clean latent only;
- previous-to-current backward warp;
- stable BG and in-bounds mask;
- detached previous frame;
- FB-reliable confidence for the first safe run;
- Charbonnier loss;
- weight 0.01 with 500-step warmup;
- no RGB decode and no cross-clip temporal loss.

### Phase 2 — controlled ablations

- temporal weights 0.005/0.01/0.02;
- reliable support versus geometric-only support;
- previous-only versus bidirectional;
- direct latent warp versus rescaled latent warp for the loss target;
- detached versus symmetric gradients.

### Phase 3 — optional extensions

Only if Phase 1 improves validation videos:

- output-RGB temporal loss on a small late-step subset;
- cross-clip overlap consistency;
- multi-scale U-Net feature temporal loss;
- adaptive weighting based on gradient norm.

These extensions must not be combined in the first experiment, otherwise the
source of any improvement or regression cannot be identified.

## 15. Recommended first experiment summary

```text
Initialization: selected V8-R joint checkpoint
Flow: frozen V7 ProPainter-RAFT student already used by V8-R
Representation: differentiable predicted-clean latent z0_hat
Direction: previous -> current using backward flow
Region: stable BG, source BG, in bounds, FB-reliable for first run
Loss: Charbonnier
Previous frame: detached
Base temporal weight: 0.01
Warmup: 500 optimizer steps
Diffusion loss: retained
Inference temporal guidance during validation: disabled first
Checkpoint selection: temporal + spatial metrics and sequence macro-average
```

This experiment directly tests whether supervision after the frozen
BrushNet/U-Net mapping can solve the gap observed between improved
intermediate alignment and limited final-video temporal improvement.
