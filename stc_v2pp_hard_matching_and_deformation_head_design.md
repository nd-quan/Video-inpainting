# STC-v2++ Motion-Aligned Sequence Noise

## Canonical design for the Hard-Matching and Deformation-Head variants

Status: Phase D1 core and Phase D2 trainer implemented; two-GPU smoke passed  
Implementation order: **Deformation Head first, Hard Matching second**

This document supersedes conflicting equations and background-only assumptions in
`sequence_level_motion_aligned_noise_codex_handoff.md`. It defines the two
STC-v2++ variants that will be implemented on top of the trained RGB STC-v2 and
shared-noise V8 checkpoints.

The two variants use different mechanisms to estimate correspondence, but share
the same sequence-noise state, variance-preserving fusion law, diffusion training
contract, and inference-time overlap reuse.

---

## 1. Problem statement

The current V8 shared-background-noise baseline uses one Gaussian field at the
same latent coordinates for every frame in a source sequence. For background
location `(x, y)`, all frames receive a correlated component sampled from the
same `(x, y)` location, even if the visible scene content has moved.

STC-v2++ replaces fixed-coordinate sharing with motion-aligned noise transport:

1. sample one Gaussian lineage anchor;
2. infer reference-to-target correspondence from frozen STC-v2 features;
3. transport the lineage noise along that correspondence;
4. fuse the transported lineage with independent Gaussian noise;
5. use the resulting tensor as the diffusion noise;
6. cache both lineage and final noise by absolute frame ID so overlapping clips
   reuse exactly the same tensors.

The correspondence is a noise-transport mechanism. It must not be described as
physical optical flow unless it is separately supervised and evaluated as such.

---

## 2. Verified repository contracts

### 2.1 STC-v2 inputs and outputs

The existing `RGBSTCConditionAdapter` receives:

```text
degraded RGB sequence: [B, T, 3, H, W]
background mask:       [B, T, 1, H, W]
```

Mask semantics after dataset loading are:

```text
M_BG = 1: strongly degraded background that must be restored
M_BG = 0: high-quality ROI that should be preserved
```

At `H = W = 512`, STC-v2 returns:

```text
features:        [B, T, 64, 64, 64]
delta_bg:        [B, T,  4, 64, 64]
latent_bg_mask:  [B, T,  1, 64, 64]
```

The restoration-condition branch remains unchanged:

```text
z_condition_aug = z_input + stc_injection_scale * delta_bg
BrushNet condition = concat(z_condition_aug, M_BG_latent)
```

Only `delta_bg` is gated by `M_BG`. The STC feature tensor itself is available
over the complete frame.

### 2.2 Training and inference composition

Training does not composite the ROI from the input. It applies diffusion noise
to the full latent, predicts the training target over the full latent, and uses
a global diffusion loss.

Inference produces two conceptually different outputs:

```text
raw output:   direct decoded diffusion prediction
final output: generated * M_BG + input * (1 - M_BG)
```

Therefore, hard ROI preservation is a post-generation operation. It is not a
constraint on the raw diffusion prediction during training or denoising.

---

## 3. Common STC-v2++ noise contract

### 3.1 Sequence lineage

Let the sequence lineage anchor be:

\[
L_1 \sim \mathcal N(0,I).
\]

For an adjacent reference frame `r = t - 1`, transport the lineage using a
backward sampling correspondence defined on target coordinates:

\[
\widetilde L_t(q)
=
\mathcal W\!\left(L_r,\Delta_{t\leftarrow r}\right)(q)
=
L_r\!\left(q+\Delta_{t\leftarrow r}(q)\right).
\]

`Delta_(t<-r)` tells each target location where to sample the reference. It is
not a forward splatting field.

Out-of-bound or invalid target locations receive a deterministic fresh Gaussian
field generated from the sequence seed and absolute target frame ID.

### 3.2 Transported-lineage normalization

Bilinear resampling can reduce Gaussian variance. Before fusion, normalize each
latent channel independently over spatial dimensions:

\[
L_t^{(c)}
=
\frac{\widetilde L_t^{(c)}-\mu_{HW}(\widetilde L_t^{(c)})}
{\sigma_{HW}(\widetilde L_t^{(c)})+\varepsilon}.
\]

The implementation must log statistics before and after normalization.

### 3.3 Variance-preserving fusion

Use exactly the existing shared-noise parameterization:

\[
E_t
=
\sqrt{\alpha}\,L_t
+
\sqrt{1-\alpha}\,\eta_t,
\qquad
\eta_t\sim\mathcal N(0,I),
\quad \alpha\in[0,1].
\]

`alpha` has the same semantics as the current
`shared_bg_noise_strength`/target correlated-variance fraction. For
`alpha = 0.9`, the lineage coefficient is `sqrt(0.9)` and the independent
coefficient is `sqrt(0.1)`.

Do not replace this equation with:

```text
alpha * lineage + sqrt(1 - alpha**2) * independent
```

and do not use linear interpolation weights `(alpha, 1 - alpha)`.

### 3.4 No final global normalization by default

Do not normalize `E_t` again over the full frame or clip by default. If `L_t`
and `eta_t` both have unit variance and are independent, the fusion equation
already has unit variance. A final global normalization would alter the intended
component balance and introduce sample-wide statistical coupling.

Final-noise mean and standard deviation must still be logged. A final global
normalization may exist only as an explicit ablation flag, disabled by default.

### 3.5 Warp scope

The primary STC-v2++ experiment is:

```text
warp_scope = full
```

For this mode, `E_t` is applied over the entire latent frame. Hard ROI
compositing still occurs only after inference.

The implementation must preserve a matched ablation:

```text
warp_scope = bg
```

For the background-only ablation:

\[
E_t^{used}
=
M_{BG,t}\odot E_t
+
(1-M_{BG,t})\odot\eta_t.
\]

The internal lineage may still be maintained over the full frame so a location
can retain lineage if its mask label later changes.

---

## 4. Sequence-state contract

The inference implementation must keep two different caches:

```python
@dataclass
class SequenceNoiseState:
    sequence_id: str
    seed: int
    anchor_frame_id: int
    lineage_noise: dict[int, torch.Tensor]
    final_noise: dict[int, torch.Tensor]
```

`lineage_noise[frame_id]` is the normalized transported state used as the
reference for later frames. `final_noise[frame_id]` is the fused noise actually
passed to the diffusion pipeline.

The final fused noise must not be recursively used as the next lineage source.
Otherwise the sequence-anchor contribution is multiplied by another
`sqrt(alpha)` at every transition and decays with sequence length.

Required behavior:

1. create state once per `sequence_id`;
2. sample exactly one lineage anchor for the first absolute frame;
3. cache every generated lineage and final noise tensor;
4. never resample an already cached absolute frame;
5. reuse overlap-frame tensors exactly, with maximum absolute difference zero;
6. select the latest cached adjacent/overlap frame as the reference for new
   frames;
7. generate all fallback and independent noise deterministically from sequence
   seed and absolute frame ID;
8. isolate state between sequences;
9. release cache entries only after they can no longer appear in a later overlap.

For `clip_length = 8` and `stride = 6`:

```text
clip 1:  1  2  3  4  5  6  7  8
clip 2:                    7  8  9 10 11 12 13 14
clip 3:                                      13 14 15 16 ...
```

Frames 7 and 8 in clip 2 must reuse both cached tensors from clip 1. New lineage
is then propagated locally from frame 8 to frame 9, from frame 9 to frame 10,
and so on.

---

## 5. Variant A: STC-v2++ Deformation Head

This is the first implementation target.

### 5.1 Purpose

Learn a bounded dense backward sampling field from an ordered adjacent pair of
frozen STC-v2 features, without RAFT, cached teacher flow, or a separately
supervised optical-flow model.

### 5.2 Model flow

```text
degraded RGB + M_BG
        |
        v
frozen trained STC-v2
        |
        +---------------------------> frozen delta_bg condition branch
        |
        v
h_r, h_t: [B, 64, H_l, W_l]
        |
        v
concat(h_r, h_t, h_t - h_r): [B, 192, H_l, W_l]
        |
        v
NoiseDeformationHead
        |
        v
raw backward offset: [B, 2, H_l, W_l]
        |
        v
tanh * max_displacement
        |
        v
pixel-center sampling grid (align_corners=False)
        |
        v
bilinear warp of reference lineage noise
        |
        v
invalid/out-of-bound fallback + per-channel spatial normalization
        |
        v
sqrt(alpha) * lineage + sqrt(1-alpha) * independent
        |
        v
full-frame diffusion noise E_t
```

### 5.3 Deformation-head architecture contract

The first head must be lightweight and reference-target conditioned:

```text
input channels:  3 * STC feature channels = 192
output channels: 2 (dx, dy in latent-pixel units)
```

The final offset layer must be zero-initialized, so the first forward pass is an
identity deformation rather than a random warp. The output must be bounded:

\[
\Delta
=
\Delta_{max}\odot\tanh(D_{raw}).
\]

Use one sampling coordinate per target location. Do not use modulated deformable
convolution for Gaussian lineage transport because it samples and mixes multiple
kernel locations with learned weights, which changes the noise distribution and
breaks the pure-transport interpretation.

### 5.4 Correct grid convention

Use `grid_sample(..., align_corners=False)`. The normalized coordinate of pixel
center `(x, y)` must be:

\[
x_n=\frac{2(x+0.5)}{W}-1,
\qquad
y_n=\frac{2(y+0.5)}{H}-1.
\]

Pixel displacement is normalized as:

\[
\Delta x_n=\frac{2\Delta x}{W},
\qquad
\Delta y_n=\frac{2\Delta y}{H}.
\]

The implementation must not combine a `linspace(-1, 1)` endpoint grid with
`align_corners=False`.

### 5.5 Feature-matching loss

Warp the frozen reference STC feature using the predicted target-to-reference
sampling field:

\[
\widehat h_{r\rightarrow t}(q)
=
h_r(q+\Delta_{t\leftarrow r}(q)).
\]

Then minimize cosine feature mismatch over valid locations:

\[
\mathcal L_{match}
=
\frac{1}{|\Omega|}
\sum_{q\in\Omega}
\left[
1-
\cos\!\left(\widehat h_{r\rightarrow t}(q),h_t(q)\right)
\right].
\]

Reference and target features are detached/frozen for the first implementation.
This prevents representation collapse and ensures the loss trains the
deformation head rather than changing the matching space.

### 5.6 Edge-aware spatial smoothness

Let `Delta = (u, v)`. First-order spatial smoothness is:

\[
\mathcal L_{smooth}
=
\operatorname{mean}
\left(
w_x|\partial_x\Delta|
+
w_y|\partial_y\Delta|
\right).
\]

Weights are derived from frozen target STC feature edges:

\[
w_x=\exp(-\gamma\|\partial_x h_t\|),
\qquad
w_y=\exp(-\gamma\|\partial_y h_t\|).
\]

The loss encourages locally coherent offsets inside a feature region while
allowing discontinuities at strong feature boundaries. It does not determine
the correct displacement direction and cannot be used without a matching or
task loss. Zero offset also has zero smoothness loss.

### 5.7 Diffusion loss and gradient contract

Construct shaped noise `E_t`, retain its gradient in `scheduler.add_noise`, and
detach it only when it is used as the diffusion target:

\[
z_t
=
\operatorname{add\_noise}(z_0,E,t),
\]

\[
\mathcal L_{diff}
=
\operatorname{MSE}
\left(
\epsilon_\theta(z_t,t,c),
\operatorname{stopgrad}(E)
\right).
\]

This allows the restoration objective to reach the deformation head through
the noisy-latent input while preventing a moving-target shortcut through the
noise target.

The first total objective is:

\[
\mathcal L
=
\mathcal L_{diff}
+
\lambda_{match}\mathcal L_{match}
+
\lambda_{smooth}\mathcal L_{smooth}.
\]

All raw and weighted components must be logged separately.

### 5.8 First-stage trainable and frozen components

Initialize from:

```text
V8 checkpoint:
experiments/checkpoint_sharedNoise_sameBG_0.9/checkpoint-2000

STC-v2 checkpoint:
experiments/checkpoint_rgb_stc_v2_sharedNoise_0.9/checkpoint-2000/stc_adapter
```

First-stage component policy:

```text
trainable:
  - NoiseDeformationHead
  - BrushNet

frozen:
  - STC-v2 spatial encoder
  - STC-v2 temporal blocks
  - STC-v2 delta condition branch for the first controlled run
  - base U-Net weights
  - IP attention processors and image projection for the first controlled run
  - FGBG fusion for the first controlled run
  - VAE, text encoder, and image encoder
```

If the controlled run underfits, later ablations may unfreeze the STC condition
output head or the same IP/fusion components trained in the original V8 stage.
They must not be silently included in the first run.

### 5.9 Deformation-head training flow

```text
clip batch [B, T]
    |
    +--> VAE encode GT/input (no grad)
    +--> build frozen V8 text/image context (no grad)
    |
    v
frozen STC-v2 forward once
    |
    +--> fixed BrushNet condition
    +--> detached STC features
    |
    v
predict adjacent backward offsets for 1<-0, 2<-1, ..., T-1<-T-2
    |
    +--> warp reference STC feature --> L_match
    +--> spatial offset gradients --> L_smooth
    +--> recurrently warp lineage noise
    |
    v
normalize lineage + variance-preserving fusion
    |
    v
scheduler.add_noise(GT latent, shaped noise, timestep)
    |
    v
trainable BrushNet + frozen U-Net
    |
    v
L_diff + lambda_match * L_match + lambda_smooth * L_smooth
```

Training initially uses local recurrent transport inside each shuffled clip. It
does not maintain mutable sequence state across independently shuffled DDP
batches. Sequence-level state is applied during ordered inference. A later
sequence-aware training milestone would require sequence ownership per rank,
ordered sampling, balanced DDP steps, and resume-safe state serialization.

### 5.10 Deformation-head inference flow

```text
ordered clips for one sequence
    |
    v
load/create SequenceNoiseState
    |
    v
frozen STC-v2 features + fixed condition
    |
    v
reuse exact cached lineage/final noise for overlap frame IDs
    |
    v
for every unseen frame:
    previous cached frame --> predicted deformation --> warped lineage
    invalid coordinates --> deterministic fresh Gaussian
    normalize --> fuse --> cache lineage and final noise
    |
    v
pass final clip noise as pipeline `latents`
disable the old shared-background mixer for this call
    |
    v
raw generated frames
    |
    v
optional final hard ROI composite
```

---

## 6. Variant B: STC-v2++ Hard Matching

This is the second implementation target and the non-parametric comparison to
the learned deformation head.

### 6.1 Purpose

Build a dense discrete correspondence directly from frozen STC-v2 feature
similarity, without a learned offset head.

### 6.2 Hard-matching equation

Normalize features channel-wise. For each target location `q`, search a local
window in the adjacent reference feature map:

\[
p^*(q)
=
\arg\max_{p\in\mathcal N(q)}
\cos(h_t(q),h_r(p)).
\]

Transport lineage with an integer gather:

\[
L_t(q)=L_r(p^*(q)).
\]

The dense integer offset is only a derived diagnostic:

\[
\Delta_{hard}(q)=p^*(q)-q.
\]

It is not the output of a trainable network.

### 6.3 Full-frame matching policy

For the primary `warp_scope = full` experiment:

```text
region_policy = unconstrained
search_scope = local_window
```

Cross-region correspondence is allowed because a physical point may change
between BG and ROI labels across frames. “Unconstrained” does not mean a global
whole-frame search. Candidates remain restricted to the configured local
window.

Required reliability controls:

1. top-1 similarity threshold;
2. top-1 versus top-2 margin;
3. mutual reference-target consistency;
4. in-bound validity;
5. deterministic fresh-noise fallback for rejected matches;
6. diagnostics for cross-region ratio, invalid ratio, source-collision ratio,
   and unique-source ratio.

A soft region-mismatch penalty may be exposed as an ablation, but strict
same-region matching is not the primary full-frame policy.

### 6.4 Why hard transport has no transport loss

The hard matcher contains no `nn.Parameter`, and `argmax` does not provide a
useful gradient for changing the selected discrete index. The transport map is
computed, not learned.

Therefore the hard-matching variant has no deformation optimizer,
`L_match`, or `L_smooth`. Matching confidence is a diagnostic and rejection
criterion, not a training loss.

The restoration network still requires adaptation because hard-warped noise is
different from the checkpoint's fixed-coordinate shared-background noise.

### 6.5 Hard-matching training flow

```text
clip batch [B, T]
    |
    +--> VAE encode GT/input (no grad)
    +--> build frozen V8 context (no grad)
    |
    v
frozen trained STC-v2
    |
    +--> fixed BrushNet condition
    +--> frozen STC features
    |
    v
adjacent local cosine search + hard argmax + confidence rejection
    |
    v
integer-gather recurrent lineage transport
    |
    v
invalid fallback + per-channel normalization
    |
    v
sqrt(alpha) * lineage + sqrt(1-alpha) * independent
    |
    v
scheduler.add_noise(GT latent, shaped noise, timestep)
    |
    v
trainable BrushNet + frozen U-Net
    |
    v
standard diffusion MSE against the hard-warped noise target
```

First-stage component policy:

```text
trainable:
  - BrushNet

frozen:
  - hard matcher (non-parametric)
  - all STC-v2 components
  - base U-Net
  - IP processors/projection and FGBG fusion for the controlled first run
  - VAE, text encoder, and image encoder
```

### 6.6 Hard-matching inference flow

```text
ordered overlapping clips
    |
    v
reuse exact cached overlap lineage/final noise
    |
    v
for unseen adjacent frames:
    frozen STC features
        --> local hard match
        --> confidence/mutual validation
        --> integer lineage gather or deterministic fresh fallback
        --> normalize and fuse
        --> cache
    |
    v
pipeline latents --> raw generation --> optional hard ROI composite
```

---

## 7. Shared checkpoint and resume contract

The new stage must distinguish initialization from exact resume:

```text
--baseline_checkpoint         frozen/trainable V8 initialization
--init_stc_v2_adapter_path    trained STC-v2 initialization
--resume_from_checkpoint      exact resume of the same v2++ topology only
```

Deformation-head checkpoints must contain:

```text
checkpoint-N/
  brushnet/
  stc_adapter/                 self-contained frozen component copy or hash
  noise_deformation/
  accelerator_state/
  metadata.json
```

Hard-matching checkpoints must contain:

```text
checkpoint-N/
  brushnet/
  stc_adapter/                 self-contained frozen component copy or hash
  accelerator_state/
  metadata.json
```

Metadata must include all trajectory- or inference-critical values:

```text
variant
warp_scope
alpha
clip length and stride
reference mode
lineage normalization mode
final global normalization flag
grid convention and align_corners
maximum deformation displacement
matching window radius
matching confidence and margin thresholds
mutual-consistency policy
invalid fallback policy
loss weights
frozen/trainable component list
initial checkpoint paths and hashes
seed and refresh policy
optimizer/scheduler/mixed-precision/DDP settings
```

---

## 8. Evaluation matrix

Use the same input sequence, checkpoint initialization, prompt, diffusion
scheduler, number of steps, seed, and ROI-compositing policy for:

1. independent noise;
2. current fixed-coordinate shared BG noise;
3. deformation-head transport with `warp_scope = bg`;
4. deformation-head transport with `warp_scope = full`;
5. hard matching with `warp_scope = bg`;
6. hard matching with `warp_scope = full`.

Report raw and final-composited metrics separately:

```text
BG reconstruction quality
raw ROI preservation
final ROI preservation
within-clip temporal consistency
clip-boundary temporal consistency
performance versus distance from the sequence anchor
noise mean/std per channel
spatial noise autocorrelation
overlap-noise maximum absolute difference
invalid/fresh-noise ratio
runtime and peak memory
```

Additional deformation-head diagnostics:

```text
offset magnitude
offset spatial gradient
in-bound sampling ratio
raw and weighted L_match
raw and weighted L_smooth
```

Additional hard-matching diagnostics:

```text
top-1 similarity
top-1/top-2 margin
mutual-match acceptance ratio
cross-region match ratio
source-collision ratio
unique-source ratio
```

---

## 9. Required tests before a full training run

### Common noise and state tests

- zero/identity correspondence preserves the reference lineage;
- the fusion endpoints `alpha = 0` and `alpha = 1` are exact;
- `alpha = 0.9` maintains expected variance within tolerance;
- per-channel normalization is finite and restores unit spatial variance;
- no final global normalization occurs by default;
- overlap frame lineage and final noise are exactly reused;
- deterministic reruns reproduce every absolute-frame noise tensor;
- different sequence IDs never share state;
- invalid locations use deterministic fresh Gaussian noise;
- `warp_scope = full` affects ROI noise;
- `warp_scope = bg` leaves final ROI noise independent;
- mask polarity remains `M_BG = 1`, `M_ROI = 0`.

### Deformation-head tests

- the zero-initialized head produces an identity sampling grid;
- a known one-pixel backward displacement samples the expected source pixel;
- the `align_corners=False` pixel-center grid is correct;
- offset bounds are enforced;
- `L_match` decreases for a correct synthetic displacement;
- constant translation has zero first-order `L_smooth`;
- irregular neighboring offsets have positive `L_smooth`;
- feature-edge weighting permits a discontinuity at an edge;
- gradients reach the deformation head through `L_match`;
- gradients reach the deformation head through the noisy-latent diffusion path;
- the detached diffusion target has no gradient path;
- frozen STC-v2 and base U-Net parameters receive no gradients.

### Hard-matching tests

- a known translated feature patch produces the expected integer offset;
- search never leaves the configured local window;
- low-confidence and non-mutual matches are rejected;
- cross-region matching is permitted in full-frame unconstrained mode;
- invalid matches use the deterministic fallback;
- the matcher has no trainable parameters;
- integer gather preserves the selected reference values exactly before optional
  normalization.

### Integration tests

- existing STC-v2 shared-noise evaluation remains unchanged;
- each v2++ variant completes a two-clip dry run;
- deformation-head single-GPU backward/optimizer smoke test passes;
- deformation-head two-GPU DDP smoke test passes;
- hard-matching BrushNet adaptation single- and two-GPU smoke tests pass;
- checkpoint save/load and exact resume pass for each topology;
- diagnostics are written without changing generation results.

---

## 10. Planned implementation order

### Phase D1: Deformation-head core

Implemented in
`examples/brushnet/STC_encoder_v2pp_deformation/noise_deformation.py` with
focused tests in `test_noise_deformation.py`:

1. pure bounded deformation head, pixel-center grid, backward warp, invalid
   fallback, lineage normalization, and variance-preserving fusion;
2. recurrent within-clip lineage construction with separate lineage and final
   noise tensors;
3. default full-frame scope and explicit background-only ablation;
4. zero-offset identity, coordinate, Gaussian-statistics, loss, and gradient
   tests;
5. no Hard-Matching implementation and no changes to the STC-v2 baseline.

### Phase D2: Deformation-head training

Implemented in
`examples/brushnet/STC_encoder_v2pp_deformation/train_noise_deformation.py`
with launcher `run_train_noise_deformation.sh`:

1. new v2++ trainer; the STC-v2 baseline script is unchanged;
2. V8 and trained STC-v2 are separate initialization arguments;
3. frozen STC features and condition are computed before differentiable shaped
   noise construction;
4. shaped noise remains differentiable through `scheduler.add_noise`, while the
   epsilon/v target uses `noise.detach()`;
5. only BrushNet and `NoiseDeformationHead` are optimized;
6. checkpoints explicitly save BrushNet, frozen STC-v2, deformation head,
   Accelerator state, and strict metadata;
7. schema/data/mask preflight passes. A one-step two-GPU smoke run completed
   with effective batch 6, finite loss/noise statistics, synchronized DDP
   checkpoint state for both models, and one RNG state per rank.

### Phase D3: Sequence-level deformation inference

1. add ordered sequence state and exact overlap reuse;
2. pass complete shaped clip noise through pipeline `latents`;
3. disable the old shared-background mixer for v2++ calls;
4. save correspondence, lineage, validity, statistics, raw output, and final
   composite diagnostics;
5. run matched-seed `bg` versus `full` scope comparisons.

### Phase H1: Hard-matching core and adaptation training

1. add local-window hard similarity matching and reliability checks;
2. reuse the common lineage/state/fusion utilities;
3. add hard-matching-specific tests and diagnostics;
4. create the BrushNet adaptation training path with standard diffusion MSE;
5. keep STC-v2 frozen and do not add deformation losses.

### Phase H2: Sequence-level hard-matching inference

1. add exact overlap reuse through the common state manager;
2. run the same evaluation matrix used for the deformation head;
3. compare quality, temporal consistency, collision statistics, and runtime.

The design flow has been approved. D3 and Hard Matching remain separate later
phases and must not be folded into the first deformation-head training run.
