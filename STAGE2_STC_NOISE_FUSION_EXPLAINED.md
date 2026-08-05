# Stage 2 — STC Flow-Guided Noise Fusion

## 1. Short answer

Yes. During Stage-2 training, the model first constructs a temporally shaped
noise clip and then uses that noise in the forward-diffusion equation together
with the encoded clean/GT latent:

\[
z_\tau
=
\sqrt{\bar\alpha_\tau}\,z_0^{GT}
+
\sqrt{1-\bar\alpha_\tau}\,\widetilde\epsilon.
\]

Here:

- \(z_0^{GT}\) is the VAE latent of the clean ground-truth video;
- \(\widetilde\epsilon\) is the STC flow-guided fusion noise;
- \(z_\tau\) is the noisy latent passed to BrushNet and the diffusion U-Net;
- \(\tau\) is a randomly sampled diffusion timestep.

The degraded/input latent is **not** the latent being corrupted in this
equation. It is used as structural conditioning for the STC encoder, flow
prediction, beta prediction, and BrushNet.

## 2. Two different meanings of “forward”

These operations should not be confused:

1. **Forward diffusion**: analytically adds noise to the clean latent and
   produces \(z_\tau\). It has no neural-network parameters.
2. **Denoiser forward pass**: sends \(z_\tau\) through frozen BrushNet and the
   frozen diffusion U-Net to predict the noise.

Stage 2 performs both operations in this order.

## 3. Inputs of one Stage-2 training batch

For the current full configuration:

```text
GT video             [B,T,3,512,512]
Degraded/input video [B,T,3,512,512]
ROI mask             [B,T,1,512,512], 1=ROI and 0=background
Clip length          T=8
```

The frozen VAE produces:

\[
z_0^{GT}=E_{VAE}(I^{GT}),
\qquad
z^D=E_{VAE}(I^D),
\]

with latent tensors approximately shaped as:

```text
[B,T,4,64,64]
```

The clean latent may use a posterior sample during training. The degraded
latent uses the posterior mode so that the STC condition remains deterministic.

The mask passed to the STC/BrushNet path is converted to:

\[
M^{BG}=1-M^{ROI}.
\]

## 4. Frozen Stage-1 motion representation

The Stage-1 checkpoint supplies:

```text
STC encoder
Adjacent bidirectional flow decoder
```

The STC input is:

\[
H^{STC}=E_{STC}([z^D,M^{BG}]).
\]

The shared directional decoder predicts adjacent flow in both directions. The
backward flow used for noise transport is:

\[
F_{t\rightarrow t-1}.
\]

During standard Stage 2, the STC encoder and flow decoder are frozen. Flow is
therefore computed on every batch but its weights are not updated.

## 5. Independent Gaussian noise

Stage 2 first samples independent noise for every frame:

\[
\epsilon_t^{ind}\sim\mathcal N(0,I),
\qquad t=0,\ldots,T-1.
\]

The first frame noise \(\epsilon_0^{ind}\) is used as the temporal anchor.

## 6. Anchor-noise warping

Adjacent predicted flows are composed into a current-to-anchor flow
\(F_{t\rightarrow0}\). Anchor noise is then transported to frame \(t\):

\[
\epsilon_t^{warp}
=W(\epsilon_0^{ind},F_{t\rightarrow0}).
\]

Bilinear resampling reduces variance because the output is a weighted sum of
four samples. The implementation divides by the square root of the sum of the
squared interpolation weights so that valid warped samples recover unit
marginal variance.

The current configuration uses:

```json
"warp_region": "all"
```

Therefore noise is warped over the whole frame. The mask is STC semantic
context; it does not restrict warping to the background. Out-of-frame samples
are still marked invalid because they have no anchor-noise source.

## 7. Learned beta and noise fusion

The Stage-2 beta head receives the STC feature and predicts a spatial beta map:

\[
\beta_t
=
\beta_{min}
+
(\beta_{max}-\beta_{min})
\sigma(f_\beta(H_t^{STC})).
\]

The current bounds are:

```text
beta_min     = 0.05
beta_max     = 0.95
initial_beta = 0.50
```

Let \(V_t\) be geometric warp validity. The effective coefficient is:

\[
c_t=\beta_tV_t.
\]

The fusion noise is:

\[
\widetilde\epsilon_0=\epsilon_0^{ind},
\]

\[
\widetilde\epsilon_t
=
c_t\epsilon_t^{warp}
+
\sqrt{1-c_t^2}\,\epsilon_t^{ind},
\qquad t>0.
\]

This creates temporal correlation while retaining fresh stochastic innovation.
For an invalid warp, \(V_t=0\), so the frame automatically falls back to fresh
independent Gaussian noise.

## 8. Forward diffusion with the fusion noise

One diffusion timestep \(\tau\) is sampled per video clip and shared by every
frame in that clip. The scheduler then applies:

\[
z_{\tau,t}
=
\sqrt{\bar\alpha_\tau}\,z_{0,t}^{GT}
+
\sqrt{1-\bar\alpha_\tau}\,\widetilde\epsilon_t.
\]

In code, this is:

```python
noisy = noise_scheduler.add_noise(
    gt_latents.flatten(0, 1),
    shaped_noise.flatten(0, 1),
    timesteps.repeat_interleave(T),
)
```

Thus the fusion noise is not simply added to the latent with coefficient one.
The scheduler scales the clean latent and noise according to the selected
diffusion timestep.

## 9. Frozen denoiser forward pass

The resulting noisy latent is passed through:

```text
z_tau
  -> frozen BrushNet with degraded latent + BG mask condition
  -> BrushNet down/mid/up residuals
  -> frozen diffusion U-Net with V8 IP-Adapter conditioning
  -> predicted noise epsilon_hat
```

The V8 image condition follows:

\[
e_{image}
=e_{base}+\alpha_{fusion}e_{fusion(FG,BG)}.
\]

BrushNet, the U-Net, IP-Adapter, FGBG fusion, VAE, and text encoder are frozen
in the standard Stage-2 configuration.

## 10. Diffusion target and gradient path

For epsilon prediction, the target is the fusion noise:

\[
L_{noise}
=
\lVert\widehat\epsilon_\theta(z_\tau,\tau,c)
-\operatorname{stopgrad}(\widetilde\epsilon)\rVert_2^2.
\]

The target is explicitly detached. This is important: beta cannot reduce loss
by moving both the noisy input and its own training label simultaneously.
Instead, beta is learned through this path:

```text
beta head
  -> fusion noise
  -> noisy latent z_tau
  -> frozen BrushNet/U-Net
  -> diffusion loss
```

Gradients pass through the frozen denoiser to the beta head, but frozen model
weights do not receive optimizer updates.

## 11. Temporal loss

The predicted clean latent is reconstructed from \(z_\tau\) and the predicted
noise. In epsilon-prediction form:

\[
\widehat z_0
=
\frac{z_\tau-\sqrt{1-\bar\alpha_\tau}\,\widehat\epsilon}
{\sqrt{\bar\alpha_\tau}}.
\]

The previous predicted-clean latent is warped into the current frame using the
frozen Stage-1 backward flow:

\[
L_{temp}
=
\frac{1}{N}
\sum_{t=1}^{T-1}
V_t\,ho\left(
\widehat z_{0,t}
-W(\widehat z_{0,t-1},F_{t\rightarrow t-1})
\right),
\]

where \(\rho\) is the Charbonnier penalty and \(N\) is the number of valid
latent elements used in the average.

The flow is detached in this loss. Therefore Stage 2 adjusts beta/noise fusion,
not the Stage-1 flow predictor.

## 12. Complete Stage-2 objective

The current full configuration uses:

\[
L_{Stage2}
=
1.0L_{noise}
+0.05L_{temp}^{weighted}
+0.01L_{noise\_stats}
+0.001L_{\beta\_prior}
+0.001L_{\beta\_smooth}.
\]

The auxiliary terms are:

- \(L_{noise\_stats}\): encourages mean near zero and standard deviation near
  one;
- \(L_{\beta\_prior}\): keeps mean beta near the configured target, currently
  0.5;
- \(L_{\beta\_smooth}\): discourages abrupt spatial changes in beta.

## 13. What is actually trained?

With `train_vcm_stc_noise_fusion_2gpu.json`, only the beta head is trainable:

```text
Trainable:
  beta_head

Frozen:
  STC encoder
  bidirectional flow decoder
  VAE
  text encoder
  BrushNet
  IP-Adapter and FGBG fusion
  diffusion U-Net
```

The trainer also contains an optional `temporal_adapter` path, but it is not
enabled by the standard Stage-2 config. It is a separate ablation and is not
part of beta-only Stage 2 described here.

## 14. What does the checkpoint contain?

Each Stage-2 checkpoint stores:

```text
checkpoint-XXXXXXX/
  noise_shaper/
    config.json
    diffusion_pytorch_model.safetensors
  trainer_state.pt
  config.json
```

`noise_shaper` contains:

```text
Frozen Stage-1 STC encoder weights
Frozen Stage-1 flow-decoder weights
Learned Stage-2 beta-head weights
Noise-fusion configuration
```

`trainer_state.pt` contains the optimizer, LR scheduler, AMP scaler, current
step, and best validation score. `latest.json` and `best.json` point to the
corresponding checkpoint directories.

## 15. Training versus inference

### Training

GT is available, so forward diffusion constructs a noisy training latent:

```text
GT -> VAE -> clean latent z0
fusion noise + z0 -> scheduler.add_noise -> z_tau
z_tau -> frozen denoiser -> training losses
```

### Inference

GT is unavailable. There is no operation that adds noise to a GT latent.
Instead, the fusion noise itself initializes the reverse diffusion process:

```text
degraded video -> STC -> predicted flow and beta
Gaussian anchor + warping + fusion -> shaped initial noise z_T
z_T -> reverse denoising -> reconstructed video
```

## 16. Fixed-beta variant

If beta is fixed to 0.5, 0.7, or 0.9, beta-head training and therefore Stage-2
optimization can be skipped. However, the following Stage-2 computations are
still required during training and inference:

```text
STC feature extraction
flow prediction
flow composition
anchor-noise warping
bilinear variance correction
fixed-beta fusion with independent noise
```

The resulting shaped noise can then be used directly when training Stage 3.
Stage 3 should be trained and evaluated with the same fixed-beta setting unless
beta randomization/conditioning is explicitly introduced.

## 17. Code map

- Batch construction, shaped noise, forward diffusion and losses:
  `examples/brushnet/train_stc_noise_fusion_vcm.py`, `compute_batch()`.
- Stage-1-to-Stage-2 transfer and beta-only freezing:
  `examples/brushnet/stc_noise_fusion_training.py`.
- Flow composition, Gaussian noise warping and beta fusion:
  `examples/brushnet/diffusers/models/stc_noise_shaper.py`.
- Standard configuration:
  `examples/brushnet/configs/train_vcm_stc_noise_fusion_2gpu.json`.

