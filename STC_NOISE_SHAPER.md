# STC-Conditioned Noise Shaping for VCM BrushNet

> **Legacy external-flow ablation.** This document describes the earlier
> nine-channel STC experiment that consumes raw/refined flow. It is retained
> for controlled comparisons, but it is **not** the current full-flow training
> plan. The implementation and review checklist for flow-from-scratch plus
> whole-frame noise warping are in
> [`STC_FULL_FLOW_DEVELOPER_HANDOFF.md`](STC_FULL_FLOW_DEVELOPER_HANDOFF.md).

## Purpose

This is an STC-inspired adaptation for the current image-diffusion BrushNet
pipeline, not a literal reproduction of VideoComposer. VideoComposer uses its
STC features as U-Net control signals; this module uses STC features to shape
the initial latent-noise covariance.

The module replaces frame-independent or fixed-coordinate shared background
noise with condition-aware latent noise. It uses the degraded background,
an input optical flow, motion confidence, and stable-background confidence to
move a Gaussian anchor noise field through the clip.

The module is opt-in. When disabled, the original motion-adapter training and
shared-background inference paths are unchanged.

## Conditions

At latent resolution, the input for frame `t` is:

```text
[decoded_background_latent (4), background_mask (1),
 normalized_backward_flow (2), motion_confidence (1), stable_bg (1)]
```

The mask convention is `1 = background`, `0 = ROI`. Pairwise flow and
confidence maps are padded with zeros for frame zero.

## Architecture

```text
[B,T,9,H,W]
  -> frame-wise Conv2D + SiLU
  -> AvgPool2D
  -> frame-wise Conv2D + SiLU
  -> reshape [B*h*w,T,C]
  -> temporal Transformer
  -> resize to latent resolution
  -> flow-residual head + beta-gate head
```

### VideoComposer STC architecture

There is no `SFCEncoder` in the cloned official repository; `SFC` is treated
as a typo for `STC`. VideoComposer implements STC inline in its U-Net as one
spatial-CNN plus per-location temporal-Transformer branch for every condition.
The branches are added before entering the denoiser.

The `encoder_architecture="videocomposer"` option ports that small pattern in
pure PyTorch. It does not import the VideoComposer U-Net. The nine VCM
condition channels are split into three independent branches:

```text
structure:   [decoded BG latent (4), BG mask (1)]
motion:      [normalized raw RAFT backward flow (2)]
reliability: [flow confidence (1), stable BG (1)]
```

Each branch follows:

```text
frame-wise Conv2D spatial encoder
  -> reshape [B*h*w,T,C]
  -> VideoComposer-style temporal self-attention
  -> resize to latent resolution
```

The three outputs are summed, matching VideoComposer's condition fusion, and
the fused feature predicts the beta gate used by the noise warper. Unlike the
official model, the feature is not concatenated into the SD1.5 U-Net input, so
the frozen BrushNet/U-Net architecture remains unchanged.

The local clone currently has no `non_ema_228000.pth`, only the source code.
Consequently this reuses the official architecture, not pretrained STC
weights. This is also safer because official weights were trained on sparse
MPEG-4 block motion vectors with SD2.1, whereas this project uses dense,
normalized RAFT flow with BrushNet/SD1.5.

The optional flow head predicts a bounded residual around the input flow:

```text
predicted_flow = input_backward_flow + max_delta * tanh(flow_head(STC))
```

Its final layer is initialized to zero. In the phase-1 noise-only experiment,
`predict_flow_residual=false`, so the raw RAFT flow is fixed and the STC path
learns only the beta gate. This makes the source of any improvement identifiable.

The gate predicts `beta` in `[0, beta_max]`. Two warp-region policies are
available. `stable_bg` reproduces the conservative hard gate. `all` applies
the learned beta over the complete frame and uses only geometric in-bounds
validity as a mandatory gate. The VideoComposer-STC configs use `all` because
the final ROI is composited from the input and a hand-designed stable-region
threshold can be too conservative. `stable_bg` and flow confidence remain STC
conditions, so the network can lower beta softly instead of being forced to
zero outside a binary region.

## Noise construction

Adjacent backward flows are composed into a current-to-anchor field. Every
frame is warped directly from frame-zero anchor noise, avoiding repeated
interpolation of already-warped noise.

For the all-frame configuration:

```text
epsilon_t = valid_t * beta_t * warp_G(epsilon_0, flow_t_to_0)
          + sqrt(1 - (valid_t * beta_t)^2) * eta_t
```

`valid_t` is one only where the backward sampling coordinate is inside the
anchor frame. It cannot be removed safely because out-of-frame coordinates
have no source noise. `warp_G` analytically corrects the variance loss caused by bilinear
interpolation using the sum of squared interpolation weights. Channel and
global sample normalization are available for ablation but disabled by
default because they are nonlinear and do not restore spatial independence.

The correct probabilistic interpretation is conditionally correlated noise:

```text
epsilon | conditions ~ N(0, Sigma(conditions)), diag(Sigma) approximately 1
```

It is intentionally not an independent `N(0, I)` video tensor.

## Phase 1: raw-flow STC noise only

This is the recommended isolated experiment. It does not instantiate a motion
adapter, load a flow-completion checkpoint, read `refined_*`, or use
`teacher_*`. It reads only `raw_f/raw_b` from `flows_432x240`; forward/backward
confidence and stable BG are computed online.

Run the VideoComposer-STC two-GPU smoke test first:

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet

CUDA_VISIBLE_DEVICES=1,2 \
/home/cilab/ndquan/envs/guided_diff/bin/python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port=29517 \
  examples/brushnet/train_stc_noise_vcm.py \
  --config examples/brushnet/configs/train_vcm_videocomposer_stc_noise_only_raw_smoke_2gpu.json
```

After the 20-step smoke test saves `noise_shaper/`, run the full configuration:

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet

CUDA_VISIBLE_DEVICES=1,2 \
/home/cilab/ndquan/envs/guided_diff/bin/python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port=29517 \
  examples/brushnet/train_stc_noise_vcm.py \
  --config examples/brushnet/configs/train_vcm_videocomposer_stc_noise_only_raw_2gpu.json
```

Checkpoints contain:

```text
checkpoint-XXXXXXX/
  noise_shaper/
  trainer_state.pt
```

The diffusion regression target uses `shaped_noise.detach()`. Gradients still
reach the shaper through both the frozen BrushNet and frozen U-Net input paths.
The aggregate beta-budget loss prevents the trivial `beta -> 0` solution.

TensorBoard:

```bash
/home/cilab/ndquan/envs/guided_diff/bin/python -m tensorboard.main \
  --logdir experiments/vcm_videocomposer_stc_noise_only_raw_2gpu/tensorboard \
  --port 6008 \
  --bind_all
```

The older `train_vcm_stc_noise_only_raw_*.json` configs retain the original
joint nine-channel STC encoder as a controlled architecture ablation. Do not
resume one architecture from a checkpoint produced by the other.

Important new scalars are `noise_beta` (beta averaged over the active warp
region), `noise_effective_beta` (including zero-valued out-of-bounds regions),
`noise_warp`, `flow_prior`, `gate_prior`, `shaped_noise_mean_abs`, and
`shaped_noise_std`.

`noise_shaper_min_lr` is an absolute LR floor. An AMP-overflowed update does
not advance the global step or learning-rate scheduler.

## Optional later phase: joint motion adapter

The earlier `train_vcm_stc_noise_shaper_2gpu.json` remains available for a
separate joint ablation. Do not mix its results with phase 1: it loads the old
motion adapter, refined flow, and trains the flow-residual head.

## Inference

For the isolated noise-only checkpoint:

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet

CUDA_VISIBLE_DEVICES=1 \
STC_NOISE_ONLY_MODE=1 \
MOTION_ADAPTER_PATH="" \
USE_PROPAINTER_FLOW_COMPLETION=0 \
TEMPORAL_GUIDANCE_SCALE=0 \
RAFT_BACKEND=propainter \
STC_NOISE_SHAPER_PATH=/path/to/checkpoint-XXXXXXX/noise_shaper \
STC_NOISE_SHAPER_STRENGTH=1.0 \
/home/cilab/ndquan/envs/guided_diff/bin/python \
  examples/brushnet/test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion_fixedBG_temporal_propainter_v0.py
```

`STC_NOISE_ONLY_MODE=1` rejects a motion-adapter path, disables recurrent flow
completion and temporal-gradient guidance, and uses bundled ProPainter RAFT
without loading `RecurrentFlowCompleteNet`. The old shared-background noise is
also disabled. The only temporal intervention is the initial shaped noise.

`RAFT_BACKEND=propainter` uses the same `raft-things.pth`, 240x432 resize,
`[-1,1]` normalization, and 20 RAFT iterations as the raw training cache.

Noise coherence in the current runner is clip-local: non-overlapping clips
start from different anchor noise. Use overlapping windows with one carried
context frame, or evaluate clips independently, before claiming whole-video
boundary consistency.
