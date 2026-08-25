# RGB-STC V4++: BG-focused flow alignment and feature-alignment loss

## Purpose

V4++ targets local temporal consistency in the degraded background without
changing the V8 shared-noise law, DDPM/DDIM scheduler, 2D U-Net, or BrushNet
interface.  It is a strict extension of V4 and must be warm-started from a
trained RGB-STC V2 adapter.

Mask convention throughout this package:

- raw mask: `0=degraded BG`, `255=high-quality ROI`;
- internal mask: `M_BG=1` on degraded BG, `M_BG=0` on high-quality ROI.

## Change 1: BG-focused alignment

V4 predicts adjacent bidirectional flow from raw spatial STC features and
forms a confidence-aware residual alignment before temporal attention.  V4++
hard-gates that residual at feature resolution:

```text
F_aligned[t] = F_raw[t]
               + M_BG[t] * (F_v4_aligned[t] - F_raw[t]).
```

Consequences:

- degraded BG receives the V4 motion-aligned feature;
- high-quality ROI remains exactly on the raw V2 spatial feature path;
- frame zero remains unchanged because it has no previous reference;
- alignment stays clip-local and non-recurrent: frame `t-1` is always a raw
  feature, never an already-warped feature.

Flow and confidence are still predicted over the complete feature grid for
diagnostics and teacher supervision.  Only the residual injected into the
temporal feature is BG-gated.

## Change 2: feature-alignment loss

For L2-normalized raw features and predicted backward flow on frame `t`:

```text
F_prev_to_t = warp(normalize(F[t-1]), predicted_backward[t])
L_feature_backward = robust(F_prev_to_t, stopgrad(normalize(F[t])))
```

The forward direction is symmetric.  The final feature loss is their mean.
Weights combine:

- teacher finite/in-bounds validity;
- the query-frame internal BG mask;
- detached predicted flow confidence with a nonzero floor.

Predicted flow performs the warp.  Teacher flow is not injected into the
feature path and remains training-only supervision.  Warping uses zero padding
without `fallback=current`; otherwise OOB predictions would copy the target
and receive an artificial zero loss.

The training objective is:

```text
L_total = L_diff
        + lambda_flow * L_teacher_flow
        + lambda_feature(step) * L_feature_alignment.
```

`lambda_feature(step)` ramps linearly from zero to the configured value over
`feature_alignment_warmup_steps`.  Defaults are deliberately conservative:

```text
lambda_flow      = 0.01
lambda_feature   = 0.01
feature warm-up  = 500 optimizer steps
confidence floor = 0.1
region           = BG
```

## Gradient contract

Feature loss updates the predicted flow head through `grid_sample` and the raw
spatial STC source feature.  Each target feature is stop-gradient.  Temporal
attention, ZeroConv, and alignment fusion continue to receive the diffusion
objective.  The V8 BrushNet, U-Net, VAE, text/image encoders, IP-Adapter, and
FGBG fusion module remain frozen.

## Checkpoint and inference contract

Inference must load the complete `stc_flow_model`.  Loading the exported
`stc_adapter` alone drops the V4++ flow head and BG-focused alignment idea.
Feature-alignment loss and clean teacher flow are not required at inference.

Resume metadata records the feature loss weight, region, confidence floor,
warm-up, epsilon, and the fixed BG-focused alignment contract.  Changing these
values requires a fresh experiment rather than exact resume.

## Scope and non-goals

V4++ improves adjacent correspondence inside one clip.  It does not maintain
cross-clip state, use a first-frame sequence anchor, recursively warp noise or
features, or replace the 2D diffusion backbone.  Those remain separate future
ablations after local consistency is validated.

