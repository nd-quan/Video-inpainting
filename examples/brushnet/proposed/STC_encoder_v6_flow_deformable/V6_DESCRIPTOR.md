# RGB-STC V6: flow-guided modulated deformable alignment

## Scope

V6 extends V5 at one location only: after V5's spatial/BG flow alignment and
before relative temporal self-attention. It does not deform diffusion noise,
VAE latents, `delta_z_BG`, BrushNet features, or cross-clip memory.

```text
[RGB degraded, M_BG]
        -> spatial encoder: F_1 ... F_T
        -> bidirectional predicted flow
        -> first-order modulated DCN: F_(t-1) -> F_t <- F_(t+1)
        -> reliable BG-to-BG zero-init residual fusion
        -> V5 relative temporal attention
        -> V5 exact-overlap cross-clip memory
        -> zero_conv -> delta_z_BG
```

`M_BG=1` is the strongly degraded region to restore; `M_BG=0` is the
high-quality ROI to preserve.

## Deformable sampling

For target coordinate `q` and kernel point `k`:

```text
p_tk(q) = q + p_k + phi_(t<-s)(q) + delta_p_tk(q)
D_(s->t)(q) = sum_k m_tk(q) W_k F_s(p_tk(q))
```

- `phi` is predicted flow, in feature-grid pixels.
- `delta_p = r_max * tanh(raw_offset)` is a bounded residual.
- `m = sigmoid(mask_logit)` is the modulation mask.
- The default is `K=3`, `deform_groups=4`, `r_max=2` feature pixels.
- STC flow is `[dx,dy]`; torchvision DCN receives interleaved `[dy,dx]`.
- The same alignment module is shared for previous-to-current and
  next-to-current directions.

Each raw neighboring feature is sampled exactly once. An aligned feature is
never reused as the source for the next frame, so V6 is first-order and
non-recursive.

## Reliability and masking

The contribution from source `s` to target `t` is weighted by:

```text
R_(s->t) = M_BG_t * warp(M_BG_s, phi) * valid_in_bounds * C_FB
```

The source feature itself is not masked before DCN. The mask gates its final
contribution, preventing ROI-to-BG leakage while retaining context near the
boundary. Invalid coordinates fall back to the V5 path.

## Warm-start identity

- residual offset logits start at zero;
- modulation masks start at `0.5`;
- only the center DCN weight is nonzero and equals group-wise identity times
  two, so its candidate initially equals one base-flow warp;
- the final deformable residual projection is zero-initialized.

Therefore a newly upgraded V6 is exactly equal to V5, including when overlap
memory is present. The auxiliary deformable loss directly trains the raw DCN
candidate while the zero-initialized fusion begins learning from diffusion.

## Objective

```text
L_total = L_diff
        + lambda_flow * L_flow
        + lambda_feature * L_feature
        + lambda_deform * L_deform
        + lambda_offset * L_offset
```

`L_deform` compares normalized bidirectional DCN candidates with stop-gradient
raw target features using Charbonnier error on teacher-valid, reliable
BG-to-BG pixels. Teacher flow is used for training validity/supervision only;
inference uses predicted flow. `L_offset` regularizes the actual bounded
residual offsets in feature pixels.

Initial loss settings:

```text
lambda_flow    = 0.05
lambda_feature = 0.05
lambda_deform  = 0.05, linear warmup 500 updates
lambda_offset  = 0.001
```

## Training stages

Training stages are separate runs so DDP and optimizer schemas never change
inside a run.

### Stage A: `deform_only`

- initialize from a selected full V5 `stc_v5_model`;
- freeze V5, the spatial encoder, and repaired flow head;
- train DCN weights, offset/mask head, and deformable residual fusion;
- recommended start: LR `2e-5`, 1000 updates.

### Stage B: `joint`

- initialize from the selected full Stage-A `stc_v6_model`;
- keep spatial encoder and flow head frozen;
- train V6 plus V5 alignment fusion, temporal blocks, output norm, and
  `zero_conv`;
- recommended start: LR `5e-6`, 4000 updates.

Exact resume is supported only within the same stage and configuration.

## Required diagnostics

- deform forward/backward loss and valid ratios;
- residual offset mean, p95, maximum;
- modulation-mask mean, standard deviation, saturation ratio;
- DCN-minus-base-warp difference;
- flow EPE/magnitude and feature-alignment loss;
- V5 relative-bias and cross-clip-gate magnitude;
- diffusion loss and `delta_z_BG` magnitude.

Evaluation should compare V5, V6 with `deformable_alignment_scale=0`, and V6
with scale `1`, using identical seeds. Scale zero is an exact V5-path ablation;
it does not disable V5 flow alignment, relative bias, or cross-clip memory.

