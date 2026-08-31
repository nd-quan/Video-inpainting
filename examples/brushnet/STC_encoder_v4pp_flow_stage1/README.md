# V4++ flow-head-only Stage-1

This diagnostic stage freezes the complete trained V4++ model except for the
shared bidirectional `flow_head`. It trains on cached clean-video teacher flow
over all valid pixels and does not execute BrushNet or diffusion.

Stage-1B supports a motion-aware objective in addition to the legacy all-pixel
teacher Charbonnier term:

```text
L = L_all + lambda_large L_large + lambda_dir L_direction + lambda_mag L_magnitude
```

`L_all` remains the bidirectional teacher-flow Charbonnier loss over all valid
pixels. The three optional auxiliaries apply only to valid pixels with teacher
motion magnitude at least `--large_motion_threshold`, and, by default, only
inside the degraded BG mask:

- `L_large`: vector Charbonnier on large-motion pixels;
- `L_direction`: `1 - cosine(predicted_flow, teacher_flow)`;
- `L_magnitude`: relative flow-magnitude error.

`--motion_loss_region all` makes these three auxiliary terms global. The
direction term uses `--direction_norm_eps` (default `0.05` feature-grid pixels)
to avoid an unstable cosine derivative when the initially predicted flow is
zero.

Initial and periodic validation logs both all-pixel and large-motion metrics:
predicted / zero EPE, gain over zero, predicted and teacher magnitudes,
magnitude ratio, cosine direction, large-motion coverage, and each loss term.
TensorBoard events are written to `OUTPUT_DIR/tensorboard` by default.

Checkpoint selection is controlled with `--best_metric`. With motion-aware
weights enabled, `auto` selects the composite
`large_motion_epe_pred + best_all_epe_weight * epe_pred`; this prevents a model
from improving only static/small-motion regions. Every checkpoint contains a
compatible `stc_flow_model` that can be visualized or loaded by V4++ without
key conversion.
