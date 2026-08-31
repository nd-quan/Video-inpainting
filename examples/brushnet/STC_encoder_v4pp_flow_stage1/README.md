# V4++ flow-head-only Stage-1

This diagnostic stage freezes the complete trained V4++ model except for the
shared bidirectional `flow_head`. It trains on cached clean-video teacher flow
over all valid pixels and does not execute BrushNet or diffusion.

Default objective:

```text
L_stage1 = L_teacher_Charbonnier
```

The initial and periodic validation reports include predicted EPE, zero-flow
EPE, gain over zero flow, and predicted magnitude. A full compatible
`stc_flow_model` is saved in every checkpoint so that it can be visualized or
loaded back into V4++ without key conversion.
