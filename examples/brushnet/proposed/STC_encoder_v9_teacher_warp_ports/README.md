# V9: clean-teacher x4 warp into BrushNet residual ports

This is an oracle experiment. Cached ProPainter-RAFT flow estimated on clean
frames is an active conditioning input, not a supervision target. The flow must
come from the same dataset split and absolute frame pair as each input clip.

The model copies the V8-R spatial encoder, relative temporal attention,
and output norm. It contains no V5 lightweight flow head, deformable operator,
or four-channel latent correction. BrushNet first computes its usual residuals.
At each selected port, a learned 64-channel projection of the actual BrushNet
residual is nearest upsampled by 4, warped with clean teacher flow resized
directly from the 512x512 RGB grid, then nearest downsampled. For example,
the 64x64 port uses a 256x256 warp grid; the 32x32 port uses 128x128.
BG, cached teacher validity, and geometric validity gate both directions.
The unwarped current residual is the base stream. TCE temporal features guide
each port's fusion. The initial warp gain is `tanh(1.5) ≈ 0.91`; the
reliability-weighted current feature remains as a fallback.

After cross-clip temporal attention, zero-initialized corrections are added
to BrushNet `down[2]` (64x64/320), `down[5]` (32x32/640), `down[8]`
(16x16/1280), and `mid` (8x8/1280). BrushNet receives the original
`[z_D, M_BG]` five-channel condition. The diffusion U-Net receives the adapted
BrushNet residuals. BrushNet, U-Net, VAE, text/image encoders and IP-Adapter
remain frozen.

Training uses the shared-background-noise diffusion objective only. The new
residual adapters and warp gain are trainable. The transferred V8-R TCE,
teacher flow, and all diffusion backbone weights are frozen. This isolates
the value of warping BrushNet features in the first experiment.

## Launch

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet
bash examples/brushnet/proposed/STC_encoder_v9_teacher_warp_ports/run_train_teacher_warp_ports_t16.sh
```

For a paired no-warp control, use a different `OUTPUT_DIR` and set
`WARP_STRENGTH=0`; retain the same initialization, data, loss, and seed. The
launcher runs the teacher-cache preflight before training. `NUM_PROCESSES=1`
selects a single process. `RESUME_FROM_CHECKPOINT=latest` resumes the exact
optimizer and data schedule in the same output directory.

The checkpoint contains the entire `stc_v9_teacher_warp_ports_model` component
and records the teacher source, x4 geometry, selected ports, and warp strength.
