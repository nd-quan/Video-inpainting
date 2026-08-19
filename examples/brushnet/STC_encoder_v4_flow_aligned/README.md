# RGB-STC v4: clip-local flow-aligned features

V4 warm-starts the completed RGB-STC v2 checkpoint trained with `T=8`, then
fine-tunes the full STC on `T=16`. The old checkpoint can execute 16 tokens
because temporal attention has no fixed-length positional embedding, but it
must not be treated as a trained T16 model: the attention distribution changes
when its temporal token count doubles.

## Model path

```text
[degraded RGB_t, M_BG_t]
          |
 per-frame spatial features s_t
          |--------------------------> teacher-supervised bidirectional flow
          |
 W(s_(t-1), backward_flow_t) -- confidence -- zero-init residual fusion
          |
 aligned features a_1...a_T
          |
 VideoComposer temporal attention over the current T=16 clip
          |
 ZeroConv delta_z_BG -> frozen V8 BrushNet/2D U-Net
```

The flow is defined in latent pixels. `backward_flow[t]` lives on current-frame
coordinates and samples the previous frame. Forward/backward consistency and
in-bounds validity produce a detached confidence map for alignment.

Important constraints:

- Features are aligned only between adjacent frames inside one clip.
- Frame `t` samples the raw feature of `t-1`; an already aligned/warped feature
  is never warped again.
- There is no cross-clip state, first-frame sequence anchor, or noise warping.
- Shared BG noise, the 2D diffusion backbone, and `L_diff` remain identical to
  V3/V2. The objective is `L_diff + lambda_flow * L_flow`.
- Default flow supervision is `all`, because the aligned representation is
  consumed over the full spatial feature map. The emitted latent correction is
  still gated by internal `M_BG=1`.
- Flow is now an inference dependency. Load `stc_flow_model`, not the smaller
  diagnostic `stc_adapter` folder.

The alignment residual projection and flow head both start at exact zero. With
the V2 weights loaded, the initial V4 condition is exactly the V2 condition
evaluated at T16. This protects the frozen BrushNet contract while fine-tuning.

## Preflight and smoke test

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet

RUN_PREFLIGHT=1 \
NUM_PROCESSES=2 \
CUDA_VISIBLE_DEVICES=1,3 \
MAX_TRAIN_STEPS=1 \
CHECKPOINTING_STEPS=1 \
OUTPUT_DIR=/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/smoke_stc_v4_flow_aligned_T16 \
bash examples/brushnet/STC_encoder_v4_flow_aligned/run_train_flow_aligned_stc_t16.sh
```

The default official run uses T16/S12, two GPUs, one clip per GPU, gradient
accumulation 3, effective batch 6 clips/update, LR `1e-5`, shared-noise strength
`0.9`, and `flow_region=all`.

Do not resume a V3 checkpoint into V4. To continue V4, point
`--resume_from_checkpoint` at a V4 checkpoint and omit `INIT_STC_ADAPTER`; the
provided launcher currently targets a fresh warm-start run by default.
