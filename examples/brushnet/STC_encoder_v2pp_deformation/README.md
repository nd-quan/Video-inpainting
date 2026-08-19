# STC-v2++ NoiseDeformationHead

This folder implements the learned deformation-head variant. The primary run
warps and fuses noise over the complete four-channel latent frame. `M_BG` is
used by the frozen STC-v2 condition branch and by the optional `warp_scope=bg`
ablation, but it does not gate noise when `WARP_SCOPE=full`.

The fusion law is exactly:

\[
E_t=\sqrt{\alpha}L_t+\sqrt{1-\alpha}\eta_t.
\]

Trainable components are BrushNet and `NoiseDeformationHead`. Trained STC-v2,
U-Net, IP-Adapter, fusion, VAE, and the text/image encoders are frozen.

Run the CPU/schema/mask preflight only:

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet
RUN_PREFLIGHT=0 bash \
  examples/brushnet/STC_encoder_v2pp_deformation/run_train_noise_deformation.sh \
  --preflight_only --report_to none
```

Recommended one-step CUDA smoke run before official training:

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_PREFLIGHT=1 \
MAX_TRAIN_STEPS=1 \
CHECKPOINTING_STEPS=1 \
OUTPUT_DIR=/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/smoke_stc_v2pp_deformation \
bash examples/brushnet/STC_encoder_v2pp_deformation/run_train_noise_deformation.sh
```

Run a separate two-GPU one-step smoke before the official run:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NUM_PROCESSES=2 \
MAX_TRAIN_STEPS=1 \
CHECKPOINTING_STEPS=1 \
LOGGING_STEPS=1 \
OUTPUT_DIR=/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/smoke_stc_v2pp_deformation_2gpu \
bash examples/brushnet/STC_encoder_v2pp_deformation/run_train_noise_deformation.sh
```

Then launch the official two-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NUM_PROCESSES=2 \
bash examples/brushnet/STC_encoder_v2pp_deformation/run_train_noise_deformation.sh
```

The launcher keeps the effective batch at six clips/update by default. Thus it
uses accumulation `6` on one GPU and `3` on two GPUs, matching the completed
STC-v2 training run. The default scheduler is intentionally `constant`, so LR
stays at `1e-5` and warmup is set to zero.

Every launcher invocation also saves the complete combined terminal stream to:

```text
<OUTPUT_DIR>/terminal_logs/train_YYYYmmdd_HHMMSS.log
```

This includes preflight, both Accelerate ranks, warnings, training progress,
checkpoint messages, and tracebacks. Override `TERMINAL_LOG_FILE` when a fixed
path is preferred. TensorBoard scalar logging remains enabled separately.

Training uses recurrent transport only inside each shuffled clip. Ordered
cross-clip state and exact overlap reuse belong to the later D3 inference phase.

## Sequence-level evaluation

The completed `checkpoint-2000` can be evaluated on the flat legacy test set
with:

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet
CUDA_VISIBLE_DEVICES=1 \
bash examples/brushnet/STC_encoder_v2pp_deformation/run_evaluate_noise_deformation.sh
```

The evaluator loads BrushNet, the frozen STC-v2 adapter, and the learned
`NoiseDeformationHead` from the v2++ checkpoint. The frozen IP-Adapter and
fusion module are loaded from the baseline checkpoint recorded in
`metadata.json`. It sends `NoiseDeformationHead.final_noise` directly through
the pipeline's `latents` argument and explicitly disables the older BG-only
shared-noise mixer.

Unlike D2 training, evaluation retains two detached FP32 caches per absolute
frame ID: normalized lineage noise for recurrence and final fused noise for the
diffusion scheduler. Clips are processed in temporal order with one state owner
per sequence. Overlap noise is reused exactly, and the first clip occurrence
owns each generated output; later overlap occurrences copy that authoritative
image. Therefore the launcher is intentionally single-GPU rather than sharding
clips of one sequence across devices.

The default checkpoint contract is T=8, stride=6, `alpha=0.9`, and full-frame
warping. Raw masks are thresholded with the established convention
`0=degraded BG, 255=HQ ROI`, then inverted internally to `M_BG=1` on BG. Hard
ROI compositing is enabled by default.

Run only the CPU/model/dataset preflight:

```bash
RUN_PREFLIGHT=0 \
bash examples/brushnet/STC_encoder_v2pp_deformation/run_evaluate_noise_deformation.sh \
  --preflight_only
```

For a short end-to-end CUDA smoke test (two ordered clips, one DDIM step):

```bash
CUDA_VISIBLE_DEVICES=1 \
MAX_CLIPS=2 \
NUM_INFERENCE_STEPS=1 \
OVERWRITE=1 \
OUTPUT_DIR=/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/smoke_stc_v2pp_sequence_eval \
bash examples/brushnet/STC_encoder_v2pp_deformation/run_evaluate_noise_deformation.sh
```

The full output contains compatible per-clip directories with `raw`, `final`,
`input`, `gt`, and `mask_roi` PNGs, plus `clip_metrics.json`. Run-level files
include `run_config.json`, `model_contract.json`, `sequence_manifest.json`,
`summary.json`, and a top-level `evaluate_*.log`. Set
`SAVE_NOISE_TENSORS=1` only when per-clip noise tensors are needed, since it
increases disk usage. Re-running the same contract resumes completed clips;
use `OVERWRITE=1` only for an intentional replacement.

### RAFT temporal DDIM guidance

`run_evaluate_zero_offset_grid_sequence_temporal_and_video.sh` adds the
training-free temporal guidance used by the historical fixed-BG temporal
experiment. It estimates adjacent RAFT-Large flow from the degraded input
frames, keeps only forward/backward-visible stable `M_BG` pixels, decodes
predicted x0 during the selected DDIM steps, and applies the stable-BG temporal
gradient to x0. It does not modify the STC-v2++ checkpoint or the zero-grid
noise law.

For meaningful overlap-to-new-frame guidance, its default
`TEMPORAL_SAMPLING_SCOPE=full_clip` samples all T=8 rows of the current window.
Only first-owner/new rows are saved, so the normal sequence-state output policy
is preserved. This costs about 8/6 more diffusion work after the first window,
plus RAFT and VAE-gradient decoding.

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet
CUDA_VISIBLE_DEVICES=1 \
DATASET_ROOT=examples/brushnet/dataset/test_1 \
DATASET_LAYOUT=flat_test \
MAX_FRAMES_PER_SEQUENCE=150 \
OUTPUT_DIR=experiments/eval_stc_v2pp_T8_cosine_0.9/test_1/zero-grid-temporal-raft \
bash examples/brushnet/STC_encoder_v2pp_deformation/run_evaluate_zero_offset_grid_sequence_temporal_and_video.sh
```

The default temporal parameters reproduce the old test:
`scale=1e-4`, DDIM steps `[15, 35)`, every step, VAE decode chunks of one,
loss scale `1024`, and detached previous decoded frame. The terminal log and
each `clip_metrics.json` record temporal loss/update/skip diagnostics. The
wrapper runs `Quan_test/imgToVideo_rgb_stc_eval.py` only after evaluation exits
successfully, writing `videos/<sequence>/final.mp4`.
