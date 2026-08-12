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
