# Latent warp v1

Training and oracle inference for the 14-input BrushNet checkpoint. Masks use
BG=1 internally. CGE is disabled until codec integration is available.

## Runtime files

| File | Purpose |
|---|---|
| `train_brushnet_latent_warp_v1.py` | Training |
| `run_train_latent_warp_v1_run1.sh` | Training launcher |
| `training_config.py` | YAML/environment/CLI training settings |
| `precompute_bg52_teacher_flows_512.py` | Create GT teacher flows for training |
| `prepare_brushnet_14ch_init.py` | Prepare the 14-input initialization checkpoint |
| `visible_gpu_uuids.py` | Map visible training GPUs |
| `latent_warp_utils.py` | Shared x0 estimation and stable-BG warp math |
| `infer_latent_warp_v1.py` | Inference, flow cache, resume and image metrics |
| `latent_warp_inference.py` | Model loading and two-pass denoising |
| `original_raft_flow.py` | Load original ProPainter RAFT teacher |
| `run_independent_inference.py` | Separate one-GPU jobs, merge, videos, optional ablation |
| `inference_jobs.py` | Shared shard planning, merging and scheduler status |
| `export_inference_videos.py` | Verified MP4 export; can wait for a running evaluation |
| `run_full_reference_metrics.py` | Supplied full-reference metric evaluator adapter |

## Configuration

- `configs/run1.yaml`: training with latent warp.
- `configs/run0.yaml`: inherited training baseline.
- `configs/inference_test_oracle.json`: latent warp + latent temporal guidance.
- `configs/inference_test_oracle_no_temporal.json`: latent warp with temporal guidance off.
- `configs/inference_test_oracle_no_warp_temporal.json`: shared noise + latent temporal guidance, latent warp off.

Both inference configs use `examples/brushnet/dataset/test`, checkpoint 6750,
512-square working images, clip length 6, stride 4, seed 1234 and 50 DDIM steps.
The teacher is original ProPainter RAFT with
`video-inpainting/flow-training/flows/inference/raft-things.pth`, matching the
training checkpoint hash. It uses GT RGB, 20 iterations and [-1,1] inputs.
Cached flow is used for both latent warping and temporal guidance; GT pixels
are never used as diffusion conditioning. This is oracle evaluation.

Temporal guidance defaults: latent space, inference indices [0,50), every step,
scale 0.1. Independent switches are `latent_warp_enabled` and
`temporal_guidance_enabled`. CLI overrides:

```text
--latent-warp / --no-latent-warp
--temporal-guidance / --no-temporal-guidance
```

Turning warp off supplies zero warp channels to the same 14-input model.
Turning temporal guidance off disables its gradient updates. A zero temporal
scale also disables guidance. Older configs without the enabled flag remain valid.

## Run

From this directory:

```bash
PY=/media/ssd1/ndquan/anaconda3/envs/guided_diff/bin/python

# Training input-path check, without launching training.
CHECK_PATHS_ONLY=1 bash run_train_latent_warp_v1_run1.sh --config configs/run1.yaml

# Training under the scheduler; choose the requested GPU count.
job run -g 3 -e /media/ssd1/ndquan/anaconda3/envs/guided_diff \
  -n "latent warp training" \
  "bash $PWD/run_train_latent_warp_v1_run1.sh --config $PWD/configs/run1.yaml"

# CPU-only inference preflight.
$PY infer_latent_warp_v1.py --config configs/inference_test_oracle.json --preflight

# Submit independent one-GPU inference jobs through a CPU coordinator.
# The optional second config starts after the first evaluation/videos succeed.
job run --cpu -e /media/ssd1/ndquan/anaconda3/envs/guided_diff \
  -n "latent warp inference and temporal ablation" \
  "$PY $PWD/run_independent_inference.py --config $PWD/configs/inference_test_oracle.json --workers 4 --then-config $PWD/configs/inference_test_oracle_no_temporal.json"
```

Use a new output directory for a new experiment. For interrupted evaluations,
use the saved resolved config and add `--resume` to the coordinator. It preserves
valid PNGs, original clip ownership, context and seeds. Complete shards are
reused. Saved settings must match. Each scheduler job reserves only one GPU
and releases it independently; there is no shared four-GPU allocation.

Inference generates/reuses GT flow automatically. `--prepare-flows-only` on
`infer_latent_warp_v1.py` generates flow without diffusion inference.

## Results and postprocessing

Each evaluation root contains `run_status.json`, `independent_jobs.json`,
`shard_plan.json`, `summary.json` and `frame_metrics.json`.
PNG predictions are under `shards/shard-XX/test_1/<sequence>/` or `test_2/`.
At overlapping windows, the first window owns each output frame. Native-size
outputs preserve the decoded input's ROI exactly when `preserve_roi` is true.

Video export is automatic after all shards complete. It reuses
`Quan_test/encode_v8r_original_eval_videos.py`, verifies count/size/FPS and writes
`videos/test_1/*.mp4`, `videos/test_2/*.mp4` and `videos/video_manifest.json`.
MP4 is MPEG-4 quality 2, yuv420p. PNGs and image metrics are unchanged.
For completed or running results, use `export_inference_videos.py --eval-root DIR`
(add `--wait --task-id ID` to wait for inference).

Full-reference metrics use the supplied script unchanged:
`Quan_test/metric/full_reference_metrics/evaluate.py`. The adapter creates
non-destructive links to original PNGs/GT, requires complete sequences and
matching native dimensions, and reuses cached per-sequence metrics/features.

```bash
job run --cpu -e /media/ssd1/ndquan/anaconda3/envs/guided_diff \
  -n "Full-reference metrics after inference" \
  "$PY $PWD/run_full_reference_metrics.py --after-task CONTROLLER_ID --eval-roots /path/to/eval-on /path/to/eval-off"
```

It submits independent one-GPU metric jobs after all requested roots complete.
Results are in `metrics_full_reference/metrics.csv`, `metrics.json`, `source.json`.
Metrics: PSNR, SSIM, LPIPS, WE, WE_L2, FS, VFID_Clips. ALL uses frame weighting
for image metrics, pair weighting for temporal metrics and pooled-feature VFID.
The supplied evaluator's WE uses Farneback on predictions; definitions are in
its README. Evaluation uses PNGs, not lossy MP4s.

## No-warp ablation

`inference_test_oracle_no_warp_temporal.json` retains checkpoint 6750, shared
noise .95, seed 1234, GT flow and latent temporal guidance .1 at all 50 steps,
with CGE off. Only `latent_warp_enabled` and the output directory differ from
the combined reference run. The same 14-input model receives zero warp/validity
channels; its zero-warp prepass is skipped.

Controller 736 runs two independent one-GPU jobs (737, 738), then exports videos.
CPU controller 739 waits to submit the full-reference metric job. Results are
under `checkpoint-6750_oracle_raft_things_no_warp_shared_noise_latent_tg0p1_2gpu`.
The previous temporal ON/OFF comparison is saved under
`experiments/latent_warp_v1_eval/comparison_temporal_on_vs_off/`.
