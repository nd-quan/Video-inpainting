# Fixed-beta Test 1

## Purpose

This experiment measures the effect of flow-shaped initial diffusion noise
without training Stage 2 and without injecting a Stage-3 condition adapter.

```text
degraded clip + ROI mask
        |
        v
frozen Stage-1 STC full-flow predictor
        |
        v
fixed-beta anchor-noise warping over the whole frame
        |
        v
shaped initial latent z_T
        |
        v
frozen SD1.5 U-Net + frozen V8 BrushNet/IP-Adapter/FGBG fusion
        |
        v
generated background + decoded-input ROI composite
```

Only the initial noise is shaped. The noise is not warped again during DDIM
denoising. Final post-processing uses the same asymmetric Gaussian soft seam as
the legacy null-text evaluator: BG remains fully generated, the ROI interior
comes from the decoded input, and only the inner ROI boundary is blended.

## Checkpoints

- Stage 1 pointer:
  `/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/vcm_stc_flow_stage1/checkpoints/best.json`
- V8 deployment checkpoint:
  `/home/cilab/ndquan/NAS_ndq/model_base/Checkpoint/fine-tuning-v8/checkpoint-2000`

The evaluator resolves `best.json` once at process startup, records its step and
validation EPE, and then locks that concrete checkpoint for the complete run.
If an existing beta directory was created with another Stage-1 checkpoint or
generation setting, the evaluator refuses to mix results and requests a new
`--output-dir`.

The V8 directory supplies BrushNet, gated IP-Adapter and FGBG fusion weights.
It does not contain a fine-tuned U-Net: V8 kept the base SD1.5 U-Net frozen.

## Fixed-beta equation

For frame `t`, the predicted backward flow warps the anchor noise `epsilon_0`:

```text
epsilon_warp,t = Warp(epsilon_0, phi_t)
epsilon_t = beta * epsilon_warp,t + sqrt(1 - beta^2) * epsilon_ind,t
```

The implementation also applies the warp-validity map, so invalid sampling
locations fall back to independent Gaussian noise. The default is `beta=0.5`.
No beta head and no optimizer are created.

## Preflight

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet

/home/cilab/ndquan/envs/guided_diff/bin/python \
  examples/brushnet/evaluate_stc_fixed_beta_test1_vcm.py \
  --config examples/brushnet/configs/eval_vcm_stc_fixed_beta_test1.json \
  --preflight-only
```

## One-clip smoke test

```bash
CUDA_VISIBLE_DEVICES=1 \
/home/cilab/ndquan/envs/guided_diff/bin/python \
  examples/brushnet/evaluate_stc_fixed_beta_test1_vcm.py \
  --config examples/brushnet/configs/eval_vcm_stc_fixed_beta_test1_smoke.json
```

## Full Test 1

```bash
CUDA_VISIBLE_DEVICES=1 \
/home/cilab/ndquan/envs/guided_diff/bin/python \
  examples/brushnet/evaluate_stc_fixed_beta_test1_vcm.py \
  --config examples/brushnet/configs/eval_vcm_stc_fixed_beta_test1.json
```

To run another fixed-beta value without editing the checkpoint or JSON:

```bash
CUDA_VISIBLE_DEVICES=1 \
/home/cilab/ndquan/envs/guided_diff/bin/python \
  examples/brushnet/evaluate_stc_fixed_beta_test1_vcm.py \
  --config examples/brushnet/configs/eval_vcm_stc_fixed_beta_test1.json \
  --fixed-beta 0.9
```

Each beta value gets a separate output directory. Clip seeds do not depend on
beta, which makes `0.5`, `0.7` and `0.9` comparisons use identical independent
Gaussian noise.

## Outputs

The experiment writes:

- `test1.log`: progress and per-clip metrics;
- `resolved_config.json`: the exact executed configuration;
- `preflight.json`: resolved checkpoints and dataset coverage;
- one `clip_metrics.json` per clip, including noise-shaper diagnostics;
- lossless output PNG frames under each class/sequence/clip directory;
- `final.mp4` for every evaluated clip; the smoke config also saves raw PNGs
  and `raw.mp4` before ROI compositing;
- `summary.json`: mean PSNR, BG-PSNR and temporal-delta error.

The current test data contains 1,466 aligned frames. `Class_C/BasketballDrill`
currently has no test segment, so it is explicitly reported and skipped unless
`require_all_sequences` is enabled.
