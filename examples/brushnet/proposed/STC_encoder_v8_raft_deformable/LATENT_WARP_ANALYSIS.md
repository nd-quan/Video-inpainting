# Predicted-clean latent correspondence: Branch B/C

## Separate BG-only variant (GPU 5)

`evaluate_v8_latent_warp_bg_only_analysis.py` and
`run_evaluate_v8_latent_warp_bg_only_analysis.sh` reuse the same diagnostic but
composite `where(A, warped_previous, current)`. B uses primary BG support; C
also requires rescaled validity. Thus ROI and invalid BG retain the original
current predicted-clean latent exactly. Sampling is still a dense tensor
operation; this is restricted application, not a sparse or per-tap BG-only
interpolator. It does not inject anything into inference.

Metrics remain on common support, hence agree with full-latent metrics for
identical captures. Additional `direct_roi_changed_elements` and
`rescaled_roi_changed_elements` must be zero. JSON identifies `bg_only_composite`
and the formula; run_config identifies `latent_bg_only`.

```bash
OUTPUT_DIR=experiments/v8_latent_bg_only_smoke_BasketballPass_gpu5 \
bash examples/brushnet/STC_encoder_v8_raft_deformable/run_evaluate_v8_latent_warp_bg_only_analysis.sh \
  --include_branches BasketballPass --max_clips 1
```

To run full test_1 explicitly, use a fresh output directory and omit the two
selection options. This separate launcher fixes GPU 5; the original remains GPU 4.

Branch A is unchanged (`visualize_v5_v6_v8_feature_alignment.py`,
`visualize_student_flow_before_after.py`, `feature_alignment_detail.py`).
Those feature diagnostics use normalized channel cosine; B/C uses the same
normalization epsilon (1e-6), plus target-energy-normalized MSE. No existing
motion bins were found there, so B/C uses [0,4), [4,16), [16,32), [32,infinity)
in original RGB pixel units.

The evaluator reuses standard V8's flat-test/cross-clip loader, frame IDs,
checkpoint loading, exact frozen V7 output and existing geometric warp/resize
utilities. It does not import or call any CGE evaluator. Neither CGE nor
temporal-guidance correction is enabled. No architecture/training code changes.

## Capture contract

An instance-local step wrapper calls the ORIGINAL standard DDIM step once with
`return_dict=True` to access its `pred_original_sample`. This is the scheduler's
predicted-clean latent (including its normal clipping/thresholding, if enabled),
not noisy z_t. The wrapper returns the unchanged `prev_sample` using the original
caller's return format. Captures immediately detach/copy to CPU. The adapter
forward hook similarly copies `output.raft_flow_backward_rgb`; it does not run
another flow estimator. Previous-clip calls are overwritten by the current clip
output. The wrapper is removed after each pipeline call.

Tests establish exact scheduler trajectory equality with/without capture over
50 steps for eta=0 and eta=0.5, including identical RNG consumption. Full GPU
image-level bitwise comparison is not automatically run. Diagnostics do not
consume random values or edit model outputs; all metrics run on CPU afterwards.

## Geometry

- B: original z0[k] -> backward warp at native latent resolution -> compare z0[k+1].
- C: original z0[k] -> nearest upsample x4 -> backward warp -> nearest downsample
  -> compare the SAME z0[k+1]. No recursively warped sources.
- Both resize the SAME RGB backward flow using the existing resize utility,
  including scaling dx by width ratio and dy by height ratio.
- Backward convention: current-frame coordinates sample previous-frame pixels.

Native target BG and direct-warped native source BG (threshold >=0.5) define
one shared semantic support. Primary support additionally requires direct warp
in-bounds. Common support intersects this with nearest-downsampled high-resolution
warp validity. ALL no-warp/B/C metrics use that common support.

There is NO FB confidence, visibility or deform-reliability input to metrics.
FB filtering preferentially removes difficult motion and occlusion, which are
exactly what this experiment diagnoses. High errors under disocclusion do not
automatically indicate a bug. Auxiliary FB analysis is not implemented.

Motion magnitude is computed in original RGB pixels, then nearest sampled at
latent positions for bin assignment. Support ratios use all native latent pixels
as denominator; bin ratios use all pixels belonging to that motion bin. Empty
support produces null metrics, not misleading zero error.

## Commands (repository root)

Quick single-sequence/one-clip smoke, GPU 4 only:

```bash
OUTPUT_DIR=experiments/v8_latent_warp_smoke_BasketballPass_gpu4 \
bash examples/brushnet/STC_encoder_v8_raft_deformable/run_evaluate_v8_latent_warp_analysis.sh \
  --include_branches BasketballPass --max_clips 1
```

Full BasketballPass: omit `--max_clips 1` and use a fresh output directory.

Full allowed evaluation (test_1, NOT validation/train/other test folders):

```bash
OUTPUT_DIR=experiments/v8_latent_warp_full_test1 \
LATENT_CAPTURE_STEPS=0,10,25,40,49 LATENT_WARP_SCALE=4 \
bash examples/brushnet/STC_encoder_v8_raft_deformable/run_evaluate_v8_latent_warp_analysis.sh
```

Add `--preflight_only` to validate selection/config. The script explicitly fixes
CUDA_VISIBLE_DEVICES=4. Default checkpoint is joint-4000; `STC_V8_MODEL` overrides
its component path. Use fresh output directories: inference can skip saved clips,
but diagnostic aggregation does not restore records from an earlier invocation.

## Files and interpretation

Existing normal evaluation images, `run_config.json`, `model_contract.json`,
and normal summary/metrics remain produced by standard V8.

`latent_analysis/` contains:

- `clips/<video>/<start>_<end>.json`: IDs, each capture's actual training timestep,
  per-pair metrics and clip means. Capture keys are INFERENCE indices.
- `pairs.json`: unique (video, source ID, target ID, inference index) records;
  overlapping clips select first occurrence, consistently with earlier V8 reports.
- `sequences.json`: equal-pair means per sequence at each capture index, with bins.
- `summary.json`: global equal-pair means and sequence macro means, separately
  for each capture index/bin; support definitions and selected clip files.
  Updated incrementally; standard evaluation summary signals successful completion.
- `terminal_logs/`: timestamped console log.

Read direct_warp_gain for Q1, rescaled_vs_direct_gain for Q2; positive means
improvement. Compare bins >=16 and >=32 for Q3, step_0 through step_49 for Q4.
Always inspect common_valid_ratio, primary_bg_valid_ratio and support_pixels
for Q5. Without an FB auxiliary analysis, Q6 cannot be answered by this run.
Error metrics and gain values are averaged per pair, not ratios of globally
pooled pixel errors. Sequence macro gives each supported sequence equal weight.
