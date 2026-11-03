# BG52: CGE then temporal guidance on the equivalent clean latent

This experiment uses new files and a separate output directory. It does not
modify the original CGE scheduler, parallel combined scheduler, or baseline evaluator.

For deterministic DDIM (`eta=0`), let `a=sqrt(alpha_previous)` and let
`delta_C` be the actual direct CGE correction returned by `cond_fn`:

```text
prev_C = prev_DDIM + delta_C
x0_C = predicted_x0 + delta_C / a
x0_CT = x0_C - temporal_scale * normalized_masked_temporal_gradient(x0_C)
prev_CT = prev_C + a * (x0_CT - x0_C)
```

The temporal gradient is evaluated on a fresh detached leaf at `x0_C`.
No second UNet call is made. The CGE correction is added exactly once.
The frozen V7 flow and stable background conditions are identical to baseline.

## Settings

The launcher reads the original BG52 final `run_config.json` directly.
It retains checkpoint-2250, flow checkpoint-0004750, BG52/test, 512x512,
clip size/stride 4/4, 50 DDIM steps, temporal scale 0.1, shared-noise strength
1.0, variance preservation, detach-previous, pair-intersection masks, and seeds
1234/6789. Both guidance windows are `[25,40)`: zero-based step indices 25..39.

Added CGE settings are scale 0.0001, fixed schedule, bg_only, BG QP52,
train_match profile, RandomAccess, intra period 64, NN intra QP offset -5,
30 fps, and one codec subprocess at a time per GPU worker.

## Run

From the BrushNet root:

```bash
bash Quan_test/run_bg52_cge_then_temporal_4gpu.sh --gpus 0 1 2 3
```

Use `--preflight-only` to validate the dataset without loading GPU models.
Use `--output /absolute/new/path` for a separate experiment directory.
The launcher checks that the selected four GPUs are idle before inference.
It assigns complete sequences to workers and resumes complete existing clips.
It rejects an output directory containing a different experiment config.

Default output:

```text
experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_cge_then_tg_equivclean_C25-40_T25-40_cge1em4_tg0p1_v7student_512
```

`status.json` records worker PIDs, exit codes, and PNG counts. Each sequence has
`source_metadata.json` and `guidance_trace.jsonl`. Each completed clip records
all 50 steps, CGE corrections, equivalent-clean shifts, temporal application,
and temporal loss. An audit rejects CGE/TGE calls outside their configured windows.
The window planner shifts the final window to cover the tail with a full
four-frame clip when possible; overlapping frames are saved only once.
With detach-previous enabled, the first frame of each clip has no temporal
gradient and the other three frames receive temporal corrections.

After verifying all 14 sequences / 1198 PNGs at 512x512, the launcher computes
the same full-reference metrics as the baseline under `test/metrics_full_reference`.
The real VCM-RS round trips make this substantially slower than temporal-only inference.

## Validation

```bash
PYTHONPATH=examples/brushnet /home/cilab/ndquan/envs/guided_diff/bin/python \
  -m unittest discover -s examples/brushnet/tests \
  -p test_cge_then_temporal_scheduler.py -v
```

The five tests check the new temporal evaluation point and no double-counted
CGE update, equivalence with CGE-only and temporal-only transitions, exact window
boundaries/no stale correction, and callback restoration on failure.
