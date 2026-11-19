# Latent warp v1 + CGE -> temporal guidance

New files only; the existing evaluator, configs and README are unchanged.

| File | Purpose |
|---|---|
| `infer_latent_warp_v1_cge.py` | Entry point. Reuses `infer_latent_warp_v1.py` for data, GT RAFT flow cache, clip windows, seeds, ROI handling and metrics |
| `latent_warp_cge_inference.py` | `LatentWarpCGEThenTemporalScheduler`, codec construction, `denoise_clip_cge` |
| `configs/inference_test_oracle_cge.json` | `inference_test_oracle.json` + CGE settings, new output folder |

## Per DDIM step

1. Two-pass latent-warp noise prediction (same as `denoise_clip`).
2. DDIM step with that noise.
3. CGE, only at `cge_steps` ([20, 30]): codec-consistency gradient through the VAE
   decoder, added to the DDIM sample.
4. Latent temporal guidance at the CGE-equivalent clean latent
   `x0 + delta_cge / sqrt(alpha_prev)`; only its own correction is added.

This is the production `cge_then_tg` order (`inference/restoration_engine.py`,
`CGEThenTemporalDDIMScheduler`). The temporal gradient uses each frame's own BG
mask, as in the no-CGE latent-warp evaluator. With `--no-cge` the result is
bit-identical to `denoise_clip` (checked on CPU).

CGE settings match production (`inference/settings.json`): scale 0.001 fixed,
`bg_only` operator (VCM-RS BG branch at QP52 + direct ROI fidelity), train_match,
RandomAccess, intra period 64, NN intra QP offset -5, frame rate 30.
`cge_codec_mode: "dual_region"` encodes ROI QP20 + BG QP52 instead.

## Environments

Run the script with guided_diff. The codec runs in a subprocess with
`codec_python` (`code/codec/envs/vcm`); do not activate it. The codec subprocess
inherits the GPU of the main process.

## Run

```bash
PY=/media/ssd1/ndquan/anaconda3/envs/guided_diff/bin/python
cd /media/ssd1/ndquan/video-inpainting/code/examples/brushnet/proposed/latent_warp_v1

# CPU-only preflight (data, checkpoint, RAFT, codec import)
$PY infer_latent_warp_v1_cge.py --config configs/inference_test_oracle_cge.json --preflight

# One-clip GPU smoke test
job run -g 1 -e /media/ssd1/ndquan/anaconda3/envs/guided_diff -n "latent warp CGE smoke" \
  "$PY $PWD/infer_latent_warp_v1_cge.py --config $PWD/configs/inference_test_oracle_cge.json \
   --sequences test_1/BasketballPass --max-clips 1 \
   --output-dir /media/ssd1/ndquan/video-inpainting/code/experiments/latent_warp_v1_eval/cge_smoke"

# Full evaluation, one GPU (split with --sequences across several jobs if needed)
job run -g 1 -e /media/ssd1/ndquan/anaconda3/envs/guided_diff -n "latent warp CGE eval" \
  "$PY $PWD/infer_latent_warp_v1_cge.py --config $PWD/configs/inference_test_oracle_cge.json"
```

Switches: `--cge/--no-cge`, `--latent-warp/--no-latent-warp`,
`--temporal-guidance/--no-temporal-guidance`, `--resume`, `--max-clips N`.
`run_independent_inference.py` still launches the no-CGE evaluator only.

Outputs as in the original evaluator, plus per-step CGE fields in
`guidance_audit.jsonl` (`cge_applied`, `cge_delta_norm`,
`equivalent_clean_delta_norm`, `cge_scale_effective`) and CGE settings in
`summary.json`. Each clip is audited: CGE must run exactly at `cge_steps`.

## CPU checks (2026-11-17, in `code/codec/smoke_test/`)

- `test_cge_equivalence.py`: CGE off == original `denoise_clip` (max diff 0).
- `test_cge_codec_cpu.py`: real VAE + real VCM-RS codec, 2 BasketballPass frames;
  CGE ran at exactly the configured step with a finite nonzero correction.
