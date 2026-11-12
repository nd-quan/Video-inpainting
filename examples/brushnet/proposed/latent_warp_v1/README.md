# latent_warp_v1 — run 1 (stable-BG latent warp, oracle GT flow)

All files here are new; no existing script was modified. Env: `/home/cilab/ndquan/envs/guided_diff`.

| Phase | File | Output |
|---|---|---|
| 0 flow cache | `precompute_bg52_teacher_flows_512.py` | `/user/Cilab/Quan/experiment/flows/bvi_dvc_bg52_512/` |
| 1 14-ch init | `prepare_brushnet_14ch_init.py` | `/user/Cilab/Quan/experiment/latent_warp_run1/init_T8_ckpt2250_14ch/` (**done**, see `report.json`) |
| 2+3 trainer | `train_brushnet_latent_warp_v1.py` (generated from `training/..._sharedNoise_sameBG_v0_0.py`) | — |
| shared helpers | `latent_warp_utils.py` | — |
| launcher | `run_train_latent_warp_v1_run1.sh` | `/user/Cilab/Quan/experiment/latent_warp_run1/train/` |
| tests | `test_latent_warp_utils.py`, `dryrun_cpu_one_step.sh`, `check_flow_cache_warp.py` | `/user/Cilab/Quan/experiment/latent_warp_run1/_cpu_test/` |

## Method (per training step, shared timestep t per clip)
1. Normal V8 inputs: `z_t`, decoded-input latent, BG mask (BG = 1), shared BG noise.
2. Extra **no-grad** BrushNet + UNet pass with the 5 warp channels = 0 → `eps_hat` →
   `x0 = (z_t − sqrt(1−ᾱ_t)·eps_hat) / sqrt(ᾱ_t)` for every frame.
3. Frame k > 0: `stable_k = BG_k · [warp(BG_{k−1}, teacher_b) > 0.5] · in_bounds · valid_b` (512×512).
4. `x0_{k−1}` → nearest ×8 → `backward_warp(·, teacher_b)` → `avg_pool8(warped·stable) / avg_pool8(stable)`;
   validity = `avg_pool8(stable)`. Frame 0 → zeros.
5. BrushNet input `[z_t 4 | input latent 4 | mask 1 | warped x0 4 | validity 1]` = 14 ch; loss unchanged.

`--latent_warp_drop_prob 1.0` turns the same trainer into the run-0 baseline (same data, steps, noise).

## Verified (CPU only, 2026-11-11)
- `test_latent_warp_utils.py`: 8/8 pass (zero flow = identity, 1-latent-px shift, half-block shift is
  unbiased, ROI/FB-invalid excluded, partial blocks normalised, drop/frame-0 zeros, flow rescale, x0 inversion).
- Phase 1 `report.json`: copied weights max diff 0.0; new slices 0.0; BrushNet residuals identical
  (max diff 0.0 at t = 999/500/50) with zero **and** random warp channels.
- `dryrun_cpu_one_step.sh`: trainer runs end to end; step-0 loss warp ON = warp OFF = 0.05924587324261665
  (bit-identical); first batch stable_frac_of_bg 0.996, |x0| mean 0.62 / max 3.97.
- `check_flow_cache_warp.py` (2 test sequences): moving sequence (0.6–0.9 px/frame) pixel PSNR in stable BG
  +1.0…+4.3 dB after warping → flow direction/units correct. Sub-pixel motion gives no measurable gain at
  latent scale (±0.8 dB), as expected for motion ≪ 8 px. Re-run on random branches once the full cache exists.
- Flow speed: CPU 8 threads ≈ 2.4 s/pair (13,104 pairs ≈ 9 h; 2 shards ≈ 4.5 h). Cache ≈ 1.9 MB/pair ≈ 25 GB.

## Order of use
```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet/examples/brushnet/proposed/latent_warp_v1
PY=/home/cilab/ndquan/envs/guided_diff/bin/python

# Phase 0 (only on a FREE GPU; refuses to start on a GPU used by anyone else)
$PY precompute_bg52_teacher_flows_512.py --device auto            # or --device cuda:N
#   --wait-for-free-gpu 600  : wait until the GPU is free instead of exiting
#   --resume                 : continue an interrupted run
#   2 free GPUs: add --num-shards 2 --shard-index {0,1} (one process each)
# CPU alternative (no GPU at all, lowest priority):
#   CUDA_VISIBLE_DEVICES= nice -n 19 $PY precompute_bg52_teacher_flows_512.py --device cpu --cpu-threads 8 \
#       --num-shards 2 --shard-index 0   (and a second process with --shard-index 1)
# afterwards: $PY check_flow_cache_warp.py --flow-root /user/Cilab/Quan/experiment/flows/bvi_dvc_bg52_512

# Phase 1 (already done; CPU only)
$PY prepare_brushnet_14ch_init.py

# Run 1 (only after checking; refuses busy GPUs)
PRINT_ONLY=1 CUDA_VISIBLE_DEVICES=0 bash run_train_latent_warp_v1_run1.sh   # show command only
CUDA_VISIBLE_DEVICES=0 bash run_train_latent_warp_v1_run1.sh
```
