# Full-reference video metrics (portable)

Computes **PSNR, SSIM, LPIPS, WE, WE_L2, FS, VFID_Clips** for generated PNG sequences against GT PNGs.
It produces the same `metrics.csv` as the original BrushNet chain
(`Quan_test/evaluate_flat_sequence_metrics.py` → `DGAF_VSR_original/postprocess_eval.py`).
This folder is self-contained and needs no other repository.

```
full_reference_metrics/
  evaluate.py                    # the only entry point (all metric code inside)
  pytorch_i3d.py                 # I3D model (piergiaj/pytorch-i3d, Apache 2.0)
  LICENSE_pytorch_i3d.txt
  requirements.txt
  weights/rgb_imagenet.pt        # I3D weights for VFID (required)
  weights/torch_home/hub/checkpoints/alexnet-owt-7be5be79.pth   # LPIPS backbone (lets it run offline)
```

## Install

```bash
# Install torch + torchvision matching your CUDA first, then:
pip install -r requirements.txt
```

## Run

```bash
python evaluate.py --pred-root /path/to/generated/test --gt-root /path/to/dataset/test
```

Results are written to `<pred-root>/metrics_full_reference/` by default, or to `--output-dir`.

| Option | Meaning |
|---|---|
| `--layout auto\|hierarchical\|flat` | `auto` selects hierarchical if `pred-root` contains `Class_*` folders |
| `--output-dir DIR` | Output folder. Fails if it already exists unless `--resume` is passed |
| `--resume` | Reuse completed sequences in `--output-dir` (per-sequence JSON + I3D features) |
| `--sequences A B ...` | Evaluate only these sequences |
| `--allow-partial` | Accept predictions that cover only part of the GT frames |
| `--require-native-size` | Fail if GT size ≠ prediction size instead of resizing GT |
| `--skip-warping` | Skip WE / WE_L2 (the slowest metrics, Farneback on CPU) |
| `--device cuda\|cpu` | Default: `cuda` if available. With `cuda`, choose the GPU via `CUDA_VISIBLE_DEVICES` |
| `--weights-dir DIR` | Default: `./weights` |

## Input layout

PNG file names must be numeric frame IDs (e.g. `000471.png`). Prediction frames must be continuous, with at least 10 frames.

```
hierarchical                              flat
pred-root/Class_B/BQTerrace/000471.png    pred-root/BQTerrace/000471.png
gt-root/GT/Class_B/BQTerrace/000471.png   gt-root/BQTerrace/gt/000471.png
  (or gt-root/Class_B/BQTerrace/...)        (or gt-root/BQTerrace/...)
```

The GT folder used for each sequence is recorded in `source.json`.

## Output

- `metrics.csv` / `metrics.json`: one row per sequence plus an `ALL` row
- `<seq>.json`: per-sequence row
- `<seq>_gt_i3d.npy`, `<seq>_final_i3d.npy`: I3D clip features (used by `--resume` and the pooled VFID)
- `source.json`: inputs, GT folders, frame ranges and library versions

## Metric definitions

All metrics are computed at **prediction resolution**. When the sizes differ, GT is converted to RGB and resized with PIL bilinear.

| Metric | Definition | Better |
|---|---|---|
| PSNR | RGB uint8, MSE over all pixels, mean over frames | ↑ |
| SSIM | `torchmetrics` SSIM, `data_range=1`, mean over frames | ↑ |
| LPIPS | `lpips` v0.1, AlexNet, inputs in [-1, 1], mean over frames | ↓ |
| WE | Prediction only: Farneback backward flow (frame t → t-1), warp frame t-1 onto t, L1 over valid pixels in [0, 1], mean over pairs | ↓ |
| WE_L2 | Same as WE, but squared L2 | ↓ |
| FS | Prediction only: SSIM between adjacent frames (11×11 Gaussian, σ = 1.5, per-channel mean), mean over pairs | ↑ |
| VFID_Clips | I3D `rgb_imagenet` Logits-endpoint features of 10-frame sliding windows (stride 1, frames resized to 224×224 bicubic); exact empirical Fréchet distance between GT and prediction feature sets | ↓ |

How the `ALL` row is aggregated:
- PSNR, SSIM and LPIPS are weighted by frame count.
- WE, WE_L2 and FS are weighted by pair count.
- VFID_Clips is recomputed on the pooled I3D features of all sequences. It is **not** the mean of the per-sequence values.

## Reproducibility

The script was validated against
`experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_latent_tg0p1_v7student_512/test/metrics_full_reference/metrics.csv`
(produced by the original chain), using the reference environment listed in `requirements.txt`.
Other versions of torch, CUDA or OpenCV can change the last decimal places, mainly for LPIPS, VFID and WE.
To compare methods, evaluate all of them in the same environment.
