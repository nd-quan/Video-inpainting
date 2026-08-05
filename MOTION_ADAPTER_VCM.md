# Flow-guided motion injection for VCM video restoration

## Proposed method

For adjacent frames \(I_{t-1}\) and \(I_t\), the refined backward flow
\(\hat F_{t\rightarrow t-1}\) is used because every target pixel in frame
\(t\) must sample a location from frame \(t-1\).

At BrushNet residual level \(l\):

\[
\bar R^l_{t-1\rightarrow t}
=
\mathcal W\left(R^l_{t-1},
\operatorname{ResizeFlow}(\hat F_{t\rightarrow t-1})\right)
\]

Flow resize scales horizontal and vertical displacement independently. The
motion adapter predicts:

\[
\Delta R_t^l =
C_t^l\odot A_\theta^l\left[
R_t^l,\bar R^l_{t-1\rightarrow t},
R_t^l-\bar R^l_{t-1\rightarrow t},
E_F(\hat F_{t\rightarrow t-1}),C_t^l
\right]
\]

and injects it through the existing BrushNet residual port:

\[
\widetilde R_t^l=R_t^l+s_m\Delta R_t^l,\qquad
\widetilde R_0^l=R_0^l .
\]

\(C_t\) is forward-backward flow consistency plus an in-bounds test. It is
defined over the complete frame, so motion is not disabled inside ROI.
`stable_bg` additionally requires aligned background in both frames and is
used only for the temporal loss.

V1 adapts all down-block residuals and the mid-block residual. Up-block
injection is disabled initially because it has the highest feature
resolution and therefore increases VRAM and the risk of texture artifacts.
Every output projection is initialized to zero, so an untrained adapter is
exactly equivalent to the current image-only baseline. With the current
12 down ports, one mid port, 64 bottleneck channels, and 16 flow channels,
V1 has 3,808,717 trainable parameters.

## Training objective

BrushNet, diffusion U-Net, VAE, text encoder, flow completion, and the current
IP-Adapter remain frozen. Only the motion adapter is optimized. A single
diffusion timestep is sampled per video clip, then repeated for its frames.
The trainer loads the same IP-Adapter and foreground/background fusion weights
as the deployed pipeline and appends their four image tokens to every frame's
text tokens.

\[
\mathcal L =
\lambda_\epsilon\|\epsilon-\epsilon_\theta\|_2^2
+\lambda_T\sqrt{\bar\alpha_\tau}
\frac{\sum C_t^{BG}\rho(
\hat z^0_t-\mathcal W(\hat z^0_{t-1},\hat F_{t\rightarrow t-1}))}
{\sum C_t^{BG}+\varepsilon}
+\lambda_R\sum_l\|\Delta R_t^l\|_2^2 .
\]

The default weights are
\(\lambda_\epsilon=1\), \(\lambda_T=0.05\), and
\(\lambda_R=10^{-4}\). The temporal term is computed in predicted-clean
latent space to avoid back-propagating through the VAE decoder. Its
\(\sqrt{\bar\alpha_\tau}\) weight prevents high-noise diffusion timesteps from
dominating after the predicted-clean conversion.

## Implemented files

- `examples/brushnet/diffusers/models/brushnet_motion_adapter.py`: multi-scale
  warp, confidence-gated residual adapters, and CFG-safe clip reshaping.
- `examples/brushnet/diffusers/models/motion_adapter_training.py`: frozen
  BrushNet/U-Net path and temporal loss.
- `examples/brushnet/motion_adapter_dataset.py`: sequential VCM clips,
  refined flows, and confidence maps.
- `examples/brushnet/train_motion_adapter_vcm.py`: AMP training, validation,
  TensorBoard, checkpointing, and resume.
- `examples/brushnet/configs/train_vcm_motion_adapter.json`: first experiment.
- `examples/brushnet/diffusers/pipelines/brushnet/pipeline_brushnet_sharedNoise_sameBG_v0_0.py`:
  inference-time injection.
- `pretrained/ProPainter/scripts/export_vcm_refined_flows.py`: offline export
  from the fine-tuned flow-completion model.

The implementation borrows ProPainter's recurrent flow completion and the
warping/forward-backward consistency principle used by FFF-VDI. It does not
copy FFF-VDI's deformable propagation in V1 because the existing BrushNet
residual ports provide a smaller, safer trainable interface.

## Step 1: export refined flows

Run from the ProPainter directory:

```bash
cd /home/cilab/ndquan/videoInpainting/pretrained/ProPainter

CUDA_VISIBLE_DEVICES=2 \
python scripts/export_vcm_refined_flows.py \
  --dataset-root /home/cilab/ndquan/videoInpainting/SFU_train \
  --flow-root /home/cilab/ndquan/videoInpainting/SFU_train/flows_432x240 \
  --checkpoint /home/cilab/ndquan/NAS_ndq/model_base/Checkpoint/vcm_flowcomp_full_t10_b4/best.pt \
  --output-root /home/cilab/ndquan/videoInpainting/SFU_train/refined_flows_432x240_t10 \
  --device cuda \
  --clip-length 10 \
  --window-stride 5 \
  --resume
```

The selected checkpoint is the T=10, B=4 best model at iteration 3000 with
validation EPE 1.1514. Each pair file contains:

- `refined_f`, `refined_b`: `[2, 240, 432]`;
- `motion_confidence`: `[1, 240, 432]`;
- `stable_bg`: `[1, 240, 432]`.

## Step 2: smoke training

Use the supplied smoke config. It runs 20 steps on sparsely sampled clips.

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet

CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=. \
/home/cilab/ndquan/envs/guided_diff/bin/python \
  examples/brushnet/train_motion_adapter_vcm.py \
  --config examples/brushnet/configs/train_vcm_motion_adapter_smoke.json
```

After it succeeds, use `train_vcm_motion_adapter.json` for the 10,000-step
experiment. It starts conservatively with `clip_length=4`, `batch_size=1`,
and gradient accumulation 4.

Training logs are written to:

```text
experiments/vcm_motion_adapter/logs/train.log
```

TensorBoard:

```bash
/home/cilab/ndquan/envs/guided_diff/bin/python -m tensorboard.main \
  --logdir /home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/vcm_motion_adapter/tensorboard \
  --port 6006 \
  --bind_all
```

Checkpoints are saved every 500 optimizer steps. `latest.json` and
`best.json` point to a directory containing `motion_adapter/` and
`trainer_state.pt`.

Resume:

```bash
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=. \
/home/cilab/ndquan/envs/guided_diff/bin/python \
  examples/brushnet/train_motion_adapter_vcm.py \
  --config examples/brushnet/configs/train_vcm_motion_adapter.json \
  --resume /absolute/path/to/checkpoint-0000500
```

### Two-GPU DDP

The DDP config keeps the same effective batch as the one-GPU configuration:

```text
2 GPUs × 1 clip/GPU × gradient accumulation 2 = 4 clips/optimizer step
```

```bash
CUDA_VISIBLE_DEVICES=1,2 \
torchrun \
  --standalone \
  --nproc_per_node=2 \
  examples/brushnet/train_motion_adapter_vcm.py \
  --config examples/brushnet/configs/train_vcm_motion_adapter_2gpu_smoke.json
```

Physical GPU 1 becomes `cuda:0`/`LOCAL_RANK=0`, and physical GPU 2 becomes
`cuda:1`/`LOCAL_RANK=1`. Only rank 0 writes logs, TensorBoard files, and
checkpoints. Validation metrics are averaged across ranks. After the smoke run,
replace the config with `train_vcm_motion_adapter_2gpu.json` for the full
10,000-step run.

## Step 3: inference

The active temporal test loads the fine-tuned flow checkpoint bundle directly.
Set `MOTION_ADAPTER_PATH` to the saved Diffusers adapter directory:

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet/examples/brushnet

CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=. \
MOTION_ADAPTER_PATH=/absolute/path/to/checkpoint-0010000/motion_adapter \
MOTION_ADAPTER_SCALE=1.0 \
TEMPORAL_CLIP_SIZE=4 \
/home/cilab/ndquan/envs/guided_diff/bin/python \
  test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion_fixedBG_temporal_propainter_v0.py
```

## Required ablations

Use the fixed test split and identical random seeds:

1. current image-only BrushNet/IP-Adapter baseline;
2. shared background noise only;
3. shared noise plus scheduler temporal guidance;
4. shared noise plus motion adapter;
5. full system;
6. raw flow versus refined flow;
7. down only versus down+mid versus down+mid+up.

Report PSNR, SSIM, LPIPS, temporal warping error on stable background, and
runtime/peak VRAM. Select the checkpoint using validation loss on Horses,
never using the three test sequences.
