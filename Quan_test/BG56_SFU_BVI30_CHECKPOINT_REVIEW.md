# BG56 SFU+BVI30 curriculum: checkpoint review and evaluation plan

Inspected run: `/user/Cilab/Quan/experiment/finetune_sfu_bvi30_bg56_512_from_coco8000_T8S4_ind5000_shared3000_ddp4`.

Training finished at step 8000. The first 5000 updates used independent noise (shared BG noise strength 0), and updates 5001–8000 used variance-preserving shared BG noise with strength 0.95. All 16 checkpoints (500–8000, interval 500) contain the required BrushNet, IP-Adapter and fusion files. The safetensors headers for candidates 6000 and 8000 are readable on CPU.

## Checkpoint evidence

The production TensorBoard event files contain only `loss` and `lr`, with 8000 recorded steps. There are no validation metrics or validation images. The smoke-test report verifies execution/resume, not image quality. Consequently, no checkpoint can currently be called validation-best.

The following means are calculated from the logged training loss over the updates immediately preceding each checkpoint, within the shared-noise phase:

| Checkpoint | Shared-noise updates completed | Last 100 updates: mean loss | Last 500 updates: mean loss |
|---|---:|---:|---:|
| 5500 | 500 | 0.117950 | 0.102445 |
| 6000 | 1000 | 0.085107 | 0.090863 |
| 6500 | 1500 | 0.140640 | 0.106937 |
| 7000 | 2000 | 0.099072 | 0.107271 |
| 7500 | 2500 | 0.108198 | 0.092053 |
| 8000 | 3000 | 0.103916 | 0.106952 |

Checkpoint 6000 is the proposed first evaluation candidate because it has the lowest trailing 100- and 500-update losses among the shared-noise checkpoints. This is a training-loss proxy, not evidence of superior PSNR, LPIPS, temporal consistency or video quality. The logged loss is the trainer's minibatch diffusion MSE and depends on sampled data, noise and timesteps. Checkpoint 8000 is the completed-curriculum reference. A held-out validation comparison is needed to establish a best checkpoint; do not select a reported test checkpoint by its test-set metric.

## Proposed evaluation, pending the user's start instruction

Use the same SFU BG56 test set as the prior final evaluations, not the mixed SFU+BVI training dataset:

- Dataset: `/home/cilab/ndquan/vcm-rs/output/sfu_qp0_bg_table_window_test/bg56/test`.
- Scope: 14 SFU sequences, 1198 frames.
- Proposed GPUs: 0 and 1, with complete sequences assigned to each worker.
- Inference resolution: 512×512; clip length 8, stride 2; 50 denoising steps.
- Shared BG noise strength 0.95, variance-preserving enabled.
- Latent temporal guidance scale 0.1 over steps `[0, 50)`, applied every step.
- Flow: V7 RAFT student `checkpoint-0004750/raft_student`.
- BG mask: `pair_intersection`; detach previous enabled.
- Main/shared noise seeds: 1234/6789. CGE disabled.
- Postprocessing: retain 512 PNGs, resize with Lanczos to original dimensions, copy original BG56 input pixels inside the native mask (`sum(mask channels) > 255`, no feather), then encode verified native-size MPEG-4 videos at the SFU source FPS.
- Metrics: the same 512 PNG metric domain as the previous final runs; PSNR, SSIM, LPIPS, WE, WE-L2, FS and pooled VFID. Native composited PNGs are separate outputs and do not change the 512 metrics.

Output directories, relative to the new evaluation root:

```text
test/Class_*/<sequence>/*.png
test/resized_to_original/Class_*/<sequence>/*.png
test/resized_to_original_native_roi/Class_*/<sequence>/*.png
test/videos/resized_to_original_native_roi/Class_*/*.mp4
test/metrics_full_reference/metrics.csv
terminal_logs/gpu0.log
terminal_logs/gpu1.log
terminal_logs/metrics.log
```

Prepared command (not started during this review):

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet
CHECKPOINT_STEP=6000 EVAL_GPU_0=0 EVAL_GPU_1=1 \
  bash Quan_test/run_bg56_sfu_bvi30_curriculum_eval_2gpu.sh
```

The default output root is `experiments/EVAL_FOR_FINAL/eval_bg56_sfu_bvi30_ind5000_shared3000_ckpt6000_latent_tg0p1_v7student_T8S2_512`. Setting `CHECKPOINT_STEP=8000` selects the final checkpoint and a separate checkpoint-8000 output directory.
