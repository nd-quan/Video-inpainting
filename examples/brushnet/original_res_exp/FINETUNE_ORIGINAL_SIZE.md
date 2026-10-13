# Shared-noise finetuning at native size

Entry point: `finetune_shared_noise_original_size.py`. This is a separate trainer;
it imports the existing BrushNet/IP-Adapter/fusion definitions but does not use
the STC V8 adapter. Existing training and evaluation scripts are unchanged.

## Defaults

- Data: `/home/cilab/ndquan/videoInpainting/SFU_STC_flow/train`, all branches,
  including overlap with evaluation, as requested. Scores on overlapping frames
  measure adaptation to those frames, not held-out generalization.
- Initialization: `experiments/train_sharedNoise_sameBG_0.95_T8/checkpoint-2250`.
  Only model weights are transferred; optimizer and schedule start fresh.
- Output: `experiments/finetune_shared_noise_original_size`.
- LR `1e-6`, 50 warmup updates, 1000 updates, checkpoint every 250 updates.
- One clip/device, T=4, stride=1, one frame/forward-backward, accumulation=4.
  Effective frames/update = T * accumulation * number of GPUs (except a short
  final accumulation group). These are starting settings, not tuned results.
- Shared BG strength=0.95, non-variance-preserving mixture, new sequence field
  each epoch, shared timestep across the clip. Set strength=1 explicitly if the
  intended experiment is to match the evaluation mixture exactly.
- Train BrushNet, IP-Adapter projection/attention, and foreground/background
  fusion. Freeze base U-Net weights, VAE, CLIP text and vision encoders.
- Temporal guidance and RAFT are not run in this diffusion training objective.

## Native geometry and memory

GT/input are normalized to [-1,1] and padded on the right/bottom with edge pixels
to multiples of 64. ROI masks use nearest semantics and zero padding (background).
No diffusion-image resize or crop occurs. Frozen CLIP vision preprocessing keeps
its normal fixed input size. The loss uses real-image coverage per latent cell
so fully padded cells contribute zero; partial edge cells are weighted by area.
This masks the loss; it does not remove convolutional influence of padding.

Headers of all GT/input/mask triplets are checked before GPU models load.
Clips cannot cross gaps in frame IDs or changes in stored dimensions. A sequence
may contain both 512 files and native-size files: those form separate geometry
groups and shared-noise keys. A 512 source file remains 512; this script does not
recover its missing native detail. `dataset_report.json` records counts.

Frozen VAE encoding runs per frame. Gradient checkpointing is enabled and the
trainable wrapper is in train mode. Each frame chunk backpropagates immediately;
its loss is weighted so chunks together equal the whole-clip noise objective.
Noise is still constructed jointly for the clip before chunking. DDP wraps all
trainable components together. More GPUs increase throughput, not per-GPU VRAM.

`--vae_tiling` optionally reduces VAE encoding memory, but alters encoding near
tile boundaries. It does not solve U-Net/BrushNet activation memory by itself.
Full 2560x1600 training must be measured on the actual GPU; no automatic resize
or fallback to crops is performed.

## Commands

From `examples/brushnet`, inspect all data without loading GPU models:

```bash
bash run_finetune_shared_noise_original_size.sh --preflight_only
```

Train on one selected GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_finetune_shared_noise_original_size.sh
```

Train with all six GPUs when available:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NUM_PROCESSES=6 \
  bash run_finetune_shared_noise_original_size.sh
```

Use `--include_branches Class_D/BasketballPass` to select a branch for a small
run. Use a separate `OUTPUT_DIR` for each experiment. Resume the native trainer
using `--resume_from_checkpoint /absolute/path/checkpoint-N`; keep the dataset,
clip/sampling settings, world size and accumulation unchanged. The native
checkpoint includes optimizer, scheduler, RNG and next-batch state. Periodic
and final checkpoints include eval-compatible `brushnet/` and
`ipadapter/{model,fusion_module}.safetensors`. Use the existing original-size
evaluator for validation; this script does not run the old square-image preview.

## Student flow: match the actual inference geometry

`test_brushnet_fixedBG_temporal_v7_student_original_size.py` currently sets
`FrozenV7RAFTFlowProvider(max_input_side=512)`. It first uses the padded native
canvas, resizes the RAFT input to a long side of at most 512 with dimensions
rounded to multiples of 8, then restores flow to the padded canvas. Restoration
multiplies dx by W_original/W_flow and dy by H_original/H_flow. The diffusion
model sees native pixels; RAFT generally sees smaller rectangular images.

The existing V7 pair loader instead resizes every pair to 512x512. Recommended
first adaptation is to retain the current inference cap and finetune the V7
student on the same rectangular geometry, using native sources:

1. Extend `proposed/STC_encoder_v7_raft_flow_distillation/raft_teacher_pair_data.py`
   (or a separate loader) to pad native GT/input consistently and perform exactly
   the provider's aspect-preserving resize. Batch one pair or bucket same shapes.
2. Generate a NEW teacher cache from clean GT using the same geometry. Match
   forward/backward direction, pixel units, validity, crop/padding and frame IDs.
   Resizing old square teacher flows rescales coordinates but does not restore
   motion details absent from that teacher inference. Keep the old cache separate.
   The existing builder is
   `proposed/STC_encoder_v1/precompute_sfu_stc_teacher_flows.py`; its preprocessing
   and cache metadata would need to describe the new per-sequence geometry.
3. Warm-start student weights using the existing `--init_from_checkpoint`
   option pointing to `experiments/train_v7_raft_student_flow/checkpoint-0004750`.
   Use a fresh optimizer for the new finetuning run. The existing trainer already
   supports this initialization independently from resuming training state.
4. Keep the current flow-distillation objective initially: teacher supervision,
   forward/backward consistency and edge-aware smoothness. Exclude padded and
   invalid/occluded correspondences from relevant losses and metrics, including
   smoothness edges if they touch invalid padding.
5. Compare old/new students with the SAME diffusion checkpoint and RGB guidance
   scale/window. Measure flow EPE/warp error as well as final video quality.

If the intention is RAFT inference at full native resolution, both teacher and
student geometry/cache and the eval provider cap must change. Full-resolution
all-pairs RAFT correlation grows quadratically in spatial feature count, so
full-HD/2560x1600 backpropagation may be impractical even when BrushNet fits.
Training on corresponding native-scale patches is another experiment: teacher
and input crops must agree, and matches whose endpoints leave the crop need
validity filtering. It is not equivalent to full-frame RAFT training. Changing
the flow inference cap to 768/1024 is also a separate geometry experiment that
requires matched training/cache and memory checks.

The files added here implement shared-noise finetuning only. Student-flow
training changes above are a plan, not an already implemented native loader.

## Verification (2026-10-05)

- CPU header preflight: 4,906 training frames, 4,849 T=4 clips, all frames covered;
  three candidate clips crossing a stored-size change removed. 150 source images
  are 512x512; the remaining files span five native sizes.
- Synthetic checks: original pixel values retained, mask padding, mixed-size
  boundaries, partial latent-cell weights, zero loss/gradient on fully padded
  cells, uneven frame-chunk gradient equivalence, shared-noise identity.
- Actual checkpoint forward/backward with frozen VAE/CLIP on GPU: 416x240 and
  2560x1600 passed with finite gradients. The latter took approximately 32 seconds
  and peaked at 30.25 GiB allocated VRAM for one synthetic frame. This excludes
  Adam moment state after an optimizer update, full-clip input storage and DDP
  overhead; it is not a full training throughput or quality measurement.
- Actual checkpoint on two GPUs: two forward/backward iterations, two frame
  chunks each, different native shapes per rank; gradient synchronization passed.
- No optimizer update, finetuning job or new model checkpoint was created by
  these checks. Python compilation and launcher shell syntax checks passed.
