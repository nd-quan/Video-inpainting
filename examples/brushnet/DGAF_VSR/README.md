# DGAF-inspired BrushNet: direct warp and rescaled warp

Hai version **không dùng STC encoder**, dùng cùng BrushNet architecture và frozen
V7 RAFT student. Chỉ khác phép warp clean latent:

| Version | Warp |
|---|---|
| `direct` | Bilinear backward warp tại latent resolution gốc |
| `dgaf` | Nearest upscale (mặc định 4×) → bilinear backward warp → nearest downscale |

Đây là adaptation của [DGAF-VSR](https://arxiv.org/abs/2511.16928) vào baseline
SD1.5/BrushNet của project, **không phải reproduction nguyên bản**. Guidance dùng
snapshot của bước denoising trước; các frame chạy song song trong clip. Mặc định
đổi chiều previous/next giữa các bước; không propagate tuần tự trong một bước.

## Isolation và model contract

Tất cả file mới nằm trong `examples/brushnet/DGAF_VSR/`. Không sửa model, pipeline,
trainer hoặc evaluator cũ. `brushnet_dgaf.py` và `pipeline_dgaf.py` là class mới,
kế thừa utilities có sẵn; không monkey-patch class/module cũ. Chỉ tái sử dụng
dataset, frozen IP/fusion helpers, metrics/compositing và frozen RAFT provider.
Không load hoặc khởi tạo STC model/checkpoint nào.

```text
degraded RGB clip ── frozen RAFT student ── forward/backward flow (tính một lần)
                                               │
cached predicted-clean latent từ step trước ── warp ── BG/validity gate
                                               │
condition = [input_latent(4), M_BG(1), aligned_clean(4), support(1)]
                                               │
noisy latent z_t(4) ── concatenate ── trainable BrushNet (14 input channels)
                                               │
                      down / mid / up residuals ── frozen U-Net
                                               │
                             DDIM step → z_next + predicted_clean
```

- Initialize toàn bộ BrushNet từ shared-noise baseline. Copy 9 input channels cũ;
  5 channel guidance mới được zero-initialize. Fresh model khớp baseline trong
  sai số số học; sau training, `temporal_guidance_scale=0` chỉ là ablation của
  **BrushNet đã fine-tune**, không tái tạo baseline gốc.
- Train toàn bộ BrushNet. Freeze U-Net/IP, VAE, CLIP text/image, projection,
  fusion và RAFT. Không có flow loss, teacher-flow dependency hoặc noise warping.
- Mặc định baseline: `train_sharedNoise_sameBG_0.95_T8/checkpoint-2250`;
  RAFT: `train_v7_raft_student_flow/checkpoint-0004750/raft_student`.
- Shared BG noise rho=0.95 theo công thức variance-preserving của baseline.
- DDIM eta=0; `clip_sample=False`, `thresholding=False`. Epsilon, v_prediction
  và sample prediction đều được xử lý theo scheduler config.
- Cache reset mỗi clip/pipeline call. Không có cross-clip memory.
- Guidance giống nhau ở hai CFG branches, cache lấy từ clean prediction sau CFG.

## Confidence / occlusion option

Train flags: `--use_confidence_mask` (default) hoặc `--no-use_confidence_mask`.
Launcher environment: `USE_CONFIDENCE_MASK=1` hoặc `0`.

Forward/backward consistency đo một lần tại **RGB resolution**:

```text
error = ||backward + warp(forward, backward)||²
visible = error <= fb_alpha * (||backward||² + ||warped_forward||²) + fb_beta
```

Làm tương tự cho chiều ngược. Default alpha=0.01, beta=0.5 (squared RGB pixels).
Đây là occlusion proxy từ flow, không phải visibility ground truth. Hai version
dùng chung confidence map; thay scale warp không thay phép tính confidence.

Tắt confidence vẫn loại mẫu ngoài ảnh và vẫn gate theo **BG đích**. Source ROI
được phép cung cấp thông tin. Channel support phân biệt missing/occluded guidance
với latent có giá trị zero thật. BG gating tại input không đảm bảo raw decoded
ROI bất biến vì receptive fields; evaluator có `hard`, `blurred`, `none` composite.

## Training: predicted guidance, không dùng GT làm guidance

Default dùng truncated rollout một bước:

1. Chọn một vị trí trên DDIM grid. Noisify GT tại timestep đứng trước vị trí đó,
   dùng noise/CLIP conditioning của baseline.
2. Chạy online model `no_grad` và DDIM transition. Cache predicted clean latent
   của model; detach cả noisy state mới lẫn cache.
3. Warp cache của adjacent frame và train BrushNet tại timestep kế tiếp.

GT chỉ tạo initial noised state và loss target. RAFT chỉ thấy degraded RGB.
Không warp GT latent hoặc clean-teacher flow để làm model condition.

Vì noisy state ở bước 3 đã qua DDIM của model, **không dùng lại initial noise
làm target**. Tính effective epsilon `(z_t - sqrt(alpha)*GT)/sqrt(1-alpha)`;
chuyển sang v/sample target nếu scheduler yêu cầu. Loss là MSE trên prediction.

`--rollout_steps N` tăng độ dài đoạn warmup detached. Default một bước là một
xấp xỉ huấn luyện tiết kiệm compute; không train full inference trajectory.
`--bootstrap_probability 0.1` dành 10% batches để học timestep đầu khi chưa có
cache. Train không CFG, eval default CFG=7.5, theo baseline.

Điều này không chứng minh hiệu quả restoration; cần train và so sánh chất lượng.
Các ablation nên giữ seed, split, clip length, noise rho, steps, LR, direction,
rollout length và confidence setting giống nhau giữa direct và dgaf.

## Chạy train

Từ repo `videoInpainting/code/BrushNet`:

```bash
# Direct warp + confidence
CUDA_VISIBLE_DEVICES=0 USE_CONFIDENCE_MASK=1 \
bash examples/brushnet/DGAF_VSR/run_train_direct.sh

# DGAF-style warp + confidence
CUDA_VISIBLE_DEVICES=2 USE_CONFIDENCE_MASK=1 \
bash examples/brushnet/DGAF_VSR/run_train_dgaf.sh

# Tắt confidence (output default sẽ có suffix conf0 riêng)
CUDA_VISIBLE_DEVICES=0 USE_CONFIDENCE_MASK=0 \
bash examples/brushnet/DGAF_VSR/run_train_direct.sh
```

Default: T16/S12, resolution512, 1 clip/GPU, accumulation3, constant LR1e-5,
2000 updates, checkpoint mỗi250 updates. Launcher mặc định 1 GPU, không tự chọn
GPU index. Override `CUDA_VISIBLE_DEVICES` theo GPU rảnh.

```bash
CUDA_VISIBLE_DEVICES=0,2 NUM_PROCESSES=2 \
GRADIENT_ACCUMULATION_STEPS=3 MAIN_PROCESS_PORT=29680 \
bash examples/brushnet/DGAF_VSR/run_train_dgaf.sh
```

Hai GPU với accumulation3 tương ứng effective batch6 clips/update; một GPU
muốn giữ batch6 cần accumulation6. Footprint lớn hơn STC-only vì train toàn bộ
BrushNet. Dùng smoke cùng T/resolution/GPU count dự kiến trước long run.

Các env khác: `OUTPUT_DIR`, `DATASET_ROOT`, `BASELINE_CHECKPOINT`,
`RAFT_STUDENT_PATH`, `UPSCALE_FACTOR`, `CLIP_LENGTH`, `CLIP_STRIDE`, `RESOLUTION`,
`CLIPS_PER_DEVICE`, `MAX_TRAIN_STEPS`, `LEARNING_RATE`, `CHECKPOINTING_STEPS`,
`MIXED_PRECISION`, `SHARED_BG_NOISE_STRENGTH`, `RESUME_FROM_CHECKPOINT`.
Các CLI argument truyền cuối launcher, ví dụ `--direction bidirectional`.

## Preflight / smoke / tests

```bash
/home/cilab/ndquan/envs/guided_diff/bin/python \
  examples/brushnet/DGAF_VSR/test_dgaf.py -v

/home/cilab/ndquan/envs/guided_diff/bin/python \
  examples/brushnet/DGAF_VSR/train_dgaf.py \
  --warp_mode dgaf --output_dir experiments/preflight_dgaf --preflight_only

CUDA_VISIBLE_DEVICES=0 /home/cilab/ndquan/envs/guided_diff/bin/python \
  examples/brushnet/DGAF_VSR/train_dgaf.py \
  --warp_mode dgaf --use_confidence_mask \
  --output_dir experiments/smoke_dgaf_new \
  --resolution 128 --clip_length 2 --clip_stride 2 \
  --smoke_test --report_to none
```

`--smoke_test` chạy đúng một optimizer update, assert guidance gradients hữu
hạn và khác zero, ghi metrics nhưng không lưu multi-GB checkpoint. Preflight
chỉ kiểm tra path/config/dataset, không load toàn bộ model hoặc allocate GPU.

## Checkpoint và resume

```text
checkpoint-N/
  brushnet_dgaf/          # Complete native model config + safetensors
  guidance_config.json   # Warp mode / scale / confidence / direction
  training_state/        # Accelerate model, optimizer, scaler, per-rank RNG
  metadata.json          # Completion marker, lineage, epoch, next batch
```

Resume cùng output/args bằng `RESUME_FROM_CHECKPOINT=latest`. Metadata so sánh
model/data dependencies, warp/mask, world size, optimization, seed và dataset
index hash. Không ghi đè checkpoint, không rewind qua checkpoint mới hơn.
Resume khôi phục optimizer/scaler/RNG và thứ tự batch; không đảm bảo bitwise
reproducibility giữa các lần chạy CUDA khi không bật deterministic kernels.
Đổi warp hoặc confidence setting phải dùng output mới và train lại để so sánh.
`--stop_after_steps N` lưu checkpoint rồi dừng tại global step N để kiểm tra
resume; không thay đổi `--max_train_steps` trong contract.

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/brushnet/DGAF_VSR/run_evaluate.sh \
  --checkpoint experiments/train_dgaf_direct_conf1_T16_S12_095/checkpoint-1000 \
  --dataset_root examples/brushnet/dataset/test_1 --split test \
  --output_dir experiments/eval_dgaf_direct_conf1_test1 \
  --clip_length 16 --clip_stride 12
```

Đổi checkpoint/output để eval version dgaf. Model dependencies được đọc từ
checkpoint. Warp/mask mặc định dùng training config; có thể override
`--no-use_confidence_mask`, `--use_confidence_mask`, hoặc `--warp_mode direct`
để ablate inference, và run config sẽ ghi rõ override. Không xem inference-only
override như kết quả của model được train theo setting đó.

Lưu raw/final/GT/input, per-clip metrics và summary. Cùng run config cho phép
resume clip hoàn tất; đổi config cần output khác. Summary là mean theo clip,
overlap frames được đếm theo từng clip. Các metric sẵn có gồm PSNR/BG PSNR,
ROI errors và temporal-delta L1; **không gọi temporal-delta L1 là tLPIPS/tOF**.
Muốn kết luận về perceptual/temporal quality cần đánh giá LPIPS/tLPIPS/tOF riêng.
