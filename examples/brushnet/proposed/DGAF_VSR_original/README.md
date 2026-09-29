# DGAF original-style scheduling for BrushNet

Bản mới độc lập với `../DGAF_VSR`; chạy các launcher trong **thư mục này**.
Không có STC encoder. RAFT student, SD U-Net, IP-Adapter, VAE, CLIP/fusion và
BrushNet baseline đều đóng băng. Chỉ BrushNet có thêm temporal condition được train.

## Inference: toàn sequence, cùng timestep

Ở mỗi diffusion step:
1. Quét frame 0 → N−1 ở step chẵn, N−1 → 0 ở step lẻ.
2. Frame đầu lượt quét dùng frozen baseline BrushNet + U-Net.
3. Scheduler trả clean estimate của frame vừa xử lý. Warp estimate đó để guide
   frame kế tiếp **ngay tại cùng timestep**.
4. Ghi đè clean cache sau mỗi frame; reset khi bắt đầu lượt quét tiếp theo.

Hai cách warp:
- `direct`: resize flow về latent grid rồi bilinear warp clean estimate.
- `dgaf`: nearest upscale clean latent ×4 → bilinear warp → nearest downscale.

Evaluator gom các cặp liền kề thành toàn bộ contiguous sequence, gồm cả cặp nối
giữa các clip cũ. Không duplicate frame do overlap. Không nối hai video hoặc qua
khoảng thiếu frame. `--max_clips` trong evaluator giới hạn **số sequence hoàn chỉnh**;
`clip_length/clip_stride` không cắt nhỏ sequence khi eval.
Các run chỉ có frame đơn lẻ chưa được dataset hỗ trợ và sẽ báo lỗi.

Clean cache chỉ giữ một frame mỗi sequence. Noisy latents, RGB, flow và embeddings
vẫn có bộ nhớ tăng theo chiều dài sequence; chưa có CPU offload cho các tensor này.
VAE dùng slicing, CLIP được encode từng frame. Các frame phụ thuộc nhau nên
không batch độc lập các frame liền kề khi denoise; có thể batch các sequence độc lập.

## Training: triplet, random neighbor

Mặc định T3/S1. Với [i−1, i, i+1], chọn trái/phải xác suất 0.5 cho từng mẫu.
Lấy cùng timestep t cho nguồn và đích, tạo noisy latent từ GT mỗi frame.
Frozen baseline dự đoán clean estimate nguồn; detach → warp → điều kiện temporal
cho frame giữa. Loss MSE dùng target theo scheduler (epsilon/v_prediction/sample).
Không rollout DDIM trong train, không cache qua optimizer step.
`--num_diffusion_steps` chỉ đặt mặc định số sampling steps lưu trong checkpoint;
training lấy t trên toàn bộ lịch noise training.

Noise giữa frame mặc định độc lập (`shared_bg_noise_strength=0`), như cách train gốc.
Có thể đặt `SHARED_BG_NOISE_STRENGTH=0.95` để làm ablation riêng.
Confidence bật/tắt bằng `USE_CONFIDENCE_MASK=1/0`.
Tắt confidence vẫn giữ sampling validity và destination BG mask cho bài toán inpainting.

## Train 2 GPU, không confidence

Chạy từ thư mục BrushNet. Hai lệnh dùng cùng GPU nên chạy lần lượt.

```bash
CUDA_VISIBLE_DEVICES=0,2 NUM_PROCESSES=2 USE_CONFIDENCE_MASK=0 \
MAIN_PROCESS_PORT=29680 \
bash examples/brushnet/DGAF_VSR_original/run_train_direct.sh
```

```bash
CUDA_VISIBLE_DEVICES=0,2 NUM_PROCESSES=2 USE_CONFIDENCE_MASK=0 \
MAIN_PROCESS_PORT=29681 \
bash examples/brushnet/DGAF_VSR_original/run_train_dgaf.sh
```

Mặc định resolution 512, một triplet/GPU, accumulation 3, LR 1e-5, 2000 updates.
Output lần lượt:
- `experiments/train_dgaf_original_direct_conf0_T3_S1`
- `experiments/train_dgaf_original_dgaf_conf0_T3_S1`

Đặt `OUTPUT_DIR` riêng khi đổi noise, stride hoặc tham số ablation.
Resume bằng `RESUME_FROM_CHECKPOINT=latest` cùng output và hyperparameters.
Checkpoint có schema_version=2; không resume hoặc eval nhầm checkpoint
snapshot/rollout từ thư mục DGAF_VSR cũ.

## Eval

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/brushnet/DGAF_VSR_original/run_evaluate.sh \
  --checkpoint experiments/train_dgaf_original_dgaf_conf0_T3_S1/checkpoint-2000 \
  --dataset_root /path/to/testset --dataset_layout flat_test \
  --output_dir experiments/eval_dgaf_original_dgaf_conf0 \
  --num_inference_steps 50
```

Warp mode, confidence, noise strength và fusion mặc định lấy từ training contract.
Có thể override warp/confidence khi làm ablation, được ghi vào run_config.
Direction mặc định alternating; previous/next là các ablation một chiều.
Không hỗ trợ simultaneous bidirectional trong pipeline tuần tự.

## Mức độ bám repo gốc

Đối chiếu `pretrained/DGAF-VSR/src/diffusers/pipelines/dgafnet/pipeline_DGAFFeaAlign.py`
(vòng lặp 918, warp 925, scheduler 1055, đảo chiều 1070) và
`examples/dgafvsr/train_dgafvsr.py` (random neighbor 890, frozen source 926, warp 938).

Đã bám lịch quét tuần tự cùng timestep và cách train random neighbor.
Đây vẫn là adaptation cho BrushNet inpainting, không phải tái tạo nguyên model SR:
- Frozen BrushNet baseline + SD1.5/IP-Adapter đóng vai trò model phục hồi nền.
  Frame đầu lượt quét dùng model nền; frame sau dùng BrushNet temporal thay baseline.
- Giữ DDIM eta=0 của thí nghiệm BrushNet; script test gốc dùng DDPMScheduler.
- Giữ RAFT student của dự án, BG/support channels và tùy chọn confidence.
- Giữ quy ước grid_sample align_corners=False nhất quán với pixel-center grid;
  không sao chép bất nhất normalization W−1/H−1 trong flow utility gốc.
- Không sử dụng consistency decoder hoặc trọng số DGAF-VSR.

## Kiểm tra

```bash
/home/cilab/ndquan/envs/guided_diff/bin/python examples/brushnet/DGAF_VSR_original/test_dgaf.py
```

`--preflight_only` chỉ kiểm tra cấu hình/dataset.
`--smoke_test --report_to none` train một update, kiểm tra temporal gradient, không lưu checkpoint lớn.
`smoke_inference.py` chạy model thật trên một triplet, 3 diffusion steps và lưu preview;
cần output_dir mới. Xem VALIDATION.md cho kết quả đã chạy.
