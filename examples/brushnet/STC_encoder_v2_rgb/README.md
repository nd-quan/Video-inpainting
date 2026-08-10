# RGB-STC v2 — frozen V8 shared-noise baseline

Đây là stage tối giản để đo riêng đóng góp của STC encoder, không mang flow
predictor/noise shaper từ `STC_encoder_v1` sang.

## Contract của thí nghiệm

```text
full degraded RGB       [B,T,3,512,512]
M_BG sequence           [B,T,1,512,512]  (1=degraded BG, 0=HQ ROI)
        │ concat
        ▼
3 spatial down blocks + temporal attention tại 64x64
        │
        ▼
ZeroConv C→4
        │ gate bằng M_BG latent
        ▼
delta_z_BG              [B,T,4,64,64]

z_condition_aug = z_input + scale * delta_z_BG
BrushNet condition = concat(z_condition_aug, M_BG_latent)  # vẫn 5 channel
```

`full_rgb_bg_mask` là experiment chính: encoder thấy toàn bộ ảnh degraded để
nhận biết artifact, đồng thời mask cho biết vùng nào cần sửa. Mode
`videocomposer_roi_masked` chỉ là ablation gần với condition masked-frame của
VideoComposer.

Checkpoint V8 được load từ ba component explicit dưới
`experiments/train_sharedNoise_sameBG_0.9/checkpoint-2000`. Không gọi
`accelerator.load_state()` trên checkpoint V8 vì optimizer/topology của stage
cũ khác hoàn toàn.

Toàn bộ VAE, BrushNet, U-Net, IP attention processors, image projection, FGBG
fusion, CLIP text và CLIP image đều frozen. Optimizer chỉ chứa RGB-STC. Tuy
nhiên BrushNet/U-Net không chạy trong `torch.no_grad()`, vì gradient phải đi
xuyên qua chúng về adapter. Để khớp numerical path của V8, BrushNet/U-Net/IP và
fusion giữ FP32 weights dưới autocast; chỉ VAE/text/image encoder frozen được
cast cố định sang mixed-precision dtype.

Loss duy nhất ở phase này là diffusion MSE của baseline:

```text
L = MSE(epsilon_pred, epsilon_shared_noise)
```

Không có optical-flow loss và chưa thêm `L_bg`/L1 reconstruction penalty. Như
vậy kết quả có thể quy trực tiếp cho RGB-STC condition thay vì trộn nhiều thay
đổi cùng lúc. BG gate bảo đảm bốn channel condition tại ROI không bị sửa trực
tiếp; đây không phải bảo đảm cứng rằng output RGB tại ROI bất biến, vì các
convolution của denoiser vẫn có receptive field đi qua biên mask.

## Preflight

Launcher mặc định dùng environment đã được full-GPU smoke-test trên máy này tại
`/home/cilab/ndquan/envs/guided_diff`. Có thể override bằng `PYTHON_BIN` và
`ACCELERATE_BIN`. Environment cần có `torch`, `accelerate`, `transformers`,
`safetensors` và dùng vendored `diffusers` của repo.

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet
python examples/brushnet/STC_encoder_v2_rgb/train_rgb_stc_shared_noise.py \
  --preflight_only
```

Preflight kiểm tra checkpoint đúng schema 5-channel BrushNet, IP-Adapter đủ
100 tensor, fusion đủ 10 tensor, dataset alignment, mask nhị phân và một clip
`T=8` thực tế mà không load full SD weights lên GPU.

## Train mặc định

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet
bash examples/brushnet/STC_encoder_v2_rgb/run_train_rgb_stc_shared_noise.sh
```

Single-GPU chạy trực tiếp bằng Python để không vô tình kế thừa DeepSpeed/FSDP
từ Accelerate config. Đường DDP đã smoke-test với hai GPU:

```bash
NUM_PROCESSES=2 \
bash examples/brushnet/STC_encoder_v2_rgb/run_train_rgb_stc_shared_noise.sh
```

Launcher kế thừa đúng cấu hình checkpoint nền: `T=8`, stride 6, rho 0.9,
variance-preserving shared noise, refresh theo epoch, seed 1234, batch một clip
và gradient accumulation 6. Các giá trị có thể override bằng environment:

```bash
MAX_TRAIN_STEPS=3000 LEARNING_RATE=2e-5 \
OUTPUT_DIR=/home/cilab/ndquan/videoInpainting/code/BrushNet/experiments/my_rgb_stc_run \
bash examples/brushnet/STC_encoder_v2_rgb/run_train_rgb_stc_shared_noise.sh
```

Nếu executable không nằm trong `PATH`, truyền rõ:

```bash
PYTHON_BIN=/path/to/env/bin/python \
ACCELERATE_BIN=/path/to/env/bin/accelerate \
bash examples/brushnet/STC_encoder_v2_rgb/run_train_rgb_stc_shared_noise.sh
```

## Checkpoint và resume

Mỗi checkpoint mới chỉ chứa RGB-STC cùng optimizer/scheduler/scaler/RNG của
stage này:

```text
experiments/train_rgb_stc_v2_sharedNoise_0.9/
├── checkpoint-N/
│   ├── stc_adapter/
│   ├── accelerator_state/
│   └── metadata.json
├── stc_adapter/              # final inference component
└── final_metadata.json
```

Không sao chép lại BrushNet/U-Net frozen. Mặc định chỉ giữ năm checkpoint gần
nhất. Resume:

```bash
bash examples/brushnet/STC_encoder_v2_rgb/run_train_rgb_stc_shared_noise.sh \
  --resume_from_checkpoint latest
```

Không truyền `checkpoint-2000` của V8 vào `--resume_from_checkpoint`; nó đã là
`--baseline_checkpoint`. Resume được kiểm tra theo contract nghiêm ngặt, nên
nếu run đầu đã override `MAX_TRAIN_STEPS`, LR, mask threshold hoặc tham số khác
ảnh hưởng trajectory thì lệnh resume cũng phải truyền lại đúng giá trị đó.
`latest` chỉ nhận checkpoint đã ghi đủ adapter, optimizer, scheduler, RNG và
metadata; folder đang ghi dở do disconnect sẽ bị bỏ qua.

## Phạm vi hiện tại

Script này hoàn thiện đường train và component save/load. Pipeline inference
hiện hữu chưa được sửa ngầm: bước tiếp theo cần hook RGB-STC một lần trên
`[B,T]` trước denoising, sau đó duplicate condition đúng theo CFG. Việc tách
hai bước giúp checkpoint train không bị đánh giá bằng inference path vốn lệch
contract của V8.
