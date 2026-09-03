# CGE một frame với VCM-RS: ROI QP20 + BG QP52

## Toán tử degradation dùng khi restoration

Mask có quy ước cố định: trắng/`1` là ROI, đen/`0` là BG. Cả hai encoder nhận **cùng một ảnh full-frame**; không nhân ảnh với mask trước khi encode:

```text
R_roi = VCM-RS(full_frame, ROI descriptor, QP 20)
R_bg  = VCM-RS(full_frame, BG descriptor,  QP 52)
D_M(x) = M * R_roi + (1 - M) * R_bg
```

Phép ghép là hard binary, không blur/feather. Đây là cùng quy tắc với `vcm-rs/Plot/combine_roi_bg_images.py` và pipeline tạo data train ROI/BG.

CGE dùng loss consistency toàn ảnh:

```text
L = ||D_M(I_hat) - x_observed||^2
```

Vì sai phân trung tâm cần `D_M(x)`, `D_M(x+h)` và `D_M(x-h)`, một CGE evaluation chạy **6 encoder call**:

```text
base:  ROI QP20 + BG QP52
plus:  ROI QP20 + BG QP52
minus: ROI QP20 + BG QP52
```

`x_observed` phải là ảnh degraded/composite đưa vào restoration. CGE không encode lại observation; nó áp forward operator lên ảnh model dự đoán rồi so sánh với observation.

## Chạy mẫu thật một frame

Lệnh sau đã được kiểm tra với frame và mask PartyScene 512×512 có sẵn:

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet

PATH=/home/cilab/ndquan/envs/vcm/bin:$PATH \
CUDA_VISIBLE_DEVICES=3 \
python examples/brushnet/vcmrs_codec_adapter.py \
  examples/brushnet/dataset/test_1/PartyScene/gt/000000.png \
  /home/cilab/ndquan/vcm-rs/working/cge_one_frame_dual_qp_train_match/merged_qp20_qp52.png \
  --mask examples/brushnet/dataset/test_1/PartyScene/masks/000000.png \
  --profile train_match \
  --configuration RandomAccess \
  --intra-period 64 \
  --frame-rate 50 \
  --keep-artifacts \
  --artifact-root /home/cilab/ndquan/vcm-rs/working/cge_one_frame_dual_qp_train_match \
  --cuda-visible-devices 3
```

Đầu ra chính:

```text
/home/cilab/ndquan/vcm-rs/working/cge_one_frame_dual_qp_train_match/merged_qp20_qp52.png
```

Vì bật `--keep-artifacts`, cùng thư mục còn có hai job riêng:

```text
cge_vcmrs_..._roi_qp20_.../output/bitstream/coded.bin
cge_vcmrs_..._roi_qp20_.../output/recon/coded/frame_000000.png
cge_vcmrs_..._bg_qp52_.../output/bitstream/coded.bin
cge_vcmrs_..._bg_qp52_.../output/recon/coded/frame_000000.png
cge_vcmrs_..._descriptors_.../roi_key0.txt
cge_vcmrs_..._descriptors_.../bg_key0.txt
```

Kết quả test hiện tại đã xác nhận ảnh merged khớp từng pixel với:

```text
where(mask > 127, ROI_reconstruction, BG_reconstruction)
```

Nếu chỉ muốn kiểm tra hai QP bằng VTM và bỏ qua spatial retargeting/restoration, NNLF:

```bash
python examples/brushnet/vcmrs_codec_adapter.py INPUT.png OUTPUT.png \
  --mask MASK.png \
  --profile vtm_only \
  --configuration AllIntra \
  --intra-period 1
```

## Descriptor được tạo như thế nào

Với `train_match`, mặc định adapter tự làm đúng quy trình cho từng mask inference:

1. Resize mask bằng nearest-neighbor ở script BrushNet.
2. Threshold một lần tại `0.5`.
3. Tách chính xác `M` và `1-M` thành các scanline rectangle có tọa độ inclusive.
4. Ghi hai descriptor với frame key số nguyên `0` và `scaling_method=1`.
5. Kiểm tra ROI/BG không overlap và không bỏ trống pixel nào.
6. Cache và dùng lại đúng cặp descriptor đó cho `base/plus/minus` và các CGE step của frame.

Không dùng trực tiếp descriptor `832×480` cho inference `512×512`. Cũng không dùng connected-component bounding box cho mask bất quy tắc vì bbox làm thay đổi hình học vùng ROI.

Có thể tắt auto-generation và cung cấp descriptor đã chuẩn bị sẵn:

```bash
export CGE_VCMRS_AUTO_DESCRIPTORS=0
export CGE_VCMRS_ROI_DESCRIPTOR=/absolute/path/frame_roi.txt
export CGE_VCMRS_BG_DESCRIPTOR=/absolute/path/frame_BG.txt
```

Khi chạy nhiều ảnh, dùng hai template có placeholder `{stem}`, `{basename}`, `{index}`:

```bash
export CGE_VCMRS_ROI_DESCRIPTOR_TEMPLATE='/absolute/path/ROI/{stem}_roi.txt'
export CGE_VCMRS_BG_DESCRIPTOR_TEMPLATE='/absolute/path/BG/{stem}_BG.txt'
```

Hai template phải được đặt cùng nhau và mỗi file phải dùng key `0` trong hệ tọa độ inference.

## Chạy BrushNet inference một frame

Các thư mục input phải chứa ảnh observation đã degraded/composite nếu mục tiêu là đánh giá restoration. Ví dụ local PartyScene dùng `inputs`, không phải `gt`:

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet/examples/brushnet

export CGE_IMAGE_DIR=/home/cilab/ndquan/videoInpainting/code/BrushNet/examples/brushnet/dataset/test_1/PartyScene/inputs
export CGE_MASK_DIR=/home/cilab/ndquan/videoInpainting/code/BrushNet/examples/brushnet/dataset/test_1/PartyScene/masks
export CGE_CAPTION_FILE=/absolute/path/to/captions.txt
export CGE_OUTPUT_DIR=/absolute/path/to/output

export CGE_BASE_MODEL_PATH=/absolute/path/or/huggingface-id
export CGE_BRUSHNET_PATH=/absolute/path/checkpoint/brushnet
export CGE_IP_ADAPTER_PATH=/absolute/path/checkpoint/ipadapter/model.safetensors
export CGE_FUSION_PATH=/absolute/path/checkpoint/ipadapter/fusion_module.safetensors

CGE_MAX_FRAMES=1 \
CGE_START_STEP=25 \
CGE_MAX_EVALS=1 \
CGE_VCMRS_PROFILE=train_match \
CGE_VCMRS_ROI_QUALITY=20 \
CGE_VCMRS_BG_QUALITY=52 \
CGE_VCMRS_CONFIGURATION=RandomAccess \
CGE_VCMRS_INTRA_PERIOD=64 \
CGE_VCMRS_FRAME_RATE=50 \
CGE_VCMRS_AUTO_DESCRIPTORS=1 \
CGE_VCMRS_ROOT=/home/cilab/ndquan/vcm-rs \
CGE_VCMRS_PYTHON=/home/cilab/ndquan/envs/vcm/bin/python \
CGE_VCMRS_CUDA_VISIBLE_DEVICES=3 \
python test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion_fixedBG_CGE_v0_2_.py
```

`CGE_MAX_EVALS=1` là một CGE evaluation và tương ứng 6 lần encode. Đặt `-1` hoặc bỏ biến để chạy theo toàn bộ lịch. Có thể giảm chi phí bằng:

```bash
CGE_START_STEP=20 CGE_END_STEP=50 CGE_EVERY_N_STEPS=5
```

Step là thứ tự denoising bắt đầu từ `0`; `CGE_END_STEP` là cận trên exclusive.

## Giữ ROI hay restoration cả ROI

Script cũ paste phần ROI observation trở lại ảnh cuối. Hành vi đó vẫn là mặc định vì yêu cầu trước đó là giữ ROI:

```bash
CGE_PRESERVE_INPUT_ROI=1
```

Nếu muốn model restoration cả phần ROI đã nén QP20, phải tắt final paste:

```bash
CGE_PRESERVE_INPUT_ROI=0
```

Trong cả hai chế độ, forward operator của CGE vẫn dùng QP20 cho ROI và QP52 cho BG. Khác biệt chỉ nằm ở bước ghép output cuối cùng của BrushNet.

## Profile và giới hạn one-frame

| Profile | Hai nhánh QP | Descriptor/spatial tools | Mục đích |
|---|---|---|---|
| `train_match` (mặc định) | ROI 20, BG 52 | Có; tự tạo ROI/BG descriptor, bật spatial retargeting/restoration và NNLF | Khớp pipeline tạo data train theo vùng |
| `vtm_only` | ROI 20, BG 52 | Bypass | Smoke test nhanh, gần vòng VVC/VTM cũ |

Pipeline SFU train dùng RandomAccess/IntraPeriod 64. Một input chỉ có một frame vẫn là I-frame; nó không thể tái tạo artifact temporal của P/B-frame. Muốn khớp video RA hoàn toàn phải đổi operator từ `frame -> frame` thành `clip/GOP -> clip/GOP` và giữ reference frames. Với restoration ảnh độc lập, dual one-frame operator ở trên là cấu hình phù hợp.

Không dùng ảnh quá nhỏ (ví dụ `64×64`) cho smoke test `train_match`: VCM spatial retargeting có thể từ chối cấu trúc bitstream ROI ở kích thước đó. Dùng ảnh inference thật `512×512`, hoặc dùng `vtm_only` cho smoke test nhỏ.

`vcmrs.encoder` đã sinh reconstruction sau decoder/post-chain, nên adapter không gọi `vcmrs.decoder` lần hai.

## Biến môi trường chính

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `CGE_VCMRS_PROFILE` | `train_match` | `train_match` hoặc `vtm_only` |
| `CGE_VCMRS_ROI_QUALITY` | `20` | QP nhánh ROI |
| `CGE_VCMRS_BG_QUALITY` | `52` | QP nhánh BG |
| `CGE_VCMRS_CONFIGURATION` | `RandomAccess` | Cấu hình VTM |
| `CGE_VCMRS_INTRA_PERIOD` | `64` | IntraPeriod |
| `CGE_VCMRS_FRAME_RATE` | `30` | Frame rate; đặt `50` cho PartyScene |
| `CGE_VCMRS_AUTO_DESCRIPTORS` | `1` | Tạo descriptor key `0` từ mask |
| `CGE_VCMRS_MASK_THRESHOLD` | `0.5` | Threshold nhị phân mask tensor |
| `CGE_VCMRS_TIMEOUT` | `600` | Timeout cho một encoder call, giây |
| `CGE_VCMRS_MAX_PARALLEL` | `1` | Giới hạn chung số VCM-RS subprocess |
| `CGE_VCMRS_KEEP_ARTIFACTS` | `0` | Giữ job thành công để debug |
| `CGE_VCMRS_ARTIFACT_ROOT` | system temp | Nơi tạo job/descriptor |
| `CGE_VCMRS_CUDA_VISIBLE_DEVICES` | kế thừa | GPU cho VCM tool |
| `CGE_START_STEP` | `0` | Step denoising đầu tiên chạy CGE |
| `CGE_END_STEP` | không giới hạn | Step kết thúc, exclusive |
| `CGE_EVERY_N_STEPS` | `1` | Chu kỳ gọi CGE |
| `CGE_MAX_EVALS` | `-1` | Số CGE evaluation tối đa mỗi frame |
| `CGE_MAX_FRAMES` | `0` | Số frame tối đa; `0` là toàn bộ |
| `CGE_PRESERVE_INPUT_ROI` | `1` | Paste ROI observation vào output cuối |

Job lỗi luôn được giữ lại. Exception chứa `cwd`, command, return code, job path và phần cuối stdout/stderr. Job thành công bị xóa nếu không bật `CGE_VCMRS_KEEP_ARTIFACTS=1`.
