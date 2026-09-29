# STC encoder V4++

This package adds BG-focused V4 feature fusion and a robust adjacent
feature-alignment loss.  See [V4PP_DESCRIPTOR.md](V4PP_DESCRIPTOR.md) for the
complete mathematical and gradient contract.

Train after the V2 warm-start checkpoint exists:

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet

CUDA_VISIBLE_DEVICES=1,3 \
NUM_PROCESSES=2 \
BASELINE_CHECKPOINT=/path/to/v8/checkpoint \
INIT_STC_ADAPTER=/path/to/v2/checkpoint/stc_adapter \
bash examples/brushnet/STC_encoder_v4pp_bg_feature/run_train_v4pp_bg_feature_t16.sh
```

Important defaults: `T=16`, stride `12`, shared-noise strength `0.95`,
`FLOW_REGION=bg`, `FEATURE_ALIGNMENT_REGION=bg`, feature loss weight `0.01`,
and a 500-step linear feature-loss ramp.

