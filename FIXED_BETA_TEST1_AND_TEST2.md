# Fixed-beta STC experiments

## Test 1: shaped initial noise, no Stage-3 adapter

Test 1 is an inference-only experiment. It loads the best Stage-1 full-flow
predictor, reconstructs it with a fixed beta, and uses it once to build the
initial diffusion noise:

\[
\epsilon_t = c_t\,W(\epsilon_0,\phi_{t\leftarrow0})
             + \sqrt{1-c_t^2}\,\epsilon_t^{ind},
\qquad c_t=\beta V_t.
\]

`V_t` is the valid support returned by warping. Warping is applied over the
whole frame. The original decoded ROI is composited over the generated result
after inference.

The following components are frozen:

- Stage-1 STC encoder and bidirectional flow decoder;
- fixed beta (there is no `beta_head`);
- SD1.5 VAE, text encoder and diffusion U-Net;
- V8 BrushNet, gated IP-Adapter and FGBG fusion module.

The V8 checkpoint does not contain a separately fine-tuned U-Net. Test 1 loads
the base SD1.5 U-Net and installs the V8 IP-Adapter processors into it. V8
conditioning is reproduced as:

\[
e_{image}=e_{full}+1.0\,e_{fusion}(e_{BG},e_{ROI}),
\]

where the branch order preserves the historical V8 mask convention.

Run preflight first:

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet

PYTHONPATH=examples/brushnet \
/home/cilab/ndquan/envs/guided_diff/bin/python \
examples/brushnet/evaluate_stc_fixed_beta_test1_vcm.py \
  --config examples/brushnet/configs/eval_vcm_stc_fixed_beta_test1.json \
  --preflight-only
```

Run one short clip:

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=examples/brushnet \
/home/cilab/ndquan/envs/guided_diff/bin/python \
examples/brushnet/evaluate_stc_fixed_beta_test1_vcm.py \
  --config examples/brushnet/configs/eval_vcm_stc_fixed_beta_test1_smoke.json
```

Run the complete test split:

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=examples/brushnet \
/home/cilab/ndquan/envs/guided_diff/bin/python \
examples/brushnet/evaluate_stc_fixed_beta_test1_vcm.py \
  --config examples/brushnet/configs/eval_vcm_stc_fixed_beta_test1.json
```

`--fixed-beta 0.0`, `0.5`, `0.7`, or `0.9` creates separate result folders.
Every beta uses the same deterministic per-clip seed, allowing a fair ablation.
Each run saves generated frames, per-clip metrics, noise-shaper statistics, a
JSONL table, a summary JSON, resolved configuration and a log file.

## Test 2: fixed shaped noise plus a trained Stage-3 adapter

Test 2 keeps the same fixed-beta construction as Test 1 and adds only a
trainable STC condition adapter. It must not train beta or reuse the old motion
adapter.

The Stage-3 training implementation accepts the Stage-1 `best.json` pointer,
rebuilds its flow predictor as a fixed-beta noise shaper and checks that no
`beta_head` exists. It then reuses `shaped["stc_features"]`, converts them into
multi-scale BrushNet residual corrections and adds them at the frozen U-Net's
down/mid residual ports.

The following components remain frozen:

- Stage-1 STC encoder and full-flow decoder;
- fixed beta and the complete noise-shaping law;
- VAE, text/image encoders, V8 BrushNet, IP-Adapter/FGBG fusion and U-Net.

Only the zero-initialized `STCBrushNetConditionAdapter` is optimized. The
learned-beta Stage-2 checkpoint and the legacy motion adapter are not used.

Run preflight after the NAS checkpoint is mounted:

```bash
cd /home/cilab/ndquan/videoInpainting/code/BrushNet

PYTHONPATH=examples/brushnet \
/home/cilab/ndquan/envs/guided_diff/bin/python \
examples/brushnet/train_stc_condition_adapter_vcm.py \
  --config examples/brushnet/configs/train_vcm_stc_condition_adapter_fixed_beta_test2_smoke_2gpu.json \
  --preflight-only
```

Run the 20-step two-GPU smoke test first. This diagnostic config deliberately
uses FP32 so that numerical/model errors can be separated from FP16 gradient
overflow before enabling mixed precision for the full run:

```bash
CUDA_VISIBLE_DEVICES=1,3 \
PYTHONPATH=examples/brushnet \
/home/cilab/ndquan/envs/guided_diff/bin/torchrun \
  --standalone \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port=29521 \
  examples/brushnet/train_stc_condition_adapter_vcm.py \
  --config examples/brushnet/configs/train_vcm_stc_condition_adapter_fixed_beta_test2_smoke_2gpu.json
```

After the smoke validation succeeds, start the full run from scratch:

```bash
CUDA_VISIBLE_DEVICES=1,3 \
PYTHONPATH=examples/brushnet \
/home/cilab/ndquan/envs/guided_diff/bin/torchrun \
  --standalone \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port=29522 \
  examples/brushnet/train_stc_condition_adapter_vcm.py \
  --config examples/brushnet/configs/train_vcm_stc_condition_adapter_fixed_beta_test2_2gpu.json
```

The full run uses `T=8`, one clip per GPU and gradient accumulation of two,
which gives an effective batch of four clips (32 frames) per optimizer step.
It validates and saves every 500 optimizer steps. To resume an interrupted
run, append:

```text
--resume experiments/vcm_stc_condition_adapter_fixed_beta_test2/checkpoints/latest.json
```

The fair Test-2 comparison is therefore:

```text
Test 1: Stage1 flow -> fixed-beta shaped z_T -> frozen V8 stack
Test 2: Stage1 flow -> fixed-beta shaped z_T -> trained STC adapter -> frozen V8 stack
```

Do not use the learned-beta Stage-2 checkpoint for this branch.
