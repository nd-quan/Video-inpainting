# V8 clean-teacher-flow ablation

Entry point: `evaluate_v8_oracle_from_config.py` replays a saved shard's
`run_config.json` through `evaluate_v8_raft_deformable_cge_temporal_bg_only.py`.
It changes only output location and V8 conditioning flow source. It does not
launch a student evaluation or change any model weights. The reference config
SHA256 and oracle scope are recorded in `oracle_provenance.json`.

## Controlled comparison

- Current and predecessor clips receive cached teacher forward/backward flow.
- `augment_brushnet_condition_v8` passes these directly as
  `raft_flow_forward_rgb` / `raft_flow_backward_rgb`; no student provider is
  called for conditioning. Previous-clip memory is still built in the same way.
- V8 still performs its original RGB-pixel-to-feature-pixel flow conversion,
  confidence calculation, DCN alignment, fusion, attention, and latent injection.
- **Sampling temporal guidance still uses the original V7 student flow.**
  Replacing that too would change a second component and would not isolate V8.
- CGE, DDIM, seeds, masks, scales, clip/shard selection and metrics are replayed.
- Existing `v7_raft_*` condition metric names are retained for compatibility;
  in an oracle run these describe the teacher tensors. Check provenance.

## Cache contract

V7 training uses `/home/cilab/ndquan/videoInpainting/SFU_STC_flow/teacher_flows_512x512`.
Its metadata specifies clean ProPainter RAFT `raft-things.pth`, 20 iterations,
512x512, RGB normalization to [-1,1]. Files are
`<split>/<class>/<sequence>/<frame0>_<frame1>.npz`, with `teacher_f` and
`teacher_b` of shape [2,H,W], channels [dx,dy] in RGB pixels. Forward is defined
on frame0, backward on frame1. The reader does not normalize or rescale them.
Teacher validity masks are deliberately not substituted for V8 confidence.

Missing/nonadjacent pairs, ambiguous branches, invalid tensors and resolution
mismatches are errors; there is no student fallback. Repeated tail-frame IDs
receive zero motion. Absolute IDs must match the source clean frames, not the
position inside a clip. Never relabel train/valid pairs to fill a test cache.

As inspected on 2026-09-09, the V7 cache has train/valid only. The separate legacy
`examples/brushnet/dataset_test_stc/teacher_flows_512x512` has only 545/945
filename-matching adjacent test_1 pairs and no test_2 pairs. Filename coverage
does not establish image identity; its source metadata must also be verified.
Full oracle evaluation requires a cache generated from the exact test GT frames.

## Run one matching shard (after supplying the complete cache)

```bash
CUDA_VISIBLE_DEVICES=0 /home/cilab/ndquan/envs/guided_diff/bin/python \
  examples/brushnet/STC_encoder_v8_raft_deformable/evaluate_v8_oracle_from_config.py \
  --reference_run_config experiments/eval_v8_joint4000_cge1_temporal1e3_bgonly/test_1/shard-0/run_config.json \
  --teacher_flow_root /path/to/complete/clean_teacher_cache \
  --teacher_flow_split test \
  --output_dir experiments/eval_v8_oracle/test_1/shard-0
```

Use the corresponding reference config for each other shard and test set;
do not reuse shard-0's config for shard-1. Add `--preflight_only` to check
coverage before inference. Aggregate completed shards with the existing
`aggregate_v8_sharded_metrics.py` using `--split_root` and `--num_shards`.

Tests: `test_oracle_teacher_flow.py` covers replay and cache semantics;
`test_v8_raft_deformable.py` verifies exact condition parity with identical
external flow, no provider use, and preservation of predecessor memory.
