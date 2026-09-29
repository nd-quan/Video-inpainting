# RGB-STC V5: Relative Temporal Bias and Cross-Clip Overlap Memory

## Scope

V5 extends the complete V4++ model. It does **not** implement deformable
alignment; that is reserved for V6 after V5 has been trained and evaluated.

```text
degraded RGB + M_BG
        |
spatial encoder
        |
repaired bidirectional flow head
        |
confidence-aware BG-gated adjacent alignment       (V4++)
        |
relative-bias local temporal attention              (V5)
        +
zero-gated predecessor-overlap memory attention     (V5)
        |
zero-conv -> delta_z_BG -> [z_D + delta_z_BG, M_BG]
```

Mask semantics remain unchanged: source mask `0=degraded BG, 255=HQ ROI` is
converted exactly once to internal `M_BG=1, M_ROI=0`.

## 1. Relative temporal position bias

For absolute frame IDs `tau_i` and `tau_j`, local attention uses

```text
S_ij^h = (Q_i^h K_j^h^T / sqrt(C))
         + b_h[clip(tau_j - tau_i, -D, D) + D].
```

- The table shape is `[num_heads, 2D+1]`.
- Default `D=32`, which covers every T=16 local offset and the first
  cross-clip experiment.
- Absolute IDs make the bias invariant to a global index shift and correct for
  shifted tail windows.
- The table is initialized to exact zero, so the local path initially equals
  V4++.

## 2. Cross-clip overlap memory

Every dataset item owns one current clip and, when valid, its exact predecessor
from the same manifest-contiguous run. For T=16/S=12, the clips share four
absolute frame IDs.

The predecessor is encoded under `no_grad`. For each temporal layer, the last
`O=T-S=4` post-layer features and their frame IDs form an explicit detached
`TemporalMemoryState`. Current queries read this memory through a separate
attention branch:

```text
Y = X + A_local(X; tau)
      + tanh(g) * A_memory(X, M_prev; tau, tau_prev),
Y = Y + FFN(Y).
```

The scalar `g` is initialized to zero. Supplying predecessor memory therefore
cannot change the V4++ result before V5 learning begins. Memory keys are enabled
only when their absolute IDs overlap the current clip; a sequence gap or
segment boundary automatically produces no memory contribution.

This first V5 is **paired cross-clip**, not persistent cross-sequence state.
The previous/current pair is self-contained, so random sampling, two-GPU DDP,
and checkpoint resume do not require rank-local caches or sequence ownership.

## 3. Training objective

V5 keeps the V4++ objective; only the conditioning path changes:

```text
L_total = L_diff
        + lambda_flow * L_flow
        + lambda_feature(step) * L_feature.
```

All losses are computed for the current clip only. The predecessor pass only
builds detached temporal memory, so there is no second BrushNet/U-Net diffusion
graph and no moving target across batches.

The real T16/S12 training index currently contains:

- 387 current clips;
- 24 manifest-contiguous run starts with empty memory;
- 363 valid predecessor/current transitions;
- 6 candidate clips rejected because they cross a manifest segment boundary.

The V5 DataLoader uses `drop_last=True`; under two processes this avoids
Accelerate duplicating one item to pad the odd 387-item epoch.

## 4. Initialization and checkpoint contract

Fresh V5 must upgrade the **full** repaired V4++ component:

```text
experiments/train_v4pp_flow_head_stage1_continue_lr2e5/
  checkpoint-5000/stc_flow_model
```

The current `best.json` points to this checkpoint. Loading only `stc_adapter`
would lose the repaired flow head and alignment fusion and is rejected.

The transfer audit permits exactly two new keys per temporal block:

- `attention.relative_position_bias`;
- `cross_clip_gate`.

V5 checkpoints save only the complete `stc_v5_model` for inference/resume.
The legacy nested `stc_adapter` is intentionally not exported because its V2
config cannot reconstruct V5 temporal blocks.

## 5. What to monitor

In addition to V4++ loss/flow/feature metrics, training logs:

- `train/relative_bias_abs_mean`;
- `train/cross_clip_gate_abs_mean`;
- `train/memory_overlap_mean` and `memory_overlap_ratio`;
- `train/predecessor_sample_ratio`.

If both new parameter magnitudes stay exactly zero after several updates, the
new temporal path is not learning. If they grow abruptly while diffusion loss
or spatial metrics deteriorate, lower the LR or the auxiliary loss weights and
inspect the first checkpoints before continuing.

