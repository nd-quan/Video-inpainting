# V8: frozen V7-RAFT-guided deformable STC alignment

## Goal

V6 learned a first-order deformable feature alignment, but its DCN base
offset came from the lightweight STC flow head.  Its measured magnitude was
near zero on the shared validation frames.  V7 separately distills a
ProPainter-RAFT student from clean teacher flow and predicts motion directly
from the degraded RGB pair.  V8 connects those two results without changing
the frozen 2D diffusion/BrushNet backbone.

```text
degraded RGB pair ──> frozen V7 RAFT ──> F^R_RGB [dx,dy]
                                            │ resize + vector scaling
[degraded RGB, M_BG] ──> V5 spatial path ──> S_t, A^V5_t
                                            │
                 raw adjacent S_(t-1), S_(t+1) + F^R_feature
                                            │
                    first-order modulated DCN + BG-to-BG reliability
                                            │
                 zero-init residual fusion into the V5 base A^V5_t
                                            │
                 relative temporal attention + cross-clip overlap memory
                                            │
                                      delta_z_BG
```

`M_BG=1` is the degraded region to restore; `M_BG=0` is the high-quality ROI.
The output remains `z_condition = z^D + M_BG * delta_z_BG`.

## Flow contract

- V7 consumes only the **degraded** RGB pair in `[-1,1]`.
- `F_f^R(t)` is `t -> t+1` defined on frame `t`; `F_b^R(t)` is `t+1 -> t`
  defined on frame `t+1`.
- Both are `[dx,dy]` in RGB-pixel units.
- Before DCN, `resize_flow_sequence` resizes RGB flow to the actual STC grid
  and multiplies `dx,dy` by width/height ratios.  Thus 8 RGB pixels at
  `512x512` becomes exactly 1 feature pixel at `64x64`.
- The DCN converts `[dx,dy]` to torchvision's interleaved `[dy,dx]` internally.

Clean GT and the cached clean teacher flow are never used at inference.

## Identity-safe design

V8 deliberately has two different streams:

1. **Base stream:** the inherited frozen V5 light-flow alignment provides
   `A^V5_t`, preserving the verified V5 condition path.
2. **New DCN stream:** V7 flow is only the base sampling offset for a
   bidirectional DCN candidate.  Its residual offsets are bounded and its
   final fusion projection starts at zero.

Consequently, at a V5-to-V8 warm start the final V8 feature, `delta_z_BG`, and
overlap-memory features are bit-identical to V5 even if the supplied V7 flow
is nonzero.  This is a critical ablation: V8 cannot be credited with a gain
merely because it silently replaced V5 before training.

## Stage-A objective

V7 is frozen, so it has no `L_flow` gradient in V8.  The initial DCN-only run
uses:

\[
L = L_{diff} + \lambda_{deform} L_{deform} + \lambda_{offset} L_{offset}.
\]

`L_deform` aligns normalized raw-source DCN candidates to stop-gradient target
STC features on V7-in-bounds, reliable BG-to-BG support.  `L_offset` penalizes
the actual bounded residual DCN offsets.  The inherited clean-teacher flow
metric can still be logged to audit V7 flow quality, but its coefficient is
fixed to zero because RAFT is frozen.

Default Stage-A settings: `lambda_deform=0.05` with a 500-step linear warmup,
and `lambda_offset=0.001`.  `L_feature` defaults to zero in Stage A because it
does not touch DCN parameters when the STC spatial path is frozen.

## Training and inference dependency

Every V8 checkpoint stores the STC/DCN model as `stc_v8_model`; it does **not**
embed V7 RAFT weights.  The exact V7 `raft_student` component path is recorded
in `metadata.json` and the resume contract.  Evaluation must supply that same
V7 checkpoint.  This makes the external flow dependency visible and prevents
accidentally evaluating a V8 STC model with a different RAFT student.

V7 is run online, once for every local pair in the current clip and, when
cross-clip memory is present, once for the predecessor clip.  This is the
correct first integration test, but it is expensive.  If Stage A is promising,
the next optimization is a versioned cache of frozen V7 degraded-frame flow;
that cache must preserve absolute pair IDs, RGB resolution, V7 checkpoint
identity, and the `[dx,dy]` convention.
