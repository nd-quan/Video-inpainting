# V8-R: rescaled RAFT pre-warp + native DCNv2 joint fine-tuning

V8-R is a separate experiment variant. It retains the frozen V8 spatial
encoder, the frozen V7 ProPainter-RAFT student, V5 relative temporal attention,
and cross-clip memory. It changes only the motion-alignment geometry and staged
joint optimization. This launcher intentionally rejects a SEA-RAFT checkpoint
so the flow source cannot silently change between experiments.

## Alignment path

For a source/target pair at the STC feature grid:

1. nearest-neighbor upsample the raw source feature and source BG mask by
   `rescaled_warp_scale` (default `4`);
2. resize the original RGB-pixel RAFT flow directly to that high-resolution
   grid, including the required `dx/dy` unit scaling;
3. backward-warp the high-resolution source;
4. nearest-neighbor downsample the warped feature, warped source BG, and
   geometric-valid mask to the native STC feature grid;
5. run native-grid modulated DCNv2 using only a learned bounded residual
   offset; and
6. fuse the bidirectional candidates before temporal attention.

The active reliability is

`target_BG * warped_source_BG * in_bounds`.

Forward/backward flow consistency is still returned as a detached diagnostic,
but it is not an input to the V8-R DCN head, fusion reliability, or alignment
loss. The RAFT flow is not added to the DCN offset after pre-warping, so motion
is applied only once.

## Default staged joint schedule

- optimizer steps `0..499`: train the reset DCN/alignment fusion only;
- from optimizer step `500`: also update the pretrained temporal branch;
- deformable branch LR: `2e-5` with 100-step warmup;
- temporal branch LR: `5e-6` with its own 100-step warmup after unfreezing;
- scheduler: constant after warmup;
- objective: `L_diff + ramp(0.1) * L_deform + 0.0005 * L_offset`;
- feature-alignment loss: disabled for the first experiment.

The default initialization is native V8 joint checkpoint 4000. Its temporal
weights are retained, while the geometry-dependent DCN and deformable fusion
are reset to their neutral zero-initialized state. The default flow pointer is
`experiments/train_v7_raft_student_flow/best.json`, currently resolving to the
validated V7 RAFT checkpoint 4750.

## Launch

Run `run_train_v8_rescaled_joint_t16.sh`. Every important setting is exposed
as an environment variable. Set `RESUME_FROM_CHECKPOINT=latest` for exact
resume in the same output directory; do not change the optimizer or geometry
settings when resuming.
