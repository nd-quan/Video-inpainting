# Codex Handoff: Sequence-Level Motion-Aligned Noise Propagation for STC-v2

## 1. Purpose

Implement and evaluate a sequence-level noise initialization mechanism for video restoration in the VCM setting.

The current shared-background-noise baseline correlates frames at identical latent coordinates. The proposed mechanism instead transports one Gaussian noise lineage along spatiotemporal correspondences inferred from STC-v2 features, while the video is still processed as overlapping clips.

The first prototype must not depend on a separately trained, teacher-supervised optical-flow predictor.

## 2. Verified design assumptions

- `STC-v2` produces spatiotemporal features but does **not** contain the explicit flow predictor used in `v3`.
- The diffusion/restoration pipeline processes multiple overlapping clips from one longer sequence.
- The current method already supports shared background noise, but shares it at fixed latent coordinates.
- Background masks and STC features are available or can be made available at the point where initial diffusion noise is constructed.
- Exact class names, file paths, mask polarity, feature scales, and tensor layouts must be verified from the repository before editing.

## 3. Research hypothesis

Let one sequence-level anchor noise be sampled once:

\[
\epsilon_1^{\mathrm{seq}} \sim \mathcal N(0,I).
\]

For each target frame, transport noise from a nearby reference frame using correspondence derived from STC-v2 features:

\[
\epsilon_t = \mathcal T_{r\rightarrow t}(\epsilon_r),
\]

where \(r\) is an overlap or local-reference frame whose noise already descends from \(\epsilon_1^{\mathrm{seq}}\). Consequently, all frame noises retain one sequence-level lineage even though STC-v2 processes short clips.

The key distinction is:

\[
\text{one sequence anchor} + \text{clip-wise local transport}
\neq
\text{independent anchor per clip}.
\]

## 4. Scope of the first implementation

### In scope

- Add a sequence-scoped noise state.
- Reuse exactly the stored noise for overlapping frames.
- Derive local correspondence from frozen STC-v2 features without teacher optical flow.
- Transport background noise only where correspondence is valid.
- Fill newly visible or invalid regions with deterministic fresh Gaussian noise.
- Correct or preserve noise statistics after interpolation/transport.
- Keep the existing shared-noise path available as a baseline through configuration.
- Add unit tests and diagnostic logging.

### Out of scope for the first implementation

- Replacing STC-v2 with v3.
- Teacher-flow generation or optical-flow supervision.
- Jointly fine-tuning the full diffusion U-Net.
- Removing the existing shared-noise baseline.
- Claiming the learned or inferred displacement is physical optical flow.
- Changing ROI/BG semantics without first verifying the repository implementation.

## 5. Recommended development strategy

Implement the idea in two milestones. Do not begin with a frozen random `Conv2D(C, 2)` head, because its output has no reason to represent useful motion.

### Milestone 1 — Training-free STC feature transport

Use STC-v2 feature similarity to estimate local correspondence between a reference frame and a target frame.

For normalized features \(h_r\) and \(h_t\):

\[
S_t(q,p)
=
\frac{\langle h_t(q),h_r(p)\rangle}
{\|h_t(q)\|\,\|h_r(p)\|}.
\]

Search only in a configurable local window around each target location. Convert similarity to either:

- hard correspondence with `argmax`, useful for preserving Gaussian statistics; or
- soft correspondence with `softmax(S/temperature)`, followed by variance correction.

Use the correspondence to sample reference noise at the target locations.

This milestone tests the central hypothesis without adding a trainable flow/offset predictor.

### Milestone 2 — Optional task-driven deformation head

Only if Milestone 1 is promising, add a lightweight head defined relative to a reference frame:

\[
\Delta_{t\leftarrow r}
=
D_{\mathrm{offset}}
\left([h_r,h_t,h_t-h_r]\right).
\]

The head may be trained using restoration/temporal objectives without teacher optical flow. Call its output a `noise deformation field` or `noise transport offset`, not optical flow.

## 6. Sequence-level state contract

Introduce a sequence-scoped state similar to:

```python
@dataclass
class SequenceNoiseState:
    sequence_id: str
    seed: int
    anchor_frame_id: int
    frame_noise: dict[int, torch.Tensor]
```

Required behavior:

1. Initialize the state once at the beginning of a sequence.
2. Sample exactly one Gaussian anchor noise for the first sequence frame.
3. Cache every produced frame noise by absolute frame ID.
4. For a new clip, load cached noise for all overlap frames.
5. Select one cached overlap frame as the local reference for unseen frames.
6. Never resample noise for an already cached frame.
7. Clear the state only when the sequence ends or `sequence_id` changes.
8. Make fresh-noise generation deterministic from `(sequence_seed, frame_id)`.

Example with `clip_length=8`, `stride=6`:

```text
clip 1: frames 1..8    -> anchor noise at frame 1, generate/cache 2..8
clip 2: frames 7..14   -> reuse cached 7 and 8, generate/cache 9..14
clip 3: frames 13..20  -> reuse cached 13 and 14, generate/cache 15..20
```

Frames 7 and 8 in clip 2 must receive exactly the same noise tensors previously produced in clip 1.

## 7. Tensor and module contracts

Codex must record the actual repository shapes before implementation. The expected abstract contracts are:

```python
stc_features: [B, T, C, Hs, Ws]
latent_bg_mask: [B, T, 1, Hl, Wl]
frame_ids: [B, T]  # absolute indices within each sequence
noise: [B, T, Cl, Hl, Wl]
```

Recommended modules/functions:

```python
extract_stc_transport_features(...)
match_local_stc_features(reference_feature, target_feature, ...)
transport_noise(reference_noise, correspondence, ...)
preserve_noise_statistics(warped_noise, valid_mask, ...)
build_sequence_level_noise(clip, frame_ids, sequence_state, ...)
```

Keep feature matching and sequence-state management separate from the diffusion pipeline so they can be unit-tested independently.

## 8. Noise construction

For target frame \(t\), let \(V_t\) be a valid, in-bound, background-to-background transport mask. Construct:

\[
\epsilon_t
=
V_t\odot
\operatorname{Norm}
\left[\mathcal T_{r\rightarrow t}(\epsilon_r)\right]
+
(1-V_t)\odot\eta_t,
\]

where:

- \(\epsilon_r\) is cached reference noise;
- \(\eta_t\sim\mathcal N(0,I)\) is deterministic fresh noise for invalid/new regions;
- `Norm` corrects interpolation-induced changes in mean or variance;
- ROI noise behavior must remain identical to the existing pipeline unless explicitly configured otherwise.

When mixing transported and independent noise, use a variance-preserving parameterization:

\[
\epsilon_t^{\mathrm{mix}}
=
\alpha\epsilon_t^{\mathrm{transport}}
+
\sqrt{1-\alpha^2}\epsilon_t^{\mathrm{ind}}.
\]

Do not use `alpha * transported + (1-alpha) * independent` unless the resulting variance change is intentional and documented.

## 9. Configuration

Add configuration without breaking old checkpoints or commands. Suggested options:

```text
noise_mode:
  independent | shared_bg | stc_feature_transport

sequence_noise_seed: int
transport_reference: first_overlap | last_overlap
transport_match_mode: hard | soft
transport_window_radius: int
transport_temperature: float
transport_mix_strength: float
transport_variance_correction: bool
transport_detach_stc_features: bool
transport_debug_dir: optional path
```

Defaults must preserve the current behavior.

## 10. Repository inspection required before coding

Before editing, Codex must find and report:

1. The STC-v2 encoder class and its exact output tensor(s).
2. The call site where STC-v2 features are consumed by the condition adapter.
3. The existing shared-background-noise construction function.
4. The training and inference entry points that call it.
5. How clips are grouped by sequence and how absolute frame IDs are represented.
6. Clip length, stride, overlap behavior, and batch sampler behavior.
7. BG-mask polarity at RGB and latent resolutions.
8. Whether multiple sequences can coexist in one batch.
9. Where RNG generators/seeds are currently created.
10. Existing tests or evaluation scripts that can verify temporal behavior.

No implementation plan is final until these facts are mapped to concrete files and symbols.

## 11. Safety and compatibility constraints

- Do not silently change the existing shared-noise baseline.
- Do not modify STC-v3 or introduce teacher-flow dependencies in Milestone 1.
- Do not mix state between different `sequence_id` values in the same batch.
- Do not rely on clip order unless the sampler explicitly guarantees sequential clips.
- Training dataloaders that shuffle independent clips cannot use persistent cross-clip state without a sequence-aware sampler.
- Inference may maintain state naturally when clips are processed sequentially.
- Avoid in-place operations that can corrupt cached overlap noise.
- Use detached cached tensors by default during inference.
- If used in training, explicitly define whether gradients may cross clip boundaries; the default should be no cross-clip autograd graph.

## 12. Tests and acceptance criteria

### Unit tests

- Identity correspondence returns the original reference noise.
- An overlap frame returns the exact cached tensor without resampling.
- Re-running a sequence with the same seed produces identical noise.
- Different `sequence_id` values do not share state.
- Invalid/out-of-bound regions use fresh deterministic Gaussian noise.
- Transported noise contains no NaN or Inf.
- Noise mean and standard deviation remain within configured tolerances.
- Mask broadcasting and resolution conversion are correct.

### Integration tests

- Existing `shared_bg` inference still runs unchanged.
- `stc_feature_transport` runs on at least two overlapping clips.
- Logged overlap-noise difference is exactly zero before any later diffusion operation.
- Memory use stays bounded because old cached entries can be released after they can no longer overlap future clips.
- A short dry run saves correspondence, validity-mask, and transported-noise diagnostics.

### Experimental comparison

Use the same checkpoint, prompt, input sequence, seed, and diffusion settings for:

1. independent noise;
2. current shared background noise;
3. STC feature transport with independent clip anchors, diagnostic only;
4. proposed STC feature transport with one sequence-level noise lineage;
5. optional STC-v3/teacher-supervised flow transport as an upper or comparative baseline.

Report separately:

- reconstruction quality;
- temporal consistency within clips;
- temporal consistency at clip boundaries;
- performance as temporal distance from the initial anchor grows;
- transported-noise mean/std and overlap identity;
- runtime and memory overhead.

## 13. Milestone stopping conditions

### Inspection checkpoint

Stop and report before editing if any of these are unknown:

- STC-v2 output shape;
- absolute frame identity across clips;
- sequence ordering guarantee;
- mask polarity;
- actual shared-noise integration point.

### Milestone 1 complete when

- the training-free transport path is config-gated;
- the old shared-noise path is unchanged;
- overlap reuse is exact;
- deterministic and statistics tests pass;
- a two-clip inference dry run completes and saves diagnostics.

### Milestone 2 complete when

- the deformation head is explicitly reference-target conditioned;
- its training objective and frozen/trainable components are documented;
- comparison against Milestone 1 uses matched seeds and settings.

## 14. Suggested `AGENTS.md` addition

Keep repository-wide agent instructions short and point to this document:

```markdown
## STC-v2 sequence-noise work

- Before modifying STC-v2, shared-noise initialization, or clip inference, read `docs/sequence_level_motion_aligned_noise_codex_handoff.md`.
- Preserve the current `shared_bg` behavior as the default baseline.
- Verify tensor shapes, mask polarity, sequence IDs, and clip ordering from code before implementing transport.
- Implement and validate one milestone at a time; do not introduce STC-v3 teacher-flow dependencies into Milestone 1.
```

## 15. Starter prompt for Codex — inspection only

```text
Read AGENTS.md and docs/sequence_level_motion_aligned_noise_codex_handoff.md.

First perform the repository-inspection checkpoint only. Do not edit files yet.

Map the ten inspection questions in Section 10 to exact file paths, classes, functions, call sites, tensor shapes, and mask semantics. Confirm whether clips from one sequence are processed in deterministic temporal order during training and inference. Identify the smallest integration point for a config-gated `stc_feature_transport` noise mode while preserving the current `shared_bg` baseline.

Return:
1. an evidence-backed code map;
2. risks or mismatches between the spec and current code;
3. a file-by-file Milestone 1 plan;
4. exact commands for unit tests and a two-clip inference dry run.

Stop after the plan and wait for approval before implementing.
```

## 16. Starter prompt for Codex — implementation after inspection approval

```text
Implement Milestone 1 from docs/sequence_level_motion_aligned_noise_codex_handoff.md using the approved code map.

Constraints:
- preserve current behavior by default;
- do not touch STC-v3 or add teacher-flow dependencies;
- keep sequence state separate per sequence ID;
- reuse overlap-frame noise exactly;
- use frozen STC-v2 features for local feature matching;
- use deterministic fresh Gaussian noise for invalid/new regions;
- add variance/statistics diagnostics;
- add focused unit and integration tests;
- do not start Milestone 2.

Run the approved checks, review the diff, and report changed files, test results, remaining risks, and the command for a short matched-seed comparison against `shared_bg`.
```

## 17. Naming guidance

Preferred research name:

**Sequence-Level Motion-Aligned Noise Propagation**

Preferred implementation terms:

- `sequence noise anchor`
- `noise lineage`
- `STC feature transport`
- `noise transport correspondence`
- `noise deformation field` for the optional learned head

Avoid calling Milestone 1 correspondence or Milestone 2 offsets optical flow unless they are explicitly supervised and evaluated as optical flow.
