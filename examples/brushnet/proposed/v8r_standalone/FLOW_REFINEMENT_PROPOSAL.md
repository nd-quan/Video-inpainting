# Flow refinement proposal for V8-R / standalone V8-R

Research and source inspection: 2026-09-28. This is a design proposal, not an implemented or experimentally validated model.

## Recommendation

Start with a separately trained, image-conditioned residual flow refiner after the frozen V7 RAFT student. Supervise it using the existing clean-frame RAFT teacher cache. Keep the original student as the baseline, then test a global-motion branch for coherent BG errors. Use the existing ProPainter VCM completion pipeline as a reuse baseline.

The diagram also permits replacing student distillation with an original frozen RAFT plus refiner. Treat these as distinct experiments: (1) frozen original RAFT + refiner; (2) frozen distilled V7 student + refiner. The second is the least disruptive extension of current V8-R. Neither is guaranteed to outperform the other.

## What the code currently does

- `../STC_encoder_v8_raft_deformable/raft_flow_provider.py`: `FrozenV7RAFTFlowProvider` freezes/evaluates the student and returns adjacent bidirectional flow under `torch.inference_mode()`. Inputs are `[B,T,3,H,W]` in `[-1,1]`; each flow is `[B,T-1,2,H,W]` in RGB-pixel `[dx,dy]` units.
- `model.py`, `StandaloneV8RAdapter._prewarp`: nearest-upsample features by x4, resize flow with displacement scaling, warp, then nearest-downsample. The flow is explicitly detached here.
- `model.py`, `ResidualDCN`: the DCN produces feature-sampling offsets. It does not return a corrected two-channel optical flow field. Feature alignment and optical-flow refinement are different outputs.
- Standalone fusion reliability is `target_BG * warped_source_BG * in_bounds`; its returned `alignment_confidence` is all ones. Legacy V8-R likewise excludes FB confidence from active DCN/fusion reliability, although it retains diagnostics.
- `model.py`, `augment_brushnet_condition`: accepts `external_current_flow` and `external_previous_flow`. These provide a convenient insertion point for a frozen refiner after its separate training.
- `../STC_encoder_v8_raft_deformable/evaluate_v8_raft_deformable_temporal_student.py`, `before_pipeline_call`: separately predicts flow for DDIM guidance. Updating only the TCE condition builder would leave guidance using the old student flow.
- Existing reusable code: `../../../../../../pretrained/ProPainter/README_VCM_FLOW.md`, `core/trainer_vcm_flow.py`, and `model/recurrent_flow_completion.py` under the same ProPainter root. The existing VCM trainer masks/replaces only BG flow failing its FB check. Its reported raw/refined EPE is restricted to that completion region with teacher validity, so it is insufficient by itself to establish improvement over all BG.

Check paths relative to the workspace if moving this document; canonical ProPainter root is `/home/cilab/ndquan/videoInpainting/pretrained/ProPainter`.

## Literature to borrow from

| Source | Useful component | Adaptation proposed here |
|---|---|---|
| [PWC-Net, CVPR 2018](https://arxiv.org/abs/1709.02371) | Feature pyramids, warping, local matching cost volumes | Match around the frozen estimate to obtain evidence for correcting it |
| [IRR, CVPR 2019](https://arxiv.org/abs/1904.05290) | Shared iterative residual refinement; bidirectional/occlusion estimation | Reuse a small update block for 2–3 corrections |
| [RAFT, ECCV 2020](https://arxiv.org/abs/2003.12039) | Recurrent correlation-based flow updates | Architectural context: RAFT already refines internally; the added block must learn useful domain-specific corrections |
| [ProPainter, ICCV 2023](https://arxiv.org/html/2309.03897v1) | Recurrent flow completion with neighboring-flow propagation | Existing pretrained baseline and a later temporal extension |
| [GMA, ICCV 2021](https://arxiv.org/abs/2104.02409) | Long-range aggregation of motion features using image self-similarity | Possible later alternative for textureless/occluded areas |

GMA does not mean estimating a single camera transform. The explicit global-transform branch below is our task-specific proposal, not a claim about GMA's architecture. These papers motivate components; they do not establish performance on this codec-degradation dataset.

## Candidate A: image-conditioned local residual refinement

For one direction, let `F0` map target-grid locations to the source frame. Define `W(X,F)(p) = X(p + F(p))`.

```text
degraded target/source RGB ── shared small image encoder ── feature pyramid
          │                                                   │
          └── frozen RAFT/student ── F0 ── local matching ──────┤
BG masks + FB residual + geometric validity ──────────────────┤
                                                             ▼
                                                  residual update block
                                                             │
                                                        F1 = F0 + ΔF
                                                             │
                                               refresh matching; repeat
                                                             ▼
                                                      refined full flow
```

Recommended prototype choices, to be tuned rather than treated as established settings:

- Pairwise training first; shared weights for forward/backward directions by swapping the image pair and direction-specific inputs.
- Small independent feature encoder with H/8 and H/4 features, approximately 32–64 feature channels. Keeping this encoder separate avoids changing the frozen RAFT implementation.
- At H/8, radius-4 correlation lookup around `p + Fk(p)`; at H/4, radius 2. A radius-4 window has 81 offsets and covers about ±32 input pixels at H/8 around the current estimate. This is a per-step search neighborhood, not a guaranteed recovery range.
- A 64- or 96-channel update network, initially a residual CNN; add ConvGRU state only when testing shared repeated updates. Start with 2 updates at H/8, then one H/4 correction.
- Inputs: target features, source features sampled under current flow, feature difference, local correlations, current flow, warped opposite flow or FB residual, BG masks, and in-bounds flags. Reverse-direction quantities must be sampled onto the same grid before combining them.
- Use both directions from the previous iteration when computing FB cues; update both directions together to avoid accidental direction-order dependence.
- Recompute the local correlation samples and FB cues after each update.
- Zero-initialize the last residual layer. Start with `F_ref = F0 + ΔF`; avoid a hard confidence mask controlling whether updates are possible.
- Upsample the residual with displacement scaling, add it to the original full-resolution `F0`, and optionally use a shallow image-guided full-resolution correction. This preserves details already present in the original estimate better than reconstructing the entire field from a coarse grid.

Why include RGB/features? A flow-only CNN can regularize implausible fields, but an incorrect constant translation can look perfectly plausible. Matching evidence helps determine the direction of correction. Severe information loss can still make the correct motion ambiguous.

Do not multiply the flow vectors themselves by a reliability mask. A zero flow vector is a motion hypothesis, not a representation of uncertainty.

## Candidate B: add global-motion correction

This branch directly targets the coherent BG-motion case in the slide.

1. Estimate an initial BG transform `G0` using robustly weighted coarse correspondences. Start with translation, then ablate a six-parameter affine transform if rotations/zoom matter.
2. Obtain independent matching evidence at low resolution, using a wider search or coarse global correlation. Regress or robustly fit a corrected transform `G1`.
3. Form `ΔG(p) = G1(p) - G0(p)` as a two-channel displacement correction on the target grid.
4. Combine it with local refinement: `F_ref(p) = F0(p) + a(p) ΔG(p) + ΔL(p)`, where `a` is a learned soft spatial gate conditioned on BG, match evidence, and fit quality.

Fit/supervise global parameters with sufficiently supported teacher-valid BG matches during training; use only degraded-image evidence at inference. Initialize the correction to zero. Keep forward/backward geometry consistent; for affine motion, use the inverse transform rather than simply negating displacement vectors.

Merely fitting an affine transform to an entirely wrong coarse flow cannot discover the correct camera motion. The global branch needs image matching or reliable external correspondences. A single transform also cannot represent arbitrary depth-dependent parallax or independently moving BG objects. Preserve the local residual and permit rejection of the global correction.

Use robust teacher BG fits to supervise global parameters only when there is enough spatial support and low fit residual. The direct dense-flow loss remains the main objective.

## Candidate C: reuse ProPainter completion

This is the quickest existing-code baseline. Adapt its cached raw flows to the selected V7 student and match the V8-R input resolution, teacher checkpoint, normalization, and data split. Existing 432x240 original-RAFT caches are not equivalent to 512x512 distilled-student outputs.

The existing hard replacement rule cannot modify flow outside its completion mask. Include an ablation allowing a dense residual update and a later image-feature conditioning branch. Changing the input channel count requires deliberate partial checkpoint loading; a changed residual output interpretation also requires retraining/calibration.

Use 5-frame clips initially if testing temporal propagation. Ensure context and direction conventions match at inference. Assess overlap/clip-boundary disagreement, and avoid averaging neighboring flow fields at fixed image coordinates without alignment.

## Why FB consistency alone is insufficient

For an incorrect constant translation `Ff = c` and `Fb = -c`, their FB cycle residual is zero wherever the warp is in bounds. Thus both can be wrong while passing the consistency test. This is a mathematical counterexample, not a measured diagnosis of the current checkpoints.

Use FB error as one cue alongside matching evidence. A separate confidence head may predict teacher-error-based reliability, but correction strength and reliability of the corrected result are different quantities.

After refinement, recompute visibility and stable BG for downstream use:

`S = B_target * W(B_source, F_ref) * in_bounds(F_ref) * C_ref`.

First retain current TCE fusion behavior while replacing only the flow. Then test adding `C_ref` as a separate ablation. This separates flow accuracy gains from gains due to rejecting unreliable pixels.

## Training objective and schedule

The clean-frame RAFT output is a pseudo-label, not true ground-truth optical flow. Keep clean RGB, teacher flow, and teacher validity on the supervision path only.

For each direction and refinement iteration k:

`L_flow = Σ_k w_k * Σ_p Q(p) [1 + η B(p)] ρ(Fk(p) - F_teacher(p)) / (Σ_p Q(p) [1 + η B(p)] + ε)`.

Here `Q` is fixed teacher/cache validity, `ρ` is a robust Charbonnier penalty, and normalized iteration weights can use the existing gamma=0.8 convention. BG weighting emphasizes the task while retaining nonzero supervision elsewhere. Evaluate both all-region and BG errors regardless of training weighting. Audit what teacher-valid masks exclude, especially occlusions.

Stage 1: use only this flow loss, freeze RAFT, and train the refiner/its image encoder. Compare with the frozen student under identical resolution and masks. Offline coarse-flow caching can remove RAFT inference from training; record checkpoint hash, RAFT iterations, resolution, normalization, and flow convention. Geometric augmentations must transform both teacher and cached coarse flow, including vector signs/scales and crop validity; changes to degraded appearance may require regenerating coarse predictions.

Stage 2: add auxiliary terms separately:

- Clean-RGB warping loss with fixed teacher-derived visible/in-bounds support, robust photometric penalty, and exclusion of true occlusion. Let gradients flow through the refined warp. Clean frames are needed only while training.
- FB cycle loss restricted to trusted visible support. Do not apply indiscriminately across occlusions; zero flow can satisfy cycle consistency.
- Teacher-gradient matching for motion boundaries. Prefer this ablation before strong generic smoothness that can oversmooth independently moving objects.
- Optional confidence loss: predict a detached soft target such as `exp(-EPE(F_ref,F_teacher)/τ)` on teacher-valid pixels. Do not let predicted confidence directly suppress the primary flow-supervision weights.

Inspect actual loss magnitudes before assigning auxiliary weights. Existing V7 weights are a reference, not calibrated weights for the new architecture. A new small refiner could start with AdamW around 1e-4; a pretrained completion network could start around 1e-5. Use held-out validation and early stopping rather than a promised training duration.

Stage 3: freeze the validated refiner, test it with the current TCE checkpoint, then fine-tune TCE on its outputs. Report both the immediate swap and adapted result. Keep downstream diffusion/noise/guidance settings fixed during comparisons. Joint refiner/TCE training is a later experiment that retains flow supervision.

## Integration details

Proposed new files: `flow_refiner.py`, `train_flow_refiner.py`, `evaluate_flow_refiner.py`, and a provider/wrapper shared by temporal training and evaluation. Keep the refiner checkpoint separate from student/TCE checkpoints and save the base-model identity in its config.

Preserve the current full-resolution bidirectional output contract. The wrapper requires degraded RGB plus BG masks; the current provider accepts RGB only, so make the new dependency explicit. Initially inject outputs through `external_current_flow`/`external_previous_flow`. Apply refinement consistently to prior clips used for memory and to DDIM temporal guidance, not just current-clip TCE pre-warp.

During refiner-only training, keep its forward pass outside `inference_mode`/`no_grad`. Coarse RAFT inference outputs should be cloned outside `inference_mode` into ordinary detached tensors before entering operations that save them for backward. Alternatively use a no-grad coarse path rather than inference tensors.

Existing `_prewarp` detaches flow in both V8-R implementations. That is compatible with a separately trained/frozen refiner, but blocks a future gradient path through flow-based pre-warping. Joint training needs an explicit gradient audit, including other detached flow outputs and no-grad blocks; merely setting `requires_grad=True` is insufficient.

Preserve `[dx,dy]`, pixel-unit scaling, mask polarity (`BG=1` internally), and the existing target-to-source sampling convention. Apply motion once: the current post-warp DCN should continue predicting residual offsets only.

## Experiments and decision criteria

First run a teacher-flow substitution diagnostic on the coherent-motion sequences. Clean-teacher flow is an imperfect privileged reference, not a strict upper bound. If it improves flow metrics but barely changes TCE restoration, inspect fusion/temporal modeling/shared-noise behavior before investing in a larger refiner. Standalone preflight currently accepts only student flow, so this diagnostic needs explicit evaluator support; legacy oracle scripts are a starting point.

| Experiment | Question |
|---|---|
| Frozen student | Current baseline |
| Frozen original RAFT + the same refiner | Can the slide's replacement avoid student distillation? |
| Frozen student + existing completion | How much does reuse buy? |
| Frozen student + image-conditioned residual refiner | Does new matching evidence improve correspondence? |
| Previous row + global branch | Does coherent BG motion improve specifically? |
| Best refiner + confidence gating | Does calibrated rejection help beyond flow correction? |
| Best refiner + temporal context | Does neighboring-frame information justify its cost? |
| Clean-teacher flow substitution | How sensitive is current TCE to better correspondences? |

Report EPE against the pseudo-teacher with fixed support, both full-frame and BG, boundary bands, motion bins, and coherent-motion sequence subsets. Separate mean flow bias from spatially varying residual error. Track the fraction of low-FB-error pixels that nevertheless have high teacher EPE. Evaluate all BG as well as any selected completion region.

For TCE, report BG PSNR/SSIM/LPIPS, temporal warp error with a fixed independent evaluator flow/mask, available clip-level video metrics, per-sequence results, and visible artifacts. Evaluating temporal quality using each method's own changing flow/confidence can reward masking out difficult pixels, so also report coverage and fixed-support scores.

Measure added latency and peak memory for the same clip length/resolution. Use independent sequences for held-out generalization where possible; if existing splits use different segments of the same sequence, describe that limitation explicitly. No expected numerical gain or runtime is claimed by this proposal.

Recommended order: teacher-flow diagnostic → existing completion baseline → candidate A → candidate B → confidence/temporal ablations. The main research hypothesis is that image-conditioned residual correction, with a separately tested global-motion prior, is better suited to codec-corrupted coherent motion than hard-mask flow completion alone.
