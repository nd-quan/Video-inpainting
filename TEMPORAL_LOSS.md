


Codex Handoff: Implement Temporal Consistency Training Loss for V8-R
1. Goal
Implement an explicit temporal consistency training loss on the predicted-clean diffusion latent for the current V8-R rescaled deformable model.

The first experiment must keep the V8-R architecture unchanged and add only an output-level temporal objective:

L_{diff}
+
L_{deform}
+
L_{offset}
+
L_{temp,weighted}
]

The purpose is to directly optimize local temporal coherence of the diffusion prediction instead of relying only on intermediate feature/deformable alignment losses.

2. Repository scope
Branch:

stc-v8
Main V8-R directory:

examples/brushnet/STC_encoder_v8_rescaled_deformable/
Important files:

examples/brushnet/STC_encoder_v8_rescaled_deformable/train_v8_rescaled_joint.py
examples/brushnet/STC_encoder_v8_rescaled_deformable/rescaled_raft_deformable_stc_adapter.py
examples/brushnet/STC_encoder_v8_rescaled_deformable/run_train_v8_rescaled_joint_t16.sh
examples/brushnet/STC_encoder_v8_rescaled_deformable/TEMPORAL_TRAINING_LOSS_PROPOSAL.md

examples/brushnet/STC_encoder_v3_rgb_flow/train_rgb_stc_flow_shared_noise.py
V8-R wraps the shared V3 trainer. Do not duplicate the complete training loop.

3. Existing behavior that must remain unchanged
Current V8-R path:

degraded RGB + BG mask
    -> frozen V7 ProPainter-RAFT student
    -> V8-R spatial feature
    -> x4 feature upsampling
    -> flow pre-warp
    -> feature downsampling
    -> residual DCNv2 refinement
    -> temporal attention / cross-clip memory
    -> delta_z_BG
    -> frozen BrushNet + frozen U-Net
    -> model_prediction
Current trainable modules remain:

deformable_alignment
deformable_fusion
alignment_fusion
stc_adapter.temporal_blocks
stc_adapter.output_norm
stc_adapter.zero_conv
Keep frozen:

V7 RAFT student
VAE
BrushNet
diffusion U-Net
text encoder
image encoder
IP-Adapter projection
baseline fusion module
Important: frozen BrushNet/U-Net must remain inside the autograd graph. Their parameters are frozen, but their forward pass must not be wrapped in torch.no_grad() because the temporal gradient must reach the V8-R condition path:

L_temp
 -> z0_hat
 -> model_prediction
 -> frozen U-Net / BrushNet operations
 -> V8-R condition
 -> trainable V8-R modules
4. First-experiment scope
Implement only:

predicted-clean latent temporal loss
clean-frame teacher backward flow for temporal correspondence
teacher valid mask for temporal support
previous -> current warping
stable BG geometric support
Charbonnier penalty
detach previous latent
temporal warmup
SNR weighting
TensorBoard diagnostics
Do not implement in this task:

JFFRA / cost-volume flow refinement
new flow predictor
RGB temporal loss
bidirectional temporal loss
cross-clip temporal loss
recursive latent warping
inference scheduler changes
new inference temporal guidance
These are future ablations.

Teacher flow policy for this implementation
For this version, use the clean-frame teacher flow directly for the temporal
loss only. The goal is to make the temporal correspondence used by L_temp
as reliable as possible.

This does not change the V8-R forward path:

degraded RGB -> frozen V7 student RAFT -> V8-R alignment/attention -> diffusion
Teacher flow is used only inside the training loss:

clean GT frames -> teacher backward flow -> warp(previous z0_hat) -> L_temp
Therefore:

student RAFT remains the inference-time motion source inside V8-R

teacher flow is used only to build the temporal target/support during training

there is no teacher-flow dependency at inference

5. Create a dedicated temporal-loss module
Create:

examples/brushnet/STC_encoder_v8_rescaled_deformable/temporal_training_loss.py
Recommended interface:

from dataclasses import dataclass
import torch


@dataclass
class TemporalTrainingLossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


def compute_temporal_training_loss(
    *,
    model_prediction: torch.Tensor,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler,
    batch,
    stc_output,
    bg_mask_sequence: torch.Tensor,
    num_clips: int,
    num_frames: int,
    global_step: int,
    weight: float,
    warmup_steps: int,
    charbonnier_eps: float,
    detach_previous: bool,
    snr_gamma: float | None,
) -> TemporalTrainingLossOutput:
    ...
loss returned by this function should be the already weighted temporal contribution that is added directly to loss_total.

The metrics dictionary must contain both raw and weighted temporal values.

6. Construct the predicted-clean latent
The shared trainer already has:

model_prediction
noisy_latents
timesteps
noise_scheduler
Do not decode through the VAE.

For epsilon prediction:

\frac{
z_t-\sqrt{1-\bar{\alpha}t}\hat{\epsilon}{\theta}
}{
\sqrt{\bar{\alpha}_t}
}
]

For velocity prediction:

\sqrt{\bar{\alpha}_t}z_t
\sqrt{1-\bar{\alpha}t}\hat v{\theta}
]

Use noise_scheduler.alphas_cumprod[timesteps] and broadcast to latent dimensions.

Requirements:

z0_hat.shape == noisy_latents.shape == [B*T, 4, H_lat, W_lat]
z0_hat.requires_grad == True
Do not detach:

model_prediction
z0_hat
current z0 latent
Then reshape:

[B, T, 4, H_lat, W_lat]
Timestep safety
The loss requires the same diffusion timestep for every frame inside one clip.

The shared trainer already defaults to:

share_across_clip = not args.independent_frame_timesteps
For this V8-R temporal-loss experiment, explicitly reject:

--independent_frame_timesteps
Also validate in the loss:

clip_t = timesteps.reshape(B, T)
if not torch.all(clip_t == clip_t[:, :1]):
    raise ValueError(
        "Temporal training loss requires one shared diffusion timestep per clip"
    )
Different clips in the same batch may use different timesteps.

7. Use clean-frame teacher backward flow for temporal correspondence
For the temporal loss, use the clean-frame teacher flow already present in the training batch:

batch["teacher_flow_backward"]
batch["teacher_valid_backward"]
Expected shapes are approximately:

teacher_flow_backward: [B, T-1, 2, H_rgb, W_rgb]
teacher_valid_backward: [B, T-1, 1, H_rgb, W_rgb]
For adjacent frames (k, k+1), the backward teacher flow is defined on frame k+1
coordinates and samples frame k.

W(\hat z_{0,k},F^{b,*}_k)
]

and:

\hat z_{0,k+1}
\tilde z_{0,k\rightarrow k+1}
]

Use only the original previous predicted-clean latent for each pair.

Do not recursively warp an already warped latent.

The student V7 RAFT flow inside stc_output remains part of the V8-R forward path,
but it is not the default correspondence source for L_temp.

8. Resize teacher flow correctly to latent resolution
Preferred source:

batch["teacher_flow_backward"]
Expected shape:

[B, T-1, 2, H_rgb, W_rgb]
Resize directly from RGB flow to z0_hat resolution with the repository's flow-resize utility. The helper must resize both:

spatial resolution
dx/dy displacement units
For 512x512 -> 64x64:

dx_latent = dx_rgb / 8
dy_latent = dy_rgb / 8
Do not use plain F.interpolate(flow, ...) without displacement scaling.

Expected result:

flow_backward_latent:
[B, T-1, 2, H_lat, W_lat]
Teacher flow is frozen supervision geometry and must be detached.

9. Warp previous predicted-clean latent
Reuse the repository's existing differentiable backward warp, preferably:

backward_warp_feature(...)
because it already follows the repository flow sign/coordinate convention and returns an in-bounds mask.

For the first experiment:

previous = z0_hat[:, :-1]
current = z0_hat[:, 1:]

if detach_previous:
    previous = previous.detach()
Flatten pairs:

previous: [B*(T-1), 4, H_lat, W_lat]
current:  [B*(T-1), 4, H_lat, W_lat]
flow_b:   [B*(T-1), 2, H_lat, W_lat]
Then:

warped_previous, valid_bounds = backward_warp_feature(
    previous,
    flow_b,
    fallback=current.detach(),
)
If the helper does not require fallback, omit it. valid_bounds must be used to exclude invalid sampling.

Expected:

warped_previous: [B*(T-1), 4, H_lat, W_lat]
valid_bounds:    [B*(T-1), 1, H_lat, W_lat] bool
10. Build the stable-BG geometric support
Current dataset convention is:

BG = 1
ROI = 0
Resize bg_mask_sequence to latent resolution with nearest-neighbor:

bg_latent:
[B, T, 1, H_lat, W_lat]
Threshold at 0.5.

For pair (k, k+1):

M^{BG}_{k+1}
\cdot
W(M^{BG}_k,F^b_k)
\cdot
V^{bounds}_k
]

Implementation:

previous_bg = bg_latent[:, :-1]
current_bg = bg_latent[:, 1:]

warped_previous_bg, bg_valid = backward_warp_feature(
    previous_bg.reshape(-1, 1, H_lat, W_lat),
    flow_b,
)

warped_previous_bg = (
    warped_previous_bg >= 0.5
).to(current.dtype)

m_geo = (
    current_bg.reshape(-1, 1, H_lat, W_lat)
    * warped_previous_bg
    * valid_bounds.to(current.dtype)
)
All masks and flow must be detached.

11. Apply teacher valid support
For the teacher-flow version, use the clean teacher valid mask already present in the batch:

batch["teacher_valid_backward"]
Resize it to the latent resolution with nearest-neighbor and threshold if needed.

Recommended support for the first experiment:

M^{geo}_k
\cdot
V^{teacher}_k
]

where V^{teacher}_k is the resized backward valid mask from the clean-frame teacher flow.

This is the default reliability mechanism for the teacher-flow temporal loss.
Do not use stc_output.alignment_confidence as the default support in this version.

Detach:

teacher_valid = teacher_valid.detach()
Keep M_geo and the final valid support ratio as separate diagnostics.

12. Temporal residual and Charbonnier loss
Temporal residual:

\hat z_{0,k+1}
W(
\operatorname{sg}(\hat z_{0,k}),
F^b_k
)
]

Charbonnier:

[
\rho(r)=\sqrt{r^2+\epsilon^2}
]

Default:

epsilon = 1e-3
Raw masked temporal loss:

\frac{
\sum_{k,x,c}M_k(x)\rho(r_{k,c}(x))
}{
C\sum_{k,x}M_k(x)+\varepsilon
}
]

where:

C = 4 latent channels
Handle empty support safely.

If support is zero, return a zero scalar connected safely to the current graph, e.g.:

loss_raw = current.sum() * 0.0
Never return NaN/Inf.

13. Log a relative temporal metric
Do not optimize this metric initially, but log it for comparison with current latent warp evaluation.

\frac{
\sum M_k|r_k|2^2
}{
\sum M_k|\hat z{0,k+1}|_2^2+\varepsilon
}
]

Also compute a no-warp baseline on the same support:

\hat z_{0,k+1}
\operatorname{sg}(\hat z_{0,k})
]

Then define:

E_{no_warp}
E_{warp}
]

Positive warp_gain means optical-flow motion compensation improves temporal correspondence.

These metrics must be logging-only.

14. Temporal weight, warmup, and SNR weighting
Add the following V8-R arguments:

--temporal_train_loss_weight
--temporal_train_loss_warmup_steps
--temporal_train_charbonnier_eps
--temporal_train_detach_previous
--temporal_train_no_detach_previous
--temporal_train_snr_gamma
Recommended defaults:

temporal_train_loss_weight = 0.1
temporal_train_loss_warmup_steps = 500
temporal_train_charbonnier_eps = 0.001
temporal_train_detach_previous = True
temporal_train_snr_gamma = 5.0
Direction is fixed to:

previous_only
for the first experiment.

Warmup:

\min
\left(
1,\frac{s+1}{N_{warmup}}
\right)
]

If warmup_steps <= 0, use ramp = 1.

SNR:

\frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}
]

\min
\left(
1,\frac{SNR(t)}{\gamma}
\right)
]

Each clip has one timestep shared across frames. If multiple clips in one batch have different timesteps, apply SNR weights per clip/pair, not from only the first clip.

The effective temporal contribution is conceptually:

\lambda_{temp}
\cdot ramp(s)
\cdot w_{snr}(t)
\cdot L_{temp}
]

Keep raw Charbonnier loss separately for logging.

15. Add a post-prediction loss hook to the shared trainer
Current EXTRA_TRAIN_LOSS_FN does not receive the tensors needed to construct z0_hat.

Modify:

examples/brushnet/STC_encoder_v3_rgb_flow/train_rgb_stc_flow_shared_noise.py
Add a backward-compatible optional hook near the existing hook declarations:

POST_PREDICTION_LOSS_FN: Optional[Callable] = None
Default must remain None, so predecessor training behavior is unchanged.

Call this hook after model_prediction and loss_diff exist and before accelerator.backward(loss):

post_prediction_output = None
loss_post_prediction = loss_diff.new_zeros(())

if POST_PREDICTION_LOSS_FN is not None:
    post_prediction_output = POST_PREDICTION_LOSS_FN(
        model_prediction=model_prediction,
        noisy_latents=noisy_latents,
        timesteps=timesteps,
        noise_scheduler=noise_scheduler,
        batch=batch,
        stc_output=stc_output,
        bg_mask_sequence=bg_mask_sequence,
        num_clips=num_clips,
        num_frames=num_frames,
        args=args,
        global_step=global_step,
    )

    loss_post_prediction = post_prediction_output.loss

    if (
        loss_post_prediction.ndim != 0
        or not loss_post_prediction.is_floating_point()
    ):
        raise ValueError(
            "Post-prediction loss must be a floating scalar"
        )
Change total loss from:

loss = (
    loss_diff
    + args.flow_loss_weight * loss_flow
    + feature_weight * loss_feature
    + loss_extra
)
to:

loss = (
    loss_diff
    + args.flow_loss_weight * loss_flow
    + feature_weight * loss_feature
    + loss_extra
    + loss_post_prediction
)
Do not alter behavior when the hook is unset.

16. Install the hook only in V8-R
In:

examples/brushnet/STC_encoder_v8_rescaled_deformable/train_v8_rescaled_joint.py
import the new loss function and add a wrapper:

def _build_temporal_training_loss(
    *,
    model_prediction,
    noisy_latents,
    timesteps,
    noise_scheduler,
    stc_output,
    bg_mask_sequence,
    num_clips,
    num_frames,
    args,
    global_step,
):
    return compute_temporal_training_loss(
        model_prediction=model_prediction,
        noisy_latents=noisy_latents,
        timesteps=timesteps,
        noise_scheduler=noise_scheduler,
        batch=batch,
        stc_output=stc_output,
        bg_mask_sequence=bg_mask_sequence,
        num_clips=num_clips,
        num_frames=num_frames,
        global_step=global_step,
        weight=args.temporal_train_loss_weight,
        warmup_steps=args.temporal_train_loss_warmup_steps,
        charbonnier_eps=args.temporal_train_charbonnier_eps,
        detach_previous=args.temporal_train_detach_previous,
        snr_gamma=args.temporal_train_snr_gamma,
    )
Inside _install_variant():

trainer.POST_PREDICTION_LOSS_FN = _build_temporal_training_loss
Do not install this hook in V3-V8 predecessor variants.

17. Add CLI validation
In V8-R argument validation:

temporal_train_loss_weight >= 0
temporal_train_loss_warmup_steps >= 0
temporal_train_charbonnier_eps > 0
temporal_train_snr_gamma > 0
independent_frame_timesteps == False
If:

temporal_train_loss_weight == 0
the objective must reduce to the existing V8-R training objective.

18. Update metadata and exact-resume contract
Update both:

_checkpoint_metadata(...)
_resume_contract(...)
in train_v8_rescaled_joint.py.

Record:

temporal_train_loss_type = "charbonnier_predicted_clean_latent"
temporal_train_loss_weight
temporal_train_loss_warmup_steps
temporal_train_charbonnier_eps
temporal_train_detach_previous
temporal_train_direction = "previous_only"
temporal_train_snr_gamma
temporal_train_flow_source = "clean_teacher_backward_flow"
temporal_train_support = "stable_bg_x_teacher_valid"
Update the metadata loss description to include the temporal term.

Exact resume must reject changing temporal-loss settings mid-run.

19. Add a separate run script
Keep the current baseline script reproducible.

Create:

examples/brushnet/STC_encoder_v8_rescaled_deformable/run_train_v8_rescaled_joint_temporal_t16.sh
based on:

run_train_v8_rescaled_joint_t16.sh
Add explicitly:

--temporal_train_loss_weight 0.1
--temporal_train_loss_warmup_steps 500
--temporal_train_charbonnier_eps 0.001
--temporal_train_detach_previous
--temporal_train_snr_gamma 5.0
Do not enable:

--independent_frame_timesteps
Use a new output directory.

20. Required TensorBoard metrics
Log at least:

train/loss_temporal
train/loss_temporal_weighted
train/temporal_effective_weight
train/temporal_charbonnier
train/temporal_relative_mse
train/temporal_no_warp_relative_mse
train/temporal_warp_gain
train/temporal_geometric_support_ratio
train/temporal_teacher_valid_support_ratio
train/temporal_flow_magnitude_teacher_rgb_px
train/temporal_snr_weight
Retain all current V8-R metrics.

If practical, also log at low frequency:

train/temporal_grad_norm
Do not perform a costly extra full backward pass every training step only for this diagnostic.

The essential correctness checks are:

z0_hat.requires_grad == True
weighted temporal loss is non-zero on valid clips
at least one trainable V8-R parameter receives temporal gradient
21. Integrate post-prediction metrics into shared logging
The shared trainer already gathers scalar mappings returned by extra_loss_output.metrics.

Add equivalent handling for:

post_prediction_output.metrics
Use accelerator.gather before converting to Python floats.

Every metric value must be a scalar tensor.

22. Tests
Add focused tests for the following cases.

A. Zero motion
previous == current
flow == 0
BG == 1
confidence == 1
Expected:

temporal residual ~ 0
finite Charbonnier loss
warp_gain ~ 0
B. Known translation
Create a synthetic latent translated by known (dx, dy) and use the corresponding backward flow.

Expected:

warp error < no-warp error
warp_gain > 0
C. Flow displacement scaling
Example:

RGB displacement = 16 px
RGB resolution = 512
latent resolution = 64
expected latent displacement = 2 px
The test must fail if the resized flow still contains 16 px.

D. ROI exclusion
Set source or current region to:

BG = 0
Expected:

M_geo = 0 there
no temporal penalty there
E. Out-of-bounds exclusion
Use flow that samples outside the source.

Expected:

valid_bounds = 0
no temporal penalty there
F. teacher valid
Set teacher valid to zero in a region.

Expected:

geometric support may remain valid
reliable support becomes zero
no temporal gradient from that region
G. Previous detach
With detach_previous=True:

gradient exists through current z0
gradient does not propagate through previous-reference z0
H. Independent timestep rejection
If frames inside one clip use different timesteps:

raise ValueError
I. Zero temporal weight regression
With temporal weight 0, current V8-R behavior should be preserved within numerical tolerance.

23. Runtime checks for the first debug run
Validate once near the start of training:

model_prediction requires grad
z0_hat requires grad
z0_hat shape == [B*T,4,H_lat,W_lat]
flow pair count == T-1
resized flow matches latent spatial resolution
BG convention is BG=1 / ROI=0
flow/mask/confidence are detached
temporal loss is finite
support ratio is non-zero on normal clips
Do not print full tensors.

24. Final implemented objective
The final first-stage temporal training objective should be:

L_{diff}
+
\lambda_{def}(s)L_{def}
+
\lambda_{off}L_{off}
+
\lambda_{temp}(s,t)L_{temp}
]

where:

\hat z_{0,k+1}
W(
\operatorname{sg}(\hat z_{0,k}),
F^b_k
)
]

M^{BG}_{k+1}
\cdot
W(M^{BG}_k,F^b_k)
\cdot
V^{bounds}_k
\cdot
C^{FB}_k
]

and:

\frac{
\sum M_k\sqrt{r_k^2+\epsilon^2}
}{
4\sum M_k+\varepsilon
}
]

r_k is the differentiable motion-compensated temporal residual. It is conceptually similar to the latent warp error used during evaluation, but here it is used to update V8-R parameters.

25. Critical constraints
Do not:

wrap frozen BrushNet/U-Net forward in torch.no_grad()
detach model_prediction
detach current z0_hat
backpropagate into frozen V7 RAFT or into teacher-flow tensors
instantiate another flow network
apply the RAFT displacement twice
use forward flow for previous -> current backward sampling
recursively warp previous latents
include ROI pixels in temporal supervision
penalize out-of-bounds samples
silently allow independent per-frame diffusion timesteps
modify inference DDIM temporal guidance
replace the current V8-R architecture
Preserve:

current V8-R checkpoint loading
DDP behavior
gradient accumulation
mixed precision
exact-resume behavior
current optimizer parameter groups
existing deformable losses
existing logging
existing evaluation scripts
26. First experiment configuration
Use:

representation = predicted-clean latent
flow = clean-frame teacher backward flow
direction = previous -> current
previous latent = detached
support = stable BG x in-bounds x teacher valid
penalty = Charbonnier
epsilon = 1e-3
lambda_temp_max = 0.1
warmup = 500 optimizer steps
SNR gamma = 5.0
inference temporal guidance for isolation test = OFF
Compare:

A. existing V8-R objective
B. same initialization + temporal training loss
Use identical validation sequences and seeds.

Do not combine this first experiment with JFFRA, RGB temporal loss, or other architecture changes.

27. Acceptance criteria
Implementation correctness:

no NaN/Inf
exact resume still works
predecessor variants are unchanged when hook is unset
z0_hat keeps autograd
V7 RAFT stays frozen and teacher-flow tensors remain detached
ROI and invalid samples are excluded
temporal loss reaches V8-R trainable parameters
warmup works
raw and weighted temporal metrics are logged
Research acceptance:

validation motion-compensated temporal error decreases
flicker/temporal metrics improve
valid support does not collapse
large-motion regions do not systematically degrade
spatial quality remains acceptable
no obvious ghosting, dragging, frozen texture, or ROI/BG boundary leakage
28. Deliverables from Codex
After implementation, report:

1. Modified/created files.
2. The new post-prediction hook and where it runs.
3. Exact temporal-loss formula implemented.
4. Tensor shapes of z0_hat, flow, support mask, and residual.
5. Confirmation that BrushNet/U-Net are frozen but differentiable w.r.t. V8-R input.
6. Confirmation that V7 RAFT remains frozen.
7. Test results.
8. Example training command.
9. Any mismatch discovered between this specification and the actual repository.
If the repository contradicts this handoff, do not silently guess. Make the smallest compatible change and report the mismatch.