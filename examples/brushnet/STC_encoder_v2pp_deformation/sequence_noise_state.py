"""Ordered sequence-level inference state for STC-v2++ deformation noise.

Training deliberately resets its recurrent lineage for every shuffled clip.
Inference is different: clips of one source video are processed in temporal
order and overlapping absolute frame IDs reuse the exact same lineage and final
noise tensors.  This module is independent of the diffusion pipeline so the
state, RNG, and overlap contracts can be tested on CPU.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import torch

from STC_encoder_v2pp_deformation.noise_deformation import (
    NoiseDeformationHead,
    backward_warp,
    normalize_lineage_channels,
    variance_preserving_fusion,
)


def _stable_frame_seed(
    base_seed: int,
    sequence_id: str,
    frame_id: int,
    purpose: str,
) -> int:
    """Return a process/order-independent seed for one random-noise purpose."""
    payload = (
        f"{int(base_seed)}:{sequence_id}:{int(frame_id)}:{str(purpose)}"
    ).encode("utf-8")
    digest = hashlib.blake2b(
        payload,
        digest_size=8,
        person=b"STCV2PPNoise",
    ).digest()
    return int.from_bytes(digest, byteorder="little") % (2**63 - 1)


def deterministic_frame_noise(
    *,
    state: "SequenceNoiseState",
    frame_id: int,
    purpose: str,
    device: torch.device,
) -> torch.Tensor:
    """Generate canonical CPU FP32 Gaussian noise, then move it to ``device``.

    Anchor, independent, and fallback streams use different ``purpose`` values
    so their tensors cannot alias or depend on traversal order.  CPU generation
    also keeps a run reproducible when the chosen CUDA device changes.
    """
    if purpose not in {"anchor", "independent", "fallback"}:
        raise ValueError(f"Unknown noise purpose: {purpose!r}")
    generator = torch.Generator(device="cpu").manual_seed(
        _stable_frame_seed(
            state.seed,
            state.sequence_id,
            int(frame_id),
            purpose,
        )
    )
    noise = torch.randn(
        state.frame_shape,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    return noise.to(device=device)


@dataclass
class SequenceNoiseState:
    """Detached FP32 lineage/final-noise cache for one ordered source video."""

    sequence_id: str
    seed: int
    channels: int
    height: int
    width: int
    anchor_frame_id: Optional[int] = None
    lineage_noise: Dict[int, torch.Tensor] = field(default_factory=dict)
    final_noise: Dict[int, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        self.sequence_id = str(self.sequence_id)
        self.seed = int(self.seed)
        self.channels = int(self.channels)
        self.height = int(self.height)
        self.width = int(self.width)
        if not self.sequence_id:
            raise ValueError("sequence_id must be non-empty")
        if min(self.channels, self.height, self.width) < 1:
            raise ValueError("noise frame dimensions must be positive")

    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        return self.channels, self.height, self.width

    def contains(self, frame_id: int) -> bool:
        frame_id = int(frame_id)
        in_lineage = frame_id in self.lineage_noise
        in_final = frame_id in self.final_noise
        if in_lineage != in_final:
            raise RuntimeError(
                f"Incomplete two-cache state for {self.sequence_id}:{frame_id}"
            )
        return in_lineage

    def store(
        self,
        frame_id: int,
        lineage: torch.Tensor,
        final: torch.Tensor,
    ) -> None:
        frame_id = int(frame_id)
        if self.contains(frame_id):
            raise ValueError(
                f"Refusing to overwrite cached frame {self.sequence_id}:{frame_id}"
            )
        expected = self.frame_shape
        if tuple(lineage.shape) != expected or tuple(final.shape) != expected:
            raise ValueError(
                f"Cached tensors must have shape {expected}; got "
                f"lineage={tuple(lineage.shape)}, final={tuple(final.shape)}"
            )
        if lineage.dtype != torch.float32 or final.dtype != torch.float32:
            raise TypeError("Sequence noise caches must remain FP32")
        if not torch.isfinite(lineage).all() or not torch.isfinite(final).all():
            raise FloatingPointError("Cannot cache non-finite sequence noise")
        self.lineage_noise[frame_id] = lineage.detach().clone()
        self.final_noise[frame_id] = final.detach().clone()
        if self.anchor_frame_id is None:
            self.anchor_frame_id = frame_id


@dataclass
class SequenceClipNoiseOutput:
    final_noise: torch.Tensor
    lineage_noise: torch.Tensor
    offsets: torch.Tensor
    valid_masks: torch.Tensor
    generated_frame_ids: Tuple[int, ...]
    reused_frame_ids: Tuple[int, ...]
    transition_target_frame_ids: Tuple[int, ...]
    pre_normalization_mean: torch.Tensor
    pre_normalization_std: torch.Tensor
    post_normalization_mean: torch.Tensor
    post_normalization_std: torch.Tensor
    overlap_max_abs_difference: float


def _empty_transition_tensor(
    reference: torch.Tensor,
    channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    return reference.new_empty((1, 0, channels, height, width))


def build_sequence_deformed_noise(
    *,
    deformation_head: NoiseDeformationHead,
    stc_features: torch.Tensor,
    frame_ids: Sequence[int],
    state: SequenceNoiseState,
    alpha: float,
    warp_scope: str = "full",
    bg_mask: torch.Tensor = None,
    normalization_eps: float = 1e-6,
    offset_mode: str = "learned",
) -> SequenceClipNoiseOutput:
    """Build/retrieve shaped noise for one ordered overlapping inference clip.

    ``stc_features`` must contain exactly one clip.  A new target frame is
    transported only from its immediately preceding absolute frame, which must
    both be present in the current STC context and already have cached lineage.
    Cached overlap frames are returned verbatim and are never warped, normalized,
    or fused a second time.
    """
    if stc_features.ndim != 5 or stc_features.shape[0] != 1:
        raise ValueError("stc_features must have shape [1,T,C,H,W]")
    _, frames, _, height, width = stc_features.shape
    ids = tuple(int(value) for value in frame_ids)
    if len(ids) != frames:
        raise ValueError(
            f"frame_ids length {len(ids)} does not match feature T={frames}"
        )
    if frames < 1:
        raise ValueError("An inference clip must contain at least one frame")
    if any(current != previous + 1 for previous, current in zip(ids[:-1], ids[1:])):
        raise ValueError(f"frame_ids must be contiguous and increasing: {ids}")
    if (height, width) != (state.height, state.width):
        raise ValueError(
            "STC/state spatial mismatch: "
            f"features={(height, width)}, state={(state.height, state.width)}"
        )
    if warp_scope not in {"full", "bg"}:
        raise ValueError("warp_scope must be 'full' or 'bg'")
    if offset_mode not in {"learned", "zero_grid"}:
        raise ValueError("offset_mode must be 'learned' or 'zero_grid'")
    if warp_scope == "bg":
        expected_mask = (1, frames, 1, height, width)
        if bg_mask is None or tuple(bg_mask.shape) != expected_mask:
            raise ValueError(f"bg_mask must have shape {expected_mask}")

    features = stc_features.detach().float()
    device = features.device
    generated = []
    reused = []
    transition_targets = []
    clip_lineage = []
    clip_final = []
    offsets = []
    valid_masks = []
    pre_means = []
    pre_stds = []
    post_means = []
    post_stds = []
    overlap_max = 0.0

    for index, frame_id in enumerate(ids):
        if state.contains(frame_id):
            cached_lineage = state.lineage_noise[frame_id]
            cached_final = state.final_noise[frame_id]
            if cached_lineage.device != device or cached_final.device != device:
                raise ValueError("All sequence-state tensors must stay on one device")
            # Compare the exact values returned for the overlap against their
            # authoritative caches.  This remains zero unless a future edit
            # accidentally reprocesses an overlap tensor.
            overlap_max = max(
                overlap_max,
                float((cached_lineage - state.lineage_noise[frame_id]).abs().max()),
                float((cached_final - state.final_noise[frame_id]).abs().max()),
            )
            lineage = cached_lineage
            final = cached_final
            reused.append(frame_id)
        elif not state.lineage_noise:
            if index != 0:
                raise RuntimeError("The sequence anchor must be the clip's first frame")
            lineage = deterministic_frame_noise(
                state=state,
                frame_id=frame_id,
                purpose="anchor",
                device=device,
            )
            independent = deterministic_frame_noise(
                state=state,
                frame_id=frame_id,
                purpose="independent",
                device=device,
            )
            frame_mask = None if bg_mask is None else bg_mask[:, index]
            final = variance_preserving_fusion(
                lineage.unsqueeze(0),
                independent.unsqueeze(0),
                alpha,
                warp_scope=warp_scope,
                bg_mask=frame_mask,
            )[0]
            state.store(frame_id, lineage, final)
            lineage = state.lineage_noise[frame_id]
            final = state.final_noise[frame_id]
            generated.append(frame_id)
        else:
            if index == 0:
                raise RuntimeError(
                    "A new clip must overlap the cached sequence by at least one "
                    "adjacent reference frame"
                )
            reference_id = ids[index - 1]
            if frame_id != reference_id + 1 or not state.contains(reference_id):
                raise RuntimeError(
                    f"Cannot extend {state.sequence_id} to frame {frame_id}; "
                    f"adjacent cached reference {reference_id} is unavailable"
                )
            if frame_id <= max(state.lineage_noise):
                raise RuntimeError(
                    f"Out-of-order unseen frame {frame_id} for {state.sequence_id}"
                )
            if offset_mode == "learned":
                offset = deformation_head(
                    features[:, index - 1],
                    features[:, index],
                ).float()
            else:
                # Keep the exact backward_warp -> grid_sample path while
                # disabling geometric deformation for the zero-offset ablation.
                offset = features.new_zeros((1, 2, height, width))
            fallback = deterministic_frame_noise(
                state=state,
                frame_id=frame_id,
                purpose="fallback",
                device=device,
            )
            warped, valid = backward_warp(
                state.lineage_noise[reference_id].unsqueeze(0),
                offset,
                fallback=fallback.unsqueeze(0),
                mode="bilinear",
            )
            pre_means.append(warped.mean(dim=(-2, -1)))
            pre_stds.append(warped.std(dim=(-2, -1), unbiased=False))
            lineage_batch = normalize_lineage_channels(
                warped,
                eps=normalization_eps,
            )
            post_means.append(lineage_batch.mean(dim=(-2, -1)))
            post_stds.append(
                lineage_batch.std(dim=(-2, -1), unbiased=False)
            )
            independent = deterministic_frame_noise(
                state=state,
                frame_id=frame_id,
                purpose="independent",
                device=device,
            )
            frame_mask = None if bg_mask is None else bg_mask[:, index]
            final_batch = variance_preserving_fusion(
                lineage_batch,
                independent.unsqueeze(0),
                alpha,
                warp_scope=warp_scope,
                bg_mask=frame_mask,
            )
            state.store(frame_id, lineage_batch[0], final_batch[0])
            lineage = state.lineage_noise[frame_id]
            final = state.final_noise[frame_id]
            generated.append(frame_id)
            transition_targets.append(frame_id)
            offsets.append(offset)
            valid_masks.append(valid)

        clip_lineage.append(lineage)
        clip_final.append(final)

    if offsets:
        offset_tensor = torch.stack(offsets, dim=1)
        valid_tensor = torch.stack(valid_masks, dim=1)
        pre_mean_tensor = torch.stack(pre_means, dim=1)
        pre_std_tensor = torch.stack(pre_stds, dim=1)
        post_mean_tensor = torch.stack(post_means, dim=1)
        post_std_tensor = torch.stack(post_stds, dim=1)
    else:
        offset_tensor = _empty_transition_tensor(
            features, 2, height, width
        )
        valid_tensor = torch.empty(
            (1, 0, 1, height, width),
            device=device,
            dtype=torch.bool,
        )
        stat_shape = (1, 0, state.channels)
        pre_mean_tensor = features.new_empty(stat_shape)
        pre_std_tensor = features.new_empty(stat_shape)
        post_mean_tensor = features.new_empty(stat_shape)
        post_std_tensor = features.new_empty(stat_shape)

    final_tensor = torch.stack(clip_final, dim=0).unsqueeze(0)
    lineage_tensor = torch.stack(clip_lineage, dim=0).unsqueeze(0)
    if not torch.isfinite(final_tensor).all() or not torch.isfinite(lineage_tensor).all():
        raise FloatingPointError("Non-finite sequence noise was produced")
    return SequenceClipNoiseOutput(
        final_noise=final_tensor,
        lineage_noise=lineage_tensor,
        offsets=offset_tensor,
        valid_masks=valid_tensor,
        generated_frame_ids=tuple(generated),
        reused_frame_ids=tuple(reused),
        transition_target_frame_ids=tuple(transition_targets),
        pre_normalization_mean=pre_mean_tensor,
        pre_normalization_std=pre_std_tensor,
        post_normalization_mean=post_mean_tensor,
        post_normalization_std=post_std_tensor,
        overlap_max_abs_difference=float(overlap_max),
    )
