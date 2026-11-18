"""Conditioned inversion prepass for the 14-channel latent warp model."""

import torch


def invert_decoded_clip(scheduler, predict_noise, condition, steps, stop_index):
    """Reuse the sampled decoded latent and fill the five warp channels with zeros."""
    if condition.shape[1] != 5:
        raise ValueError("Expected four decoded latent channels and one mask channel")
    clean = condition[:, :4]
    zero_warp = torch.zeros_like(condition)
    inversion_condition = torch.cat((condition, zero_warp), dim=1)
    return scheduler.invert(
        clean, lambda state, t: predict_noise(state, t, inversion_condition), steps, stop_index
    )
