"""Deterministic approximate DDIM inversion on the reverse scheduler grid."""

import torch

from .scheduling_ddim import DDIMScheduler


class LatentWarpDDIMInversionV1(DDIMScheduler):
    @torch.no_grad()
    def invert(self, clean, predict_noise, steps, stop_index):
        self.set_timesteps(steps, device=clean.device)
        grid = self.timesteps.flip(0)
        if not 0 <= stop_index < len(grid):
            raise ValueError("Inversion index outside DDIM grid")
        if self.config.prediction_type != "epsilon":
            raise ValueError("Inversion requires epsilon prediction")
        alphas = self.alphas_cumprod.to(clean.device, torch.float32)
        current_alpha = self.final_alpha_cumprod.to(clean.device, torch.float32)
        state = clean.float()
        timestep_in = grid[0]
        for timestep_out in grid[: stop_index + 1]:
            next_alpha = alphas[timestep_out]
            if not torch.equal(current_alpha, next_alpha):
                epsilon = predict_noise(state.to(clean.dtype), timestep_in).float()
                estimate = (state - (1 - current_alpha).sqrt() * epsilon) / current_alpha.sqrt()
                state = next_alpha.sqrt() * estimate + (1 - next_alpha).sqrt() * epsilon
            current_alpha = next_alpha
            timestep_in = timestep_out
        return state.to(clean.dtype), grid[stop_index]


def epsilon_target(noisy, ground_truth, timesteps, scheduler):
    alphas = scheduler.alphas_cumprod.to(noisy.device, torch.float32)
    alpha = alphas[timesteps].view(-1, 1, 1, 1)
    return (noisy.float() - alpha.sqrt() * ground_truth.float()) / (1 - alpha).sqrt()
