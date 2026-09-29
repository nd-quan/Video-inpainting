#!/usr/bin/env python
"""Focused tests for the V8-R predicted-clean temporal training loss."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (
    backward_warp_feature,
)
from STC_encoder_v8_rescaled_deformable import temporal_training_loss as temporal


class _Scheduler:
    def __init__(self, alphas=(0.5,), prediction_type="epsilon"):
        self.alphas_cumprod = torch.tensor(alphas, dtype=torch.float32)
        self.config = SimpleNamespace(prediction_type=prediction_type)


class TemporalTrainingLossTests(unittest.TestCase):
    def _compute(
        self,
        predicted_clean,
        flow,
        *,
        bg=None,
        valid=None,
        timesteps=None,
        detach_previous=True,
        weight=1.0,
        warmup_steps=0,
        global_step=0,
        scheduler=None,
    ):
        batch, frames, _, height, width = predicted_clean.shape
        scheduler = scheduler or _Scheduler()
        timesteps = (
            torch.zeros(batch * frames, dtype=torch.long)
            if timesteps is None
            else timesteps
        )
        alpha = scheduler.alphas_cumprod[timesteps].reshape(-1, 1, 1, 1)
        prediction = torch.zeros(
            batch * frames, 4, height, width, dtype=torch.float32, requires_grad=True
        )
        # For epsilon prediction with epsilon_hat=0, this reconstructs the
        # requested predicted-clean latent exactly.
        noisy = predicted_clean.reshape(-1, 4, height, width) * alpha.sqrt()
        bg = bg if bg is not None else torch.ones(batch, frames, 1, height, width)
        valid = (
            valid
            if valid is not None
            else torch.ones(batch, frames - 1, 1, *flow.shape[-2:])
        )
        output = temporal.compute_temporal_training_loss(
            model_prediction=prediction,
            noisy_latents=noisy,
            timesteps=timesteps,
            noise_scheduler=scheduler,
            batch={
                "teacher_flow_backward": flow,
                "teacher_valid_backward": valid,
            },
            stc_output=None,
            bg_mask_sequence=bg,
            num_clips=batch,
            num_frames=frames,
            global_step=global_step,
            weight=weight,
            warmup_steps=warmup_steps,
            charbonnier_eps=1e-3,
            detach_previous=detach_previous,
            snr_gamma=1.0,
        )
        return output, prediction

    def test_zero_motion_is_finite_and_has_zero_warp_gain(self):
        frame = torch.randn(1, 1, 4, 5, 6)
        clean = frame.expand(1, 2, 4, 5, 6).clone()
        flow = torch.zeros(1, 1, 2, 5, 6)
        output, _ = self._compute(clean, flow)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertAlmostEqual(
            float(output.metrics["train/temporal_charbonnier"]), 1e-3, places=6
        )
        self.assertAlmostEqual(
            float(output.metrics["train/temporal_warp_gain"]), 0.0, places=7
        )

    def test_known_translation_improves_over_no_warp(self):
        previous = torch.randn(1, 4, 7, 8)
        flow_flat = torch.zeros(1, 2, 7, 8)
        flow_flat[:, 0] = 1.0
        current, _ = backward_warp_feature(previous, flow_flat)
        clean = torch.stack((previous, current), dim=1)
        output, _ = self._compute(clean, flow_flat[:, None])
        self.assertLess(
            float(output.metrics["train/temporal_relative_mse"]),
            float(output.metrics["train/temporal_no_warp_relative_mse"]),
        )
        self.assertGreater(float(output.metrics["train/temporal_warp_gain"]), 0.0)

    def test_rgb_flow_displacement_is_scaled_to_latent_units(self):
        clean = torch.zeros(1, 2, 4, 64, 64)
        flow = torch.zeros(1, 1, 2, 512, 512)
        flow[:, :, 0] = 16.0
        captured = []
        original = temporal.backward_warp_feature

        def capture(reference, resized_flow, fallback=None):
            captured.append(resized_flow.detach().clone())
            return original(reference, resized_flow, fallback=fallback)

        with patch.object(temporal, "backward_warp_feature", side_effect=capture):
            self._compute(clean, flow)
        self.assertGreaterEqual(len(captured), 1)
        torch.testing.assert_close(
            captured[0][:, 0], torch.full_like(captured[0][:, 0], 2.0)
        )

    def test_roi_exclusion_removes_temporal_gradient(self):
        clean = torch.zeros(1, 2, 4, 4, 4)
        clean[:, 1, :, 2, 2] = 10.0
        flow = torch.zeros(1, 1, 2, 4, 4)
        bg = torch.ones(1, 2, 1, 4, 4)
        bg[:, 1, :, 2, 2] = 0.0
        output, prediction = self._compute(clean, flow, bg=bg)
        output.loss.backward()
        self.assertEqual(float(prediction.grad[:, :, 2, 2].abs().sum()), 0.0)

    def test_source_roi_exclusion_removes_support(self):
        clean = torch.zeros(1, 2, 4, 4, 4)
        flow = torch.zeros(1, 1, 2, 4, 4)
        bg = torch.ones(1, 2, 1, 4, 4)
        bg[:, 0, :, 1, 1] = 0.0
        output, _ = self._compute(clean, flow, bg=bg)
        expected_ratio = 15.0 / 16.0
        self.assertAlmostEqual(
            float(output.metrics["train/temporal_geometric_support_ratio"]),
            expected_ratio,
        )

    def test_out_of_bounds_flow_has_empty_support_and_zero_loss(self):
        clean = torch.randn(1, 2, 4, 4, 4)
        flow = torch.full((1, 1, 2, 4, 4), 100.0)
        output, prediction = self._compute(clean, flow)
        self.assertEqual(
            float(output.metrics["train/temporal_geometric_support_ratio"]), 0.0
        )
        self.assertEqual(float(output.loss), 0.0)
        output.loss.backward()
        self.assertEqual(float(prediction.grad.abs().sum()), 0.0)

    def test_teacher_invalid_region_has_no_temporal_gradient(self):
        clean = torch.zeros(1, 2, 4, 4, 4)
        clean[:, 1, :, 1, 2] = 8.0
        flow = torch.zeros(1, 1, 2, 4, 4)
        valid = torch.ones(1, 1, 1, 4, 4)
        valid[:, :, :, 1, 2] = 0.0
        output, prediction = self._compute(clean, flow, valid=valid)
        output.loss.backward()
        self.assertEqual(float(prediction.grad[:, :, 1, 2].abs().sum()), 0.0)
        self.assertLess(
            float(output.metrics["train/temporal_teacher_valid_support_ratio"]),
            float(output.metrics["train/temporal_geometric_support_ratio"]),
        )

    def test_detached_previous_only_backpropagates_to_current(self):
        clean = torch.zeros(1, 2, 4, 4, 4)
        clean[:, 1] = 1.0
        flow = torch.zeros(1, 1, 2, 4, 4)
        output, prediction = self._compute(clean, flow, detach_previous=True)
        output.loss.backward()
        gradient = prediction.grad.reshape(1, 2, 4, 4, 4)
        self.assertEqual(float(gradient[:, 0].abs().sum()), 0.0)
        self.assertGreater(float(gradient[:, 1].abs().sum()), 0.0)

    def test_independent_frame_timesteps_are_rejected(self):
        clean = torch.zeros(1, 2, 4, 4, 4)
        flow = torch.zeros(1, 1, 2, 4, 4)
        scheduler = _Scheduler(alphas=(0.5, 0.25))
        with self.assertRaisesRegex(ValueError, "shared diffusion timestep"):
            self._compute(
                clean,
                flow,
                timesteps=torch.tensor([0, 1]),
                scheduler=scheduler,
            )

    def test_zero_weight_has_zero_loss_and_gradient(self):
        clean = torch.randn(1, 2, 4, 4, 4)
        flow = torch.zeros(1, 1, 2, 4, 4)
        output, prediction = self._compute(clean, flow, weight=0.0)
        self.assertEqual(float(output.loss), 0.0)
        output.loss.backward()
        self.assertEqual(float(prediction.grad.abs().sum()), 0.0)

    def test_warmup_scales_weighted_loss(self):
        clean = torch.zeros(1, 2, 4, 4, 4)
        clean[:, 1] = 1.0
        flow = torch.zeros(1, 1, 2, 4, 4)
        full, _ = self._compute(clean, flow, warmup_steps=0)
        ramped, _ = self._compute(clean, flow, warmup_steps=10, global_step=0)
        self.assertAlmostEqual(float(ramped.loss), float(full.loss) * 0.1, places=6)

    def test_snr_weight_is_applied_per_clip(self):
        clean = torch.zeros(2, 2, 4, 4, 4)
        clean[:, 1] = 1.0
        flow = torch.zeros(2, 1, 2, 4, 4)
        scheduler = _Scheduler(alphas=(0.5, 0.1))
        output, _ = self._compute(
            clean,
            flow,
            timesteps=torch.tensor([0, 0, 1, 1]),
            scheduler=scheduler,
        )
        expected_mean_snr_weight = 0.5 * (1.0 + (0.1 / 0.9))
        self.assertAlmostEqual(
            float(output.loss)
            / float(output.metrics["train/temporal_charbonnier"]),
            expected_mean_snr_weight,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
