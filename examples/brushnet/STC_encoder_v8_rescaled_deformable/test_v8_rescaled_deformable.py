#!/usr/bin/env python
"""Unit tests for V8-R rescaled pre-warp and geometric-only support."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from diffusers.models.brushnet_motion_adapter import resize_flow
from STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (
    backward_warp_feature,
)
from STC_encoder_v8_rescaled_deformable.rescaled_raft_deformable_stc_adapter import (
    RescaledPrewarpModulatedDeformableAlignment,
    RescaledRAFTGuidedDeformableBGSTCAdapter,
)


class V8RescaledDeformableTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.model = RescaledRAFTGuidedDeformableBGSTCAdapter(
            hidden_channels=8,
            num_heads=1,
            num_layers=1,
            dropout=0.0,
            flow_max_displacement=(4.0, 4.0),
            relative_position_max_distance=8,
            cross_clip_memory_frames=2,
            deform_hidden_channels=16,
            deform_groups=2,
            deform_residual_max_displacement=1.5,
            rescaled_warp_scale=4,
        ).eval()

    @staticmethod
    def _aligner():
        return RescaledPrewarpModulatedDeformableAlignment(
            channels=1,
            hidden_channels=4,
            kernel_size=3,
            deform_groups=1,
            residual_max_displacement=2.0,
            flow_max_displacement=(4.0, 4.0),
        ).eval()

    def test_zero_residual_dcn_does_not_apply_flow_twice(self):
        aligner = self._aligner()
        prewarped = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
        target = torch.full_like(prewarped, -9.0)
        native_flow = torch.zeros(1, 2, 5, 5)
        native_flow[:, 0] = 2.0
        ones = torch.ones(1, 1, 5, 5)
        valid = torch.ones_like(ones, dtype=torch.bool)
        with torch.no_grad():
            output = aligner.forward_prewarped(
                prewarped, target, native_flow, ones, ones, valid
            )
        # If native_flow were added again as a DCN base offset, this equality
        # would fail.  At initialization DCN is an identity on its input.
        torch.testing.assert_close(output.aligned_source, prewarped, rtol=0, atol=1e-6)
        self.assertEqual(float(output.residual_offset.abs().max()), 0.0)

    def test_scale_one_matches_direct_native_warp(self):
        self.model.register_to_config(rescaled_warp_scale=1)
        source = torch.randn(1, 8, 4, 4)
        target = torch.randn_like(source)
        source_bg = torch.ones(1, 1, 4, 4)
        flow_rgb = torch.zeros(1, 2, 32, 32)
        flow_rgb[:, 0] = 8.0
        with torch.no_grad():
            actual, _, actual_valid = self.model._rescaled_prewarp(
                source, target, source_bg, flow_rgb
            )
            native_flow = resize_flow(flow_rgb, (4, 4))
            expected, expected_valid = backward_warp_feature(
                source, native_flow, fallback=target
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        self.assertTrue(torch.equal(actual_valid, expected_valid))

    def test_reliability_is_geometric_bg_only_without_fb_confidence(self):
        aligner = self._aligner()
        source = torch.randn(1, 1, 4, 4)
        target = torch.randn_like(source)
        flow = torch.zeros(1, 2, 4, 4)
        ones = torch.ones(1, 1, 4, 4)
        valid = torch.ones_like(ones, dtype=torch.bool)
        with torch.no_grad():
            bg = aligner.forward_prewarped(source, target, flow, ones, ones, valid)
            source_roi = aligner.forward_prewarped(
                source, target, flow, torch.zeros_like(ones), ones, valid
            )
            target_roi = aligner.forward_prewarped(
                source, target, flow, ones, torch.zeros_like(ones), valid
            )
            invalid = aligner.forward_prewarped(
                source, target, flow, ones, ones, torch.zeros_like(valid)
            )
        self.assertTrue(torch.equal(bg.reliability, ones))
        self.assertEqual(int(torch.count_nonzero(source_roi.reliability)), 0)
        self.assertEqual(int(torch.count_nonzero(target_roi.reliability)), 0)
        self.assertEqual(int(torch.count_nonzero(invalid.reliability)), 0)

    def test_rgb_flow_is_resized_directly_to_high_grid(self):
        source = torch.randn(1, 8, 4, 4)
        target = torch.randn_like(source)
        source_bg = torch.ones(1, 1, 4, 4)
        flow_rgb = torch.zeros(1, 2, 32, 32)
        flow_rgb[:, 0] = 8.0
        captured = {}
        original = backward_warp_feature

        def capture(feature, flow, fallback=None):
            captured.setdefault("flow", flow.detach().clone())
            return original(feature, flow, fallback=fallback)

        module_name = self.model.__class__.__module__
        with patch(f"{module_name}.backward_warp_feature", capture):
            self.model._rescaled_prewarp(source, target, source_bg, flow_rgb)
        # Native feature is 4x4 and x4 intermediate is 16x16.  An 8-RGB-pixel
        # displacement must therefore become 4 intermediate-grid pixels.
        self.assertEqual(tuple(captured["flow"].shape[-2:]), (16, 16))
        torch.testing.assert_close(
            captured["flow"][:, 0], torch.full_like(captured["flow"][:, 0], 4.0)
        )


if __name__ == "__main__":
    unittest.main()
