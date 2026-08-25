from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn


BRUSHNET_DIR = Path(__file__).resolve().parent.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v4pp_bg_feature.bg_focused_flow_aligned_stc_adapter import (  # noqa: E402
    BGFocusedFlowAlignedRGBSTCAdapter,
)
from STC_encoder_v4pp_bg_feature.feature_alignment import (  # noqa: E402
    compute_feature_alignment_loss,
)


class V4PPBGFeatureTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(41)
        model = BGFocusedFlowAlignedRGBSTCAdapter(
            hidden_channels=16,
            num_heads=1,
            num_layers=1,
            mlp_ratio=2.0,
            flow_max_displacement=(4.0, 4.0),
        )
        nn.init.normal_(
            model.alignment_fusion.to_residual_and_gate.weight, std=0.02
        )
        nn.init.normal_(
            model.alignment_fusion.to_residual_and_gate.bias, std=0.02
        )
        return model

    def test_bg_fusion_changes_bg_but_roi_is_exact_raw_feature(self):
        model = self.make_model().eval()
        rgb = torch.randn(1, 4, 3, 32, 32)
        mask = torch.zeros(1, 4, 1, 32, 32)
        mask[:, :, :, :, :16] = 1.0
        output = model(rgb, mask)
        difference = output.aligned_spatial_features - output.spatial_features
        feature_mask = torch.zeros(1, 4, 1, 4, 4)
        feature_mask[:, :, :, :, :2] = 1.0
        self.assertEqual(
            torch.count_nonzero(difference * (1.0 - feature_mask)).item(), 0
        )
        self.assertGreater(
            (difference[:, 1:] * feature_mask[:, 1:]).abs().sum().item(), 0.0
        )
        self.assertEqual(torch.count_nonzero(difference[:, :1]).item(), 0)

    def test_all_roi_is_exact_v2_spatial_path(self):
        model = self.make_model().eval()
        rgb = torch.randn(1, 3, 3, 32, 32)
        roi_mask = torch.zeros(1, 3, 1, 32, 32)
        output = model(rgb, roi_mask)
        torch.testing.assert_close(
            output.aligned_spatial_features,
            output.spatial_features,
            rtol=0,
            atol=0,
        )

    def test_feature_loss_reaches_raw_features_and_predicted_flow(self):
        torch.manual_seed(42)
        features = torch.randn(1, 3, 8, 5, 6, requires_grad=True)
        forward = (torch.randn(1, 2, 2, 5, 6) * 0.05).requires_grad_()
        backward = (torch.randn(1, 2, 2, 5, 6) * 0.05).requires_grad_()
        teacher_forward = torch.zeros_like(forward)
        teacher_backward = torch.zeros_like(backward)
        valid = torch.ones(1, 2, 1, 5, 6)
        bg = torch.ones(1, 3, 1, 40, 48)
        confidence = torch.ones(1, 2, 1, 5, 6)
        output = compute_feature_alignment_loss(
            features,
            forward,
            backward,
            teacher_forward,
            teacher_backward,
            bg,
            valid,
            valid,
            confidence,
            region="bg",
        )
        output.loss.backward()
        self.assertGreater(features.grad.abs().sum().item(), 0.0)
        self.assertGreater(forward.grad.abs().sum().item(), 0.0)
        self.assertGreater(backward.grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(output.loss))

    def test_bg_region_with_no_bg_has_exact_zero_loss(self):
        features = torch.randn(1, 2, 4, 4, 4)
        flow = torch.zeros(1, 1, 2, 4, 4)
        valid = torch.ones(1, 1, 1, 4, 4)
        roi_only = torch.zeros(1, 2, 1, 32, 32)
        output = compute_feature_alignment_loss(
            features,
            flow,
            flow.clone(),
            flow.clone(),
            flow.clone(),
            roi_only,
            valid,
            valid,
            torch.ones_like(valid),
            region="bg",
        )
        self.assertEqual(output.loss.item(), 0.0)
        self.assertEqual(output.valid_forward_ratio.item(), 0.0)
        self.assertEqual(output.valid_backward_ratio.item(), 0.0)

    def test_predicted_oob_does_not_receive_current_feature_fallback(self):
        features = torch.ones(1, 2, 4, 4, 4)
        predicted = torch.full((1, 1, 2, 4, 4), 20.0)
        teacher = torch.zeros_like(predicted)
        valid = torch.ones(1, 1, 1, 4, 4)
        bg = torch.ones(1, 2, 1, 32, 32)
        output = compute_feature_alignment_loss(
            features,
            predicted,
            predicted.clone(),
            teacher,
            teacher.clone(),
            bg,
            valid,
            valid,
            torch.ones_like(valid),
            region="bg",
        )
        self.assertGreater(output.loss.item(), 0.1)

    def test_save_load_round_trip(self):
        model = self.make_model().eval()
        rgb = torch.randn(1, 3, 3, 32, 32)
        mask = torch.randint(0, 2, (1, 3, 1, 32, 32)).float()
        expected = model(rgb, mask)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            restored = BGFocusedFlowAlignedRGBSTCAdapter.from_pretrained(
                directory
            ).eval()
            actual = restored(rgb, mask)
        torch.testing.assert_close(
            actual.delta_bg, expected.delta_bg, rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual.aligned_spatial_features,
            expected.aligned_spatial_features,
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()

