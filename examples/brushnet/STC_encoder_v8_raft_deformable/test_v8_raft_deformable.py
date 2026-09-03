#!/usr/bin/env python
"""Unit tests for V8's V7-flow/DCN geometry and V5 warm-start contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v5_relative_crossclip.relative_crossclip_stc_adapter import (
    RelativeCrossClipBGSTCAdapter,
)
from STC_encoder_v6_flow_deformable.deformable_alignment_loss import (
    compute_deformable_alignment_loss,
)
from STC_encoder_v8_raft_deformable.raft_guided_deformable_stc_adapter import (
    RAFTGuidedDeformableBGSTCAdapter,
)


class V8RAFTDeformableTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.v5 = RelativeCrossClipBGSTCAdapter(
            hidden_channels=8,
            num_heads=1,
            num_layers=1,
            dropout=0.0,
            flow_max_displacement=(2.0, 2.0),
            relative_position_max_distance=8,
            cross_clip_memory_frames=2,
        ).eval()
        self.v8 = RAFTGuidedDeformableBGSTCAdapter(
            hidden_channels=8,
            num_heads=1,
            num_layers=1,
            dropout=0.0,
            flow_max_displacement=(2.0, 2.0),
            relative_position_max_distance=8,
            cross_clip_memory_frames=2,
            deform_hidden_channels=16,
            deform_groups=2,
            deform_residual_max_displacement=1.5,
        ).eval()
        transfer = self.v8.load_state_dict(self.v5.state_dict(), strict=False)
        self.assertFalse(transfer.unexpected_keys)
        self.assertTrue(transfer.missing_keys)
        self.assertTrue(all(name.startswith(("deformable_alignment.", "deformable_fusion.")) for name in transfer.missing_keys))
        self.rgb = torch.randn(1, 4, 3, 32, 32)
        self.bg = torch.ones(1, 4, 1, 32, 32)
        self.ids = torch.tensor([[10, 11, 12, 13]])

    def _rgb_flow(self, dx=0.0, dy=0.0):
        flow = torch.zeros(1, 3, 2, 32, 32)
        flow[:, :, 0] = float(dx)
        flow[:, :, 1] = float(dy)
        return flow

    def test_zero_initialized_v8_is_exact_v5_even_with_external_flow(self):
        forward = self._rgb_flow(dx=5.0, dy=-3.0)
        backward = self._rgb_flow(dx=-5.0, dy=3.0)
        with torch.no_grad():
            v5_output = self.v5(self.rgb, self.bg, output_size=(4, 4), frame_ids=self.ids)
            v8_output = self.v8(
                self.rgb,
                self.bg,
                output_size=(4, 4),
                frame_ids=self.ids,
                raft_flow_forward_rgb=forward,
                raft_flow_backward_rgb=backward,
            )
        torch.testing.assert_close(v8_output.features, v5_output.features, rtol=0, atol=0)
        torch.testing.assert_close(v8_output.delta_bg, v5_output.delta_bg, rtol=0, atol=0)
        torch.testing.assert_close(v8_output.aligned_spatial_features, v5_output.aligned_spatial_features, rtol=0, atol=0)

    def test_rgb_flow_is_resized_with_displacement_scaling(self):
        forward = self._rgb_flow(dx=8.0, dy=16.0)
        backward = self._rgb_flow(dx=-8.0, dy=-16.0)
        with torch.no_grad():
            output = self.v8(
                self.rgb,
                self.bg,
                output_size=(4, 4),
                frame_ids=self.ids,
                raft_flow_forward_rgb=forward,
                raft_flow_backward_rgb=backward,
            )
        torch.testing.assert_close(
            output.predicted_flow_forward[:, :, 0],
            torch.ones_like(output.predicted_flow_forward[:, :, 0]),
        )
        torch.testing.assert_close(
            output.predicted_flow_forward[:, :, 1],
            torch.full_like(output.predicted_flow_forward[:, :, 1], 2.0),
        )
        torch.testing.assert_close(
            output.predicted_flow_backward[:, :, 0],
            -torch.ones_like(output.predicted_flow_backward[:, :, 0]),
        )
        torch.testing.assert_close(
            output.predicted_flow_backward[:, :, 1],
            torch.full_like(output.predicted_flow_backward[:, :, 1], -2.0),
        )

    def test_external_flow_is_required(self):
        with self.assertRaisesRegex(ValueError, "requires frozen V7"):
            self.v8(self.rgb, self.bg, output_size=(4, 4), frame_ids=self.ids)

    def test_deform_loss_reaches_dcn_parameters_with_frozen_external_flow(self):
        self.v8.train()
        output = self.v8(
            self.rgb,
            self.bg,
            output_size=(4, 4),
            frame_ids=self.ids,
            raft_flow_forward_rgb=self._rgb_flow(dx=1.0),
            raft_flow_backward_rgb=self._rgb_flow(dx=-1.0),
        )
        loss_output = compute_deformable_alignment_loss(
            spatial_features=output.spatial_features,
            deformed_previous=output.deformed_previous_features,
            deformed_next=output.deformed_next_features,
            residual_offset_backward=output.residual_offset_backward,
            residual_offset_forward=output.residual_offset_forward,
            reliability_backward=output.deform_reliability_backward,
            reliability_forward=output.deform_reliability_forward,
            teacher_forward=output.predicted_flow_forward,
            teacher_backward=output.predicted_flow_backward,
        )
        (loss_output.loss + 1e-3 * loss_output.loss_offset).backward()
        final_head = self.v8.deformable_alignment.offset_mask_head[-1]
        self.assertIsNotNone(final_head.weight.grad)
        self.assertGreater(float(final_head.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(self.v8.deformable_alignment.weight.grad)
        self.assertGreater(float(self.v8.deformable_alignment.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
