#!/usr/bin/env python

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (
    backward_warp_feature,
)
from STC_encoder_v5_relative_crossclip.relative_crossclip_stc_adapter import (
    RelativeCrossClipBGSTCAdapter,
)
from STC_encoder_v6_flow_deformable.deformable_alignment_loss import (
    compute_deformable_alignment_loss,
)
from STC_encoder_v6_flow_deformable.flow_guided_deformable_stc_adapter import (
    FlowGuidedDeformableBGSTCAdapter,
    FlowGuidedModulatedDeformableAlignment,
    flow_xy_to_dcn_offset,
)


class V6FlowGuidedDeformableTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.v5 = RelativeCrossClipBGSTCAdapter(
            hidden_channels=8,
            num_heads=1,
            num_layers=1,
            dropout=0.0,
            flow_max_displacement=(2.0, 2.0),
            relative_position_max_distance=8,
            cross_clip_memory_frames=2,
        ).eval()
        self.v6 = FlowGuidedDeformableBGSTCAdapter(
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
        transfer = self.v6.load_state_dict(self.v5.state_dict(), strict=False)
        self.assertFalse(transfer.unexpected_keys)
        self.assertTrue(transfer.missing_keys)
        self.assertTrue(
            all(
                name.startswith(("deformable_alignment.", "deformable_fusion."))
                for name in transfer.missing_keys
            )
        )
        self.previous_rgb = torch.randn(1, 4, 3, 32, 32)
        self.current_rgb = torch.randn(1, 4, 3, 32, 32)
        self.bg = torch.ones(1, 4, 1, 32, 32)
        self.previous_ids = torch.tensor([[0, 1, 2, 3]])
        self.current_ids = torch.tensor([[2, 3, 4, 5]])

    def test_flow_xy_is_repeated_as_interleaved_yx(self):
        flow = torch.zeros(1, 2, 2, 2)
        flow[:, 0] = 3.0
        flow[:, 1] = 7.0
        offset = flow_xy_to_dcn_offset(flow, deform_groups=2, kernel_size=3)
        self.assertEqual(tuple(offset.shape), (1, 36, 2, 2))
        torch.testing.assert_close(offset[:, 0::2], torch.full_like(offset[:, 0::2], 7))
        torch.testing.assert_close(offset[:, 1::2], torch.full_like(offset[:, 1::2], 3))

    def test_zero_residual_dcn_matches_one_base_flow_warp(self):
        aligner = FlowGuidedModulatedDeformableAlignment(
            channels=1,
            hidden_channels=4,
            kernel_size=3,
            deform_groups=1,
            residual_max_displacement=2.0,
            flow_max_displacement=(4.0, 4.0),
        ).eval()
        source = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
        target = torch.full_like(source, -9.0)
        flow = torch.zeros(1, 2, 5, 5)
        flow[:, 0] = 1.0
        ones = torch.ones(1, 1, 5, 5)
        with torch.no_grad():
            output = aligner(source, target, flow, ones, ones, ones)
            expected, _ = backward_warp_feature(source, flow, fallback=target)
        torch.testing.assert_close(
            output.aligned_source, expected, rtol=0, atol=1e-6
        )
        self.assertEqual(float(output.residual_offset.abs().max()), 0.0)
        torch.testing.assert_close(
            output.modulation_mask, torch.full_like(output.modulation_mask, 0.5)
        )

    def test_zero_initialized_v6_is_exact_v5_without_memory(self):
        with torch.no_grad():
            source = self.v5(
                self.current_rgb, self.bg, output_size=(4, 4), frame_ids=self.current_ids
            )
            v6 = self.v6(
                self.current_rgb, self.bg, output_size=(4, 4), frame_ids=self.current_ids
            )
        torch.testing.assert_close(v6.features, source.features, rtol=0, atol=0)
        torch.testing.assert_close(v6.delta_bg, source.delta_bg, rtol=0, atol=0)
        torch.testing.assert_close(
            v6.aligned_spatial_features,
            source.aligned_spatial_features,
            rtol=0,
            atol=0,
        )

    def test_zero_initialized_v6_is_exact_v5_with_overlap_memory(self):
        with torch.no_grad():
            previous_v5 = self.v5(
                self.previous_rgb,
                self.bg,
                output_size=(4, 4),
                frame_ids=self.previous_ids,
            )
            previous_v6 = self.v6(
                self.previous_rgb,
                self.bg,
                output_size=(4, 4),
                frame_ids=self.previous_ids,
            )
            source = self.v5(
                self.current_rgb,
                self.bg,
                output_size=(4, 4),
                frame_ids=self.current_ids,
                temporal_memory=previous_v5.temporal_memory,
            )
            v6 = self.v6(
                self.current_rgb,
                self.bg,
                output_size=(4, 4),
                frame_ids=self.current_ids,
                temporal_memory=previous_v6.temporal_memory,
            )
        self.assertEqual(int(v6.memory_overlap_count.item()), 2)
        torch.testing.assert_close(v6.features, source.features, rtol=0, atol=0)

    def test_deformation_is_first_order_not_recursive(self):
        changed = self.current_rgb.clone()
        changed[:, 0] += 10.0
        with torch.no_grad():
            first = self.v6(
                self.current_rgb, self.bg, output_size=(4, 4), frame_ids=self.current_ids
            )
            second = self.v6(
                changed, self.bg, output_size=(4, 4), frame_ids=self.current_ids
            )
        # Candidate for target frame 2 uses only raw frames 1 and 2. Changing
        # frame 0 may alter pair 0, but cannot propagate through pair 1.
        torch.testing.assert_close(
            first.deformed_previous_features[:, 1],
            second.deformed_previous_features[:, 1],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            first.deformed_next_features[:, 2],
            second.deformed_next_features[:, 2],
            rtol=0,
            atol=0,
        )

    def test_roi_and_source_support_disable_deform_contribution(self):
        aligner = self.v6.deformable_alignment.eval()
        source = torch.randn(1, 8, 4, 4)
        target = torch.randn_like(source)
        flow = torch.zeros(1, 2, 4, 4)
        confidence = torch.ones(1, 1, 4, 4)
        ones = torch.ones_like(confidence)
        zeros = torch.zeros_like(confidence)
        with torch.no_grad():
            target_roi = aligner(source, target, flow, confidence, ones, zeros)
            source_roi = aligner(source, target, flow, confidence, zeros, ones)
            bg_to_bg = aligner(source, target, flow, confidence, ones, ones)
        self.assertEqual(int(torch.count_nonzero(target_roi.reliability)), 0)
        self.assertEqual(int(torch.count_nonzero(source_roi.reliability)), 0)
        self.assertGreater(int(torch.count_nonzero(bg_to_bg.reliability)), 0)

    def test_residual_offsets_are_bounded_and_masks_are_valid(self):
        head = self.v6.deformable_alignment.offset_mask_head[-1]
        with torch.no_grad():
            head.weight.fill_(100.0)
            head.bias.fill_(100.0)
            output = self.v6(
                self.current_rgb, self.bg, output_size=(4, 4), frame_ids=self.current_ids
            )
        bound = float(self.v6.config.deform_residual_max_displacement)
        self.assertLessEqual(float(output.residual_offset_backward.abs().max()), bound)
        self.assertTrue(torch.isfinite(output.modulation_mask_backward).all())
        self.assertGreaterEqual(float(output.modulation_mask_backward.min()), 0.0)
        self.assertLessEqual(float(output.modulation_mask_backward.max()), 1.0)

    def test_deform_loss_reaches_dcn_offset_mask_and_fusion(self):
        self.v6.train()
        output = self.v6(
            self.current_rgb, self.bg, output_size=(4, 4), frame_ids=self.current_ids
        )
        batch, pairs, _, height, width = output.predicted_flow_forward.shape
        teacher_forward = torch.zeros(batch, pairs, 2, height, width)
        teacher_backward = torch.zeros_like(teacher_forward)
        valid = torch.ones(batch, pairs, 1, height, width)
        loss_output = compute_deformable_alignment_loss(
            spatial_features=output.spatial_features,
            deformed_previous=output.deformed_previous_features,
            deformed_next=output.deformed_next_features,
            residual_offset_backward=output.residual_offset_backward,
            residual_offset_forward=output.residual_offset_forward,
            reliability_backward=output.deform_reliability_backward,
            reliability_forward=output.deform_reliability_forward,
            teacher_forward=teacher_forward,
            teacher_backward=teacher_backward,
            valid_forward=valid,
            valid_backward=valid,
        )
        loss = (
            loss_output.loss
            + 1e-3 * loss_output.loss_offset
            + output.features.square().mean()
        )
        loss.backward()
        final_head = self.v6.deformable_alignment.offset_mask_head[-1]
        self.assertIsNotNone(final_head.weight.grad)
        self.assertGreater(float(final_head.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(self.v6.deformable_alignment.weight.grad)
        self.assertGreater(
            float(self.v6.deformable_alignment.weight.grad.abs().sum()), 0.0
        )
        fusion = self.v6.deformable_fusion.to_residual_and_gate
        self.assertIsNotNone(fusion.weight.grad)
        self.assertGreater(float(fusion.weight.grad.abs().sum()), 0.0)

    def test_t1_shape_and_all_roi_identity(self):
        rgb = self.current_rgb[:, :1]
        roi = torch.zeros(1, 1, 1, 32, 32)
        with torch.no_grad():
            output = self.v6(
                rgb, roi, output_size=(4, 4), frame_ids=torch.tensor([[9]])
            )
        self.assertEqual(tuple(output.features.shape), (1, 1, 8, 4, 4))
        self.assertEqual(tuple(output.deformed_previous_features.shape), (1, 0, 8, 4, 4))
        self.assertEqual(int(torch.count_nonzero(output.delta_bg)), 0)
        torch.testing.assert_close(
            output.aligned_spatial_features,
            output.base_aligned_spatial_features,
            rtol=0,
            atol=0,
        )

    def test_full_component_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            self.v6.save_pretrained(directory, safe_serialization=True)
            loaded = FlowGuidedDeformableBGSTCAdapter.from_pretrained(directory)
            self.assertEqual(int(loaded.config.deform_groups), 2)
            self.assertEqual(float(loaded.config.deform_residual_max_displacement), 1.5)
            self.assertEqual(set(self.v6.state_dict()), set(loaded.state_dict()))


if __name__ == "__main__":
    unittest.main()
