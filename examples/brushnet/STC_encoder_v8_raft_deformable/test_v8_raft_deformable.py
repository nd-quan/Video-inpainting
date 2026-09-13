#!/usr/bin/env python
"""Unit tests for V8's V7-flow/DCN geometry and V5 warm-start contract."""

from __future__ import annotations

import sys
import unittest
import tempfile
from unittest.mock import patch
from types import SimpleNamespace
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
    augment_brushnet_condition_v8,
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

    def test_direction_config_roundtrip_and_legacy_default(self):
        self.assertEqual(self.v8.config.deformable_alignment_direction, "bidirectional")
        for direction in ("previous_only", "next_only"):
            self.v8.register_to_config(deformable_alignment_direction=direction)
            with tempfile.TemporaryDirectory() as directory:
                self.v8.save_pretrained(directory)
                restored = RAFTGuidedDeformableBGSTCAdapter.from_pretrained(directory)
                self.assertEqual(restored.config.deformable_alignment_direction, direction)
                for name, weight in self.v8.state_dict().items():
                    self.assertTrue(torch.equal(weight, restored.state_dict()[name]))

    def test_direction_neutralizes_candidate_and_reliability(self):
        spatial = torch.randn(1, 4, 8, 4, 4)
        base = torch.randn_like(spatial)
        flow = torch.zeros(1, 3, 2, 4, 4)
        confidence = torch.ones(1, 3, 1, 4, 4)
        captured = []
        hook = self.v8.deformable_fusion.register_forward_pre_hook(
            lambda module, args: captured.append(args)
        )
        try:
            for direction, candidate_index, reliability_index, output_index in (
                ("previous_only", 2, 4, 4),
                ("next_only", 1, 3, 3),
            ):
                self.v8.register_to_config(deformable_alignment_direction=direction)
                result = self.v8._deformable_spatial_alignment(
                    spatial, base, flow, flow, confidence, self.bg, 1.0
                )
                args = captured[-1]
                self.assertTrue(torch.equal(args[candidate_index], base.flatten(0, 1)))
                self.assertEqual(torch.count_nonzero(args[reliability_index]).item(), 0)
                self.assertEqual(torch.count_nonzero(result[output_index]).item(), 0)
                active_index = 3 if reliability_index == 4 else 4
                self.assertGreater(torch.count_nonzero(args[active_index]).item(), 0)
        finally:
            hook.remove()

    def test_training_loss_only_backpropagates_active_direction(self):
        from STC_encoder_v8_raft_deformable import train_v8_raft_deformable as train
        for direction, active in (("previous_only", 0), ("next_only", 1)):
            terms = [torch.tensor(float(i + 1), requires_grad=True) for i in range(4)]
            result = SimpleNamespace(
                loss=(terms[0] + terms[1]) / 2,
                loss_backward=terms[0], loss_forward=terms[1],
                loss_offset=(terms[2] + terms[3]) / 2,
                loss_offset_backward=terms[2], loss_offset_forward=terms[3],
                valid_backward_ratio=torch.tensor(1.), valid_forward_ratio=torch.tensor(1.),
                reliability_backward_mean=torch.tensor(1.), reliability_forward_mean=torch.tensor(1.),
            )
            output = SimpleNamespace(**{key: None for key in (
                "spatial_features", "deformed_previous_features", "deformed_next_features",
                "residual_offset_backward", "residual_offset_forward", "deform_reliability_backward",
                "deform_reliability_forward", "predicted_flow_forward", "predicted_flow_backward",
            )})
            args = SimpleNamespace(deformable_alignment_direction=direction,
                deform_alignment_charbonnier_eps=0.001, deform_alignment_warmup_steps=0,
                deform_alignment_loss_weight=0.05, deform_offset_loss_weight=0.001)
            with patch.object(train, "compute_deformable_alignment_loss", return_value=result):
                loss = train._build_v8_extra_train_loss(stc_output=output, batch=None,
                    bg_mask_sequence=None, args=args, global_step=0).loss
            loss.backward()
            self.assertAlmostEqual(terms[active].grad.item(), 0.05)
            self.assertAlmostEqual(terms[active + 2].grad.item(), 0.001)
            self.assertIsNone(terms[1 - active].grad)
            self.assertIsNone(terms[3 - active].grad)

    def test_external_cache_bypasses_provider_for_current_and_memory(self):
        flow = SimpleNamespace(forward=self._rgb_flow(2, 1), backward=self._rgb_flow(-2, -1))
        class Provider:
            def __init__(self):
                self.calls = 0
            def predict_sequence(self, rgb):
                self.calls += 1
                return flow
        provider = Provider()
        kwargs = dict(model=self.v8, base_condition_latents=torch.zeros(4, 4, 4, 4),
                      rgb_sequence=self.rgb, bg_mask_sequence=self.bg,
                      frame_ids=self.ids, previous_rgb_sequence=self.rgb,
                      previous_bg_mask_sequence=self.bg, previous_frame_ids=self.ids,
                      previous_valid_mask=torch.ones_like(self.ids, dtype=torch.bool))
        with torch.no_grad():
            expected = augment_brushnet_condition_v8(raft_flow_provider=provider, **kwargs)
            actual = augment_brushnet_condition_v8(raft_flow_provider=None,
                external_current_flow=flow, external_previous_flow=flow, **kwargs)
        self.assertEqual(provider.calls, 2)
        torch.testing.assert_close(expected[0], actual[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1].raft_flow_forward_rgb, flow.forward)
        torch.testing.assert_close(expected[1].memory_overlap_count, actual[1].memory_overlap_count)
        with self.assertRaises(ValueError):
            augment_brushnet_condition_v8(raft_flow_provider=None, external_current_flow=flow, **kwargs)

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
