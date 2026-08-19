from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn


BRUSHNET_DIR = Path(__file__).resolve().parent.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v3_rgb_flow.flow_supervision import (  # noqa: E402
    compute_teacher_flow_loss,
)
from STC_encoder_v4_flow_aligned.flow_aligned_stc_adapter import (  # noqa: E402
    FlowAlignedRGBSTCAdapter,
    augment_brushnet_condition,
    backward_warp_feature,
)


class FlowAlignedRGBSTCTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(23)
        return FlowAlignedRGBSTCAdapter(
            hidden_channels=16,
            num_heads=1,
            num_layers=1,
            mlp_ratio=2.0,
            flow_max_displacement=(4.0, 3.0),
        )

    def test_t16_shapes_and_finite(self):
        model = self.make_model().eval()
        rgb = torch.randn(1, 16, 3, 32, 40)
        mask = torch.ones(1, 16, 1, 32, 40)
        base = torch.randn(16, 4, 4, 5)
        condition, output, augmented = augment_brushnet_condition(
            model, base, rgb, mask
        )
        self.assertEqual(tuple(output.features.shape), (1, 16, 16, 4, 5))
        self.assertEqual(tuple(output.spatial_features.shape), (1, 16, 16, 4, 5))
        self.assertEqual(tuple(output.predicted_flow_forward.shape), (1, 15, 2, 4, 5))
        self.assertEqual(tuple(output.predicted_flow_backward.shape), (1, 15, 2, 4, 5))
        self.assertEqual(tuple(output.alignment_confidence.shape), (1, 15, 1, 4, 5))
        self.assertEqual(tuple(condition.shape), (16, 5, 4, 5))
        self.assertTrue(torch.isfinite(condition).all())
        self.assertTrue(torch.equal(augmented.flatten(0, 1), base))

    def test_zero_init_is_exact_v2_path_at_t16(self):
        model = self.make_model().eval()
        torch.manual_seed(24)
        nn.init.normal_(model.stc_adapter.zero_conv.weight, std=0.01)
        nn.init.normal_(model.stc_adapter.zero_conv.bias, std=0.01)
        original_v2 = copy.deepcopy(model.stc_adapter).eval()
        rgb = torch.randn(1, 16, 3, 32, 32)
        mask = torch.randint(0, 2, (1, 16, 1, 32, 32)).float()
        expected = original_v2(rgb, mask, return_dict=True)
        actual = model(rgb, mask, return_dict=True)
        torch.testing.assert_close(actual.features, expected.features, rtol=0, atol=0)
        torch.testing.assert_close(actual.delta_bg, expected.delta_bg, rtol=0, atol=0)
        torch.testing.assert_close(
            actual.aligned_spatial_features,
            actual.spatial_features,
            rtol=0,
            atol=0,
        )

    def test_backward_warp_direction(self):
        reference = torch.arange(5, dtype=torch.float32).reshape(1, 1, 1, 5)
        # At query x, sample reference x+1.
        flow = torch.zeros(1, 2, 1, 5)
        flow[:, 0] = 1.0
        fallback = torch.full_like(reference, -1.0)
        warped, valid = backward_warp_feature(reference, flow, fallback=fallback)
        torch.testing.assert_close(
            warped,
            torch.tensor([[[[1.0, 2.0, 3.0, 4.0, -1.0]]]]),
            rtol=0,
            atol=1e-6,
        )
        self.assertEqual(valid.flatten().tolist(), [True, True, True, True, False])

    def test_flow_loss_and_diffusion_path_reach_expected_modules(self):
        model = self.make_model()
        nn.init.normal_(model.stc_adapter.zero_conv.weight, std=0.01)
        rgb = torch.randn(1, 4, 3, 32, 32)
        mask = torch.ones(1, 4, 1, 32, 32)
        output = model(rgb, mask)

        teacher_f = torch.randn_like(output.predicted_flow_forward) * 0.1
        teacher_b = torch.randn_like(output.predicted_flow_backward) * 0.1
        valid = torch.ones(1, 3, 1, 4, 4)
        flow_loss = compute_teacher_flow_loss(
            output.predicted_flow_forward,
            output.predicted_flow_backward,
            teacher_f,
            teacher_b,
            mask,
            valid,
            valid,
            region="all",
        ).loss
        flow_loss.backward(retain_graph=True)
        self.assertGreater(
            model.flow_head.network[-1].weight.grad.abs().sum().item(), 0.0
        )

        model.zero_grad(set_to_none=True)
        output.delta_bg.square().mean().backward()
        fusion_projection = model.alignment_fusion.to_residual_and_gate.weight
        self.assertIsNotNone(fusion_projection.grad)
        self.assertGreater(fusion_projection.grad.abs().sum().item(), 0.0)

    def test_nonzero_fusion_is_not_recurrently_warped(self):
        class UnitXFlow(nn.Module):
            def forward(self, reference, query):
                result = reference.new_zeros(reference.shape[0], 2, *reference.shape[-2:])
                result[:, 0] = 0.25
                return result

        model = self.make_model().eval()
        model.flow_head = UnitXFlow()
        nn.init.normal_(model.alignment_fusion.to_residual_and_gate.weight, std=0.01)
        spatial = torch.randn(1, 5, 16, 4, 4)
        forward, backward = model._decode_bidirectional_flow(spatial)
        aligned, _ = model._align_spatial_features(spatial, forward, backward)
        # Frame zero is the only explicit local reference. Every later fusion
        # consumes spatial[:, t-1], never aligned[:, t-1].
        self.assertTrue(torch.equal(aligned[:, 0], spatial[:, 0]))
        altered = (aligned[:, 1:] - spatial[:, 1:]).abs().sum(dim=(2, 3, 4))
        self.assertTrue(torch.all(altered > 0))

    def test_save_load_round_trip(self):
        model = self.make_model().eval()
        nn.init.normal_(model.flow_head.network[-1].weight, std=0.01)
        nn.init.normal_(model.alignment_fusion.to_residual_and_gate.weight, std=0.01)
        rgb = torch.randn(1, 3, 3, 32, 32)
        mask = torch.ones(1, 3, 1, 32, 32)
        expected = model(rgb, mask)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            restored = FlowAlignedRGBSTCAdapter.from_pretrained(directory).eval()
            actual = restored(rgb, mask)
        torch.testing.assert_close(actual.delta_bg, expected.delta_bg, rtol=0, atol=0)
        torch.testing.assert_close(
            actual.predicted_flow_backward,
            expected.predicted_flow_backward,
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
