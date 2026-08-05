import unittest

import torch
import torch.nn as nn

from diffusers.models.brushnet_motion_adapter import (
    BrushNetFlowMotionAdapter,
    backward_warp_feature,
    resize_flow,
)
from diffusers.models.motion_adapter_training import (
    FrozenBrushNetMotionModel,
    build_flow_confidence,
    build_stable_bg_confidence,
    temporal_warp_loss,
)


class BrushNetMotionAdapterTest(unittest.TestCase):
    def test_resize_flow_scales_displacements(self):
        flow = torch.zeros(1, 2, 4, 6)
        flow[:, 0] = 2.0
        flow[:, 1] = 4.0
        resized = resize_flow(flow, (2, 3))
        self.assertEqual(tuple(resized.shape), (1, 2, 2, 3))
        torch.testing.assert_close(resized[:, 0], torch.ones(1, 2, 3))
        torch.testing.assert_close(resized[:, 1], torch.full((1, 2, 3), 2.0))

    def test_backward_warp_uses_current_to_previous_flow(self):
        feature = torch.arange(5, dtype=torch.float32).view(1, 1, 1, 5)
        flow = torch.zeros(1, 2, 1, 5)
        flow[:, 0] = -1.0
        warped = backward_warp_feature(feature, flow)
        expected = torch.tensor([[[[0.0, 0.0, 1.0, 2.0, 3.0]]]])
        torch.testing.assert_close(warped, expected)

    def test_zero_initialization_exactly_preserves_baseline_with_cfg(self):
        adapter = BrushNetFlowMotionAdapter(
            down_channels=(4, 8),
            mid_channel=8,
            up_channels=(),
            bottleneck_channels=4,
            flow_channels=2,
            use_down=True,
            use_mid=True,
            use_up=False,
        )
        # CFG layout: [three unconditional frames, three conditional frames].
        down = [torch.randn(6, 4, 8, 8), torch.randn(6, 8, 4, 4)]
        mid = torch.randn(6, 8, 4, 4)
        flow = torch.randn(1, 2, 2, 16, 16)
        confidence = torch.ones(1, 2, 1, 16, 16)
        output_down, output_mid, _ = adapter(
            down, mid, [], flow, confidence, cfg_branches=2
        )
        for actual, expected in zip(output_down, down):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(output_mid, mid, rtol=0, atol=0)

    def test_first_frame_is_never_corrected(self):
        adapter = BrushNetFlowMotionAdapter(
            down_channels=(2,),
            mid_channel=2,
            up_channels=(),
            bottleneck_channels=2,
            flow_channels=2,
            use_down=True,
            use_mid=False,
            use_up=False,
        )
        torch.nn.init.zeros_(adapter.down_adapters[0].output_projection.weight)
        torch.nn.init.ones_(adapter.down_adapters[0].output_projection.bias)
        residual = torch.zeros(3, 2, 4, 4)
        flow = torch.zeros(1, 2, 2, 4, 4)
        confidence = torch.ones(1, 2, 1, 4, 4)
        output, _, _ = adapter(
            [residual], torch.zeros_like(residual), [], flow, confidence
        )
        self.assertEqual(float(output[0][0].abs().max()), 0.0)
        self.assertGreater(float(output[0][1:].abs().min()), 0.0)

    def test_stable_bg_confidence_for_zero_flow(self):
        flow = torch.zeros(1, 2, 2, 8, 8)
        bg = torch.ones(1, 3, 1, 8, 8)
        flow_confidence = build_flow_confidence(flow, flow)
        torch.testing.assert_close(
            flow_confidence, torch.ones_like(flow_confidence)
        )
        confidence = build_stable_bg_confidence(flow, flow, bg)
        torch.testing.assert_close(confidence, torch.ones_like(confidence))

    def test_temporal_loss_is_small_for_identical_frames(self):
        decoded = torch.ones(1, 3, 3, 8, 8)
        flow = torch.zeros(1, 2, 2, 8, 8)
        confidence = torch.ones(1, 2, 1, 8, 8)
        loss = temporal_warp_loss(decoded, flow, confidence)
        self.assertLess(float(loss), 0.002)

    def test_frozen_unet_still_backpropagates_to_adapter(self):
        class FakeBrushNet(nn.Module):
            def forward(self, sample, *args, **kwargs):
                return [sample], sample, []

        class FakeUNet(nn.Module):
            def forward(self, sample, *args, mid_block_add_sample=None, **kwargs):
                return (sample + mid_block_add_sample,)

        adapter = BrushNetFlowMotionAdapter(
            down_channels=(4,),
            mid_channel=4,
            up_channels=(),
            bottleneck_channels=4,
            flow_channels=2,
            use_down=True,
            use_mid=True,
            use_up=False,
        )
        model = FrozenBrushNetMotionModel(FakeBrushNet(), FakeUNet(), adapter)
        noisy = torch.randn(1, 3, 4, 8, 8)
        cond = torch.randn(1, 3, 5, 8, 8)
        text = torch.randn(1, 2, 6)
        flow = torch.zeros(1, 2, 2, 8, 8)
        confidence = torch.ones(1, 2, 1, 8, 8)
        prediction, _ = model(
            noisy, torch.tensor([10]), text, cond, flow, confidence
        )
        prediction.mean().backward()
        gradient = adapter.mid_adapter.output_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_noise_only_keeps_brushnet_input_gradient(self):
        class FakeBrushNet(nn.Module):
            def forward(self, sample, *args, **kwargs):
                return [sample], sample.square(), []

        class FakeUNet(nn.Module):
            def forward(self, sample, *args, mid_block_add_sample=None, **kwargs):
                return (sample + mid_block_add_sample,)

        model = FrozenBrushNetMotionModel(FakeBrushNet(), FakeUNet(), None)
        noisy = torch.randn(1, 3, 4, 8, 8, requires_grad=True)
        prediction, regularization = model(
            noisy,
            torch.tensor([10]),
            torch.randn(1, 2, 6),
            torch.randn(1, 3, 5, 8, 8),
            torch.zeros(1, 2, 2, 8, 8),
            torch.ones(1, 2, 1, 8, 8),
        )
        prediction.mean().backward()
        self.assertIsNotNone(noisy.grad)
        self.assertGreater(float(noisy.grad.abs().sum()), 0.0)
        self.assertEqual(float(regularization), 0.0)


if __name__ == "__main__":
    unittest.main()
