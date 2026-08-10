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

from diffusers.models.stc_flow_training import prepare_teacher_flow  # noqa: E402
from STC_encoder_v3_rgb_flow.flow_supervision import (  # noqa: E402
    compute_teacher_flow_loss,
)
from STC_encoder_v3_rgb_flow.rgb_stc_flow_adapter import (  # noqa: E402
    RGBSTCFlowAdapter,
    augment_brushnet_condition,
)
from STC_encoder_v3_rgb_flow.teacher_flow_data import (  # noqa: E402
    collate_teacher_flow_clips,
)


class RGBSTCFlowTests(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(11)
        return RGBSTCFlowAdapter(
            hidden_channels=16,
            num_heads=1,
            num_layers=1,
            mlp_ratio=2.0,
            flow_max_displacement=(4.0, 3.0),
        )

    def test_shapes_and_zero_init_identity(self):
        model = self.make_model()
        rgb = torch.randn(2, 3, 3, 32, 40)
        mask = torch.ones(2, 3, 1, 32, 40)
        base = torch.randn(6, 4, 4, 5)
        condition, output, augmented = augment_brushnet_condition(
            model, base, rgb, mask
        )
        self.assertEqual(tuple(output.features.shape), (2, 3, 16, 4, 5))
        self.assertEqual(tuple(output.predicted_flow_forward.shape), (2, 2, 2, 4, 5))
        self.assertEqual(tuple(output.predicted_flow_backward.shape), (2, 2, 2, 4, 5))
        self.assertEqual(torch.count_nonzero(output.delta_bg).item(), 0)
        self.assertEqual(torch.count_nonzero(output.predicted_flow_forward).item(), 0)
        self.assertTrue(torch.equal(augmented.flatten(0, 1), base))
        self.assertTrue(torch.equal(condition[:, :4], base))

    def test_teacher_resize_scales_pixel_displacement(self):
        teacher = torch.zeros(1, 1, 2, 16, 24)
        teacher[:, :, 0] = 8.0
        teacher[:, :, 1] = 4.0
        resized, valid = prepare_teacher_flow(
            teacher, (2, 3), torch.ones(1, 1, 1, 16, 24)
        )
        torch.testing.assert_close(
            resized[:, :, 0], torch.ones_like(resized[:, :, 0]), rtol=0, atol=1e-6
        )
        torch.testing.assert_close(
            resized[:, :, 1],
            torch.full_like(resized[:, :, 1], 0.5),
            rtol=0,
            atol=1e-6,
        )
        self.assertGreater(valid.mean().item(), 0.0)

    def test_forward_backward_direction_order(self):
        class DifferenceHead(nn.Module):
            def forward(self, reference, query):
                return (query - reference)[:, :2]

        model = self.make_model()
        model.flow_head = DifferenceHead()
        features = torch.zeros(1, 2, 16, 1, 1)
        features[:, 1, :2] = 1.0
        forward, backward = model._decode_bidirectional_flow(features)
        # Forward lives on frame 0 and queries frame 1; backward is the reverse.
        self.assertTrue(torch.all(forward < 0.0))
        self.assertTrue(torch.all(backward > 0.0))

    def test_bg_region_ignores_roi_only_error(self):
        predicted = torch.zeros(1, 1, 2, 4, 4)
        teacher = torch.zeros_like(predicted)
        teacher[..., :, 2:] = 1.0
        mask_bg = torch.zeros(1, 2, 1, 4, 4)
        mask_bg[..., :, :2] = 1.0
        validity = torch.ones(1, 1, 1, 4, 4)
        bg = compute_teacher_flow_loss(
            predicted,
            predicted.clone(),
            teacher,
            teacher.clone(),
            mask_bg,
            validity,
            validity.clone(),
            region="bg",
        )
        all_pixels = compute_teacher_flow_loss(
            predicted,
            predicted.clone(),
            teacher,
            teacher.clone(),
            mask_bg,
            validity,
            validity.clone(),
            region="all",
        )
        self.assertLess(bg.loss.item(), 0.002)
        self.assertGreater(all_pixels.loss.item(), 0.1)

    def test_empty_valid_region_returns_differentiable_zero(self):
        predicted_f = torch.randn(1, 1, 2, 4, 4, requires_grad=True)
        predicted_b = torch.randn(1, 1, 2, 4, 4, requires_grad=True)
        teacher = torch.zeros_like(predicted_f)
        no_bg = torch.zeros(1, 2, 1, 4, 4)
        invalid = torch.zeros(1, 1, 1, 4, 4)
        output = compute_teacher_flow_loss(
            predicted_f,
            predicted_b,
            teacher,
            teacher.clone(),
            no_bg,
            invalid,
            invalid.clone(),
            region="bg",
        )
        self.assertEqual(output.loss.item(), 0.0)
        output.loss.backward()
        self.assertIsNotNone(predicted_f.grad)
        self.assertEqual(torch.count_nonzero(predicted_f.grad).item(), 0)

    def test_flow_loss_reaches_rgb_stc_features(self):
        model = self.make_model()
        rgb = torch.randn(1, 3, 3, 32, 32)
        mask = torch.ones(1, 3, 1, 32, 32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        output = model(rgb, mask)
        teacher_f = torch.randn_like(output.predicted_flow_forward) * 0.1
        teacher_b = torch.randn_like(output.predicted_flow_backward) * 0.1
        valid = torch.ones(1, 2, 1, 4, 4)

        def flow_loss(model_output):
            return compute_teacher_flow_loss(
                model_output.predicted_flow_forward,
                model_output.predicted_flow_backward,
                teacher_f,
                teacher_b,
                mask,
                valid,
                valid,
                region="bg",
            ).loss

        # With a zero-initialized final flow layer, step one learns that layer.
        flow_loss(output).backward()
        first_conv = model.stc_adapter.spatial_encoder[0].block[0].weight
        if first_conv.grad is not None:
            self.assertEqual(first_conv.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            model.flow_head.network[-1].weight.grad.abs().sum().item(), 0.0
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # From step two onward, the learned head carries L_flow into RGB-STC.
        flow_loss(model(rgb, mask)).backward()
        self.assertIsNotNone(first_conv.grad)
        self.assertGreater(first_conv.grad.abs().sum().item(), 0.0)

    def test_save_load_round_trip(self):
        model = self.make_model().eval()
        torch.nn.init.normal_(model.flow_head.network[-1].weight, std=0.01)
        rgb = torch.randn(1, 2, 3, 32, 32)
        mask = torch.ones(1, 2, 1, 32, 32)
        expected = model(rgb, mask)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            restored = RGBSTCFlowAdapter.from_pretrained(directory).eval()
            actual = restored(rgb, mask)
        torch.testing.assert_close(
            actual.predicted_flow_forward,
            expected.predicted_flow_forward,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(actual.delta_bg, expected.delta_bg, rtol=0, atol=0)

    def test_collate_preserves_clipwise_teacher_flow(self):
        def example(value):
            tensors = {
                "pixel_values": torch.full((3, 3, 4, 4), value),
                "masks": torch.ones(3, 1, 4, 4),
                "conditioning_pixel_values": torch.zeros(3, 3, 4, 4),
                "input_ids": torch.zeros(3, 5, dtype=torch.long),
                "clip_images": torch.zeros(3, 3, 2, 2),
                "fg_clip_images": torch.zeros(3, 3, 2, 2),
                "bg_clip_images": torch.zeros(3, 3, 2, 2),
                "drop_image_embeds": torch.zeros(3, dtype=torch.long),
                "video": f"Class/sequence_{value}",
                "frame_ids": torch.arange(3),
                "teacher_flow_forward": torch.full((2, 2, 8, 8), value),
                "teacher_flow_backward": torch.full((2, 2, 8, 8), value),
                "teacher_valid_forward": torch.ones(2, 1, 8, 8),
                "teacher_valid_backward": torch.ones(2, 1, 8, 8),
            }
            return tensors

        batch = collate_teacher_flow_clips([example(1.0), example(2.0)])
        self.assertEqual(tuple(batch["pixel_values"].shape), (6, 3, 4, 4))
        self.assertEqual(tuple(batch["teacher_flow_forward"].shape), (2, 2, 2, 8, 8))
        self.assertEqual(batch["teacher_flow_forward"][1, 0, 0, 0, 0].item(), 2.0)


if __name__ == "__main__":
    unittest.main()
