from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


BRUSHNET_DIR = Path(__file__).resolve().parent.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v2_rgb.rgb_stc_adapter import (  # noqa: E402
    RGBSTCConditionAdapter,
    augment_brushnet_condition,
)


class RGBSTCConditionAdapterTests(unittest.TestCase):
    def make_model(self, mode="full_rgb_bg_mask"):
        torch.manual_seed(7)
        return RGBSTCConditionAdapter(
            hidden_channels=16,
            num_heads=1,
            num_layers=1,
            mlp_ratio=2.0,
            condition_mode=mode,
        )

    def test_real_contract_downsamples_by_eight(self):
        model = self.make_model()
        rgb = torch.randn(1, 3, 3, 64, 80)
        mask = torch.ones(1, 3, 1, 64, 80)
        output = model(rgb, mask, return_dict=True)
        self.assertEqual(tuple(output.delta_bg.shape), (1, 3, 4, 8, 10))
        self.assertEqual(tuple(output.features.shape), (1, 3, 16, 8, 10))

    def test_primary_condition_receives_full_rgb_and_bg_mask(self):
        model = self.make_model("full_rgb_bg_mask")
        rgb = torch.randn(2, 3, 3, 16, 24)
        mask = (torch.rand(2, 3, 1, 16, 24) > 0.5).float()
        condition = model.build_pixel_condition(rgb, mask)
        torch.testing.assert_close(condition[:, :, :3], rgb, rtol=0, atol=0)
        torch.testing.assert_close(condition[:, :, 3:], mask, rtol=0, atol=0)

    def test_videocomposer_ablation_masks_degraded_bg(self):
        model = self.make_model("videocomposer_roi_masked")
        rgb = torch.randn(1, 2, 3, 16, 16)
        bg_mask = torch.zeros(1, 2, 1, 16, 16)
        bg_mask[..., :8, :] = 1.0
        condition = model.build_pixel_condition(rgb, bg_mask)
        roi_mask = 1.0 - bg_mask
        torch.testing.assert_close(
            condition[:, :, :3], rgb * roi_mask, rtol=0, atol=0
        )
        torch.testing.assert_close(
            condition[:, :, 3:], roi_mask, rtol=0, atol=0
        )

    def test_zero_init_is_exact_baseline_identity(self):
        model = self.make_model()
        rgb = torch.randn(2, 3, 3, 32, 40)
        mask = (torch.rand(2, 3, 1, 32, 40) > 0.5).float()
        base = torch.randn(6, 4, 4, 5)
        brush_condition, output, augmented = augment_brushnet_condition(
            model, base, rgb, mask
        )
        self.assertEqual(torch.count_nonzero(output.delta_bg).item(), 0)
        self.assertTrue(torch.equal(augmented.flatten(0, 1), base))
        self.assertTrue(torch.equal(brush_condition[:, :4], base))
        expected_mask = F.interpolate(
            mask.flatten(0, 1), size=base.shape[-2:], mode="nearest"
        )
        self.assertTrue(torch.equal(brush_condition[:, 4:], expected_mask))

    def test_fifth_channel_preserves_all_roi_and_all_bg_polarity(self):
        model = self.make_model()
        rgb = torch.randn(1, 2, 3, 32, 32)
        base = torch.randn(2, 4, 4, 4)
        for mask_value in (0.0, 1.0):
            mask = torch.full((1, 2, 1, 32, 32), mask_value)
            brush_condition, _, _ = augment_brushnet_condition(
                model, base, rgb, mask
            )
            self.assertTrue(
                torch.equal(
                    brush_condition[:, 4:],
                    torch.full_like(brush_condition[:, 4:], mask_value),
                )
            )

    def test_brushnet_condition_preserves_vae_latent_dtype(self):
        model = self.make_model()
        rgb = torch.randn(1, 2, 3, 32, 32)
        mask = torch.ones(1, 2, 1, 32, 32)
        base = torch.randn(2, 4, 4, 4, dtype=torch.float16)
        brush_condition, _, augmented = augment_brushnet_condition(
            model, base, rgb, mask
        )
        self.assertEqual(brush_condition.dtype, torch.float16)
        self.assertEqual(augmented.dtype, torch.float16)
        self.assertTrue(torch.equal(brush_condition[:, :4], base))

    def test_bg_gate_preserves_hq_roi(self):
        model = self.make_model()
        torch.nn.init.normal_(model.zero_conv.weight)
        torch.nn.init.normal_(model.zero_conv.bias)
        rgb = torch.randn(1, 2, 3, 32, 32)
        bg_mask = torch.zeros(1, 2, 1, 32, 32)
        bg_mask[..., :16, :] = 1.0
        base = torch.randn(2, 4, 4, 4)
        _, output, augmented = augment_brushnet_condition(
            model, base, rgb, bg_mask
        )
        latent_mask = output.latent_bg_mask.expand_as(output.delta_bg).bool()
        self.assertEqual(
            torch.count_nonzero(output.delta_bg.masked_select(~latent_mask)).item(),
            0,
        )
        base_sequence = base.reshape(1, 2, 4, 4, 4)
        torch.testing.assert_close(
            augmented.masked_select(~latent_mask),
            base_sequence.masked_select(~latent_mask),
            rtol=0,
            atol=0,
        )
        self.assertGreater(
            torch.count_nonzero(output.delta_bg.masked_select(latent_mask)).item(),
            0,
        )

    def test_temporal_axis_does_not_mix_videos(self):
        model = self.make_model().eval()
        torch.nn.init.normal_(model.zero_conv.weight)
        mask = torch.ones(2, 3, 1, 32, 32)
        rgb = torch.randn(2, 3, 3, 32, 32)
        changed = rgb.clone()
        changed[0, 0] += 2.0
        first = model(rgb, mask)
        second = model(changed, mask)
        self.assertGreater((first[0, 1] - second[0, 1]).abs().max().item(), 0)
        torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)

    def test_first_backward_updates_zero_conv(self):
        model = self.make_model()
        rgb = torch.randn(1, 2, 3, 32, 32)
        mask = torch.ones(1, 2, 1, 32, 32)
        model(rgb, mask).sum().backward()
        self.assertIsNotNone(model.zero_conv.weight.grad)
        self.assertGreater(model.zero_conv.weight.grad.abs().sum().item(), 0)
        first_spatial_weight = model.spatial_encoder[0].block[0].weight
        if first_spatial_weight.grad is not None:
            self.assertEqual(first_spatial_weight.grad.abs().sum().item(), 0)

    def test_save_load_round_trip(self):
        model = self.make_model()
        torch.nn.init.normal_(model.zero_conv.weight)
        rgb = torch.randn(1, 2, 3, 32, 32)
        mask = torch.ones(1, 2, 1, 32, 32)
        expected = model(rgb, mask)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            restored = RGBSTCConditionAdapter.from_pretrained(directory)
            actual = restored(rgb, mask)
        self.assertEqual(restored.config.condition_mode, "full_rgb_bg_mask")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_layout_is_rejected(self):
        model = self.make_model()
        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.randn(2, 3, 32, 32), torch.ones(2, 1, 32, 32))
        with self.assertRaisesRegex(ValueError, "divisible"):
            model(
                torch.randn(1, 2, 3, 31, 32),
                torch.ones(1, 2, 1, 31, 32),
            )


if __name__ == "__main__":
    unittest.main()
