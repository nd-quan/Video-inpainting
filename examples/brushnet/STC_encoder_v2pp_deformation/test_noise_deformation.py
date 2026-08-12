from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


BRUSHNET_DIR = Path(__file__).resolve().parent.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v2pp_deformation.noise_deformation import (  # noqa: E402
    NoiseDeformationHead,
    backward_warp,
    build_deformed_clip_noise,
    cosine_feature_matching_loss,
    edge_aware_smoothness_loss,
    make_backward_sampling_grid,
    normalize_lineage_channels,
    variance_preserving_fusion,
)


class NoiseDeformationCoreTests(unittest.TestCase):
    def make_head(self, channels=4, max_displacement=3.0):
        torch.manual_seed(11)
        return NoiseDeformationHead(
            feature_channels=channels,
            hidden_channels=8,
            num_hidden_layers=2,
            max_displacement=max_displacement,
        )

    def test_zero_initialized_head_is_identity_offset(self):
        head = self.make_head()
        reference = torch.randn(2, 4, 8, 9)
        target = torch.randn_like(reference)
        offset = head(reference, target)
        self.assertTrue(torch.equal(offset, torch.zeros_like(offset)))

        warped, valid = backward_warp(reference, offset)
        self.assertTrue(valid.all())
        torch.testing.assert_close(warped, reference, rtol=0, atol=2e-6)

    def test_pixel_center_grid_matches_align_corners_false_identity(self):
        offset = torch.zeros(1, 2, 2, 4)
        grid, valid = make_backward_sampling_grid(offset)
        expected_x = torch.tensor([-0.75, -0.25, 0.25, 0.75])
        expected_y = torch.tensor([-0.5, 0.5])
        torch.testing.assert_close(grid[0, 0, :, 0], expected_x)
        torch.testing.assert_close(grid[0, :, 0, 1], expected_y)
        self.assertTrue(valid.all())

    def test_one_pixel_backward_offset_samples_expected_reference(self):
        reference = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
        offset = torch.zeros(1, 2, 3, 4)
        offset[:, 0] = 1.0
        fallback = torch.full_like(reference, -5.0)
        warped, valid = backward_warp(reference, offset, fallback=fallback)
        torch.testing.assert_close(warped[..., :-1], reference[..., 1:])
        self.assertTrue(torch.equal(warped[..., -1], fallback[..., -1]))
        self.assertTrue(valid[..., :-1].all())
        self.assertFalse(valid[..., -1].any())

    def test_offset_is_bounded(self):
        head = self.make_head(max_displacement=2.5)
        torch.nn.init.normal_(head.to_offset.weight, std=20.0)
        reference = torch.randn(2, 4, 8, 8)
        offset = head(reference, -reference)
        self.assertLessEqual(offset.abs().max().item(), 2.5)

    def test_head_checkpoint_round_trip(self):
        head = self.make_head(max_displacement=2.5)
        torch.nn.init.normal_(head.to_offset.weight, std=0.05)
        reference = torch.randn(1, 4, 6, 7)
        target = torch.randn_like(reference)
        expected = head(reference, target)
        with tempfile.TemporaryDirectory() as directory:
            head.save_pretrained(directory, safe_serialization=True)
            restored = NoiseDeformationHead.from_pretrained(directory)
            actual = restored(reference, target)
        self.assertEqual(restored.config.max_displacement, 2.5)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_channel_normalization_restores_spatial_statistics(self):
        tensor = 3.0 + 4.0 * torch.randn(3, 4, 32, 24)
        normalized = normalize_lineage_channels(tensor)
        torch.testing.assert_close(
            normalized.mean(dim=(-2, -1)),
            torch.zeros(3, 4),
            rtol=0,
            atol=2e-6,
        )
        torch.testing.assert_close(
            normalized.std(dim=(-2, -1), unbiased=False),
            torch.ones(3, 4),
            rtol=0,
            atol=2e-6,
        )

    def test_variance_preserving_law_and_endpoints(self):
        torch.manual_seed(3)
        lineage = torch.randn(2, 4, 256, 256)
        independent = torch.randn_like(lineage)
        self.assertTrue(
            torch.equal(
                variance_preserving_fusion(lineage, independent, 0.0),
                independent,
            )
        )
        self.assertTrue(
            torch.equal(
                variance_preserving_fusion(lineage, independent, 1.0),
                lineage,
            )
        )
        fused = variance_preserving_fusion(lineage, independent, 0.9)
        self.assertAlmostEqual(fused.var(unbiased=False).item(), 1.0, delta=0.015)

    def test_full_scope_ignores_mask_and_bg_scope_preserves_roi(self):
        lineage = torch.full((1, 2, 3, 4), 2.0)
        independent = torch.full_like(lineage, -1.0)
        bg_mask = torch.zeros(1, 1, 3, 4)
        bg_mask[..., :2] = 1.0
        full = variance_preserving_fusion(
            lineage, independent, 0.9, warp_scope="full", bg_mask=bg_mask
        )
        bg_only = variance_preserving_fusion(
            lineage, independent, 0.9, warp_scope="bg", bg_mask=bg_mask
        )
        self.assertGreater((full - independent).abs().min().item(), 0.0)
        roi = (1.0 - bg_mask).expand_as(independent).bool()
        self.assertTrue(torch.equal(bg_only.masked_select(roi), independent.masked_select(roi)))
        self.assertGreater(
            (bg_only - independent).masked_select(~roi).abs().min().item(), 0.0
        )

    def test_matching_loss_prefers_correct_backward_translation(self):
        torch.manual_seed(5)
        reference = torch.randn(1, 6, 5, 7)
        target = torch.zeros_like(reference)
        target[..., :-1] = reference[..., 1:]

        correct_offset = torch.zeros(1, 2, 5, 7)
        correct_offset[:, 0] = 1.0
        correct_warp, correct_valid = backward_warp(reference, correct_offset)
        correct = cosine_feature_matching_loss(correct_warp, target, correct_valid)

        zero_offset = torch.zeros_like(correct_offset)
        wrong_warp, wrong_valid = backward_warp(reference, zero_offset)
        wrong = cosine_feature_matching_loss(wrong_warp, target, wrong_valid)
        self.assertLess(correct.item(), 1e-6)
        self.assertLess(correct.item(), wrong.item())

    def test_smoothness_and_feature_edge_weighting(self):
        constant_offset = torch.ones(1, 2, 5, 6)
        features = torch.zeros(1, 3, 5, 6)
        self.assertEqual(
            edge_aware_smoothness_loss(constant_offset, features).item(), 0.0
        )

        discontinuous = constant_offset.clone()
        discontinuous[..., 3:] = 4.0
        no_edge = edge_aware_smoothness_loss(discontinuous, features, gamma=2.0)
        edge_features = features.clone()
        edge_features[..., 3:] = 10.0
        with_edge = edge_aware_smoothness_loss(
            discontinuous, edge_features, gamma=2.0
        )
        self.assertGreater(no_edge.item(), 0.0)
        self.assertLess(with_edge.item(), no_edge.item())

    def test_clip_builder_uses_full_frame_and_keeps_lineage_separate(self):
        torch.manual_seed(13)
        head = self.make_head()
        features = torch.randn(1, 3, 4, 8, 8, requires_grad=True)
        anchor = torch.randn(1, 2, 8, 8)
        independent = torch.randn(1, 3, 2, 8, 8)
        fallback = torch.randn_like(independent)
        bg_mask = torch.zeros(1, 3, 1, 8, 8)
        output = build_deformed_clip_noise(
            head,
            features,
            anchor,
            independent,
            fallback,
            alpha=0.9,
            warp_scope="full",
            bg_mask=bg_mask,
        )
        self.assertEqual(tuple(output.final_noise.shape), (1, 3, 2, 8, 8))
        self.assertEqual(tuple(output.lineage_noise.shape), (1, 3, 2, 8, 8))
        self.assertEqual(tuple(output.offsets.shape), (1, 2, 2, 8, 8))
        self.assertTrue(output.valid_masks.all())
        self.assertEqual(tuple(output.pre_normalization_mean.shape), (1, 2, 2))
        torch.testing.assert_close(
            output.post_normalization_mean,
            torch.zeros_like(output.post_normalization_mean),
            rtol=0,
            atol=2e-6,
        )
        torch.testing.assert_close(
            output.post_normalization_std,
            torch.ones_like(output.post_normalization_std),
            rtol=0,
            atol=2e-6,
        )
        self.assertIsNone(features.grad)

        # alpha=1 proves that final noise equals lineage without recursively
        # replacing the lineage state with a separately fused tensor.
        lineage_only = build_deformed_clip_noise(
            head,
            features,
            anchor,
            independent,
            fallback,
            alpha=1.0,
            warp_scope="full",
        )
        self.assertTrue(
            torch.equal(lineage_only.final_noise, lineage_only.lineage_noise)
        )

    def test_zero_grid_mode_keeps_identity_grid_sample_path(self):
        torch.manual_seed(23)
        head = self.make_head()
        # Make the learned head non-zero so this test proves zero_grid ignores
        # its prediction rather than relying on zero-initialized weights.
        torch.nn.init.normal_(head.to_offset.weight, std=0.05)
        features = torch.randn(1, 4, 4, 8, 8)
        anchor = torch.randn(1, 2, 8, 8)
        independent = torch.randn(1, 4, 2, 8, 8)
        fallback = torch.randn_like(independent)
        output = build_deformed_clip_noise(
            head,
            features,
            anchor,
            independent,
            fallback,
            alpha=0.9,
            offset_mode="zero_grid",
        )
        self.assertTrue(torch.equal(output.offsets, torch.zeros_like(output.offsets)))
        self.assertTrue(output.valid_masks.all())
        # Frame 1 is produced by identity grid_sample followed by channel norm.
        expected = normalize_lineage_channels(anchor)
        torch.testing.assert_close(
            output.lineage_noise[:, 1], expected, rtol=0, atol=3e-6
        )

    def test_invalid_offset_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "offset_mode"):
            build_deformed_clip_noise(
                self.make_head(),
                torch.randn(1, 2, 4, 4, 4),
                torch.randn(1, 2, 4, 4),
                torch.randn(1, 2, 2, 4, 4),
                torch.randn(1, 2, 2, 4, 4),
                alpha=0.9,
                offset_mode="invalid",
            )

    def test_matching_and_diffusion_paths_reach_head_not_frozen_features(self):
        torch.manual_seed(17)
        head = self.make_head()
        features = torch.randn(1, 3, 4, 8, 8, requires_grad=True)
        anchor = torch.randn(1, 2, 8, 8)
        independent = torch.randn(1, 3, 2, 8, 8)
        fallback = torch.randn_like(independent)
        output = build_deformed_clip_noise(
            head,
            features,
            anchor,
            independent,
            fallback,
            alpha=0.9,
        )
        # A stand-in differentiable noisy-latent objective exercises the same
        # path that scheduler.add_noise exposes to the diffusion loss.
        diffusion_proxy = output.final_noise[:, 1:].square().mean()
        loss = diffusion_proxy + output.match_loss + 0.1 * output.smoothness_loss
        loss.backward()
        self.assertIsNone(features.grad)
        self.assertIsNotNone(head.to_offset.weight.grad)
        self.assertGreater(head.to_offset.weight.grad.abs().sum().item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_clip_forward_backward(self):
        torch.manual_seed(19)
        device = torch.device("cuda")
        head = self.make_head().to(device)
        features = torch.randn(1, 3, 4, 8, 8, device=device)
        anchor = torch.randn(1, 2, 8, 8, device=device)
        independent = torch.randn(1, 3, 2, 8, 8, device=device)
        fallback = torch.randn_like(independent)
        output = build_deformed_clip_noise(
            head,
            features,
            anchor,
            independent,
            fallback,
            alpha=0.9,
        )
        (output.final_noise.square().mean() + output.match_loss).backward()
        self.assertGreater(head.to_offset.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
