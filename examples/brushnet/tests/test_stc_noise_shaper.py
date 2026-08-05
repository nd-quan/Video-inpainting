import tempfile
import unittest
from pathlib import Path

import torch

from diffusers.models.stc_noise_shaper import (
    STCConditionedNoiseShaper,
    VideoComposerSTCConditionEncoder,
    gaussian_backward_warp,
)


def make_inputs(batch=1, frames=4, height=16, width=16):
    noise = torch.randn(batch, frames, 4, height, width)
    decoded = torch.randn_like(noise)
    bg = torch.ones(batch, frames, 1, height, width)
    flow = torch.zeros(batch, frames - 1, 2, height, width)
    confidence = torch.ones(batch, frames - 1, 1, height, width)
    stable = torch.ones_like(confidence)
    return noise, decoded, bg, flow, confidence, stable


class STCNoiseShaperTest(unittest.TestCase):
    def make_model(self, **kwargs):
        defaults = dict(
            hidden_channels=16,
            num_attention_heads=4,
            num_transformer_layers=1,
            flow_residual_scale=1.0,
            beta_max=0.95,
            initial_beta=0.25,
            channel_normalize=False,
            global_normalize=False,
        )
        defaults.update(kwargs)
        return STCConditionedNoiseShaper(**defaults)

    def make_full_model(self, **kwargs):
        defaults = dict(
            condition_channels=5,
            hidden_channels=8,
            num_attention_heads=2,
            num_transformer_layers=1,
            flow_prediction_mode="full",
            full_flow_max_displacement=(4.0, 3.0),
            beta_mode="fixed",
            fixed_beta=0.5,
            warp_region="all",
            channel_normalize=False,
            global_normalize=False,
        )
        defaults.update(kwargs)
        return STCConditionedNoiseShaper(**defaults)

    def test_shapes_and_zero_initialized_flow_prior(self):
        model = self.make_model()
        inputs = make_inputs(batch=2)
        inputs[3][:, :, 0] = 2.0
        output = model(*inputs)
        self.assertEqual(tuple(output["noise"].shape), (2, 4, 4, 16, 16))
        self.assertEqual(tuple(output["beta"].shape), (2, 3, 1, 16, 16))
        torch.testing.assert_close(output["predicted_flow"], inputs[3])

    def test_unreliable_or_roi_pixels_keep_fresh_noise(self):
        model = self.make_model()
        inputs = list(make_inputs())
        inputs[2].zero_()
        inputs[5].zero_()
        output = model(*inputs)
        torch.testing.assert_close(output["noise"], inputs[0], rtol=0, atol=0)
        self.assertEqual(float(output["effective_beta"].abs().max()), 0.0)

    def test_all_frame_warp_does_not_hard_gate_roi_or_unstable_pixels(self):
        model = self.make_model(warp_region="all")
        inputs = list(make_inputs())
        inputs[2].zero_()
        inputs[4].zero_()
        inputs[5].zero_()
        output = model(*inputs)
        torch.testing.assert_close(
            output["effective_beta"],
            output["beta"],
            rtol=0,
            atol=0,
        )
        self.assertGreater(
            float((output["noise"][:, 1:] - inputs[0][:, 1:]).abs().max()),
            0.0,
        )

    def test_warp_region_must_be_known(self):
        with self.assertRaisesRegex(ValueError, "warp_region must be"):
            self.make_model(warp_region="unknown")

    def test_global_normalization_matches_first_two_moments(self):
        model = self.make_model(global_normalize=True)
        output = model(*make_inputs(batch=2))
        per_sample_mean = output["noise"].mean(dim=(1, 2, 3, 4))
        per_sample_std = output["noise"].std(
            dim=(1, 2, 3, 4), unbiased=False
        )
        torch.testing.assert_close(per_sample_mean, torch.zeros_like(per_sample_mean), atol=2e-5, rtol=0)
        torch.testing.assert_close(per_sample_std, torch.ones_like(per_sample_std), atol=2e-5, rtol=0)

    def test_gaussian_warp_corrects_fractional_variance(self):
        torch.manual_seed(7)
        samples = torch.randn(4096, 1, 3, 3)
        flow = torch.zeros(4096, 2, 3, 3)
        flow[:, 0] = 0.5
        flow[:, 1] = 0.5
        warped, valid = gaussian_backward_warp(samples, flow)
        center = warped[:, 0, 1, 1]
        self.assertGreater(float(valid[:, :, 1, 1].min()), 0.0)
        self.assertLess(abs(float(center.mean())), 0.04)
        self.assertLess(abs(float(center.std(unbiased=False)) - 1.0), 0.04)

    def test_gradients_reach_stc_flow_and_gate_heads(self):
        model = self.make_model()
        # Zero-init preserves the refined-flow prior on the first update and
        # intentionally blocks encoder gradients until the heads move away
        # from zero. Use a tiny nonzero head here to test the full graph.
        torch.nn.init.normal_(model.flow_head[-1].weight, std=1e-3)
        torch.nn.init.normal_(model.beta_head[-1].weight, std=1e-3)
        inputs = make_inputs(height=8, width=8)
        weight = torch.randn_like(inputs[0])
        output = model(*inputs)
        loss = (output["noise"] * weight).mean()
        loss.backward()
        self.assertGreater(float(model.flow_head[-1].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.beta_head[-1].weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.stc_encoder.spatial[0].weight.grad.abs().sum()), 0.0)

    def test_variable_clip_length_and_single_frame(self):
        model = self.make_model()
        output4 = model(*make_inputs(frames=4))["noise"]
        output10 = model(*make_inputs(frames=10))["noise"]
        output1 = model(*make_inputs(frames=1))["noise"]
        self.assertEqual(output4.shape[1], 4)
        self.assertEqual(output10.shape[1], 10)
        self.assertEqual(output1.shape[1], 1)

    def test_strength_must_be_a_correlation_scale(self):
        model = self.make_model()
        with self.assertRaisesRegex(ValueError, "strength must be in"):
            model(*make_inputs(), strength=-0.1)
        with self.assertRaisesRegex(ValueError, "strength must be in"):
            model(*make_inputs(), strength=1.1)

    def test_noise_only_phase_uses_fixed_input_flow(self):
        model = self.make_model(
            predict_flow_residual=False,
            use_input_flow_prior=True,
            beta_min=0.05,
            initial_beta=0.4,
        )
        inputs = make_inputs()
        inputs[3][:, :, 0] = 1.25
        output = model(*inputs)
        torch.testing.assert_close(output["predicted_flow"], inputs[3])
        self.assertFalse(any(p.requires_grad for p in model.flow_head.parameters()))
        self.assertGreaterEqual(float(output["beta"].min()), 0.05)
        self.assertLessEqual(float(output["beta"].max()), 0.95)

    def test_videocomposer_encoder_has_separate_condition_branches(self):
        encoder = VideoComposerSTCConditionEncoder(
            input_channels=9,
            hidden_channels=8,
            num_heads=2,
            num_layers=1,
            condition_group_channels=(5, 2, 2),
            pool_size=128,
        )
        condition = torch.randn(2, 4, 9, 16, 16)
        output = encoder(condition)
        self.assertEqual(tuple(output.shape), (2, 4, 8, 16, 16))
        self.assertEqual(len(encoder.branches), 3)
        (output * torch.randn_like(output)).mean().backward()
        for branch in encoder.branches:
            self.assertGreater(
                float(branch.spatial_in[0].weight.grad.abs().sum()),
                0.0,
            )

    def test_videocomposer_noise_shaper_save_load_round_trip(self):
        model = self.make_model(
            hidden_channels=8,
            num_attention_heads=2,
            encoder_architecture="videocomposer",
            condition_group_channels=(5, 2, 2),
            videocomposer_pool_size=128,
        ).eval()
        inputs = make_inputs(height=8, width=8)
        expected = model(*inputs)["noise"]
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(Path(directory))
            restored = STCConditionedNoiseShaper.from_pretrained(directory).eval()
            actual = restored(*inputs)["noise"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_videocomposer_condition_groups_must_cover_input(self):
        with self.assertRaisesRegex(ValueError, "must sum to input_channels"):
            VideoComposerSTCConditionEncoder(
                input_channels=9,
                hidden_channels=8,
                num_heads=2,
                num_layers=1,
                condition_group_channels=(5, 2),
            )

    def test_full_flow_predicts_both_directions_without_input_flow(self):
        torch.manual_seed(11)
        model = self.make_full_model().eval()
        torch.nn.init.normal_(model.full_flow_head.network[-1].weight, std=1e-2)
        noise, decoded, bg, _, _, _ = make_inputs(
            batch=2, frames=4, height=8, width=10
        )
        direct = model.predict_flow(decoded, bg)
        dispatched = model(
            decoded_latents=decoded,
            bg_mask=bg,
            operation="predict_flow",
        )
        output_without_prior = model(noise, decoded, bg)
        unrelated_input_flow = torch.randn(2, 3, 2, 17, 19)
        output_with_ignored_prior = model(
            noise,
            decoded,
            bg,
            backward_flow=unrelated_input_flow,
            motion_confidence=torch.zeros(1),
            stable_bg=torch.zeros(1),
        )
        self.assertEqual(
            tuple(direct["predicted_flow_backward"].shape),
            (2, 3, 2, 8, 10),
        )
        self.assertEqual(
            tuple(direct["predicted_flow_forward"].shape),
            (2, 3, 2, 8, 10),
        )
        torch.testing.assert_close(
            output_without_prior["predicted_flow_backward"],
            direct["predicted_flow_backward"],
        )
        torch.testing.assert_close(
            dispatched["predicted_flow_backward"],
            direct["predicted_flow_backward"],
        )
        torch.testing.assert_close(
            output_without_prior["predicted_flow_forward"],
            direct["predicted_flow_forward"],
        )
        torch.testing.assert_close(
            output_with_ignored_prior["predicted_flow_backward"],
            direct["predicted_flow_backward"],
        )

    def test_full_flow_condition_keeps_roi_latent_values(self):
        model = self.make_full_model().eval()
        decoded = torch.randn(1, 3, 4, 8, 8)
        bg = torch.zeros(1, 3, 1, 8, 8)
        captured = []
        hook = model.stc_encoder.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach().clone())
        )
        try:
            model.predict_flow(decoded, bg)
        finally:
            hook.remove()
        self.assertEqual(len(captured), 1)
        torch.testing.assert_close(captured[0][:, :, :4], decoded)
        torch.testing.assert_close(captured[0][:, :, 4:], bg)

    def test_full_flow_decoder_shares_ordered_pair_weights(self):
        model = self.make_full_model().eval()
        features = torch.randn(2, 3, 8, 5, 6)
        captured = []
        hook = model.full_flow_head.register_forward_pre_hook(
            lambda _module, args: captured.append(
                (args[0].detach().clone(), args[1].detach().clone())
            )
        )
        try:
            backward, forward = model._decode_full_bidirectional_flow(features)
        finally:
            hook.remove()
        previous = features[:, :-1].reshape(-1, 8, 5, 6)
        current = features[:, 1:].reshape(-1, 8, 5, 6)
        self.assertEqual(tuple(backward.shape), (2, 2, 2, 5, 6))
        self.assertEqual(tuple(forward.shape), (2, 2, 2, 5, 6))
        self.assertEqual(len(captured), 2)
        torch.testing.assert_close(captured[0][0], previous)
        torch.testing.assert_close(captured[0][1], current)
        torch.testing.assert_close(captured[1][0], current)
        torch.testing.assert_close(captured[1][1], previous)

    def test_full_flow_uses_xy_latent_pixel_bounds(self):
        model = self.make_full_model(
            full_flow_max_displacement=(3.0, 5.0)
        ).eval()
        final = model.full_flow_head.network[-1]
        torch.nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias.copy_(torch.tensor([100.0, -100.0]))
        decoded = torch.randn(1, 2, 4, 8, 12)
        bg = torch.ones(1, 2, 1, 8, 12)
        output = model.predict_flow(decoded, bg)
        for key in ("predicted_flow_backward", "predicted_flow_forward"):
            flow = output[key]
            torch.testing.assert_close(
                flow[:, :, 0], torch.full_like(flow[:, :, 0], 3.0)
            )
            torch.testing.assert_close(
                flow[:, :, 1], torch.full_like(flow[:, :, 1], -5.0)
            )

    def test_fixed_beta_matches_variance_preserving_fusion(self):
        beta = 0.6
        model = self.make_full_model(fixed_beta=beta).eval()
        noise, decoded, bg, _, _, _ = make_inputs(
            frames=4, height=8, width=8
        )
        output = model(noise, decoded, bg)
        expected = noise.clone()
        innovation = (1.0 - beta**2) ** 0.5
        expected[:, 1:] = beta * noise[:, :1] + innovation * noise[:, 1:]
        torch.testing.assert_close(output["noise"], expected)
        torch.testing.assert_close(
            output["beta"], torch.full_like(output["beta"], beta)
        )
        self.assertIsNone(model.beta_head)
        self.assertFalse(any("beta_head" in name for name, _ in model.named_parameters()))
        flow_parameter_ids = {id(parameter) for parameter in model.flow_parameters()}
        trainable_parameter_ids = {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }
        self.assertEqual(flow_parameter_ids, trainable_parameter_ids)

    def test_full_flow_gradients_reach_shared_decoder_and_encoder(self):
        model = self.make_full_model()
        torch.nn.init.normal_(model.full_flow_head.network[-1].weight, std=1e-3)
        decoded = torch.randn(1, 3, 4, 8, 8)
        bg = torch.ones(1, 3, 1, 8, 8)
        output = model.predict_flow(decoded, bg)
        loss = (
            output["predicted_flow_backward"].square().mean()
            + output["predicted_flow_forward"].square().mean()
        )
        loss.backward()
        self.assertGreater(
            float(model.full_flow_head.network[-1].weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(model.stc_encoder.spatial[0].weight.grad.abs().sum()),
            0.0,
        )

    def test_full_flow_save_load_round_trip(self):
        model = self.make_full_model(
            full_flow_max_displacement=(7.0, 6.0), fixed_beta=0.4
        ).eval()
        torch.nn.init.normal_(model.full_flow_head.network[-1].weight, std=1e-3)
        inputs = make_inputs(frames=3, height=8, width=8)
        expected = model(inputs[0], inputs[1], inputs[2])
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(Path(directory))
            restored = STCConditionedNoiseShaper.from_pretrained(directory).eval()
            actual = restored(inputs[0], inputs[1], inputs[2])
        self.assertEqual(restored.config.flow_prediction_mode, "full")
        self.assertEqual(restored.config.beta_mode, "fixed")
        self.assertEqual(
            tuple(restored.config.full_flow_max_displacement), (7.0, 6.0)
        )
        for key in (
            "noise",
            "predicted_flow_backward",
            "predicted_flow_forward",
            "beta",
        ):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_full_flow_requires_full_frame_warp(self):
        with self.assertRaisesRegex(ValueError, "requires warp_region='all'"):
            self.make_full_model(warp_region="stable_bg")

    def test_videocomposer_full_flow_uses_structure_only_branch(self):
        model = self.make_full_model(
            encoder_architecture="videocomposer",
            condition_group_channels=(5,),
            videocomposer_pool_size=16,
        ).eval()
        decoded = torch.randn(1, 3, 4, 8, 8)
        bg = torch.ones(1, 3, 1, 8, 8)
        output = model.predict_flow(decoded, bg)
        self.assertEqual(len(model.stc_encoder.branches), 1)
        self.assertEqual(
            tuple(output["predicted_flow_backward"].shape), (1, 2, 2, 8, 8)
        )
        self.assertEqual(
            tuple(output["predicted_flow_forward"].shape), (1, 2, 2, 8, 8)
        )

    def test_save_load_round_trip(self):
        model = self.make_model().eval()
        inputs = make_inputs(height=8, width=8)
        expected = model(*inputs)["noise"]
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(Path(directory))
            restored = STCConditionedNoiseShaper.from_pretrained(directory).eval()
            actual = restored(*inputs)["noise"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
