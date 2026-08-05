import unittest
from types import SimpleNamespace

import torch

from diffusers.models.stc_noise_shaper import STCConditionedNoiseShaper
from diffusers.pipelines.brushnet.pipeline_brushnet_sharedNoise_sameBG_v0_0 import (
    StableDiffusionBrushNetPipeline,
)
from diffusers.pipelines.brushnet.pipeline_brushnet_sharedNoise_sameBG_v0_0 import (
    _encode_brushnet_and_stc_condition_latents,
    _prepare_stc_noise_sequences,
)


class _LatentDistribution:
    def __init__(self, shape):
        self.shape = shape
        self.sample_calls = 0
        self.mode_calls = 0

    def sample(self):
        self.sample_calls += 1
        return torch.full(self.shape, 2.0)

    def mode(self):
        self.mode_calls += 1
        return torch.full(self.shape, 5.0)


class _VAE:
    def __init__(self, shape, scaling_factor=0.25):
        self.config = SimpleNamespace(scaling_factor=scaling_factor)
        self.distribution = _LatentDistribution(shape)

    def encode(self, _image):
        return SimpleNamespace(latent_dist=self.distribution)


class STCPipelineInputTest(unittest.TestCase):
    def test_runtime_noise_shaper_does_not_modify_pipeline_components(self):
        class DummyModule(torch.nn.Module):
            def __init__(self, **config):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))
                self.config = SimpleNamespace(**config)

            @property
            def dtype(self):
                return self.weight.dtype

        pipe = StableDiffusionBrushNetPipeline(
            vae=DummyModule(block_out_channels=(1, 1, 1, 1)),
            text_encoder=DummyModule(),
            tokenizer=DummyModule(),
            unet=DummyModule(),
            brushnet=DummyModule(),
            scheduler=DummyModule(),
            safety_checker=None,
            feature_extractor=None,
            image_encoder=None,
            requires_safety_checker=False,
        )
        before = set(pipe.components)
        shaper = STCConditionedNoiseShaper(
            latent_channels=4,
            condition_channels=5,
            hidden_channels=8,
            num_attention_heads=2,
            num_transformer_layers=1,
            encoder_architecture="videocomposer",
            condition_group_channels=(5,),
            videocomposer_pool_size=8,
            flow_prediction_mode="full",
            full_flow_max_displacement=(8.0, 8.0),
            beta_mode="fixed",
            fixed_beta=0.5,
            warp_region="all",
        )
        pipe.set_noise_shaper(shaper)
        self.assertIs(pipe.noise_shaper, shaper)
        self.assertEqual(set(pipe.components), before)
        self.assertNotIn("noise_shaper", pipe.config)
        pipe.to("cpu")

    def test_stc_receives_deterministic_vae_mode(self):
        vae = _VAE((6, 4, 8, 8))
        brushnet_latents, stc_latents = (
            _encode_brushnet_and_stc_condition_latents(
                vae,
                torch.zeros(6, 3, 64, 64),
            )
        )
        torch.testing.assert_close(
            brushnet_latents, torch.full_like(brushnet_latents, 0.5)
        )
        torch.testing.assert_close(
            stc_latents, torch.full_like(stc_latents, 1.25)
        )
        self.assertEqual(vae.distribution.sample_calls, 1)
        self.assertEqual(vae.distribution.mode_calls, 1)

    def test_full_flow_without_motion_flow_uses_clip_length_batching(self):
        noise = torch.arange(6 * 4 * 2 * 2, dtype=torch.float32).reshape(
            6, 4, 2, 2
        )
        # Twice the expected batch mimics condition duplication under CFG. Only
        # the first conditional sequence must be sent to STC.
        structural = torch.randn(12, 4, 2, 2)
        mask = torch.ones(12, 1, 2, 2)
        result = _prepare_stc_noise_sequences(
            noise,
            structural,
            mask,
            flow_input=None,
            flow_mode="full",
            clip_length=3,
        )
        self.assertEqual(result["batch"], 2)
        self.assertEqual(result["frames"], 3)
        self.assertIsNone(result["backward_flow"])
        self.assertEqual(tuple(result["independent_noise"].shape), (2, 3, 4, 2, 2))
        self.assertEqual(tuple(result["decoded_latents"].shape), (2, 3, 4, 2, 2))
        self.assertEqual(tuple(result["bg_mask"].shape), (2, 3, 1, 2, 2))
        torch.testing.assert_close(
            result["independent_noise"].reshape_as(noise), noise
        )
        torch.testing.assert_close(
            result["decoded_latents"].reshape_as(structural[:6]),
            structural[:6],
        )
        shaper = STCConditionedNoiseShaper(
            condition_channels=5,
            hidden_channels=8,
            num_attention_heads=2,
            flow_prediction_mode="full",
            full_flow_max_displacement=(4.0, 4.0),
            beta_mode="fixed",
            fixed_beta=0.5,
            warp_region="all",
        ).eval()
        shaped = shaper(
            independent_noise=result["independent_noise"],
            decoded_latents=result["decoded_latents"],
            bg_mask=result["bg_mask"],
            backward_flow=result["backward_flow"],
        )
        self.assertEqual(tuple(shaped["noise"].shape), (2, 3, 4, 2, 2))
        self.assertEqual(
            tuple(shaped["predicted_flow_backward"].shape), (2, 2, 2, 2, 2)
        )

    def test_legacy_mode_still_rejects_missing_flow(self):
        with self.assertRaisesRegex(
            ValueError, "Legacy STC noise shaping requires motion_backward_flow"
        ):
            _prepare_stc_noise_sequences(
                torch.randn(3, 4, 2, 2),
                torch.randn(3, 4, 2, 2),
                torch.ones(3, 1, 2, 2),
                flow_input=None,
                flow_mode="legacy",
                clip_length=3,
            )

    def test_full_flow_requires_explicit_clip_length(self):
        with self.assertRaisesRegex(
            ValueError, "requires noise_shaper_clip_length"
        ):
            _prepare_stc_noise_sequences(
                torch.randn(6, 4, 2, 2),
                torch.randn(6, 4, 2, 2),
                torch.ones(6, 1, 2, 2),
                flow_input=None,
                flow_mode="full",
                clip_length=None,
            )

    def test_legacy_flow_defines_batch_and_clip_length(self):
        flow = torch.zeros(2, 2, 2, 8, 8)
        result = _prepare_stc_noise_sequences(
            torch.randn(6, 4, 2, 2),
            torch.randn(6, 4, 2, 2),
            torch.ones(6, 1, 2, 2),
            flow_input=flow,
            flow_mode="legacy",
            clip_length=None,
        )
        self.assertEqual(result["batch"], 2)
        self.assertEqual(result["frames"], 3)
        self.assertIs(result["backward_flow"], flow)

    def test_clip_length_must_divide_flat_noise_batch(self):
        with self.assertRaisesRegex(ValueError, "must be positive and divide"):
            _prepare_stc_noise_sequences(
                torch.randn(5, 4, 2, 2),
                torch.randn(5, 4, 2, 2),
                torch.ones(5, 1, 2, 2),
                flow_input=None,
                flow_mode="full",
                clip_length=3,
            )


if __name__ == "__main__":
    unittest.main()
