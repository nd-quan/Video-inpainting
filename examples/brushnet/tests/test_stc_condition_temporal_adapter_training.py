import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn


BRUSHNET_EXAMPLE = Path(__file__).resolve().parents[1]
if str(BRUSHNET_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_EXAMPLE))

from diffusers.models.stc_condition_adapter import STCBrushNetConditionAdapter
from diffusers.models.unet_temporal_adapter import DiffusionUNetTemporalAdapter
from stc_condition_temporal_adapter_training import (
    FrozenBrushNetSTCConditionTemporalModel,
    joint_stage3_parameter_groups,
    load_or_build_temporal_adapter,
)


class _FakeNoiseShaper(nn.Module):
    def __init__(self):
        super().__init__()
        self.stc_encoder = nn.Conv2d(5, 3, 1)
        self.flow_head = nn.Conv2d(3, 2, 1)
        self.beta_head = nn.Conv2d(3, 1, 1)

    def forward(self, independent_noise, decoded_latents, bg_mask, **kwargs):
        batch, frames = decoded_latents.shape[:2]
        condition = torch.cat((decoded_latents, bg_mask), dim=2)
        features = self.stc_encoder(condition.flatten(0, 1)).reshape(
            batch, frames, 3, *decoded_latents.shape[-2:]
        )
        return {
            "noise": independent_noise,
            "stc_features": features,
            "predicted_flow_backward": decoded_latents.new_zeros(
                batch, frames - 1, 2, *decoded_latents.shape[-2:]
            ),
            "beta_mean": decoded_latents.new_tensor(0.5),
            "effective_beta_mean": decoded_latents.new_tensor(0.5),
        }


class _FakeBrushNet(nn.Module):
    def forward(self, sample, *args, **kwargs):
        return (sample,), sample, ()


class _FakeUNet(nn.Module):
    def forward(
        self,
        sample,
        *args,
        down_block_add_samples=None,
        mid_block_add_sample=None,
        temporal_adapter=None,
        temporal_num_frames=None,
        temporal_adapter_scale=1.0,
        **kwargs,
    ):
        hidden = sample + down_block_add_samples[0] + mid_block_add_sample
        hidden = temporal_adapter(
            hidden,
            batch_size=hidden.shape[0] // temporal_num_frames,
            num_frames=temporal_num_frames,
            stage="down",
            block_index=0,
            scale=temporal_adapter_scale,
        )
        return (hidden,)


class _FakeUNetConfig:
    block_out_channels = (4, 6)


class _FakeConfiguredUNet:
    config = _FakeUNetConfig()


class JointStage3TemporalTrainingTest(unittest.TestCase):
    def _model(self):
        condition = STCBrushNetConditionAdapter(
            input_channels=3,
            down_channels=(4,),
            mid_channel=4,
            up_channels=(),
            bottleneck_channels=2,
            use_down=True,
            use_mid=True,
            use_up=False,
        )
        temporal = DiffusionUNetTemporalAdapter(
            block_out_channels=(4,),
            down_block_indices=(0,),
            use_mid=False,
            bottleneck_channels=2,
        )
        return FrozenBrushNetSTCConditionTemporalModel(
            noise_shaper=_FakeNoiseShaper(),
            condition_adapter=condition,
            temporal_adapter=temporal,
            brushnet=_FakeBrushNet(),
            unet=_FakeUNet(),
            alphas_cumprod=torch.linspace(0.99, 0.01, 20),
        )

    def test_only_separate_adapters_are_trainable(self):
        model = self._model()
        groups = joint_stage3_parameter_groups(
            model,
            condition_lr=5e-5,
            temporal_lr=1e-4,
        )
        self.assertEqual(
            [group["name"] for group in groups],
            ["condition_adapter", "temporal_adapter"],
        )
        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(any(name.startswith("condition_adapter.") for name in trainable))
        self.assertTrue(any(name.startswith("temporal_adapter.") for name in trainable))
        self.assertFalse(any(name.startswith("noise_shaper.") for name in trainable))
        self.assertFalse(any(name.startswith("brushnet.") for name in trainable))
        self.assertFalse(any(name.startswith("unet.") for name in trainable))

    def test_joint_forward_reaches_both_adapters(self):
        model = self._model()
        joint_stage3_parameter_groups(model, condition_lr=5e-5, temporal_lr=1e-4)
        clean = torch.randn(1, 3, 4, 8, 8)
        output = model(
            clean_latents=clean,
            independent_noise=torch.randn_like(clean),
            decoded_latents=torch.randn_like(clean),
            bg_mask=torch.ones(1, 3, 1, 8, 8),
            timesteps=torch.tensor([5]),
            encoder_hidden_states=torch.randn(1, 2, 8),
            brushnet_condition=torch.randn(1, 3, 5, 8, 8),
        )
        output["prediction"].square().mean().backward()
        condition_grad = (
            model.condition_adapter.down_projections[0].output_projection.weight.grad
        )
        temporal_grad = (
            model.temporal_adapter.down_adapters["0"].output_projection.weight.grad
        )
        self.assertIsNotNone(condition_grad)
        self.assertIsNotNone(temporal_grad)
        self.assertGreater(float(condition_grad.abs().sum()), 0.0)
        self.assertGreater(float(temporal_grad.abs().sum()), 0.0)

    def test_condition_only_initialization_builds_zero_temporal_residual(self):
        config = {
            "down_block_indices": (0,),
            "use_mid": False,
            "bottleneck_channels": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            adapter = load_or_build_temporal_adapter(
                _FakeConfiguredUNet(),
                config,
                initialization=Path(directory),
            )
        projection = adapter.down_adapters["0"].output_projection
        self.assertEqual(float(projection.weight.abs().sum()), 0.0)
        self.assertEqual(float(projection.bias.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
