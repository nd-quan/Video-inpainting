import sys
import tempfile
import unittest
from pathlib import Path

import torch


BRUSHNET_EXAMPLE = Path(__file__).resolve().parents[1]
if str(BRUSHNET_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_EXAMPLE))

from diffusers.models.stc_noise_shaper import STCConditionedNoiseShaper
from stc_noise_fusion_training import (
    beta_parameters,
    build_stage2_noise_shaper,
    temporal_latent_loss,
)


class Stage2NoiseFusionTest(unittest.TestCase):
    def _stage1(self):
        return STCConditionedNoiseShaper(
            latent_channels=4,
            condition_channels=5,
            hidden_channels=8,
            num_attention_heads=2,
            num_transformer_layers=1,
            encoder_architecture="videocomposer",
            condition_group_channels=(5,),
            videocomposer_pool_size=16,
            flow_prediction_mode="full",
            full_flow_max_displacement=(8.0, 8.0),
            beta_mode="fixed",
            fixed_beta=0.5,
            warp_region="all",
        )

    def test_transfer_is_exact_and_only_beta_is_trainable(self):
        torch.manual_seed(7)
        stage1 = self._stage1()
        with tempfile.TemporaryDirectory() as directory:
            component = Path(directory) / "checkpoint-0000001" / "flow_predictor"
            stage1.save_pretrained(component)
            stage2, resolved = build_stage2_noise_shaper(
                component.parent,
                {"beta_min": 0.05, "beta_max": 0.95, "initial_beta": 0.5},
            )
        self.assertEqual(resolved, component.resolve())
        self.assertEqual(stage2.config.beta_mode, "learned")
        self.assertIsNotNone(stage2.beta_head)
        stage1_state = stage1.state_dict()
        stage2_state = stage2.state_dict()
        for name, value in stage1_state.items():
            self.assertTrue(torch.equal(value, stage2_state[name]), name)
        trainable = {
            name for name, parameter in stage2.named_parameters() if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("beta_head.") for name in trainable))
        self.assertGreater(sum(parameter.numel() for parameter in beta_parameters(stage2)), 0)

    def test_noise_fusion_backpropagates_only_to_beta(self):
        stage1 = self._stage1()
        with tempfile.TemporaryDirectory() as directory:
            component = Path(directory) / "flow_predictor"
            stage1.save_pretrained(component)
            stage2, _ = build_stage2_noise_shaper(
                component,
                {"beta_min": 0.05, "beta_max": 0.95, "initial_beta": 0.5},
            )
        independent = torch.randn(1, 3, 4, 8, 8)
        decoded = torch.randn_like(independent)
        background = torch.ones(1, 3, 1, 8, 8)
        output = stage2(
            independent_noise=independent,
            decoded_latents=decoded,
            bg_mask=background,
        )
        loss = output["noise"].square().mean() + output["beta"].mean()
        loss.backward()
        beta_gradients = [
            parameter.grad for parameter in stage2.beta_head.parameters()
        ]
        self.assertTrue(any(gradient is not None for gradient in beta_gradients))
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in stage2.named_parameters()
                if not name.startswith("beta_head.")
            )
        )

    def test_temporal_loss_accepts_adjacent_backward_flow(self):
        predicted_clean = torch.randn(2, 4, 4, 8, 8, requires_grad=True)
        backward_flow = torch.zeros(2, 3, 2, 8, 8)
        loss = temporal_latent_loss(predicted_clean, backward_flow)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(predicted_clean.grad)


if __name__ == "__main__":
    unittest.main()
