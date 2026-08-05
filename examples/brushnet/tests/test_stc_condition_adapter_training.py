import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import torch.nn as nn


BRUSHNET_EXAMPLE = Path(__file__).resolve().parents[1]
if str(BRUSHNET_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_EXAMPLE))

from diffusers.models.stc_condition_adapter import STCBrushNetConditionAdapter
from diffusers.models.stc_noise_shaper import STCConditionedNoiseShaper
from stc_condition_adapter_training import (
    FrozenBrushNetSTCConditionModel,
    build_stage3_adapter,
    configure_stage3_trainable,
    load_stage1_fixed_noise_shaper,
    load_stage2_noise_shaper,
    validate_stage3_adapter,
    validate_stage3_noise_shaper,
)
from train_stc_condition_adapter_vcm import _load_components
from train_stc_noise_fusion_vcm import build_ip_conditioner


class _FakeBrushNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.brushnet_down_blocks = nn.ModuleList(
            [nn.Conv2d(4, 4, 1), nn.Conv2d(4, 6, 1)]
        )
        self.brushnet_mid_block = nn.Conv2d(4, 6, 1)
        self.brushnet_up_blocks = nn.ModuleList([nn.Conv2d(4, 4, 1)])

    def forward(self, sample, *args, **kwargs):
        pooled = torch.nn.functional.avg_pool2d(sample, 2)
        down = (sample, torch.cat((pooled, pooled[:, :2]), dim=1))
        mid = torch.cat((pooled, pooled[:, :2]), dim=1)
        up = (sample,)
        return down, mid, up


class _FakeUNet(nn.Module):
    def forward(
        self,
        sample,
        *args,
        down_block_add_samples=None,
        mid_block_add_sample=None,
        **kwargs,
    ):
        mid = torch.nn.functional.interpolate(
            mid_block_add_sample[:, :4],
            size=sample.shape[-2:],
            mode="nearest",
        )
        return (sample + down_block_add_samples[0][:, :4] + mid,)


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
            "predicted_flow_backward": independent_noise.new_zeros(
                batch, frames - 1, 2, *independent_noise.shape[-2:]
            ),
            "beta_mean": independent_noise.new_tensor(0.5),
            "effective_beta_mean": independent_noise.new_tensor(0.5),
        }


class Stage3ConditionAdapterTest(unittest.TestCase):
    def _noise_shaper(self, beta_mode="fixed", fixed_beta=0.5):
        options = {
            "latent_channels": 4,
            "condition_channels": 5,
            "hidden_channels": 8,
            "num_attention_heads": 2,
            "num_transformer_layers": 1,
            "encoder_architecture": "videocomposer",
            "condition_group_channels": (5,),
            "videocomposer_pool_size": 8,
            "flow_prediction_mode": "full",
            "full_flow_max_displacement": (8.0, 8.0),
            "beta_mode": beta_mode,
            "warp_region": "all",
        }
        if beta_mode == "fixed":
            options["fixed_beta"] = fixed_beta
        else:
            options["initial_beta"] = fixed_beta
        return STCConditionedNoiseShaper(**options)

    def _save_component_pointer(self, root, model, component_name):
        checkpoint = root / "checkpoint-0000123"
        component = checkpoint / component_name
        model.save_pretrained(component)
        pointer = root / "best.json"
        pointer.write_text(
            '{"checkpoint": "' + str(checkpoint) + '", "step": 123}',
            encoding="utf-8",
        )
        return pointer, component

    def _adapter(self, use_up=True):
        return STCBrushNetConditionAdapter(
            input_channels=3,
            down_channels=(4, 6),
            mid_channel=6,
            up_channels=(4,),
            bottleneck_channels=4,
            use_down=True,
            use_mid=True,
            use_up=use_up,
        )

    def test_zero_initialization_exactly_preserves_brushnet_residuals(self):
        adapter = self._adapter(use_up=True)
        features = torch.randn(2, 3, 3, 8, 8)
        down = (torch.randn(6, 4, 8, 8), torch.randn(6, 6, 4, 4))
        mid = torch.randn(6, 6, 4, 4)
        up = (torch.randn(6, 4, 8, 8),)
        output = adapter(features, down, mid, up)
        for actual, expected in zip(output["down"], down):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(output["mid"], mid, rtol=0, atol=0)
        for actual, expected in zip(output["up"], up):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(float(output["correction_rms"]), 0.0)
        self.assertEqual(float(output["correction_energy"]), 0.0)
        output["correction_energy"].backward()
        gradients = [
            parameter.grad
            for parameter in adapter.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(
            all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        )

    def test_save_and_load_preserves_adapter_config(self):
        adapter = self._adapter(use_up=False)
        with tempfile.TemporaryDirectory() as directory:
            adapter.save_pretrained(directory)
            loaded = STCBrushNetConditionAdapter.from_pretrained(directory)
        self.assertEqual(tuple(loaded.config.down_channels), (4, 6))
        self.assertFalse(loaded.config.use_up)

    def test_adapter_corrections_match_brushnet_port_dtype(self):
        adapter = self._adapter(use_up=True).float()
        features = torch.randn(1, 2, 3, 8, 8, dtype=torch.float32)
        down = (
            torch.randn(2, 4, 8, 8, dtype=torch.float16),
            torch.randn(2, 6, 4, 4, dtype=torch.float16),
        )
        mid = torch.randn(2, 6, 4, 4, dtype=torch.float16)
        up = (torch.randn(2, 4, 8, 8, dtype=torch.float16),)
        output = adapter(features, down, mid, up)

        self.assertTrue(
            all(tensor.dtype == torch.float16 for tensor in output["down"])
        )
        self.assertEqual(output["mid"].dtype, torch.float16)
        self.assertTrue(
            all(tensor.dtype == torch.float16 for tensor in output["up"])
        )
        self.assertTrue(
            all(
                tensor.dtype == torch.float16
                for tensor in output["down_corrections"]
                + (output["mid_correction"],)
                + output["up_corrections"]
            )
        )

    def test_frozen_denoiser_backpropagates_to_zero_initialized_adapter(self):
        noise_shaper = _FakeNoiseShaper()
        brushnet = _FakeBrushNet()
        adapter = self._adapter(use_up=False)
        model = FrozenBrushNetSTCConditionModel(
            noise_shaper,
            adapter,
            brushnet,
            _FakeUNet(),
            alphas_cumprod=torch.linspace(0.99, 0.01, 20),
        )
        configure_stage3_trainable(model, train_stc_encoder=False)
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
        gradient = adapter.mid_projection.output_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in noise_shaper.parameters()
            )
        )

    def test_optional_stc_finetuning_exposes_only_encoder_and_adapter(self):
        model = FrozenBrushNetSTCConditionModel(
            _FakeNoiseShaper(),
            self._adapter(use_up=False),
            _FakeBrushNet(),
            _FakeUNet(),
            alphas_cumprod=torch.ones(10),
        )
        configure_stage3_trainable(model, train_stc_encoder=True)
        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(any(name.startswith("condition_adapter.") for name in trainable))
        self.assertTrue(any(name.startswith("noise_shaper.stc_encoder.") for name in trainable))
        self.assertFalse(any("flow_head" in name for name in trainable))
        self.assertFalse(any("beta_head" in name for name in trainable))

    def test_noise_shaper_exposes_the_exact_reused_stc_features(self):
        shaper = STCConditionedNoiseShaper(
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
            beta_mode="learned",
            initial_beta=0.5,
            warp_region="all",
        )
        decoded = torch.randn(1, 3, 4, 8, 8)
        background = torch.ones(1, 3, 1, 8, 8)
        expected = shaper.encode_stc_features(decoded, background)
        actual = shaper(
            independent_noise=torch.randn_like(decoded),
            decoded_latents=decoded,
            bg_mask=background,
        )["stc_features"]
        torch.testing.assert_close(actual, expected)

    def test_fixed_stage1_loader_resolves_pointer_and_preserves_flow_weights(self):
        source = self._noise_shaper(beta_mode="fixed", fixed_beta=0.5)
        with tempfile.TemporaryDirectory() as directory:
            pointer, component = self._save_component_pointer(
                Path(directory), source, "flow_predictor"
            )
            loaded, resolved = load_stage1_fixed_noise_shaper(
                pointer, fixed_beta=0.7
            )

        self.assertEqual(resolved, component.resolve())
        self.assertEqual(str(loaded.config.beta_mode).lower(), "fixed")
        self.assertAlmostEqual(float(loaded.config.fixed_beta), 0.7)
        self.assertEqual(str(loaded.config.warp_region).lower(), "all")
        self.assertIsNone(loaded.beta_head)
        for name, expected in source.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], expected)

    def test_fixed_stage1_loader_rejects_beta_outside_unit_interval(self):
        source = self._noise_shaper(beta_mode="fixed", fixed_beta=0.5)
        with tempfile.TemporaryDirectory() as directory:
            pointer, _ = self._save_component_pointer(
                Path(directory), source, "flow_predictor"
            )
            with self.assertRaisesRegex(ValueError, "fixed_beta"):
                load_stage1_fixed_noise_shaper(pointer, fixed_beta=1.01)

    def test_trainer_dispatch_builds_fixed_stage1_adapter_only(self):
        source = self._noise_shaper(beta_mode="fixed", fixed_beta=0.5)
        brushnet = _FakeBrushNet()
        with tempfile.TemporaryDirectory() as directory:
            pointer, component = self._save_component_pointer(
                Path(directory), source, "flow_predictor"
            )
            config = {
                "model": {
                    "noise_source": "fixed_stage1",
                    "stage1_checkpoint": str(pointer),
                    "fixed_beta": 0.7,
                },
                "adapter": {
                    "input_channels": 8,
                    "bottleneck_channels": 4,
                    "use_down": True,
                    "use_mid": True,
                    "use_up": False,
                },
            }
            loaded, adapter, resolved, mode = _load_components(
                config, None, brushnet, torch.device("cpu")
            )
        self.assertEqual(mode, "fixed_stage1")
        self.assertEqual(resolved, component.resolve())
        self.assertAlmostEqual(float(loaded.config.fixed_beta), 0.7)
        self.assertIsNone(loaded.beta_head)
        self.assertEqual(int(adapter.config.input_channels), 8)
        self.assertFalse(any(parameter.requires_grad for parameter in loaded.parameters()))

    def test_adapter_validation_rejects_another_stc_width(self):
        shaper = self._noise_shaper(beta_mode="fixed", fixed_beta=0.5)
        brushnet = _FakeBrushNet()
        adapter = build_stage3_adapter(
            brushnet,
            shaper,
            {"input_channels": 8, "bottleneck_channels": 4},
        )
        validate_stage3_adapter(adapter, brushnet, shaper)
        wrong = self._adapter(use_up=False)
        with self.assertRaisesRegex(ValueError, "channel mismatch"):
            validate_stage3_adapter(wrong, brushnet, shaper)

    def test_stage3_validator_accepts_both_explicit_source_modes(self):
        fixed = self._noise_shaper(beta_mode="fixed", fixed_beta=0.7)
        learned = self._noise_shaper(beta_mode="learned", fixed_beta=0.5)
        validate_stage3_noise_shaper(
            fixed, expected_mode="fixed_stage1", fixed_beta=0.7
        )
        validate_stage3_noise_shaper(learned, expected_mode="learned_stage2")

    def test_stage3_validator_rejects_wrong_mode_or_fixed_beta(self):
        fixed = self._noise_shaper(beta_mode="fixed", fixed_beta=0.7)
        with self.assertRaisesRegex(ValueError, "model.noise_source"):
            validate_stage3_noise_shaper(fixed, expected_mode="legacy_fixed")
        with self.assertRaisesRegex(ValueError, "learned_stage2"):
            validate_stage3_noise_shaper(fixed, expected_mode="learned_stage2")
        with self.assertRaisesRegex(ValueError, "Fixed-beta"):
            validate_stage3_noise_shaper(
                fixed, expected_mode="fixed_stage1", fixed_beta=0.9
            )

    def test_learned_stage2_loader_remains_learned_only(self):
        fixed = self._noise_shaper(beta_mode="fixed", fixed_beta=0.7)
        learned = self._noise_shaper(beta_mode="learned", fixed_beta=0.5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixed_pointer, _ = self._save_component_pointer(
                root / "fixed", fixed, "noise_shaper"
            )
            learned_pointer, learned_component = self._save_component_pointer(
                root / "learned", learned, "noise_shaper"
            )
            with self.assertRaisesRegex(ValueError, "learned-beta"):
                load_stage2_noise_shaper(fixed_pointer)
            loaded, resolved = load_stage2_noise_shaper(learned_pointer)

        self.assertEqual(resolved, learned_component.resolve())
        self.assertEqual(str(loaded.config.beta_mode).lower(), "learned")

    def test_fixed_beta_test2_configs_select_stage1_and_v8(self):
        config_names = (
            "train_vcm_stc_condition_adapter_fixed_beta_test2_2gpu.json",
            "train_vcm_stc_condition_adapter_fixed_beta_test2_smoke_2gpu.json",
        )
        for config_name in config_names:
            with self.subTest(config=config_name):
                config_path = BRUSHNET_EXAMPLE / "configs" / config_name
                config = json.loads(config_path.read_text(encoding="utf-8"))
                model = config["model"]
                adapter = config["adapter"]

                self.assertEqual(model["noise_source"], "fixed_stage1")
                self.assertTrue(str(model["stage1_checkpoint"]).endswith("best.json"))
                self.assertGreaterEqual(float(model["fixed_beta"]), 0.0)
                self.assertLessEqual(float(model["fixed_beta"]), 1.0)
                self.assertNotIn("stage2_checkpoint", model)
                self.assertEqual(
                    Path(model["v8_checkpoint"]).name,
                    "checkpoint-2000",
                )
                self.assertIn("fine-tuning-v8", Path(model["v8_checkpoint"]).parts)
                self.assertIs(adapter["train_stc_encoder"], False)
                self.assertGreater(
                    int(config["trainer"]["max_consecutive_amp_overflows"]), 0
                )
                expected_amp = "smoke" not in config_name
                self.assertIs(config["trainer"]["amp"], expected_amp)

    def test_ip_conditioner_modules_follow_unet_dtype(self):
        unet = nn.Linear(4, 4).float()
        conditioner = Mock()
        conditioner.image_encoder = nn.Linear(4, 4).half()
        conditioner.image_proj_model = nn.Linear(4, 4).half()
        conditioner.fusion_module = nn.Linear(4, 4).half()
        config = {
            "ip_adapter": {
                "enabled": True,
                "image_encoder": "unused",
                "weights": "unused",
                "fusion_weights": "unused",
                "num_tokens": 4,
                "scale": 1.0,
            }
        }
        with patch(
            "train_stc_noise_fusion_vcm.FusionIPAdapter",
            return_value=conditioner,
        ):
            actual = build_ip_conditioner(unet, config, torch.device("cpu"))

        self.assertIs(actual, conditioner)
        for module in (
            conditioner.image_encoder,
            conditioner.image_proj_model,
            conditioner.fusion_module,
        ):
            self.assertTrue(
                all(parameter.dtype == torch.float32 for parameter in module.parameters())
            )
            self.assertFalse(any(parameter.requires_grad for parameter in module.parameters()))
            self.assertFalse(module.training)
        conditioner.set_scale.assert_called_once_with(1.0)


if __name__ == "__main__":
    unittest.main()
