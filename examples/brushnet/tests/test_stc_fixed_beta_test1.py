import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusers.models.stc_noise_shaper import STCConditionedNoiseShaper
from evaluate_stc_fixed_beta_test1_vcm import (
    clip_output_is_complete,
    comparable_run_config,
    composite_input_roi,
    gaussian_soft_composite_input_roi,
)
from stc_fixed_beta_test1 import (
    deterministic_clip_seed,
    inspect_manifest_split,
    load_fixed_beta_stage1,
    resolve_v8_components,
)


class FixedBetaTest1Test(unittest.TestCase):
    def make_stage1(self, root: Path):
        checkpoint = root / "checkpoint-0000123"
        component = checkpoint / "flow_predictor"
        model = STCConditionedNoiseShaper(
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
        model.save_pretrained(component)
        pointer = root / "best.json"
        pointer.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "step": 123,
                    "valid_epe": 0.25,
                }
            ),
            encoding="utf-8",
        )
        return model, pointer

    def test_fixed_beta_override_preserves_strict_stage1_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            source, pointer = self.make_stage1(Path(directory))
            loaded, metadata = load_fixed_beta_stage1(pointer, fixed_beta=0.7)
            self.assertEqual(loaded.config.beta_mode, "fixed")
            self.assertAlmostEqual(float(loaded.config.fixed_beta), 0.7)
            self.assertEqual(loaded.config.warp_region, "all")
            self.assertIsNone(loaded.beta_head)
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            self.assertEqual(metadata["stage1_step"], 123)
            for key, value in source.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key], value)

    def test_custom_named_json_checkpoint_pointer_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            _, pointer = self.make_stage1(Path(directory))
            custom_pointer = pointer.with_name("best_to_test1.json")
            pointer.rename(custom_pointer)
            loaded, metadata = load_fixed_beta_stage1(
                custom_pointer, fixed_beta=0.5
            )
            self.assertEqual(metadata["stage1_step"], 123)
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))

    def test_clip_seed_is_beta_independent_and_sequence_specific(self):
        first = deterministic_clip_seed(2026, "Class_A", "Traffic", 10)
        repeated = deterministic_clip_seed(2026, "Class_A", "Traffic", 10)
        other = deterministic_clip_seed(2026, "Class_A", "Traffic", 18)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other)

    def test_v8_resolver_requires_deployment_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "brushnet").mkdir()
            (checkpoint / "brushnet" / "config.json").write_text("{}")
            (checkpoint / "ipadapter").mkdir()
            (checkpoint / "ipadapter" / "model.safetensors").touch()
            (checkpoint / "ipadapter" / "fusion_module.safetensors").touch()
            components = resolve_v8_components(checkpoint)
            self.assertEqual(components["checkpoint"], checkpoint.resolve())
            self.assertTrue(components["ip_adapter"].is_file())

    def test_manifest_reports_empty_test_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "splits": ["test"],
                        "sequences": [
                            {
                                "class": "A",
                                "name": "present",
                                "splits": {"test": {"frame_count": 8}},
                            },
                            {
                                "class": "B",
                                "name": "missing",
                                "splits": {"test": {"frame_count": 0}},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            coverage = inspect_manifest_split(manifest, "test")
            self.assertEqual(coverage["frame_count"], 8)
            self.assertEqual(coverage["empty_sequences"], ["B/missing"])

    def test_roi_compositing_copies_input_only_inside_roi(self):
        generated = [Image.fromarray(torch.zeros(2, 2, 3).byte().numpy())]
        decoded = torch.ones(1, 3, 2, 2)
        roi = torch.zeros(1, 1, 2, 2)
        roi[:, :, 0, 0] = 1
        result = torch.from_numpy(
            np.array(composite_input_roi(generated, decoded, roi)[0])
        )
        self.assertTrue(torch.equal(result[0, 0], torch.tensor([255, 255, 255])))
        self.assertTrue(torch.equal(result[1, 1], torch.tensor([0, 0, 0])))

    def test_gaussian_soft_composite_keeps_bg_and_softens_inner_roi_edge(self):
        generated = [Image.fromarray(np.zeros((7, 7, 3), dtype=np.uint8))]
        decoded = torch.ones(1, 3, 7, 7)
        roi = torch.zeros(1, 1, 7, 7)
        roi[:, :, 2:5, 2:5] = 1
        result = np.asarray(
            gaussian_soft_composite_input_roi(
                generated, decoded, roi, kernel_size=3, sigma=0.0
            )[0]
        )
        # Generated BG is unchanged, ROI center is exact input, and the inner
        # edge is a fractional Gaussian mix rather than a hard seam.
        self.assertTrue(np.array_equal(result[0, 0], np.zeros(3, dtype=np.uint8)))
        self.assertTrue(np.array_equal(result[3, 3], np.full(3, 255, dtype=np.uint8)))
        self.assertTrue(np.all(result[2, 2] > 0))
        self.assertTrue(np.all(result[2, 2] < 255))

    def test_gaussian_blending_rejects_even_kernel(self):
        with self.assertRaisesRegex(ValueError, "positive odd"):
            gaussian_soft_composite_input_roi(
                [Image.new("RGB", (2, 2))],
                torch.ones(1, 3, 2, 2),
                torch.ones(1, 1, 2, 2),
                kernel_size=4,
            )

    def test_resume_requires_all_images_to_be_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            clip_dir = Path(directory)
            final = clip_dir / "final"
            final.mkdir()
            indices = [10, 11]
            metrics = clip_dir / "clip_metrics.json"
            metrics.write_text(json.dumps({"frame_indices": indices}))
            Image.new("RGB", (2, 2)).save(final / "000010.png")
            self.assertFalse(
                clip_output_is_complete(metrics, clip_dir, indices, False)
            )
            Image.new("RGB", (2, 2)).save(final / "000011.png")
            self.assertTrue(
                clip_output_is_complete(metrics, clip_dir, indices, False)
            )
            (final / "000011.png").write_bytes(b"not an image")
            self.assertFalse(
                clip_output_is_complete(metrics, clip_dir, indices, False)
            )

    def test_overwrite_and_max_clips_do_not_change_run_identity(self):
        first = {"inference": {"overwrite": False, "max_clips": 1, "steps": 5}}
        second = {"inference": {"overwrite": True, "max_clips": 20, "steps": 5}}
        self.assertEqual(comparable_run_config(first), comparable_run_config(second))


if __name__ == "__main__":
    unittest.main()
