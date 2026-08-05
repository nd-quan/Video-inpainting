import tempfile
import unittest

import torch

from diffusers import UNet2DConditionModel
from diffusers.models.unet_temporal_adapter import (
    DiffusionUNetTemporalAdapter,
    TemporalConvResidualBlock,
)


class TemporalConvResidualBlockTest(unittest.TestCase):
    def test_zero_initialization_is_exact_identity_and_has_gradients(self):
        torch.manual_seed(3)
        block = TemporalConvResidualBlock(8, bottleneck_channels=4)
        inputs = torch.randn(6, 8, 5, 5, requires_grad=True)
        outputs = block(inputs, batch_size=2, num_frames=3)
        self.assertTrue(torch.equal(inputs, outputs))
        outputs.square().mean().backward()
        self.assertIsNotNone(block.output_projection.weight.grad)
        self.assertGreater(float(block.output_projection.weight.grad.abs().sum()), 0.0)

    def test_single_frame_path_remains_identity_after_training(self):
        block = TemporalConvResidualBlock(8, bottleneck_channels=4)
        torch.nn.init.normal_(block.output_projection.weight)
        inputs = torch.randn(2, 8, 4, 4)
        outputs = block(inputs, batch_size=2, num_frames=1)
        self.assertTrue(torch.equal(inputs, outputs))

    def test_trained_block_connects_adjacent_frames(self):
        torch.manual_seed(5)
        block = TemporalConvResidualBlock(8, bottleneck_channels=4)
        torch.nn.init.normal_(block.output_projection.weight, std=0.1)
        reference = torch.zeros(3, 8, 4, 4)
        changed = reference.clone()
        changed[0] = torch.randn(8, 4, 4)
        reference_output = block(reference, batch_size=1, num_frames=3)
        changed_output = block(changed, batch_size=1, num_frames=3)
        neighbor_change = (changed_output[1] - reference_output[1]).abs().sum()
        self.assertGreater(float(neighbor_change), 0.0)

    def test_invalid_flattened_video_shape_is_rejected(self):
        block = TemporalConvResidualBlock(8, bottleneck_channels=4)
        with self.assertRaisesRegex(ValueError, "batch_size\\*num_frames"):
            block(torch.randn(5, 8, 4, 4), batch_size=2, num_frames=3)


class DiffusionUNetTemporalAdapterTest(unittest.TestCase):
    def _adapter(self):
        return DiffusionUNetTemporalAdapter(
            block_out_channels=(8, 16),
            down_block_indices=(0,),
            use_mid=True,
            up_block_indices=(),
            bottleneck_channels=4,
        )

    def test_save_load_round_trip(self):
        adapter = self._adapter()
        with tempfile.TemporaryDirectory() as directory:
            adapter.save_pretrained(directory)
            loaded = DiffusionUNetTemporalAdapter.from_pretrained(directory)
        self.assertEqual(set(adapter.state_dict()), set(loaded.state_dict()))
        for name, value in adapter.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[name]), name)

    def test_unselected_stage_is_identity(self):
        adapter = self._adapter()
        features = torch.randn(6, 16, 4, 4)
        result = adapter(
            features,
            batch_size=2,
            num_frames=3,
            stage="up",
            block_index=0,
        )
        self.assertIs(result, features)

    def test_zero_adapter_preserves_tiny_unet_output(self):
        torch.manual_seed(9)
        unet = UNet2DConditionModel(
            sample_size=16,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
            block_out_channels=(32, 64),
            layers_per_block=1,
            norm_num_groups=8,
            mid_block_type="UNetMidBlock2D",
            cross_attention_dim=16,
        ).eval()
        adapter = DiffusionUNetTemporalAdapter.from_unet(
            unet,
            down_block_indices=(0,),
            use_mid=True,
            up_block_indices=(),
            bottleneck_channels=8,
        ).eval()
        sample = torch.randn(6, 4, 16, 16)
        timestep = torch.full((6,), 100, dtype=torch.long)
        text = torch.randn(6, 1, 16)
        with torch.no_grad():
            baseline = unet(sample, timestep, text, return_dict=False)[0]
            adapted = unet(
                sample,
                timestep,
                text,
                temporal_adapter=adapter,
                temporal_num_frames=3,
                return_dict=False,
            )[0]
        self.assertTrue(torch.equal(baseline, adapted))


if __name__ == "__main__":
    unittest.main()
