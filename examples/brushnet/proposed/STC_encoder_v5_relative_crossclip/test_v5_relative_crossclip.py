#!/usr/bin/env python

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
BRUSHNET_DIR = THIS_DIR.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v4pp_bg_feature.bg_focused_flow_aligned_stc_adapter import (
    BGFocusedFlowAlignedRGBSTCAdapter,
)
from STC_encoder_v5_relative_crossclip.cross_clip_data import (
    build_predecessor_index,
    suffix_prefix_overlap,
)
from STC_encoder_v5_relative_crossclip.relative_crossclip_stc_adapter import (
    RelativeCrossClipBGSTCAdapter,
)


class V5RelativeCrossClipTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.source = BGFocusedFlowAlignedRGBSTCAdapter(
            hidden_channels=8,
            num_heads=1,
            num_layers=1,
            dropout=0.0,
            flow_max_displacement=(2.0, 2.0),
        ).eval()
        self.v5 = RelativeCrossClipBGSTCAdapter(
            hidden_channels=8,
            num_heads=1,
            num_layers=1,
            dropout=0.0,
            flow_max_displacement=(2.0, 2.0),
            relative_position_max_distance=8,
            cross_clip_memory_frames=2,
        ).eval()
        transfer = self.v5.load_state_dict(self.source.state_dict(), strict=False)
        self.assertEqual(
            set(transfer.missing_keys),
            {
                "stc_adapter.temporal_blocks.0.cross_clip_gate",
                "stc_adapter.temporal_blocks.0.attention.relative_position_bias",
            },
        )
        self.assertEqual(transfer.unexpected_keys, [])
        self.previous_rgb = torch.randn(1, 4, 3, 32, 32)
        self.current_rgb = torch.randn(1, 4, 3, 32, 32)
        self.previous_bg = torch.ones(1, 4, 1, 32, 32)
        self.current_bg = torch.ones(1, 4, 1, 32, 32)
        self.previous_ids = torch.tensor([[0, 1, 2, 3]])
        self.current_ids = torch.tensor([[2, 3, 4, 5]])

    def test_suffix_prefix_and_predecessor_contract(self):
        self.assertEqual(suffix_prefix_overlap((0, 1, 2, 3), (2, 3, 4, 5)), 2)
        clips = (
            ("a", (), (0, 1, 2, 3)),
            ("a", (), (2, 3, 4, 5)),
            ("a", (), (10, 11, 12, 13)),
            ("b", (), (2, 3, 4, 5)),
        )
        predecessor, overlap = build_predecessor_index(clips, expected_stride=2)
        self.assertEqual(predecessor, (None, 0, None, None))
        self.assertEqual(overlap, (0, 2, 0, 0))

    def test_zero_initialized_v5_matches_v4pp_without_memory(self):
        with torch.no_grad():
            source = self.source(
                self.current_rgb, self.current_bg, output_size=(4, 4)
            )
            v5 = self.v5(
                self.current_rgb,
                self.current_bg,
                output_size=(4, 4),
                frame_ids=self.current_ids,
            )
        torch.testing.assert_close(v5.features, source.features, rtol=0, atol=0)
        torch.testing.assert_close(
            v5.aligned_spatial_features,
            source.aligned_spatial_features,
            rtol=0,
            atol=0,
        )

    def test_zero_gate_preserves_output_even_when_memory_is_supplied(self):
        with torch.no_grad():
            previous = self.v5(
                self.previous_rgb,
                self.previous_bg,
                output_size=(4, 4),
                frame_ids=self.previous_ids,
            )
            no_memory = self.v5(
                self.current_rgb,
                self.current_bg,
                output_size=(4, 4),
                frame_ids=self.current_ids,
            )
            with_memory = self.v5(
                self.current_rgb,
                self.current_bg,
                output_size=(4, 4),
                frame_ids=self.current_ids,
                temporal_memory=previous.temporal_memory,
            )
        self.assertEqual(int(with_memory.memory_overlap_count.item()), 2)
        torch.testing.assert_close(
            with_memory.features, no_memory.features, rtol=0, atol=0
        )
        self.assertFalse(with_memory.temporal_memory.layer_features[0].requires_grad)

    def test_bias_and_cross_gate_receive_gradients(self):
        self.v5.train()
        with torch.no_grad():
            previous = self.v5(
                self.previous_rgb,
                self.previous_bg,
                output_size=(4, 4),
                frame_ids=self.previous_ids,
            )
        output = self.v5(
            self.current_rgb,
            self.current_bg,
            output_size=(4, 4),
            frame_ids=self.current_ids,
            temporal_memory=previous.temporal_memory,
        )
        output.features.square().mean().backward()
        block = self.v5.stc_adapter.temporal_blocks[0]
        self.assertIsNotNone(block.attention.relative_position_bias.grad)
        self.assertGreater(float(block.attention.relative_position_bias.grad.abs().sum()), 0)
        self.assertIsNotNone(block.cross_clip_gate.grad)
        self.assertGreater(float(block.cross_clip_gate.grad.abs()), 0)

    def test_nonoverlapping_memory_is_masked_even_with_an_open_gate(self):
        with torch.no_grad():
            previous = self.v5(
                self.previous_rgb,
                self.previous_bg,
                output_size=(4, 4),
                frame_ids=self.previous_ids,
            )
            self.v5.stc_adapter.temporal_blocks[0].cross_clip_gate.fill_(1.0)
            no_memory = self.v5(
                self.current_rgb,
                self.current_bg,
                output_size=(4, 4),
                frame_ids=self.current_ids + 100,
            )
            unrelated_memory = self.v5(
                self.current_rgb,
                self.current_bg,
                output_size=(4, 4),
                frame_ids=self.current_ids + 100,
                temporal_memory=previous.temporal_memory,
            )
        self.assertEqual(int(unrelated_memory.memory_overlap_count.item()), 0)
        torch.testing.assert_close(
            unrelated_memory.features, no_memory.features, rtol=0, atol=0
        )

    def test_absolute_relative_bias_is_translation_invariant(self):
        attention = self.v5.stc_adapter.temporal_blocks[0].attention
        with torch.no_grad():
            attention.relative_position_bias.copy_(
                torch.arange(
                    attention.relative_position_bias.numel(), dtype=torch.float32
                ).reshape_as(attention.relative_position_bias)
            )
        first = attention._relative_bias(self.current_ids, self.previous_ids)
        shifted = attention._relative_bias(
            self.current_ids + 1000, self.previous_ids + 1000
        )
        torch.testing.assert_close(first, shifted, rtol=0, atol=0)

    def test_full_component_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            self.v5.save_pretrained(directory, safe_serialization=True)
            loaded = RelativeCrossClipBGSTCAdapter.from_pretrained(directory)
            self.assertEqual(int(loaded.config.cross_clip_memory_frames), 2)
            self.assertEqual(int(loaded.config.relative_position_max_distance), 8)
            self.assertEqual(set(self.v5.state_dict()), set(loaded.state_dict()))


if __name__ == "__main__":
    unittest.main()
