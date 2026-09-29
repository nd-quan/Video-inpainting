from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


BRUSHNET_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BRUSHNET_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v2pp_deformation.evaluate_noise_deformation import (  # noqa: E402
    build_clip_local_deformed_noise,
    prefix_clip_records,
)
from STC_encoder_v2pp_deformation.noise_deformation import (  # noqa: E402
    NoiseDeformationHead,
)


class ClipLocalEvaluationTests(unittest.TestCase):
    def make_head(self):
        return NoiseDeformationHead(
            feature_channels=4,
            hidden_channels=8,
            num_hidden_layers=1,
            max_displacement=3.0,
        ).eval()

    def build(self, frame_ids, mode="learned"):
        return build_clip_local_deformed_noise(
            deformation_head=self.make_head(),
            stc_features=torch.randn(1, len(frame_ids), 4, 6, 7),
            frame_ids=frame_ids,
            noise_channels=2,
            noise_seed=1234,
            video="video-A",
            alpha=0.9,
            warp_scope="full",
            bg_mask=torch.ones(1, len(frame_ids), 1, 6, 7),
            normalization_eps=1e-6,
            offset_mode=mode,
        )

    def test_overlap_is_regenerated_with_fresh_clip_anchor(self):
        first = self.build((0, 1, 2, 3))
        second = self.build((2, 3, 4, 5))
        self.assertEqual(first.generated_frame_ids, (0, 1, 2, 3))
        self.assertEqual(second.generated_frame_ids, (2, 3, 4, 5))
        self.assertEqual(second.reused_frame_ids, ())
        self.assertFalse(torch.equal(first.lineage_noise[:, 2], second.lineage_noise[:, 0]))

    def test_same_clip_is_order_independent_and_reproducible(self):
        first = self.build((6, 7, 8, 9))
        torch.manual_seed(999)
        second = self.build((6, 7, 8, 9))
        self.assertTrue(torch.equal(first.final_noise, second.final_noise))

    def test_zero_grid_generates_all_frames_with_identity_offsets(self):
        output = self.build((10, 11, 12, 13), mode="zero_grid")
        self.assertEqual(output.generated_frame_ids, (10, 11, 12, 13))
        self.assertTrue(torch.equal(output.offsets, torch.zeros_like(output.offsets)))
        self.assertTrue(output.valid_masks.all())


class PrefixFrameLimitTests(unittest.TestCase):
    def test_150_frame_nonzero_prefix_has_shifted_tail_covering_last_frame(self):
        records = [(300 + index, Path(f"Class_B/BQTerrace/{300 + index:06d}.png")) for index in range(300)]
        clips = prefix_clip_records(
            records,
            clip_length=8,
            clip_stride=6,
            max_frames=150,
        )
        self.assertEqual(clips[0][1], tuple(range(300, 308)))
        self.assertEqual(clips[-1][1], tuple(range(442, 450)))
        self.assertEqual(max(frame_id for _, ids in clips for frame_id in ids), 449)
        self.assertEqual(len({frame_id for _, ids in clips for frame_id in ids}), 150)

    def test_shorter_sequence_uses_its_complete_prefix_and_tail(self):
        records = [(120 + index, Path(f"Class_B/ParkScene/{120 + index:06d}.png")) for index in range(120)]
        clips = prefix_clip_records(
            records,
            clip_length=8,
            clip_stride=6,
            max_frames=150,
        )
        self.assertEqual(clips[-1][1], tuple(range(232, 240)))
        self.assertEqual(len({frame_id for _, ids in clips for frame_id in ids}), 120)


if __name__ == "__main__":
    unittest.main()
