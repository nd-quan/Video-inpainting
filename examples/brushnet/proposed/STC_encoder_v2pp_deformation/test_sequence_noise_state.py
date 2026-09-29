from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


BRUSHNET_DIR = Path(__file__).resolve().parent.parent
if str(BRUSHNET_DIR) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_DIR))

from STC_encoder_v2pp_deformation.noise_deformation import (  # noqa: E402
    NoiseDeformationHead,
)
from STC_encoder_v2pp_deformation.sequence_noise_state import (  # noqa: E402
    SequenceNoiseState,
    build_sequence_deformed_noise,
)


class SequenceNoiseStateTests(unittest.TestCase):
    def make_head(self):
        # A new head is exact identity warp, which makes cache/RNG assertions
        # independent of learned checkpoint values.
        return NoiseDeformationHead(
            feature_channels=4,
            hidden_channels=8,
            num_hidden_layers=1,
            max_displacement=3.0,
        ).eval()

    def make_state(self, sequence="video-A", seed=123):
        return SequenceNoiseState(
            sequence_id=sequence,
            seed=seed,
            channels=2,
            height=6,
            width=7,
        )

    def test_two_overlapping_clips_reuse_exact_noise(self):
        torch.manual_seed(7)
        head = self.make_head()
        state = self.make_state()
        clip_one_features = torch.randn(1, 4, 4, 6, 7)
        clip_two_features = torch.randn(1, 4, 4, 6, 7)

        first = build_sequence_deformed_noise(
            deformation_head=head,
            stc_features=clip_one_features,
            frame_ids=(0, 1, 2, 3),
            state=state,
            alpha=0.9,
        )
        cached_lineage_2 = state.lineage_noise[2].clone()
        cached_final_2 = state.final_noise[2].clone()
        second = build_sequence_deformed_noise(
            deformation_head=head,
            stc_features=clip_two_features,
            frame_ids=(2, 3, 4, 5),
            state=state,
            alpha=0.9,
        )

        self.assertEqual(first.generated_frame_ids, (0, 1, 2, 3))
        self.assertEqual(first.reused_frame_ids, ())
        self.assertEqual(second.generated_frame_ids, (4, 5))
        self.assertEqual(second.reused_frame_ids, (2, 3))
        self.assertEqual(second.transition_target_frame_ids, (4, 5))
        self.assertEqual(second.overlap_max_abs_difference, 0.0)
        self.assertTrue(torch.equal(second.lineage_noise[0, 0], cached_lineage_2))
        self.assertTrue(torch.equal(second.final_noise[0, 0], cached_final_2))
        self.assertTrue(torch.equal(first.lineage_noise[0, 2], second.lineage_noise[0, 0]))
        self.assertTrue(torch.equal(first.final_noise[0, 2], second.final_noise[0, 0]))
        self.assertEqual(len(state.lineage_noise), 6)
        self.assertEqual(len(state.final_noise), 6)

    def test_shifted_tail_with_larger_overlap(self):
        head = self.make_head()
        state = self.make_state()
        first = build_sequence_deformed_noise(
            deformation_head=head,
            stc_features=torch.randn(1, 4, 4, 6, 7),
            frame_ids=(0, 1, 2, 3),
            state=state,
            alpha=0.9,
        )
        tail = build_sequence_deformed_noise(
            deformation_head=head,
            stc_features=torch.randn(1, 4, 4, 6, 7),
            frame_ids=(1, 2, 3, 4),
            state=state,
            alpha=0.9,
        )
        self.assertEqual(tail.reused_frame_ids, (1, 2, 3))
        self.assertEqual(tail.generated_frame_ids, (4,))
        torch.testing.assert_close(tail.final_noise[0, :3], first.final_noise[0, 1:])

    def test_same_seed_and_sequence_are_order_deterministic(self):
        head = self.make_head()

        def run_once():
            state = self.make_state()
            first = build_sequence_deformed_noise(
                deformation_head=head,
                stc_features=torch.zeros(1, 3, 4, 6, 7),
                frame_ids=(10, 11, 12),
                state=state,
                alpha=0.75,
            )
            second = build_sequence_deformed_noise(
                deformation_head=head,
                stc_features=torch.zeros(1, 3, 4, 6, 7),
                frame_ids=(12, 13, 14),
                state=state,
                alpha=0.75,
            )
            return first.final_noise.clone(), second.final_noise.clone()

        first_run = run_once()
        # Perturb the process-global RNG; per-frame canonical RNG must ignore it.
        torch.manual_seed(9999)
        second_run = run_once()
        self.assertTrue(torch.equal(first_run[0], second_run[0]))
        self.assertTrue(torch.equal(first_run[1], second_run[1]))

    def test_different_sequence_id_has_different_anchor(self):
        head = self.make_head()
        feature = torch.zeros(1, 2, 4, 6, 7)
        a = build_sequence_deformed_noise(
            deformation_head=head,
            stc_features=feature,
            frame_ids=(0, 1),
            state=self.make_state(sequence="video-A"),
            alpha=0.9,
        )
        b = build_sequence_deformed_noise(
            deformation_head=head,
            stc_features=feature,
            frame_ids=(0, 1),
            state=self.make_state(sequence="video-B"),
            alpha=0.9,
        )
        self.assertFalse(torch.equal(a.lineage_noise[:, 0], b.lineage_noise[:, 0]))

    def test_gap_without_adjacent_reference_is_rejected(self):
        head = self.make_head()
        state = self.make_state()
        build_sequence_deformed_noise(
            deformation_head=head,
            stc_features=torch.zeros(1, 3, 4, 6, 7),
            frame_ids=(0, 1, 2),
            state=state,
            alpha=0.9,
        )
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            build_sequence_deformed_noise(
                deformation_head=head,
                stc_features=torch.zeros(1, 3, 4, 6, 7),
                frame_ids=(4, 5, 6),
                state=state,
                alpha=0.9,
            )

    def test_bg_scope_preserves_independent_roi(self):
        head = self.make_head()
        state = self.make_state()
        mask = torch.zeros(1, 2, 1, 6, 7)
        mask[..., :3, :] = 1.0
        output = build_sequence_deformed_noise(
            deformation_head=head,
            stc_features=torch.zeros(1, 2, 4, 6, 7),
            frame_ids=(0, 1),
            state=state,
            alpha=0.9,
            warp_scope="bg",
            bg_mask=mask,
        )
        self.assertEqual(tuple(output.final_noise.shape), (1, 2, 2, 6, 7))
        self.assertTrue(torch.isfinite(output.final_noise).all())

    def test_zero_grid_sequence_mode_reports_zero_offsets_and_still_transitions(self):
        head = self.make_head()
        torch.nn.init.normal_(head.to_offset.weight, std=0.1)
        output = build_sequence_deformed_noise(
            deformation_head=head,
            stc_features=torch.randn(1, 4, 4, 6, 7),
            frame_ids=(0, 1, 2, 3),
            state=self.make_state(),
            alpha=0.9,
            offset_mode="zero_grid",
        )
        self.assertEqual(output.transition_target_frame_ids, (1, 2, 3))
        self.assertTrue(torch.equal(output.offsets, torch.zeros_like(output.offsets)))
        self.assertTrue(output.valid_masks.all())


if __name__ == "__main__":
    unittest.main()
