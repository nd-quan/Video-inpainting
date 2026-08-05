"""Regression tests for SFU split planning and RAFT pair enumeration."""

from collections import Counter
import unittest

from precompute_sfu_stc_teacher_flows import iter_manifest_pairs
from prepare_sfu_stc_dataset import (
    SEQUENCES,
    index_to_split,
    iter_indices,
    manifest_entry,
)


EXPECTED_SPLIT_COUNTS = {
    "train": 4906,
    "valid": 1158,
    "test": 1466,
}

EXPECTED_PAIR_COUNTS = {
    "train": 4882,
    "valid": 1140,
    "test": 1449,
}


class SFUSplitPlanningTests(unittest.TestCase):
    def test_class_coverage_and_explicit_disabled_basketballdrill_test(self):
        self.assertEqual(len(SEQUENCES), 18)
        self.assertEqual(
            Counter(spec.class_name for spec in SEQUENCES),
            Counter({
                "Class_A": 2,
                "Class_B": 5,
                "Class_C": 4,
                "Class_D": 4,
                "Class_E": 3,
            }),
        )

        for spec in SEQUENCES:
            with self.subTest(sequence=spec.name):
                self.assertEqual(set(spec.splits), {"train", "valid", "test"})
                for split in ("train", "valid", "test"):
                    count = len(tuple(iter_indices(spec.splits[split])))
                    if spec.name == "BasketballDrill" and split == "test":
                        self.assertEqual(count, 0)
                    else:
                        self.assertGreater(count, 0)

        basketball_drill = next(
            spec for spec in SEQUENCES if spec.name == "BasketballDrill"
        )
        self.assertEqual(
            tuple(iter_indices(basketball_drill.splits["train"])),
            tuple(range(0, 140)),
        )
        self.assertEqual(
            tuple(iter_indices(basketball_drill.splits["valid"])),
            tuple(range(140, 200)),
        )
        self.assertEqual(tuple(basketball_drill.splits["test"]), ())
        self.assertIsNone(basketball_drill.sfu_source)
        self.assertEqual(manifest_entry(basketball_drill)["test_source"], "disabled")

    def test_source_frame_indices_do_not_overlap_between_splits(self):
        for spec in SEQUENCES:
            with self.subTest(sequence=spec.name):
                indices = {
                    split: set(iter_indices(spec.splits[split]))
                    for split in ("train", "valid", "test")
                }
                self.assertTrue(indices["train"].isdisjoint(indices["valid"]))
                self.assertTrue(indices["train"].isdisjoint(indices["test"]))
                self.assertTrue(indices["valid"].isdisjoint(indices["test"]))
                self.assertEqual(
                    index_to_split(spec),
                    {
                        index: split
                        for split, split_indices in indices.items()
                        for index in split_indices
                    },
                )

    def test_expected_aggregate_split_counts(self):
        actual = {
            split: sum(
                len(tuple(iter_indices(spec.splits[split])))
                for spec in SEQUENCES
            )
            for split in ("train", "valid", "test")
        }
        self.assertEqual(actual, EXPECTED_SPLIT_COUNTS)
        self.assertEqual(sum(actual.values()), 7530)

    def test_sfu_prefixes_map_to_original_frames_and_are_train_only(self):
        by_name = {spec.name: spec for spec in SEQUENCES}
        expected = {
            "BQSquare": (0, 179, 0, 179),
            "BasketballPass": (330, 479, 0, 149),
            "BlowingBubbles": (480, 629, 0, 149),
            "FourPeople": (630, 809, 0, 179),
            "ParkScene": (810, 909, 0, 99),
            "Traffic": (1150, 1209, 0, 59),
        }
        for name, (global_start, global_end, source_start, source_end) in expected.items():
            with self.subTest(sequence=name):
                source = by_name[name].sfu_source
                self.assertIsNotNone(source)
                self.assertEqual(source.global_start, global_start)
                self.assertEqual(source.global_end, global_end)
                self.assertEqual(source.source_start, source_start)
                self.assertEqual(source.source_end, source_end)
                train = set(iter_indices(by_name[name].splits["train"]))
                self.assertTrue(set(range(source_start, source_end + 1)) <= train)

        self.assertEqual(
            {spec.name for spec in SEQUENCES if spec.sfu_source is not None},
            set(expected),
        )

    def test_valid_and_test_use_03_frames_and_sources_never_share_a_segment(self):
        for spec in SEQUENCES:
            entry = manifest_entry(spec)
            for split in ("valid", "test"):
                for segment in entry["splits"][split]["segments"]:
                    with self.subTest(sequence=spec.name, split=split):
                        self.assertEqual(segment["source"], "03_frames")

            for segment in entry["splits"]["train"]["segments"]:
                self.assertIn(segment["source"], {"03_frames", "SFU_train"})

        source_counts = {
            split: Counter(
                segment["source"]
                for spec in SEQUENCES
                for segment in manifest_entry(spec)["splits"][split]["segments"]
                for _ in range(segment["end"] - segment["start"] + 1)
            )
            for split in ("train", "valid", "test")
        }
        self.assertEqual(source_counts["train"], Counter({
            "03_frames": 4086,
            "SFU_train": 820,
        }))
        self.assertEqual(source_counts["valid"], Counter({"03_frames": 1158}))
        self.assertEqual(source_counts["test"], Counter({"03_frames": 1466}))

    def test_manifest_counts_match_declared_segments(self):
        for spec in SEQUENCES:
            entry = manifest_entry(spec)
            for split in ("train", "valid", "test"):
                with self.subTest(sequence=spec.name, split=split):
                    declared = entry["splits"][split]["frame_count"]
                    segments = entry["splits"][split]["segments"]
                    counted = sum(
                        segment["end"] - segment["start"] + 1
                        for segment in segments
                    )
                    self.assertEqual(declared, counted)


class TeacherPairEnumerationTests(unittest.TestCase):
    def test_pairs_never_cross_segment_video_or_split_boundaries(self):
        manifest = {
            "sequences": [
                {
                    "name": "SequenceA",
                    "splits": {
                        "train": {"segments": [
                            {"start": 0, "end": 2},
                            {"start": 5, "end": 7},
                        ]},
                        "valid": {"segments": [{"start": 20, "end": 22}]},
                        "test": {"segments": [{"start": 30, "end": 31}]},
                    },
                },
                {
                    "name": "SequenceB",
                    "splits": {
                        "train": {"segments": [{"start": 100, "end": 102}]},
                        "valid": {"segments": [{"start": 110, "end": 111}]},
                        "test": {"segments": [{"start": 120, "end": 121}]},
                    },
                },
            ]
        }

        pairs = [
            (sequence["name"], split, index0, index1)
            for sequence, split, index0, index1 in iter_manifest_pairs(
                manifest, {"train"}
            )
        ]
        self.assertEqual(
            pairs,
            [
                ("SequenceA", "train", 0, 1),
                ("SequenceA", "train", 1, 2),
                ("SequenceA", "train", 5, 6),
                ("SequenceA", "train", 6, 7),
                ("SequenceB", "train", 100, 101),
                ("SequenceB", "train", 101, 102),
            ],
        )
        self.assertNotIn(("SequenceA", "train", 2, 5), pairs)

    def test_current_manifest_pair_counts(self):
        manifest = {"sequences": [manifest_entry(spec) for spec in SEQUENCES]}
        actual = {
            split: sum(
                1 for _ in iter_manifest_pairs(manifest, {split})
            )
            for split in ("train", "valid", "test")
        }
        self.assertEqual(actual, EXPECTED_PAIR_COUNTS)
        self.assertEqual(sum(actual.values()), 7471)


if __name__ == "__main__":
    unittest.main()
