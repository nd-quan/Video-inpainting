import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from stc_flow_dataset import STCFlowDataset


def write_triplet(root, index, height=2, width=4):
    """Write an aligned, asymmetric triplet for resize/flip assertions."""

    x = np.arange(width, dtype=np.uint8)[None, :, None]
    y = np.arange(height, dtype=np.uint8)[:, None, None]
    # OpenCV writes BGR.  Values remain asymmetric after conversion to RGB.
    gt = np.concatenate(
        (
            np.broadcast_to(10 + x, (height, width, 1)),
            np.broadcast_to(30 + y, (height, width, 1)),
            np.full((height, width, 1), 50 + index, dtype=np.uint8),
        ),
        axis=2,
    )
    decoded = np.clip(gt.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, : width // 2] = 255
    for kind, image in (("GT", gt), ("input", decoded), ("mask", mask)):
        path = root / "train" / kind / "Class_A" / "SequenceOne" / f"{index:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(path)


def write_manifest(root, segments):
    frame_count = sum(end - start + 1 for start, end in segments)
    manifest = {
        "version": 1,
        "splits": ["train", "valid", "test"],
        "mask_semantics": {"background": 0, "roi": 255},
        "sequences": [
            {
                "name": "SequenceOne",
                "class": "Class_A",
                "fps": 30,
                "splits": {
                    "train": {
                        "segments": [
                            {"start": start, "end": end}
                            for start, end in segments
                        ],
                        "frame_count": frame_count,
                        "relative_root": "train/{kind}/Class_A/SequenceOne",
                    }
                },
            }
        ],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def make_config(root, **overrides):
    data = {
        "dataset_root": str(root),
        "manifest": str(root / "manifest.json"),
        "clip_length": 3,
        "height": 2,
        "width": 4,
        "train_stride": 1,
        "validate_files": True,
        "random_horizontal_flip": False,
    }
    data.update(overrides)
    return {"seed": 1234, "data": data}


class STCFlowDatasetTest(unittest.TestCase):
    def make_tree(self, root, segments):
        write_manifest(root, segments)
        for start, end in segments:
            for index in range(start, end + 1):
                write_triplet(root, index)

    def test_clips_stay_inside_manifest_segments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_tree(root, [(0, 2), (5, 8)])
            dataset = STCFlowDataset(make_config(root), "train")

            self.assertEqual(len(dataset), 3)
            self.assertEqual(
                [record.frame_indices for record in dataset.clips],
                [(0, 1, 2), (5, 6, 7), (6, 7, 8)],
            )
            for record in dataset.clips:
                self.assertGreaterEqual(record.start, record.segment_start)
                self.assertLessEqual(record.frame_indices[-1], record.segment_end)

    def test_loads_aligned_resized_rgb_and_roi_one_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_tree(root, [(10, 12)])
            config = make_config(root, height=4, width=8)
            sample = STCFlowDataset(config, "train")[0]

            self.assertEqual(tuple(sample["gt_frames"].shape), (3, 3, 4, 8))
            self.assertEqual(
                tuple(sample["decoded_frames"].shape), (3, 3, 4, 8)
            )
            self.assertEqual(tuple(sample["roi_masks"].shape), (3, 1, 4, 8))
            self.assertGreaterEqual(float(sample["gt_frames"].min()), -1.0)
            self.assertLessEqual(float(sample["gt_frames"].max()), 1.0)
            self.assertEqual(
                set(torch.unique(sample["roi_masks"]).tolist()), {0.0, 1.0}
            )
            self.assertTrue(torch.all(sample["roi_masks"][..., :4] == 1))
            self.assertTrue(torch.all(sample["roi_masks"][..., 4:] == 0))
            torch.testing.assert_close(
                sample["frame_indices"], torch.tensor([10, 11, 12])
            )
            self.assertEqual(sample["sequence"], "SequenceOne")
            self.assertEqual(sample["class_name"], "Class_A")
            self.assertEqual(sample["metadata"]["source_indices"], (10, 11, 12))
            self.assertEqual(sample["metadata"]["segment_range"], (10, 12))
            self.assertEqual(tuple(sample["original_size"].tolist()), (2, 4))

    def test_horizontal_flip_is_deterministic_and_aligned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_tree(root, [(0, 2)])
            plain = STCFlowDataset(make_config(root), "train")[0]
            flipped_dataset = STCFlowDataset(
                make_config(root, horizontal_flip_probability=1.0), "train"
            )
            flipped = flipped_dataset[0]
            repeated = flipped_dataset[0]

            self.assertTrue(flipped["flipped"])
            for key in ("gt_frames", "decoded_frames", "roi_masks"):
                torch.testing.assert_close(
                    flipped[key], torch.flip(plain[key], dims=(-1,))
                )
                torch.testing.assert_close(repeated[key], flipped[key])

            half_config = make_config(
                root,
                random_horizontal_flip=True,
                horizontal_flip_probability=0.5,
            )
            first = STCFlowDataset(half_config, "train")
            second = STCFlowDataset(half_config, "train")
            first_read = first[0]
            repeated_read = first[0]
            second_dataset_read = second[0]
            self.assertEqual(first_read["flipped"], repeated_read["flipped"])
            self.assertEqual(first_read["flipped"], second_dataset_read["flipped"])

    def test_gt_decode_can_be_skipped_after_teacher_cache_is_built(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_tree(root, [(0, 2)])
            sample = STCFlowDataset(
                make_config(
                    root,
                    load_gt=False,
                    horizontal_flip_probability=1.0,
                ),
                "train",
            )[0]

            self.assertNotIn("gt_frames", sample)
            self.assertEqual(
                tuple(sample["decoded_frames"].shape), (3, 3, 2, 4)
            )
            self.assertEqual(tuple(sample["roi_masks"].shape), (3, 1, 2, 4))
            self.assertTrue(sample["flipped"])

    def test_optional_teacher_cache_and_flow_flip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_tree(root, [(0, 2)])
            without_teacher = STCFlowDataset(make_config(root), "train")[0]
            self.assertNotIn("teacher_flow_forward", without_teacher)

            flow_root = root / "teacher"
            pair_root = flow_root / "train" / "Class_A" / "SequenceOne"
            pair_root.mkdir(parents=True)
            original_forward = []
            original_backward = []
            for index0, index1 in ((0, 1), (1, 2)):
                teacher_f = np.zeros((2, 2, 4), dtype=np.float32)
                teacher_b = np.zeros((2, 2, 4), dtype=np.float32)
                teacher_f[0] = np.arange(4, dtype=np.float32)[None] + index0
                teacher_f[1] = 2.0
                teacher_b[0] = -teacher_f[0]
                teacher_b[1] = -2.0
                valid_f = np.tile(np.array([[0, 1, 1, 0]], np.float32), (2, 1))
                valid_b = 1.0 - valid_f
                np.savez_compressed(
                    pair_root / f"{index0:06d}_{index1:06d}.npz",
                    teacher_f=teacher_f,
                    teacher_b=teacher_b,
                    valid_f=valid_f,
                    valid_b=valid_b,
                )
                original_forward.append(torch.from_numpy(teacher_f))
                original_backward.append(torch.from_numpy(teacher_b))

            config = make_config(
                root,
                teacher_flow_root=str(flow_root),
                horizontal_flip_probability=1.0,
            )
            sample = STCFlowDataset(config, "train")[0]
            expected_forward = torch.flip(torch.stack(original_forward), dims=(-1,))
            expected_backward = torch.flip(torch.stack(original_backward), dims=(-1,))
            expected_forward[:, 0].neg_()
            expected_backward[:, 0].neg_()
            torch.testing.assert_close(
                sample["teacher_flow_forward"], expected_forward
            )
            torch.testing.assert_close(
                sample["teacher_flow_backward"], expected_backward
            )
            self.assertEqual(
                tuple(sample["teacher_valid_forward"].shape), (2, 1, 2, 4)
            )
            self.assertEqual(
                tuple(sample["teacher_valid_backward"].shape), (2, 1, 2, 4)
            )

    def test_eager_validation_detects_missing_aligned_modality(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_tree(root, [(0, 2)])
            (root / "train" / "mask" / "Class_A" / "SequenceOne" / "000001.png").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Missing aligned mask"):
                STCFlowDataset(make_config(root), "train")

    def test_overlapping_segments_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_tree(root, [(0, 3), (3, 6)])
            with self.assertRaisesRegex(ValueError, "Overlapping train segments"):
                STCFlowDataset(make_config(root), "train")


if __name__ == "__main__":
    unittest.main()
