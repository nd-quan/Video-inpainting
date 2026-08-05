import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


BRUSHNET_EXAMPLE = Path(__file__).resolve().parents[1]
if str(BRUSHNET_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(BRUSHNET_EXAMPLE))

from shared_bg_noise_training import (  # noqa: E402
    CLIP_TENSOR_KEYS,
    HierarchicalV8ClipDataset,
    SharedNoiseClipDataset,
    collate_shared_noise_clips,
    make_sequence_shared_background_noise,
    mix_shared_background_noise,
    sample_clip_timesteps,
    sample_shared_background_noise,
)


class _Tokenizer:
    model_max_length = 5

    def __call__(self, *args, **kwargs):
        return SimpleNamespace(input_ids=torch.zeros(1, 5, dtype=torch.long))


class _ImageProcessor:
    def __call__(self, images, return_tensors="pt"):
        tensors = []
        for image in images:
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(array.copy()).permute(2, 0, 1))
        return SimpleNamespace(pixel_values=torch.stack(tensors))


class _FrameDataset(Dataset):
    def __init__(self, frame_count=10):
        self.image_ids = [f"{index:06d}" for index in range(frame_count)]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        value = float(index)
        return {
            "pixel_values": torch.full((3, 2, 2), value),
            "masks": torch.ones(1, 2, 2),
            "conditioning_pixel_values": torch.full((3, 2, 2), value),
            "input_ids": torch.full((4,), index, dtype=torch.long),
            "clip_images": torch.full((3, 2, 2), value),
            "fg_clip_images": torch.full((3, 2, 2), value),
            "bg_clip_images": torch.full((3, 2, 2), value),
            "drop_image_embed": 0,
        }


class SharedBackgroundNoiseTrainingTest(unittest.TestCase):
    def test_variance_preserving_formula_matches_sampling(self):
        independent = torch.tensor(
            [[[[[1.0, 2.0]]], [[[3.0, 4.0]]]]]
        )
        shared = torch.tensor([[[[[5.0, 6.0]]]]])
        # First pixel is BG, second pixel is ROI.
        bg_mask = torch.tensor(
            [[[[[1.0, 0.0]]], [[[1.0, 0.0]]]]]
        )
        actual = mix_shared_background_noise(
            independent,
            shared,
            bg_mask,
            strength=0.25,
            variance_preserving=True,
        )
        mixed = (0.75**0.5) * independent + 0.5 * shared
        expected = independent * (1.0 - bg_mask) + mixed * bg_mask
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[..., 1], independent[..., 1])

    def test_full_strength_shares_bg_within_but_not_between_clips(self):
        template = torch.empty(2, 3, 4, 8, 8)
        bg_mask = torch.ones(2, 3, 1, 8, 8)
        generator = torch.Generator().manual_seed(123)
        noise = sample_shared_background_noise(
            template,
            bg_mask,
            strength=1.0,
            variance_preserving=True,
            generator=generator,
        )
        for clip in range(2):
            torch.testing.assert_close(noise[clip, 0], noise[clip, 1])
            torch.testing.assert_close(noise[clip, 0], noise[clip, 2])
        self.assertFalse(torch.equal(noise[0, 0], noise[1, 0]))

    def test_linear_mode_matches_legacy_sampling_lerp(self):
        independent = torch.tensor([[[[[1.0]]], [[[3.0]]]]])
        shared = torch.tensor([[[[[5.0]]]]])
        bg_mask = torch.ones(1, 2, 1, 1, 1)
        actual = mix_shared_background_noise(
            independent,
            shared,
            bg_mask,
            strength=0.25,
            variance_preserving=False,
        )
        torch.testing.assert_close(actual, 0.75 * independent + 0.25 * shared)

    def test_roi_stays_independent(self):
        template = torch.empty(1, 3, 2, 4, 4)
        bg_mask = torch.zeros(1, 3, 1, 4, 4)
        actual_generator = torch.Generator().manual_seed(77)
        expected_generator = torch.Generator().manual_seed(77)
        actual = sample_shared_background_noise(
            template,
            bg_mask,
            strength=1.0,
            generator=actual_generator,
        )
        expected = torch.randn(template.shape, generator=expected_generator)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.equal(actual[:, 0], actual[:, 1]))

    def test_variance_preserving_strength_is_empirical_correlation(self):
        rho = 0.65
        template = torch.empty(1, 2, 1, 256, 256)
        bg_mask = torch.ones(1, 2, 1, 256, 256)
        noise = sample_shared_background_noise(
            template,
            bg_mask,
            strength=rho,
            variance_preserving=True,
            generator=torch.Generator().manual_seed(9),
        )
        first, second = noise[0, 0].flatten(), noise[0, 1].flatten()
        first_centered = first - first.mean()
        second_centered = second - second.mean()
        correlation = (
            (first_centered * second_centered).mean()
            / (first_centered.square().mean() * second_centered.square().mean()).sqrt()
        )
        self.assertAlmostEqual(float(first.std(unbiased=False)), 1.0, delta=0.02)
        self.assertAlmostEqual(float(second.std(unbiased=False)), 1.0, delta=0.02)
        self.assertAlmostEqual(float(correlation), rho, delta=0.02)

    def test_clip_timesteps_repeat_inside_each_clip(self):
        timesteps = sample_clip_timesteps(
            num_clips=3,
            clip_length=4,
            num_train_timesteps=1000,
            device="cpu",
            generator=torch.Generator().manual_seed(4),
        ).reshape(3, 4)
        for clip in timesteps:
            torch.testing.assert_close(clip, clip[:1].expand_as(clip))

    def test_shared_field_is_sequence_wide_and_refreshes_by_epoch(self):
        template = torch.empty(3, 2, 2, 8, 8)
        keys = ["Class_A/SequenceA", "Class_A/SequenceB", "Class_A/SequenceA"]
        epoch_zero = make_sequence_shared_background_noise(
            template, keys, base_seed=1234, refresh_index=0
        )
        repeated = make_sequence_shared_background_noise(
            template, keys, base_seed=1234, refresh_index=0
        )
        epoch_one = make_sequence_shared_background_noise(
            template, keys, base_seed=1234, refresh_index=1
        )

        torch.testing.assert_close(epoch_zero, repeated)
        torch.testing.assert_close(epoch_zero[0], epoch_zero[2])
        self.assertFalse(torch.equal(epoch_zero[0], epoch_zero[1]))
        self.assertFalse(torch.equal(epoch_zero[0], epoch_one[0]))

        bg_mask = torch.ones(3, 2, 1, 8, 8)
        mixed = sample_shared_background_noise(
            template,
            bg_mask,
            strength=0.7,
            variance_preserving=True,
            generator=torch.Generator().manual_seed(99),
            shared_bg_noise=epoch_zero,
        )
        # The shared component is sequence-wide, while individual frame noise
        # keeps the final noises distinct when 0 < rho < 1.
        self.assertFalse(torch.equal(mixed[0, 0], mixed[0, 1]))

    def test_manifest_clips_do_not_cross_video_or_split_boundaries(self):
        manifest = {
            "video_a": {"name": "a", "start": 0, "end": 4, "split": "train"},
            "video_b": {"name": "b", "start": 5, "end": 9, "split": "valid"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            dataset = SharedNoiseClipDataset(
                _FrameDataset(),
                path,
                split="train",
                clip_length=3,
                stride=1,
            )
            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset.clips[0][2], (0, 1, 2))
            self.assertEqual(dataset.clips[-1][2], (2, 3, 4))
            self.assertTrue(all(4 not in clip[2][:-1] for clip in dataset.clips))

            batch = collate_shared_noise_clips([dataset[0], dataset[1]])
            self.assertEqual(batch["clip_batch_size"], 2)
            self.assertEqual(batch["num_frames"], 3)
            self.assertEqual(batch["videos"], ["a", "a"])
            self.assertEqual(tuple(batch["frame_ids"].shape), (2, 3))
            for key in CLIP_TENSOR_KEYS + ("drop_image_embeds",):
                self.assertEqual(batch[key].shape[0], 6)

    def test_hierarchical_loader_uses_only_aligned_branch_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sequence in ("SequenceA", "SequenceB"):
                for kind in ("GT", "input", "mask"):
                    (root / "train" / kind / "Class_A" / sequence).mkdir(
                        parents=True
                    )
                for frame_index in range(3):
                    name = f"{frame_index:06d}.png"
                    gt = np.zeros((4, 4, 3), dtype=np.uint8)
                    gt[..., 1] = 255
                    decoded = np.zeros((4, 4, 3), dtype=np.uint8)
                    decoded[..., 0] = 255
                    roi = np.zeros((4, 4), dtype=np.uint8)
                    roi[0, 0] = 255
                    Image.fromarray(gt).save(
                        root / "train" / "GT" / "Class_A" / sequence / name
                    )
                    Image.fromarray(decoded).save(
                        root / "train" / "input" / "Class_A" / sequence / name
                    )
                    Image.fromarray(roi).save(
                        root / "train" / "mask" / "Class_A" / sequence / name
                    )

            dataset = HierarchicalV8ClipDataset(
                dataset_root=root,
                split="train",
                tokenizer=_Tokenizer(),
                clip_image_processor=_ImageProcessor(),
                clip_length=2,
                stride=1,
                resolution=4,
            )
            self.assertEqual(dataset.frame_count, 6)
            self.assertEqual(dataset.covered_frame_count, 6)
            self.assertEqual(dataset.branch_count, 2)
            self.assertEqual(len(dataset), 4)
            self.assertEqual(
                {clip[0] for clip in dataset.clips},
                {"Class_A/SequenceA", "Class_A/SequenceB"},
            )

            sample = dataset[0]
            self.assertEqual(tuple(sample["pixel_values"].shape), (2, 3, 4, 4))
            self.assertEqual(tuple(sample["masks"].shape), (2, 1, 4, 4))
            self.assertEqual(tuple(sample["input_ids"].shape), (2, 5))
            # Source mask has ROI=255 at [0,0]; V8 receives BG=0 there and 1 elsewhere.
            self.assertEqual(float(sample["masks"][0, 0, 0, 0]), 0.0)
            self.assertEqual(float(sample["masks"][0, 0, 0, 1]), 1.0)
            # Preserve the V8 checkpoint's historical swapped FG/BG CLIP order.
            self.assertGreater(
                float(sample["fg_clip_images"].sum()),
                float(sample["bg_clip_images"].sum()),
            )
            batch = collate_shared_noise_clips([dataset[0], dataset[2]])
            self.assertEqual(tuple(batch["pixel_values"].shape), (4, 3, 4, 4))

    def test_invalid_strength_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "strength must be in"):
            mix_shared_background_noise(
                torch.randn(1, 2, 1, 2, 2),
                torch.randn(1, 1, 1, 2, 2),
                torch.ones(1, 2, 1, 2, 2),
                strength=1.1,
            )


if __name__ == "__main__":
    unittest.main()
