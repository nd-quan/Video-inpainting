"""Regression tests for direct ROI L2 + BG-only codec guidance.

Uses CPU tensors and a recording codec; no VCM-RS process or weights are needed.
Run in the guided_diff environment with unittest discovery for this file.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

BRUSHNET_EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "brushnet"
sys.path.insert(0, str(BRUSHNET_EXAMPLES))

from diffusers.schedulers.scheduling_ddim_CGE import (  # noqa: E402
    _codec_uses_region_composite,
    cond_fn,
    grad_codec,
)
from vcmrs_codec_adapter import VCMRSBackgroundOnlyCodec  # noqa: E402


class RecordingRegionalCodec:
    """Expose both operators, as the real wrapped regional codec does."""

    profile = "train_match"
    roi_quality = 20
    bg_quality = 52
    max_parallel = 1

    def __init__(self):
        self.calls = []

    def roundtrip_background01(self, image, *, roi_mask, call_key="frame"):
        self.calls.append(("bg", call_key))
        # Linear surrogate permits an exact check of the finite difference.
        return 0.5 * image

    def roundtrip_regions01(self, image, *, roi_mask, call_key="frame"):
        self.calls.append(("roi", call_key))
        background = self.roundtrip_background01(
            image, roi_mask=roi_mask, call_key=call_key
        )
        return torch.where(roi_mask.bool(), image, background)

    def roundtrip01(self, image, *, roi_mask, call_key="frame"):
        return self.roundtrip_regions01(image, roi_mask=roi_mask, call_key=call_key)


class BackgroundOnlyCGETest(unittest.TestCase):
    def test_facade_hides_dual_region_entry_points_but_keeps_metadata(self):
        regional = RecordingRegionalCodec()
        codec = VCMRSBackgroundOnlyCodec(regional_codec=regional)
        self.assertFalse(hasattr(codec, "roundtrip_regions01"))
        self.assertFalse(hasattr(codec, "roundtrip01"))
        self.assertFalse(_codec_uses_region_composite(codec))
        self.assertEqual(codec.profile, "train_match")
        self.assertEqual(codec.bg_quality, 52)

    def test_codec_gradient_is_zero_on_roi_and_uses_three_bg_calls(self):
        regional = RecordingRegionalCodec()
        codec = VCMRSBackgroundOnlyCodec(regional_codec=regional)
        image = torch.full((3, 2, 2), 0.2)
        observed = torch.zeros_like(image)
        roi_mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        gradient, _ = grad_codec(image, observed, roi_mask, 500, codec, "test")
        torch.testing.assert_close(
            gradient, (-0.1 * (1 - roi_mask)).expand_as(image)
        )
        self.assertEqual(regional.calls, [
            ("bg", "test_base"), ("bg", "test_plus"), ("bg", "test_minus")
        ])

    def test_guidance_retains_direct_roi_l2_in_single_and_clip_paths(self):
        # The single-frame reference uses the batch path; V8 uses per-frame
        # guidance. Both must preserve the same unnormalized ROI L2 gradient.
        for frame_count, per_frame in [(1, False), (2, False), (2, True)]:
            with self.subTest(frame_count=frame_count, per_frame=per_frame):
                regional = RecordingRegionalCodec()
                codec = VCMRSBackgroundOnlyCodec(regional_codec=regional)
                image = torch.full((frame_count, 3, 2, 2), 0.2)
                roi_mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
                roi_mask = roi_mask.repeat(frame_count, 1, 1, 1)
                if frame_count > 1:
                    image[1] = 0.4
                    roi_mask[1] = 1 - roi_mask[1]
                observed = torch.zeros_like(image)
                scale = 1.35e-4
                args = SimpleNamespace(
                    cge_codec=codec,
                    guidance_scale_cge=scale,
                    cge_scale_schedule="fixed",
                    per_frame_cge=per_frame,
                    vae_scaling_factor=1.0,
                    decode_chunk_size=1,
                    cge_codec_eval_count=0,
                )
                with torch.no_grad():
                    update = cond_fn(
                        image, 500, observed, roi_mask,
                        lambda x: SimpleNamespace(sample=x), args,
                    )
                expected_roi = 0.5 * (image - observed) * roi_mask
                expected_bg = (
                    0.5 * (0.5 * ((image + 1) / 2) - (observed + 1) / 2)
                    * (1 - roi_mask)
                )
                expected_gradient = expected_roi + expected_bg
                torch.testing.assert_close(args.last_cge_raw_grad, expected_gradient)
                torch.testing.assert_close(update, -scale * expected_gradient)
                self.assertEqual(args.cge_codec_eval_count, 1)
                self.assertEqual(len(regional.calls), 3 * frame_count)
                self.assertTrue(all(region == "bg" for region, _ in regional.calls))

    def test_explicit_dual_region_codec_still_encodes_both_regions(self):
        regional = RecordingRegionalCodec()
        self.assertTrue(_codec_uses_region_composite(regional))
        image = torch.full((3, 2, 2), 0.2)
        roi_mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        gradient, _ = grad_codec(
            image, torch.zeros_like(image), roi_mask, 500, regional, "dual"
        )
        expected = (0.1 * roi_mask - 0.1 * (1 - roi_mask)).expand_as(image)
        torch.testing.assert_close(gradient, expected)
        self.assertEqual(sum(region == "roi" for region, _ in regional.calls), 3)
        self.assertEqual(sum(region == "bg" for region, _ in regional.calls), 3)


if __name__ == "__main__":
    unittest.main()
