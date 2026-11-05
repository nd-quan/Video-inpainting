"""Native ROI and eval-compatibility contracts for the deployment entry point."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
import restoration_engine as deployment


class InferenceContracts(unittest.TestCase):
    def test_512_blend_matches_existing_evaluator_exactly(self):
        source = (deployment.BRUSHNET_EXAMPLES / "inference_experiments"
                  / "test_brushnet_fixedBG_cge_then_temporal_equivalent_clean_timed.py")
        function = next(n for n in ast.parse(source.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "blend_with_input")
        scope = {"cv2": cv2, "np": np, "Image": Image}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
        rng = np.random.RandomState(11)
        native = rng.randint(0, 256, (71, 119, 3), dtype=np.uint8)
        mask = np.zeros_like(native)
        mask[5:53, 17:93] = 255
        generated = Image.fromarray(rng.randint(0, 256, (512, 512, 3), dtype=np.uint8))
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "input.png", Path(tmp) / "mask.png"
            Image.fromarray(native).save(a)
            Image.fromarray(mask).save(b)
            expected = scope["blend_with_input"](generated, str(a), str(b))
            actual = deployment.evaluation_blend_512(generated, native, mask.sum(2) > 255)
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_native_roi_is_exact_and_background_is_lanczos(self):
        rng = np.random.RandomState(2)
        native = rng.randint(0, 256, (43, 77, 3), dtype=np.uint8)
        generated = Image.fromarray(rng.randint(0, 256, (512, 512, 3), dtype=np.uint8))
        roi = np.zeros((43, 77), dtype=bool)
        roi[7:31, 9:63] = True
        upscaled, actual = deployment.restore_native(generated, native, roi)
        pixels = np.asarray(actual)
        self.assertEqual(actual.size, (77, 43))
        np.testing.assert_array_equal(pixels[roi], native[roi])
        np.testing.assert_array_equal(pixels[~roi], np.asarray(upscaled)[~roi])
        for full_roi in [np.ones_like(roi), np.zeros_like(roi)]:
            _, result = deployment.restore_native(generated, native, full_roi)
            expected = native if full_roi.all() else np.asarray(upscaled)
            np.testing.assert_array_equal(np.asarray(result), expected)

    def test_mask_threshold_uses_wide_sum_and_matches_legacy_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(p / "input.png")
            values = np.array([[0, 85, 86], [255, 128, 1]], dtype=np.uint8)
            Image.fromarray(np.repeat(values[..., None], 3, axis=2)).save(p / "mask.png")
            _, roi = deployment.read_native(p / "input.png", p / "mask.png")
            np.testing.assert_array_equal(roi, [[False, False, True], [True, True, False]])

    def test_config_and_mode_windows(self):
        s = deployment.Settings()
        s.cge_end = s.temporal_end = 26
        for mode, expected in [("cge", ([25], [])), ("tg", ([], [25])),
                               ("cge_then_tg", ([25], [25])), ("none", ([], []))]:
            s.guidance = mode
            self.assertEqual(deployment.expected_steps(s, 4), expected)
        s.guidance = "cge_then_tg"
        self.assertEqual(deployment.expected_steps(s, 1), ([25], []))
        reference = deployment.ROOT / "experiments/EVAL_FOR_FINAL/eval_bg52_sn2250_latent_tg0p1_v7student_512/run_config.json"
        restored = deployment.settings_from_json(reference)
        self.assertEqual(restored.guidance, "tg")
        self.assertEqual((restored.temporal_start, restored.temporal_end), (0, 50))
        self.assertEqual(restored.shared_bg_noise_strength, 1.0)

    def test_tail_window_does_not_overwrite_previously_saved_frames(self):
        class FakeEngine:
            s = deployment.Settings(guidance="none")

            def generate_clip(self, pairs, clip_index, indices):
                return [Image.new("RGB", (512, 512), (10 + 10 * clip_index,) * 3) for _ in pairs], [], 1.0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "input").mkdir()
            (root / "mask").mkdir()
            pairs = []
            for i in range(5):
                a, b = root / "input" / (str(i) + ".png"), root / "mask" / (str(i) + ".png")
                Image.new("RGB", (17, 13), (140, 80, 50)).save(a)
                Image.new("RGB", (17, 13), 0).save(b)
                pairs.append((a, b))
            args = SimpleNamespace(output_dir=root / "out", resume=False, save_intermediates=True)
            result = deployment.run_sequence(FakeEngine(), Path("test"), pairs, args)
            self.assertEqual(result["frames"], 5)
            with Image.open(root / "out/final/test/3.png") as im:
                self.assertEqual(im.getpixel((0, 0)), (10, 10, 10))
            with Image.open(root / "out/final/test/4.png") as im:
                self.assertEqual(im.getpixel((0, 0)), (20, 20, 20))


if __name__ == "__main__":
    unittest.main()
