from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).with_name("evaluate_temporal_inconsistency.py")
SPEC = importlib.util.spec_from_file_location("temporal_inconsistency_evaluator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVALUATOR
SPEC.loader.exec_module(EVALUATOR)


class TemporalInconsistencyTests(unittest.TestCase):
    def setUp(self):
        self.height = 64
        self.width = 64
        self.zero = torch.zeros(3, self.height, self.width)
        self.bg = torch.ones(1, self.height, self.width)
        self.flow = torch.zeros(2, self.height, self.width)
        self.valid = torch.ones(1, self.height, self.width)
        self.kwargs = {
            "scales": (2.0, 4.0, 8.0),
            "primary_scale": 4.0,
            "erode_radius": 0,
            "min_valid_pixels": 1.0,
            "min_blurred_support": 0.0,
            "consistent_rmse_threshold": 1.0 / 255.0,
        }

    def evaluate(self, previous, current, **overrides):
        kwargs = {**self.kwargs, **overrides}
        metrics, _ = EVALUATOR.evaluate_pair(
            previous,
            current,
            self.zero,
            self.zero,
            self.bg,
            self.bg,
            self.flow,
            self.valid,
            **kwargs,
        )
        return metrics

    def test_perfect_prediction_is_consistent(self):
        metrics = self.evaluate(self.zero, self.zero)
        self.assertEqual(metrics["frequency_diagnosis"], "consistent")
        self.assertEqual(metrics["raw_temporal_rmse"], 0.0)

    def test_uniform_temporal_bias_is_coarse_global(self):
        current = torch.full_like(self.zero, 0.1)
        metrics = self.evaluate(self.zero, current)
        self.assertEqual(
            metrics["frequency_diagnosis"],
            "coarse_global_low_frequency_dominant",
        )
        self.assertLess(metrics["local_frequency_share"], 0.05)
        self.assertGreater(metrics["global_dc_l1"], 0.09)

    def test_checkerboard_temporal_bias_is_local(self):
        yy, xx = torch.meshgrid(torch.arange(self.height), torch.arange(self.width), indexing="ij")
        checker = ((yy + xx) % 2).float() * 0.2 - 0.1
        current = checker[None].expand(3, -1, -1)
        metrics = self.evaluate(self.zero, current)
        self.assertEqual(
            metrics["frequency_diagnosis"], "local_high_frequency_dominant"
        )
        self.assertGreater(metrics["local_frequency_share"], 0.95)
        self.assertLess(metrics["global_dc_l1"], 1e-5)

    def test_roi_only_artifact_is_ignored(self):
        bg = torch.zeros_like(self.bg)
        bg[:, :, : self.width // 2] = 1.0
        current = torch.zeros_like(self.zero)
        current[:, :, self.width // 2 :] = 0.25
        metrics, _ = EVALUATOR.evaluate_pair(
            self.zero,
            current,
            self.zero,
            self.zero,
            bg,
            bg,
            self.flow,
            self.valid,
            **self.kwargs,
        )
        self.assertEqual(metrics["frequency_diagnosis"], "consistent")
        self.assertEqual(metrics["raw_temporal_rmse"], 0.0)

    def test_motion_consistent_error_is_removed_by_backward_warp(self):
        torch.manual_seed(17)
        previous_error = torch.rand_like(self.zero) * 0.1
        backward = torch.zeros_like(self.flow)
        backward[0] = 2.0
        current_error = EVALUATOR.backward_warp(previous_error, backward)
        metrics, _ = EVALUATOR.evaluate_pair(
            previous_error,
            current_error,
            self.zero,
            self.zero,
            self.bg,
            self.bg,
            backward,
            self.valid,
            **self.kwargs,
        )
        self.assertEqual(metrics["frequency_diagnosis"], "consistent")
        self.assertLess(metrics["raw_temporal_rmse"], 1e-6)

    def test_empty_validity_is_skipped_not_counted_as_zero(self):
        metrics, maps = EVALUATOR.evaluate_pair(
            self.zero,
            torch.full_like(self.zero, 0.2),
            self.zero,
            self.zero,
            self.bg,
            self.bg,
            self.flow,
            torch.zeros_like(self.valid),
            **self.kwargs,
        )
        self.assertEqual(metrics["status"], "insufficient_valid_area")
        self.assertNotIn("raw_temporal_rmse", metrics)
        self.assertIsNone(maps)

    def test_cached_flow_resize_scales_displacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.npz"
            teacher = np.zeros((2, 16, 32), dtype=np.float32)
            teacher[0] = 4.0
            teacher[1] = 2.0
            np.savez(
                path,
                teacher_b=teacher,
                valid_b=np.ones((1, 16, 32), dtype=np.float32),
            )
            flow, valid = EVALUATOR.load_teacher_backward(
                path, (32, 64), torch.device("cpu")
            )
        torch.testing.assert_close(flow[0], torch.full_like(flow[0], 8.0))
        torch.testing.assert_close(flow[1], torch.full_like(flow[1], 4.0))
        self.assertGreater(float(valid.mean()), 0.5)

    def test_overlap_selection_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for clip_index, frame_ids in enumerate(([0, 1, 2], [2, 3, 4])):
                clip = root / f"video__{frame_ids[0]:06d}-{frame_ids[-1]:06d}"
                (clip / "final").mkdir(parents=True)
                for frame_id in frame_ids:
                    (clip / "final" / f"{frame_id:06d}.png").touch()
                (clip / "clip_metrics.json").write_text(
                    json.dumps({"video": "Class/Video", "frame_ids": frame_ids}),
                    encoding="utf-8",
                )
            first = EVALUATOR.discover_units(root, "final", "first")[0]
            last = EVALUATOR.discover_units(root, "final", "last")[0]
        self.assertIn("000000-000002", first.frames[2].clip_id)
        self.assertIn("000002-000004", last.frames[2].clip_id)
        self.assertNotEqual(first.frames[2].clip_id, last.frames[2].clip_id)

    def test_saved_transformed_references_take_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = root / "eval" / "clip"
            saved_gt = clip / "gt" / "000007.png"
            saved_mask = clip / "mask_roi" / "000007.png"
            canonical_gt = root / "dataset" / "valid" / "GT" / "Class" / "Video" / "000007.png"
            canonical_mask = root / "dataset" / "valid" / "mask" / "Class" / "Video" / "000007.png"
            for path in (saved_gt, saved_mask, canonical_gt, canonical_mask):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            ref = EVALUATOR.FrameRef(
                video="Class/Video",
                frame_id=7,
                prediction=clip / "final" / "000007.png",
                clip_dir=clip,
                clip_id="clip",
                clip_order=0,
            )
            gt, mask = EVALUATOR.reference_paths(ref, root / "dataset", "valid")
        self.assertEqual(gt, saved_gt)
        self.assertEqual(mask, saved_mask)

    def test_aggregate_reports_spatial_scope_without_area_squared_coverage(self):
        def row(weight, coverage, scope):
            return {
                "status": "ok",
                "raw_weight": weight,
                "raw_temporal_l1": 0.1,
                "raw_temporal_mse": 0.01,
                "raw_temporal_rmse": 0.1,
                "global_dc_l1": 0.01,
                "global_dc_rmse": 0.01,
                "motion_mean_px": 0.0,
                "valid_coverage": coverage,
                "energy_top10_fraction": 0.1,
                "energy_support90_fraction": 0.9,
                "directional_coherence": 0.9,
                "sigma_4_weight": weight,
                "sigma_4_coarse_mse": 0.01,
                "sigma_4_local_mse": 0.0,
                "frequency_diagnosis": "coarse_global_low_frequency_dominant",
                "spatial_scope_diagnosis": scope,
                "motion_band": "static",
                "window_boundary": 0,
            }

        summary = EVALUATOR.aggregate_rows(
            [row(1.0, 0.1, "spatially_localized"), row(9.0, 0.9, "globally_coherent")],
            primary_scale=4.0,
            consistent_threshold=1.0 / 255.0,
            local_share_threshold=0.6,
        )
        self.assertAlmostEqual(summary["valid_coverage"], 0.5)
        self.assertEqual(summary["spatial_scope_counts"]["globally_coherent"], 1)
        self.assertEqual(summary["spatial_scope_diagnosis"], "globally_coherent")


if __name__ == "__main__":
    unittest.main()
