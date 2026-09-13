import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from oracle_teacher_flow import OracleTeacherFlowCache
from evaluate_v8_oracle_from_config import replay_arguments


class OracleTests(unittest.TestCase):
    def test_direction_units_padding_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata.json").write_text(json.dumps({"height": 8, "width": 8}))
            branch = root / "test" / "Class_A" / "Video"
            branch.mkdir(parents=True)
            np.savez(branch / "000100_000101.npz",
                     teacher_f=np.full((2, 8, 8), 3., dtype=np.float32),
                     teacher_b=np.full((2, 8, 8), -2., dtype=np.float32))
            cache = OracleTeacherFlowCache(root, "test", 8)
            result = cache.sequence("Video", torch.tensor([100, 101, 101]), "cpu")
            self.assertEqual(tuple(result.forward.shape), (1, 2, 2, 8, 8))
            self.assertTrue((result.forward[:, 0] == 3).all())
            self.assertTrue((result.backward[:, 0] == -2).all())
            self.assertTrue((result.forward[:, 1] == 0).all())
            with self.assertRaises(FileNotFoundError):
                cache.sequence("Video", torch.tensor([101, 102]), "cpu")
            with self.assertRaises(ValueError):
                cache.sequence("Video", torch.tensor([100, 102]), "cpu")
            with self.assertRaises(ValueError):
                OracleTeacherFlowCache(root, "test", 16)

    def test_replay_preserves_parameters(self):
        config = dict(output_dir="/student", seed=1234, clip_length=16,
                      temporal_guidance_scale=0.001, raft_mixed_precision=False,
                      temporal_detach_previous=True, overwrite=False)
        args = replay_arguments(config, "/oracle", "/cache", "test")
        for key in ("seed", "clip_length", "temporal_guidance_scale"):
            self.assertEqual(args[args.index("--" + key) + 1], str(config[key]))
        self.assertIn("--raft_no_mixed_precision", args)
        self.assertNotIn("--overwrite", args)
        with self.assertRaises(ValueError):
            replay_arguments(config, "/student", "/cache", "test")


if __name__ == "__main__":
    unittest.main()
