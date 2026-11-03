"""Small end-to-end postprocessing test; no diffusion checkpoint is required."""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
from sfu_long_test_loader import load_sequences
from code.BrushNet.examples.brushnet.inference_experiments.test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion_fixedBG_temporal_v0_postprocess_native_roi_video import (
    postprocess_sequence,
    verify_source_frames,
)


def run_synthetic_pipeline(tmp_path):
    dataset = tmp_path / "dataset"
    output = tmp_path / "eval"
    relative = Path("Class_A/Traffic")
    native = np.zeros((6, 8, 3), dtype=np.uint8)
    native[:, :] = (10, 20, 30)
    native[2, 3] = (240, 15, 20)
    mask = np.zeros_like(native)
    mask[2, 3] = (255, 255, 255)
    predicted = np.full((512, 512, 3), (50, 100, 150), dtype=np.uint8)
    for frame_id in (1, 2):
        name = f"{frame_id:06d}.png"
        for subdir, pixels in (("input", native), ("mask", mask), ("GT", native)):
            path = dataset / subdir / relative / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(path)
        path = output / "test" / relative / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(predicted).save(path)

    sequence = load_sequences(dataset, output, discover_hierarchical=True)[0]
    verify_source_frames(sequence)
    record = postprocess_sequence(sequence, dataset, output, fps=30, skip_video=False)
    resized = output / "test/resized_to_original" / relative / "000001.png"
    composed = output / "test/resized_to_original_native_roi" / relative / "000001.png"
    with Image.open(resized) as image:
        pixels = np.asarray(image.convert("RGB"))
        assert image.size == (8, 6)
        assert tuple(pixels[2, 3]) == (50, 100, 150)
    with Image.open(composed) as image:
        pixels = np.asarray(image.convert("RGB"))
        assert tuple(pixels[2, 3]) == (240, 15, 20)
        assert tuple(pixels[0, 0]) == (50, 100, 150)
    assert record["frames"] == 2
    assert Path(record["video"]).is_file()


class NativeROIVideoPipelineTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "FFmpeg tools are not installed")
    def test_resize_composite_and_video(self):
        with tempfile.TemporaryDirectory(prefix="test_native_roi_video_") as directory:
            run_synthetic_pipeline(Path(directory))


if __name__ == "__main__":
    unittest.main()
