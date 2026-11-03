"""Contracts for official settings, dispatch, and native video conversion."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
import generate
from media_io import export_yuv, frame_bytes, prepare_inputs
from restoration_engine import Settings, expected_steps, run_sequence
from run_logging import RunLog


class GenerateContracts(unittest.TestCase):
    def test_official_schedule_and_qp_models(self):
        config = generate.load_config(generate.DEFAULT_CONFIG)
        for qp in ("46", "48", "52", "56"):
            s = generate.settings_for(config, qp)
            self.assertEqual(expected_steps(s, 8), ([20, 30], list(range(50))))
            self.assertEqual((s.temporal_clip_size, s.temporal_clip_stride), (8, 6))
            self.assertEqual((s.cge_codec_workers, s.cge_max_parallel), (24, 24))
            self.assertEqual((s.cge_guidance_scale, s.temporal_guidance_scale), (.001, .1))
            self.assertEqual(s.cge_bg_quality, int(qp))
            self.assertEqual(expected_steps(generate.settings_for(config, qp, "cge"), 8), ([20, 30], []))
            self.assertEqual(expected_steps(generate.settings_for(config, qp, "tg"), 8), ([], list(range(50))))
        self.assertIn("0007500", generate.settings_for(config, "52").flow_checkpoint)

    def test_selection_and_slots(self):
        config = generate.load_config(generate.DEFAULT_CONFIG)
        args = generate.parse_args(["--output-dir", "/tmp/generate-contract-no-write", "--max-clips", "1"])
        jobs = generate.build_jobs(config, args)
        self.assertEqual(len(jobs), 56)
        self.assertTrue(all(j["max_frames"] == 8 for j in jobs))
        batches = generate.make_batches(jobs, [0, 1], 1)
        self.assertEqual(len(batches), 8)
        self.assertTrue(all(len(b["jobs"]) == 7 for b in batches))
        self.assertEqual(sum(len(b["jobs"]) for b in batches), 56)
        expanded = generate.make_batches(jobs, [0, 1], 2)
        self.assertEqual(len(expanded), 16)
        dispatched = [(j["qp"], j["sequence"]) for b in expanded for j in b["jobs"]]
        self.assertEqual(len(dispatched), len(set(dispatched)))
        self.assertEqual(len(dispatched), 56)
        args.qp, args.sequences = ["52"], ["Traffic"]
        self.assertEqual(len(generate.build_jobs(config, args)), 1)
        args.sequences = None
        self.assertEqual(len(generate.build_jobs(config, args)), 14)

    def test_idle_gpu_selection_respects_allocations_and_visibility(self):
        query = SimpleNamespace(stdout="0, 16, 0\n1, 24000, 50\n2, 16, 0\n3, 100, 30\n")
        with patch("generate.subprocess.run", return_value=query), patch.dict("os.environ", {}, clear=True):
            self.assertEqual(generate.select_gpus(2), [0, 2])
            with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2,3"}):
                self.assertEqual(generate.select_gpus(2), [2])
            with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1"}):
                with self.assertRaisesRegex(RuntimeError, "No idle GPU"):
                    generate.select_gpus(2)

    def test_stride_six_tail_and_first_clip_ownership(self):
        class FakeEngine:
            s = Settings(guidance="none", temporal_clip_size=8, temporal_clip_stride=6)
            seen = []

            def generate_clip(self, pairs, clip_index, indices):
                self.seen.append(list(indices))
                return [Image.new("RGB", (512, 512), (10 + clip_index,) * 3) for _ in pairs], [], 1.0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = []
            for i in range(16):
                a, b = root / ("input%d.png" % i), root / ("mask%d.png" % i)
                Image.new("RGB", (16, 12), (140, 80, 50)).save(a)
                Image.new("RGB", (16, 12), 0).save(b)
                pairs.append((a, b))
            engine = FakeEngine()
            record = run_sequence(engine, Path("Class_Custom/Video"), pairs,
                SimpleNamespace(output_dir=root / "out", resume=False, save_intermediates=False))
            self.assertEqual(engine.seen, [list(range(8)), list(range(6, 14)), list(range(8, 16))])
            self.assertEqual((record["frames"], record["clips"]), (16, 3))
            for i, expected in ((6, 10), (8, 11), (14, 12)):
                with Image.open(root / "out/final/Class_Custom/Video" / ("input%d.png" % i)) as image:
                    self.assertEqual(image.getpixel((0, 0)), (expected,) * 3)

    def test_png_yuv_mp4_alignment_and_export(self):
        ffmpeg, ffprobe = "/usr/local/bin/ffmpeg", "/usr/local/bin/ffprobe"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("input", "mask"):
                (root / name).mkdir()
            images = []
            for i in range(8):
                name = str(117 + i) + ".png"
                path = root / "input" / name
                Image.new("RGB", (32, 24), (i * 20, 60, 130)).save(path)
                Image.new("RGB", (32, 24), 0).save(root / "mask" / name)
                images.append(path)
            meta = dict(width=32, height=24, fps=30, input_pixel_format="yuv420p")
            direct, info = prepare_inputs(root / "input", root / "mask", root / "unused", meta, ffmpeg, ffprobe)
            self.assertFalse((root / "unused").exists())
            self.assertEqual(direct[0][0], images[0])
            raw = root / "input.yuv"
            report = export_yuv(images, raw, info, ffmpeg)
            self.assertEqual(report["bytes"], 8 * 32 * 24 * 3 // 2)
            decoded, media = prepare_inputs(raw, root / "mask", root / "yuv_png", meta, ffmpeg, ffprobe)
            self.assertEqual([p.name for p, _ in decoded], [p.name for p in images])
            self.assertEqual(media["selected_frames"], 8)
            mp4 = root / "input.mp4"
            subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-f", "rawvideo", "-pixel_format", "yuv420p",
                            "-video_size", "32x24", "-framerate", "30", "-i", str(raw), "-c:v", "mpeg4",
                            "-threads", "1", str(mp4)], check=True)
            decoded_mp4, mp4_info = prepare_inputs(mp4, root / "mask", root / "mp4_png", meta, ffmpeg, ffprobe, 8)
            self.assertEqual(len(decoded_mp4), 8)
            self.assertEqual(mp4_info["fps"], "30/1")
            self.assertEqual(mp4_info["conversion"], "mp4 -> PNG")
            raw.write_bytes(raw.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "Truncated"):
                prepare_inputs(raw, root / "mask", root / "bad", meta, ffmpeg, ffprobe)
            self.assertEqual(frame_bytes(32, 24, "yuv420p10le"), 32 * 24 * 3)
            with self.assertRaises(ValueError):
                frame_bytes(31, 24, "yuv420p")

    def test_public_output_is_only_native_png_and_yuv(self):
        class FakeEngine:
            s = Settings(guidance="none", temporal_clip_size=8, temporal_clip_stride=6)

            def generate_clip(self, pairs, clip_index, indices):
                return [Image.new("RGB", (512, 512), (30, 50, 70)) for _ in pairs], [], 1.0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = []
            public, work = root / "output", root / ".output.work"
            relative = Path("Class_Custom/Video")
            for i in range(8):
                a, b = root / ("input%d.png" % i), root / ("mask%d.png" % i)
                Image.new("RGB", (32, 24), (140, 80, 50)).save(a)
                Image.new("RGB", (32, 24), 0).save(b)
                pairs.append((a, b))
            final = public / "png/bg52" / relative
            log = RunLog(public / "generate.log")
            log.write("RUN_START")
            record = run_sequence(FakeEngine(), relative, pairs, SimpleNamespace(output_dir=work,
                final_dir=final, resume=False, save_intermediates=True,
                progress_callback=lambda event, **details: log.write(event, **details)))
            pngs = [final / (p.stem + ".png") for p, _ in pairs]
            export_yuv(pngs, public / "yuv/bg52/Class_Custom/Video.yuv", dict(width=32, height=24, fps=30),
                       "/usr/local/bin/ffmpeg", write_sidecar=False)
            self.assertEqual(record["output"], str(final))
            self.assertEqual({p.name for p in public.iterdir()}, {"png", "yuv", "generate.log"})
            self.assertEqual(len(list(public.rglob("*.png"))), 8)
            self.assertEqual(len(list(public.rglob("*.yuv"))), 1)
            self.assertFalse(list(public.rglob("*.json")))
            self.assertTrue(list(work.rglob("*.json")))
            self.assertTrue((work / "512" / relative).is_dir())
            content = (public / "generate.log").read_text()
            self.assertIn("CLIP_START", content)
            self.assertIn("CLIP_COMPLETE", content)
            self.assertIn("total_clips=1", content)

    def test_shared_log_keeps_concurrent_process_lines_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "generate.log"
            code = ("import sys; sys.path.insert(0,sys.argv[1]); from run_logging import RunLog; "
                    "log=RunLog(sys.argv[2]); [log.write('EVENT',worker=sys.argv[3],index=i) for i in range(25)]")
            procs = [subprocess.Popen([sys.executable, "-c", code, str(generate.HERE), str(path), worker]) for worker in ("a", "b")]
            for proc in procs:
                self.assertEqual(proc.wait(timeout=15), 0)
            lines = path.read_text().splitlines()
            self.assertEqual(len(lines), 50)
            self.assertTrue(all(" EVENT " in line and "worker=" in line and "index=" in line for line in lines))
            self.assertEqual(sum("worker=a " in line for line in lines), 25)
            self.assertEqual(sum("worker=b " in line for line in lines), 25)


if __name__ == "__main__":
    unittest.main()
