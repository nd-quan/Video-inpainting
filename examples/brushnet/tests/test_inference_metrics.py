"""Packaged metric protocol, input scope, and reporting contracts."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from metrics import METRICS, RD_METRICS
from metrics import evaluate as evaluation
from metrics import core, media_io
from metrics.plot_metric_comparison import plot


class MetricContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = evaluation.load_config(evaluation.HERE / 'config.json', SimpleNamespace())
        cls.manifest, cls.before, cls.cases, cls.baseline_id = evaluation.baseline_data(cls.config)

    def test_kid_is_full_unbiased_cubic_mmd_and_can_be_negative(self):
        rng = np.random.RandomState(4)
        a, b = rng.randn(13, 9), rng.randn(11, 9) + .01
        kernel = lambda x, y: (x @ y.T / 9 + 1) ** 3
        xx, yy, xy = kernel(a, a), kernel(b, b), kernel(a, b)
        expected = (sum(xx[i, j] for i in range(13) for j in range(13) if i != j) / (13 * 12)
                    + sum(yy[i, j] for i in range(11) for j in range(11) if i != j) / (11 * 10)
                    - 2 * xy.mean())
        self.assertAlmostEqual(core.unbiased_kid(a, b), expected, places=12)
        self.assertLess(core.unbiased_kid(a, a), 0)
        with self.assertRaises(ValueError):
            core.unbiased_kid(a[:1], b)

    def test_warping_l2_excludes_outside_pixels(self):
        previous = np.zeros((8, 8, 3), dtype=np.float32)
        current = np.full_like(previous, .2)
        flow = np.zeros((8, 8, 2), dtype=np.float32)
        flow[:, 0, 0] = -2
        current[:, 0] = .9
        with patch('cv2.calcOpticalFlowFarneback', return_value=flow):
            self.assertAlmostEqual(core.warping_l2(previous, current), .04, places=7)

    def fixture(self, root):
        spec = dict(width=12, height=10, first_frame=5, last_frame=7, frames=3,
                    **{'class': 'Class_D'})
        folder = root / 'png/bg46/Class_D/Example'
        folder.mkdir(parents=True)
        for i in range(5, 8):
            Image.new('RGB', (12, 10), (i, 20, 30)).save(folder / f'{i:06d}.png')
        return folder, {'sequences': {'Example': spec}}

    def test_input_selection_preserves_original_ids_and_rejects_wrong_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder, manifest = self.fixture(root)
            job = evaluation.discover(root, manifest, [46])[0]
            self.assertEqual([Path(p).stem for p in job['pngs']], ['000005', '000006', '000007'])
            self.assertEqual(evaluation.discover(root/'png', manifest, [46]), [job])
            with self.assertRaisesRegex(ValueError, 'Missing restored PNGs'):
                evaluation.discover(root, manifest, [46, 48])
            Image.new('RGB', (13, 10)).save(folder / '000006.png')
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                evaluation.discover(root, manifest, [46])
            (folder / '000006.png').unlink()
            with self.assertRaisesRegex(ValueError, 'frame IDs/count'):
                evaluation.discover(root, manifest, [46])

    def test_baseline_has_exact_four_qp_scope_and_is_checksum_protected(self):
        self.assertEqual(len(self.manifest['sequences']), 14)
        self.assertEqual(set(self.before), {(n, qp) for n in self.manifest['sequences'] for qp in [46, 48, 52, 56]})
        for spec in self.manifest['sequences'].values():
            self.assertEqual(spec['last_frame'] - spec['first_frame'] + 1, spec['frames'])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = Path(self.config['baseline'])
            (root/'manifest.json').write_bytes((source/'manifest.json').read_bytes())
            (root/'metrics.json').write_text('{"results": []}')
            with self.assertRaisesRegex(ValueError, 'cache changed'):
                evaluation.baseline_data(dict(self.config, baseline=str(root)))

    def records(self):
        return [dict(sequence='RaceHorsesD', bg_qp=qp, frames=self.before[('RaceHorsesD', qp)]['frames'],
                     baseline_id=self.baseline_id, protocol_id='test-protocol', **{key: .1 for key in METRICS})
                for qp in [46, 48, 52, 56]]

    def test_merge_rejects_conflicting_points_and_different_protocols(self):
        records = self.records()
        self.assertEqual(evaluation.merge_records([records[:2], records[2:]]), records)
        self.assertEqual(len(evaluation.merge_records([records, records])), 4)
        changed = deepcopy(records)
        changed[0]['FID'] = 99
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
            evaluation.merge_records([records, changed])
        changed = deepcopy(records[2:])
        changed[0]['protocol_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'different metric protocols'):
            evaluation.merge_records([records[:2], changed])

    def test_partial_qp_exports_five_metric_table_and_defers_rd(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            media_io.save_json(root/'results.json', {'results': self.records()[:1]})
            with redirect_stdout(io.StringIO()):
                status = evaluation.export_reports(root, self.config)
            self.assertEqual(status['rd_curves'], 'pending_all_four_qps')
            self.assertEqual(status['computed_metrics'], list(METRICS))
            self.assertEqual(status['plotted_metrics'], list(RD_METRICS))
            self.assertFalse(status['WE_L2_plotted'])
            after = json.loads((root/'metrics_after.json').read_text())['results']
            self.assertEqual(len(after), 1)
            self.assertEqual(after[0]['method'], 'After restore')
            for ext in ['png', 'pdf']:
                self.assertGreater((root/f'AFTER_RESTORATION_METRICS.{ext}').stat().st_size, 1000)
            self.assertFalse((root/'rd_curves').exists())
            with self.assertRaisesRegex(ValueError, 'all four QPs'):
                evaluation.export_reports(root, self.config, 'rd')

    def test_bar_chart_is_unweighted_sequence_mean_and_omits_psnr_ssim_we(self):
        rows = [dict(sequence=name, bg_qp=46, method=method, frames=frames,
                     **{key: value for key in RD_METRICS})
                for name, frames, value in [('short', 3, .2), ('long', 100, .6)]
                for method in ['Before restore', 'After restore']]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plot(rows, root)
            data = json.loads((root/'metric_comparison.json').read_text())['results']
            for row in data:
                self.assertAlmostEqual(row['FID'], .4)
                self.assertEqual(row['sequences'], 2)
                self.assertFalse({'WE_L2', 'PSNR', 'SSIM'} & set(row))

    def test_cli_rejects_duplicate_qps_before_reading_input(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluation.main(['run', '--input-root', '/unused', '--output-dir', '/unused2', '--qps', '46', '46'])


if __name__ == '__main__':
    unittest.main()
