"""Patch alignment, native crop handling and streaming batch invariance."""
from pathlib import Path
from copy import deepcopy
import json
import tempfile
import unittest

import numpy as np
import torch

from metrics.patch_fid import configuration, coordinates, crop_signature, PatchAccumulator, extract_features


class PatchFIDContracts(unittest.TestCase):
    def test_crop_plan_is_content_and_qp_independent_but_seed_and_frame_sensitive(self):
        config = configuration()
        a = coordinates('Traffic', 117, 2560, 1600, config)
        np.testing.assert_array_equal(a, coordinates('Traffic',117,2560,1600,dict(config)))
        self.assertFalse(np.array_equal(a,coordinates('Traffic',118,2560,1600,config)))
        self.assertFalse(np.array_equal(a,coordinates('Traffic',117,2560,1600,dict(config,seed=5))))
        for y,x in coordinates('RaceHorsesD',235,416,240,config):
            self.assertTrue(0 <= y <= 112 and 0 <= x <= 288)
        with self.assertRaisesRegex(ValueError,'exceeds'):
            coordinates('small',0,127,200,config)
        for invalid in [dict(patch_size=True),dict(patches_per_frame=0),dict(seed=-1),dict(region='bg'),dict(version=2)]:
            with self.assertRaises(ValueError):configuration(invalid)

    def test_streaming_crops_match_explicit_crop_and_preserve_order_across_batches(self):
        from cleanfid.resize import build_resizer
        class FakeInception(torch.nn.Module):
            def forward(self,x):
                return x[:,0].flatten(1)[:,:2048]
        rng=np.random.RandomState(32)
        frames=[rng.randint(0,256,(140,170,3),dtype=np.uint8) for _ in range(3)]
        spec=dict(width=170,height=140,frames=3,first_frame=117)
        config=configuration(dict(patches_per_frame=3))
        resize=build_resizer('clean')
        expected=np.stack([resize(frame[y:y+128,x:x+128])[:,:,0].reshape(-1)[:2048]
            for index,frame in enumerate(frames)
            for y,x in coordinates('Fixture',117+index,170,140,config)]).astype(np.float64)
        for batch in [1,4,32]:
            actual=extract_features(iter(frames),FakeInception(),'cpu','Fixture',spec,config,batch)
            np.testing.assert_array_equal(actual,expected)
        self.assertEqual(actual.shape,(9,2048))
        self.assertNotEqual(crop_signature('Fixture',spec,config),crop_signature('Fixture',dict(spec,first_frame=118),config))
        accumulator=PatchAccumulator(FakeInception(),'cpu','Fixture',spec,config)
        with self.assertRaisesRegex(ValueError,'frame order'):accumulator.add(frames[0],118)
        with self.assertRaisesRegex(ValueError,'uint8'):accumulator.add(frames[0].astype(np.float32),117)
        accumulator.add(frames[0],117)
        with self.assertRaisesRegex(ValueError,'Incomplete'):accumulator.finish()

    def test_provider_and_packaged_crop_implementations_are_identical(self):
        from metrics import patch_fid
        provider=Path('/home/cilab/ndquan/vcm-rs/metrics_rd/patch_fid.py')
        if provider.exists():
            self.assertEqual(Path(patch_fid.__file__).read_bytes(),provider.read_bytes())

    def package_fixture(self,root):
        from metrics import patch_fid
        from metrics.media_io import save_json, sha256
        baseline,source,old=root/'baseline',root/'measured',root/'old_metrics'
        baseline.mkdir();source.mkdir();old.mkdir()
        protocol=dict(patch_fid=configuration(dict(patches_per_frame=2)),implementation_sha256=sha256(patch_fid.__file__))
        spec=dict(width=170,height=140,frames=2,first_frame=5,last_frame=6,
                  reference_video='Fixture.mp4',reference_video_sha256='reference-hash',
                  reference_features={},reference_feature_sha256={})
        rows=[dict(sequence='Fixture',bg_qp=qp,method='Before restore',frames=2,total_kbps=100,
                   LPIPS=.1,FID=9,KID=.01,**{'MS-SSIM':.8,'WE_L2':.02}) for qp in [46,48]]
        save_json(old/'rd_metrics_before_after.json',dict(results=rows))
        save_json(baseline/'metrics.json',dict(results=deepcopy(rows),metric_backends={}))
        save_json(baseline/'manifest.json',dict(sequences={'Fixture':spec},cases=[dict(sequence='Fixture',bg_qp=qp) for qp in [46,48]],metrics_sha256=sha256(baseline/'metrics.json')))
        record=dict(sequence='Fixture',method='Before restore',frames=2,patches=4,
                    reference_sha256='reference-hash',crop_sha256=crop_signature('Fixture',spec,protocol['patch_fid']),protocol_id='id')
        save_json(source/'patchfid.json',dict(results=[dict(record,bg_qp=qp,PatchFID=value) for qp,value in [(46,2.0),(48,3.0)]],
                 protocol=protocol,protocol_id='id',source_metrics_sha256=sha256(old/'rd_metrics_before_after.json')))
        save_json(source/'request.json',dict(config=dict(source=str(old),batch_size=32)))
        path=source/'features/reference/Fixture.npz';path.parent.mkdir(parents=True)
        identity=dict(protocol_id='id',video_sha256='reference-hash',crop_sha256=record['crop_sha256'])
        np.savez(path,features=np.ones((4,2048),np.float64),identity=json.dumps(identity))
        return baseline,source,rows

    def test_packaging_preserves_original_scores_and_verifies_patch_identity(self):
        from metrics.package_patchfid import attach_patchfid
        from metrics.media_io import sha256
        with tempfile.TemporaryDirectory() as tmp:
            baseline,source,old=self.package_fixture(Path(tmp))
            attach_patchfid(baseline,source)
            rows=json.loads((baseline/'metrics.json').read_text())['results']
            self.assertEqual([{k:v for k,v in r.items() if k!='PatchFID'} for r in rows],old)
            self.assertEqual([r['PatchFID'] for r in rows],[2.0,3.0])
            manifest=json.loads((baseline/'manifest.json').read_text())
            self.assertEqual(manifest['metrics_sha256'],sha256(baseline/'metrics.json'))
            self.assertEqual(manifest['patch_fid_batch_size'],32)
            self.assertEqual(np.load(baseline/manifest['sequences']['Fixture']['reference_features']['patch']).shape,(4,2048))
        with tempfile.TemporaryDirectory() as tmp:
            baseline,source,_=self.package_fixture(Path(tmp))
            measured=json.loads((source/'patchfid.json').read_text())
            measured['results'][0]['crop_sha256']='wrong'
            (source/'patchfid.json').write_text(json.dumps(measured))
            with self.assertRaisesRegex(ValueError,'crop identity mismatch'):attach_patchfid(baseline,source)

    def test_patchfid_reports_include_before_after_and_all_four_qps(self):
        from metrics import evaluate
        from metrics.media_io import save_json
        from metrics import report_metrics
        from unittest.mock import patch
        from contextlib import redirect_stdout
        import io
        config=evaluate.load_config(evaluate.HERE/'config.json',type('Args',(),{})())
        # A small synthetic score fixture exercises reports without GPU extraction.
        config['patch_fid']={'enabled':False}
        manifest,before,cases,baseline_id=evaluate.baseline_data(config)
        for row in before.values():row['PatchFID']=2.0
        records=[dict(sequence='RaceHorsesD',bg_qp=qp,frames=before[('RaceHorsesD',qp)]['frames'],
                 baseline_id=baseline_id,protocol_id='fixture',LPIPS=.1,FID=3,KID=.001,
                 **{'MS-SSIM':.9,'WE_L2':.01,'PatchFID':1.0}) for qp in [46,48,52,56]]
        with self.assertRaisesRegex(ValueError,'Incomplete PatchFID'):report_metrics([records[0],dict(LPIPS=.1)])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            save_json(root/'results.json',dict(results=records))
            with patch('metrics.evaluate.baseline_data',return_value=(manifest,before,cases,baseline_id)),redirect_stdout(io.StringIO()):
                status=evaluate.export_reports(root,config)
            self.assertIn('PatchFID',status['computed_metrics'])
            self.assertIn('PatchFID',status['plotted_metrics'])
            self.assertEqual(status['rd_curves'],'ready')
            for relative in ['AFTER_RESTORATION_METRICS.png','metric_comparison.png',
                             'rd_curves/all_sequences_PatchFID.png',
                             'rd_curves/all_sequences_before_after_restore_5metric.pdf']:
                self.assertTrue((root/relative).is_file(),relative)
            after=json.loads((root/'metrics_after.json').read_text())['results']
            self.assertEqual([r['PatchFID'] for r in after],[1.0]*4)

    def test_evaluator_rejects_changed_patch_protocol_and_batch_size(self):
        from metrics import evaluate
        from metrics.package_patchfid import attach_patchfid
        from metrics.media_io import sha256
        config=evaluate.load_config(evaluate.HERE/'config.json',type('Args',(),{})())
        with tempfile.TemporaryDirectory() as tmp:
            baseline,source,_=self.package_fixture(Path(tmp))
            measured=json.loads((source/'patchfid.json').read_text())
            measured['protocol']['weights_sha256']=sha256(Path(config['weights'])/'clean/inception-2015-12-05.pt')
            (source/'patchfid.json').write_text(json.dumps(measured))
            attach_patchfid(baseline,source)
            cfg=dict(config,baseline=str(baseline),patch_fid=dict(enabled=True,patches_per_frame=2,batch_size=32))
            evaluate.baseline_data(cfg)
            with self.assertRaisesRegex(ValueError,'settings differ'):
                evaluate.baseline_data(dict(cfg,patch_fid=dict(cfg['patch_fid'],seed=4)))
            with self.assertRaisesRegex(ValueError,'batch size differs'):
                evaluate.baseline_data(dict(cfg,patch_fid=dict(cfg['patch_fid'],batch_size=16)))


if __name__ == '__main__':unittest.main()
