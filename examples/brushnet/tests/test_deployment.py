"""Delivery names, asset copying, RD scope, and host launcher contracts."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import package_assets
import run_pipeline
import evaluation_reports as reports

DEPLOYMENT = Path(run_pipeline.__file__).parent


class DeploymentContracts(unittest.TestCase):
    def test_evaluation_adapts_native_png_layout_to_packaged_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'frames/bg46/final/Class_D/RaceHorsesD'
            source.mkdir(parents=True)
            config = json.loads((DEPLOYMENT/'config.json').read_text())
            config['brushnet_root'] = str(DEPLOYMENT.parents[3])
            args = SimpleNamespace(max_frames=0, sequences=['RaceHorsesD'], qps=[46],
                                   sub_eval='both', dry_run=False, resume=True)
            with patch('run_pipeline.call') as call, redirect_stdout(io.StringIO()):
                run_pipeline.evaluate(config, args, root)
            command = call.call_args.args[0]
            self.assertTrue(command[1].endswith('/inference/metrics/evaluate.py'))
            self.assertEqual(command[2], 'run')
            self.assertIn('--resume', command)
            self.assertEqual(command[command.index('--reports')+1], 'all')
            staged = root/'metric_input/png/bg46/Class_D/RaceHorsesD'
            self.assertEqual(staged.resolve(), source.resolve())

    def test_packaged_compare_preserves_sequence_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = json.loads((DEPLOYMENT/'config.json').read_text())
            config.update(output_root=str(root), brushnet_root=str(DEPLOYMENT.parents[3]))
            for name in ['first', 'second']:
                rd = root/name/'rd';rd.mkdir(parents=True)
                (rd/'results.json').write_text('{}')
                (rd/'complete.json').write_text('{}')
            args = SimpleNamespace(source_runs=['first', 'second'], run_name='combined',
                                   sub_eval='both', sequences=['Traffic'], dry_run=False)
            with patch('run_pipeline.call') as call:
                run_pipeline.compare_runs(config, args, root/'combined')
            command = call.call_args.args[0]
            self.assertEqual(command[2], 'merge')
            self.assertEqual(command[command.index('--sequences')+1], 'Traffic')

    def test_explicit_single_qp_cli_and_conflicting_options(self):
        base=['--config',str(DEPLOYMENT/'config.json'),'infer','--dry-run']
        for option,expected in [(['--qp','46'],[46]),(['--qps','48','56'],[48,56]),([],list(run_pipeline.QPS))]:
            with patch('run_pipeline.infer') as infer:
                run_pipeline.main(base+option)
            self.assertEqual(infer.call_args.args[1].qps,expected)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_pipeline.main(base+['--qp','46','--qps','48'])

    def comparison_fixture(self, root):
        config=json.loads((DEPLOYMENT/'config.json').read_text())
        config['output_root']=str(root)
        for qp in run_pipeline.QPS:
            source=root/('bg'+str(qp));rd=source/'rd';rd.mkdir(parents=True)
            case=dict(sequence='Traffic',**{'class':'Class_A'},bg_qp=qp,roi_qp=20,frames=11,first_frame=0,last_frame=10,fps=30,
                      total_bytes=1000+qp,total_kbps=(1000+qp)*8*30/11/1000,reference_video='reference/Class_A/Traffic.mp4')
            rows=[{**{k:case[k] for k in ['sequence','class','bg_qp','roi_qp','frames','fps','total_bytes','total_kbps']},
                   'method':method,**{k:float(qp)/100 for k in reports.METRICS}}
                  for method in ['Before restore','After restore']]
            run_pipeline.write_csv(rd/'rd_metrics_before_after.csv',rows)
            run_pipeline.save_json(rd/'complete.json',{'qps':[qp]})
            run_pipeline.save_json(rd/'manifest.json',{'domain':'native decoded H264 CRF18/yuv420p','cases':[case]})
            settings=run_pipeline.effective_settings(config,qp,SimpleNamespace(guidance=None,single_step=False))
            run_pipeline.save_json(source/'request.json',{'settings':{str(qp):settings}})
        return config,SimpleNamespace(source_runs=['bg46','bg48','bg52','bg56'],run_name='comparison',sequences=None,dry_run=False)

    def test_compare_reuses_four_completed_qp_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,args=self.comparison_fixture(root)
            def fake_plot(rows, output):
                run_pipeline.save_json(output/'plot_manifest.json',{'qps':list(run_pipeline.QPS)})
            with patch('run_pipeline.plot_curves',side_effect=fake_plot) as plot, patch('run_pipeline.call') as call, redirect_stdout(io.StringIO()):
                run_pipeline.compare_runs(config,args,root/'comparison')
            call.assert_not_called();self.assertEqual(len(plot.call_args.args[0]),8)
            rows=run_pipeline.csv_rows(root/'comparison/rd/rd_metrics_before_after.csv')
            self.assertEqual(len(rows),8)
            self.assertEqual({int(r['bg_qp']) for r in rows},set(run_pipeline.QPS))
            self.assertTrue(json.loads((root/'comparison/rd/complete.json').read_text())['full_four_qp_curves'])
            self.assertEqual(len(json.loads((root/'comparison/rd/manifest.json').read_text())['sources']),4)
            self.assertTrue((root/'comparison/rd/sequence_metrics/all_sequences_metrics.png').is_file())
            self.assertIn('WE(10-2)', rows[0])
            self.assertNotIn('WE_L2', rows[0])

    def test_four_decimal_display_scales_warping_before_rounding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            row=dict(sequence='Traffic',bg_qp=52,frames=11,method='After restore',total_kbps=123.12345678,
                     **{k:.12345678 for k in reports.METRICS})
            row['WE_L2']=.0012345678
            reports.table_report(root/'table',[row],'Metric display contract')
            exported=run_pipeline.csv_rows(root/'table.csv')[0]
            self.assertEqual(exported['WE(10-2)'],'0.1235')
            self.assertEqual(exported['LPIPS'],'0.1235')
            self.assertEqual(exported['total_kbps'],'123.1235')
            self.assertEqual(exported['frames'],'11')
            self.assertEqual(row['WE_L2'],.0012345678)
            self.assertGreater((root/'table.png').stat().st_size,1000)

    def test_partial_qp_produces_tables_and_defers_rd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,args=self.comparison_fixture(root)
            rows=reports.read_measurements(root/'bg52/rd')
            with patch('run_pipeline.plot_curves') as plot, redirect_stdout(io.StringIO()):
                status=run_pipeline.export_reports(rows,root/'reports','both')
            plot.assert_not_called()
            self.assertEqual(status['rd_curves'],'pending_all_four_qps')
            self.assertEqual(status['sequence_metrics'],'ready')
            with self.assertRaisesRegex(ValueError,'all four QPs'):
                run_pipeline.export_reports(rows,root/'reports','rd')

    def test_raw_json_preserves_precision_and_legacy_csv_scale_is_reversible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            row=dict(sequence='Traffic',bg_qp=52,method='After restore',**{k:.00123456789 for k in reports.METRICS})
            run_pipeline.write_csv(root/'rd_metrics_before_after.csv',[row])
            self.assertAlmostEqual(reports.read_measurements(root)[0]['WE_L2'],.001235)
            run_pipeline.save_json(root/'rd_metrics_before_after.json',{'results':[row]})
            self.assertEqual(reports.read_measurements(root)[0]['WE_L2'],row['WE_L2'])
            (root/'rd_metrics_before_after.json').unlink()
            row.pop('WE_L2')
            run_pipeline.write_csv(root/'rd_metrics_before_after.csv',[row])
            with self.assertRaisesRegex(ValueError,'current evaluator'):
                reports.read_measurements(root)

    def test_compare_rejects_missing_qp_and_mismatched_guidance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,args=self.comparison_fixture(root)
            args.source_runs=args.source_runs[:3]
            with self.assertRaisesRegex(ValueError,'all four'):
                run_pipeline.compare_runs(config,args,root/'comparison')
            args.source_runs.append('bg56')
            path=root/'bg56/request.json';request=json.loads(path.read_text())
            request['settings']['56']['cge_guidance_scale']=.1
            run_pipeline.save_json(path,request)
            with self.assertRaisesRegex(ValueError,'Guidance/configuration'):
                run_pipeline.compare_runs(config,args,root/'comparison')

    def test_compare_rejects_duplicate_points_and_changed_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config,args=self.comparison_fixture(root)
            path=root/'bg56/rd/rd_metrics_before_after.csv'
            rows=run_pipeline.csv_rows(path)
            run_pipeline.write_csv(path,rows+[rows[0]])
            with self.assertRaisesRegex(ValueError,'duplicates'):
                run_pipeline.compare_runs(config,args,root/'comparison')
            run_pipeline.write_csv(path,rows)
            manifest=root/'bg56/rd/manifest.json'
            data=json.loads(manifest.read_text());data['cases'][0]['reference_video']='other_reference.mp4'
            run_pipeline.save_json(manifest,data)
            with self.assertRaisesRegex(ValueError,'Reference or frame scope'):
                run_pipeline.compare_runs(config,args,root/'comparison')
            self.assertFalse((root/'comparison').exists())

    def test_resume_starts_unstarted_qps_and_resumes_existing_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            existing=run/'frames/bg52'
            existing.mkdir(parents=True);(existing/'run_config.json').write_text('{}')
            config=json.loads((DEPLOYMENT/'config.json').read_text())
            args=SimpleNamespace(qps=[52,56],guidance=None,single_step=False,sequences=None,max_frames=0,resume=True,dry_run=False)
            with patch('run_pipeline.call') as mocked:
                run_pipeline.infer(config,args,run)
            self.assertIn('--resume',mocked.call_args_list[0].args[0])
            self.assertNotIn('--resume',mocked.call_args_list[1].args[0])

    def test_short_names_and_single_step(self):
        config = json.loads((DEPLOYMENT/'config.json').read_text())
        args = SimpleNamespace(guidance=None, single_step=True)
        for qp in run_pipeline.QPS:
            s = run_pipeline.effective_settings(config, qp, args)
            self.assertEqual(s['checkpoint'], '/models/bg{}/restorer'.format(qp))
            self.assertEqual(s['flow_checkpoint'], '/models/bg{}/flow'.format(qp))
            self.assertEqual((s['cge_start'], s['cge_end'], s['temporal_start'], s['temporal_end']), (25,26,25,26))
            self.assertEqual(s['cge_bg_quality'], qp)

    def test_model_copy_renames_directories_and_rewrites_only_copied_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init = root/'init.pth'; init.write_bytes(b'raft-init')
            entries = {}
            original_configs = []
            for qp in run_pipeline.QPS:
                checkpoint = root/'long_training_name_bg{}'.format(qp)/'checkpoint-1234'
                for folder in ['brushnet', 'ipadapter']:
                    (checkpoint/folder).mkdir(parents=True)
                    (checkpoint/folder/'weights.bin').write_bytes(b'unchanged-weights')
                flow = root/'long_flow_training_bg{}'.format(qp)/'checkpoint-0123/raft_student'
                flow.mkdir(parents=True)
                cfg = dict(source_checkpoint=str(init),propainter_root='/old/ProPainter',architecture='propainter_raft_large')
                (flow/'config.json').write_text(json.dumps(cfg))
                original_configs.append(flow/'config.json')
                (flow/'pytorch_model.bin').write_bytes(b'student')
                entries['ckpt_bg'+str(qp)] = str(checkpoint)
                entries['student_flow_bg'+str(qp)] = str(flow.parent)
            catalog = root/'catalog.yaml'
            catalog.write_text('\n'.join(k+': '+v for k,v in entries.items()))
            sd = root/'sd'; sd.mkdir()
            (sd/'model_index.json').write_text('{}')
            for name in ['feature_extractor','safety_checker','scheduler','text_encoder','tokenizer','unet','vae']:
                (sd/name).mkdir()
            clip=root/'clip';clip.mkdir();(clip/'config.json').write_text('{}')
            i3d=root/'i3d.pt';i3d.write_bytes(b'i3d')
            cache=root/'hub/checkpoints';cache.mkdir(parents=True)
            for name in ['alexnet-owt-7be5be79.pth','pt_inception-2015-12-05-6726825d.pth']:
                (cache/name).write_bytes(b'cached')
            args=SimpleNamespace(catalog=catalog,output=root/'models',sd15=sd,clip=clip,i3d=i3d,torch_cache=cache.parent,copy=False)
            with redirect_stdout(io.StringIO()):package_assets.models(args)
            self.assertFalse(args.output.exists())
            args.copy=True
            with redirect_stdout(io.StringIO()):package_assets.models(args)
            for qp in run_pipeline.QPS:
                self.assertEqual((args.output/'bg{}'.format(qp)/'restorer/brushnet/weights.bin').read_bytes(),b'unchanged-weights')
                flow=json.loads((args.output/'bg{}'.format(qp)/'flow/config.json').read_text())
                self.assertEqual(flow['source_checkpoint'],'/models/common/raft_init.pth')
                self.assertEqual(flow['propainter_root'],'/app/ProPainter')
            for p in original_configs:self.assertEqual(json.loads(p.read_text())['propainter_root'],'/old/ProPainter')

    def test_rd_manifest_rejects_wrong_rate_or_cross_qp_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cases=[]
            for qp in run_pipeline.QPS:
                (root/'reference.mp4').write_bytes(b'video')
                (root/'before{}.mp4'.format(qp)).write_bytes(b'video')
                cases.append(dict(sequence='Traffic',**{'class':'Class_A'},bg_qp=qp,frames=11,first_frame=0,last_frame=10,fps=30,
                                  total_bytes=1000,total_kbps=1000*8*30/11/1000,reference_video='reference.mp4',baseline_video='before{}.mp4'.format(qp)))
            args=SimpleNamespace(qps=list(run_pipeline.QPS),sequences=None)
            path=root/'rd_manifest.json'
            path.write_text(json.dumps({'cases':cases}))
            self.assertEqual(len(run_pipeline.selected_cases({'rd_root':str(root)},args)),4)
            cases[0]['total_kbps']=99
            path.write_text(json.dumps({'cases':cases}))
            with self.assertRaisesRegex(ValueError,'Bitrate'):run_pipeline.selected_cases({'rd_root':str(root)},args)
            cases[0]['total_kbps']=1000*8*30/11/1000
            cases[0]['first_frame']=1;cases[0]['last_frame']=11
            path.write_text(json.dumps({'cases':cases}))
            with self.assertRaisesRegex(ValueError,'scope'):run_pipeline.selected_cases({'rd_root':str(root)},args)

    def test_host_launcher_preserves_spaces_and_gpu_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);binary=root/'bin';binary.mkdir()
            capture=root/'args.txt'
            docker=binary/'docker'
            docker.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" > "$CAPTURE_ARGS"\n')
            docker.chmod(0o755)
            envfile=root/'deploy.env'
            envfile.write_text('\n'.join(['IMAGE="test-image:1"','GPU_ID="3"','CONTAINER_PYTHON="/opt/python"'] +
                                         [name+'="'+str(root/(name+' with spaces'))+'"' for name in ['MODEL_DIR','DATA_DIR','RD_DIR','OUTPUT_DIR']]))
            env=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ['PATH'],BRUSHNET_DEPLOY_ENV=str(envfile),CAPTURE_ARGS=str(capture))
            subprocess.run(['bash',str(DEPLOYMENT/'run.sh'),'infer','--qps','52'],env=env,check=True)
            argv=capture.read_text().splitlines()
            self.assertEqual(argv[argv.index('--gpus')+1],'device=3')
            self.assertIn('type=bind,source='+str(root/'MODEL_DIR with spaces')+',target=/models,readonly',argv)
            self.assertEqual(argv[-3:],['infer','--qps','52'])


if __name__=='__main__':unittest.main()
