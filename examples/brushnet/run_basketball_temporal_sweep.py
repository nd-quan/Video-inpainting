"""150-frame BasketballPass sweep, one worker per GPU, with persistent logs."""
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'examples/brushnet'
OUT = Path(os.environ['SWEEP_OUTPUT_ROOT']).resolve()
DATA = SCRIPTS / 'dataset/test_1'
TEACHER = ROOT.parents[1] / 'SFU_STC_flow/teacher_flows_512x512'
CHECKPOINT = ROOT / 'experiments/train_sharedNoise_sameBG_0.95_T8/checkpoint-2250'
STUDENT = ROOT / 'experiments/train_v7_raft_student_flow/checkpoint-0004750/raft_student'
BASELINE = SCRIPTS / 'test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion_fixedBG_nulltext_v0_0.py'
TEMPORAL = SCRIPTS / 'test_brushnet_VCM_final_ddim_brushnet_ipadapter_v2_plus_fusion_fixedBG_temporal_v0.py'


def metrics(case_dir):
    import cv2
    import numpy as np
    from skimage.metrics import structural_similarity
    output = case_dir / 'test_1/BasketballPass'
    if not output.is_dir():
        candidates = list(case_dir.rglob('BasketballPass'))
        if len(candidates) != 1:
            raise RuntimeError(f'Cannot resolve output directory: {candidates}')
        output = candidates[0]
    rows = []
    prev = prev_bg = None
    for i in range(150):
        def read(path):
            x = cv2.imread(str(path))
            if x is None:
                raise FileNotFoundError(path)
            return x
        pred = read(output / f'{i:06d}.png').astype(np.float32) / 255
        gt = cv2.resize(read(DATA / f'BasketballPass/gt/{i:06d}.png'), (512, 512)).astype(np.float32) / 255
        mask = read(DATA / f'BasketballPass/masks/{i:06d}.png').sum(-1) > 255
        bg = 1 - cv2.resize(mask.astype(np.float32), (512,512), interpolation=cv2.INTER_NEAREST)
        err = (pred - gt) ** 2
        mse = float(err.mean())
        row = dict(frame=i, psnr=float(-10*np.log10(max(mse,1e-12))),
                   ssim=float(structural_similarity(gt, pred, data_range=1, channel_axis=2)),
                   bg_mse=float((err*bg[...,None]).sum()/max(3*bg.sum(),1)))
        if prev is not None:
            with np.load(TEACHER / f'train/Class_D/BasketballPass/{i-1:06d}_{i:06d}.npz') as z:
                flow = z['teacher_b'].astype(np.float32)
            yy, xx = np.mgrid[:512,:512].astype(np.float32)
            mx, my = xx+flow[0], yy+flow[1]
            warped = cv2.remap(prev,mx,my,cv2.INTER_LINEAR)
            support = bg * cv2.remap(prev_bg,mx,my,cv2.INTER_LINEAR) * ((mx>=0)&(mx<=511)&(my>=0)&(my<=511))
            row['teacher_warp_bg_mse'] = float((((pred-warped)**2)*support[...,None]).sum()/max(3*support.sum(),1))
        rows.append(row)
        prev, prev_bg = pred, bg
    (case_dir/'frame_metrics.json').write_text(json.dumps(rows,indent=2))
    result = {key: float(np.mean([r[key] for r in rows if key in r])) for key in rows[-1] if key!='frame'}
    result.update(frames=150,pairs=149)
    (case_dir/'metrics.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [dict(name='baseline')]
    for flow in ['teacher_cache','v7_student']:
        for label,start,end in [('full',0,50),('25-35',25,35),('0-15',0,15),('35-50',35,50)]:
            for scale in ['0.001','0.005','0.01','0.05','0.1']:
                cases.append(dict(name=f'{flow}_{label}_scale_{scale}',flow=flow,start=start,end=end,scale=scale))
    (OUT/'manifest.json').write_text(json.dumps(dict(
        dataset=str(DATA/'BasketballPass'),frame_range=[0,149],gpus=[1,2],
        checkpoint=str(CHECKPOINT),student=str(STUDENT),teacher=str(TEACHER),
        teacher_split='train',teacher_gt_verified_pixel_identical=True,
        clip_size=4,seed=1234,shared_seed=6789,shared_bg_strength=1,
        variance_preserving_shared_noise=True,steps=50,cases=cases,
        metric_note='PSNR/SSIM on 512px blended output; common clean teacher backward warp, pair BG intersection, all 149 pairs. Baseline samples singly, temporal in clips of 4.'),indent=2))
    pending = queue.Queue()
    for case in cases:
        pending.put(case)
    def worker(gpu):
        while True:
            try:
                case=pending.get_nowait()
            except queue.Empty:
                return
            folder=OUT/case['name']; folder.mkdir(exist_ok=True)
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),LONG_TEST_ROOT=str(DATA),
                     SEQUENCES='BasketballPass', MAX_IMAGES='150', OUTPUT_ROOT=str(folder),
                     CHECKPOINT_DIR=str(CHECKPOINT),TEMPORAL_CLIP_SIZE='4',
                     NUM_INFERENCE_STEPS='50',MAIN_NOISE_SEED='1234',SHARED_NOISE_SEED='6789',
                     SHARED_BG_NOISE_STRENGTH='1',VARIANCE_PRESERVING_SHARED_NOISE='1',
                     TEMPORAL_BG_MASK_MODE='pair_intersection',TEMPORAL_GUIDANCE_SPACE='latent',
                     V7_RAFT_STUDENT_PATH=str(STUDENT),TEACHER_FLOW_ROOT=str(TEACHER),
                     TEACHER_FLOW_SPLIT='train',GUIDANCE_MODE='temporal',FORCE_REGENERATE='1',
                     OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            if case['name']=='baseline':
                cmd=[sys.executable,'-u',str(BASELINE),'--long_test_root',str(DATA),
                     '--sequences','BasketballPass','--output_root',str(folder),
                     '--max_images','150','--checkpoint_dir',str(CHECKPOINT),
                     '--shared_bg_noise_strength','1','--matched_temporal_clip_size','4','--overwrite']
            else:
                env.update(TEMPORAL_FLOW_BACKEND=case['flow'],TEMPORAL_START_STEP=str(case['start']),
                           TEMPORAL_END_STEP=str(case['end']),TEMPORAL_GUIDANCE_SCALE=case['scale'])
                cmd=[sys.executable,'-u',str(TEMPORAL)]
            status=dict(case=case,gpu=gpu,status='running',started=time.time(),command=cmd)
            path=folder/'status.json'; path.write_text(json.dumps(status,indent=2))
            print(f"GPU {gpu}: START {case['name']}",flush=True)
            try:
                with (folder/'run.log').open('w') as log:
                    subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                status['metrics']=metrics(folder)
                status['status']='complete'
            except Exception as exc:
                status.update(status='failed',error=str(exc))
            status['finished']=time.time(); path.write_text(json.dumps(status,indent=2))
            print(f"GPU {gpu}: {status['status']} {case['name']}",flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker,[1,2]))
    rows=[]
    for case in cases:
        status=json.loads((OUT/case['name']/'status.json').read_text())
        rows.append(dict(name=case['name'],status=status['status'],gpu=status['gpu'],**status.get('metrics',{})))
    with (OUT/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader();writer.writerows(rows)
    if any(row['status']!='complete' for row in rows):
        raise SystemExit('Some cases failed; inspect status.json and run.log')

if __name__=='__main__':
    main()
