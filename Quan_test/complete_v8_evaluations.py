#!/usr/bin/env python
"""Complete videos/PNG-based metrics, verify, then remove only inventoried V8 PNGs."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import cv2
from imgToVideo_rgb_stc_eval import discover_sources, select_unique_frames, fps_for_sequence, encode_video

ROOT = Path(__file__).resolve().parents[1]
KINDS = ('final', 'raw', 'input', 'gt', 'mask_roi')
METRICS = ('PSNR', 'SSIM', 'LPIPS', 'WE', 'FS', 'VFID_Clips')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def validate_video(path, frames, fps):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise ValueError(f'Cannot open {path}')
    first = cv2.imread(str(frames[0][1]))
    expected_hw = first.shape[:2]
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if frame.shape[:2] != expected_hw: raise ValueError(f'Wrong video dimensions: {path}')
            count += 1
    finally:
        cap.release()
    if count != len(frames) or abs(actual_fps - fps) > .01:
        raise ValueError(f'Video mismatch: {path}: frames={count}/{len(frames)}, fps={actual_fps}/{fps}')
    return {'decoded_frames': count, 'width': expected_hw[1], 'height': expected_hw[0], 'fps': actual_fps}


def process(root):
    root = root.resolve()
    # Paths come from the fixed inventory, not arbitrary recursive delete targets.
    root.relative_to(ROOT / 'experiments')
    audit = root / 'postprocess_v8'
    done = audit / 'complete.json'
    if done.exists():
        if list(root.rglob('*.png')): raise ValueError(f'PNG files reappeared after completion: {root}')
        print(f'ALREADY COMPLETE {root}', flush=True); return
    cleanup_plan = audit / 'cleanup_plan.json'
    if (audit / 'ready_to_delete.json').exists():
        finish_cleanup(root, audit)
        return
    clips = sorted(root.rglob('clip_metrics.json'))
    summary = json.loads((root / 'summary.json').read_text())
    if len(clips) != summary['clips']: raise ValueError(f'Incomplete clip count: {root}')
    sources = {kind: discover_sources(root, kind, recursive=True) for kind in KINDS}
    selected = {kind: {seq: select_unique_frames(items, 'first') for seq, items in sources[kind].items()} for kind in KINDS}
    sequences = sorted(selected['final'])
    for kind in KINDS:
        if sorted(selected[kind]) != sequences: raise ValueError('Sequence mismatch')
        for seq in sequences:
            if [f[0] for f in selected[kind][seq]] != [f[0] for f in selected['final'][seq]]:
                raise ValueError(f'Frame alignment mismatch: {seq} {kind}')
    pngs = sorted(root.rglob('*.png'))
    allowed = {p for kind in KINDS for entries in sources[kind].values() for _, _, _, p in entries}
    if set(pngs) != allowed: raise ValueError('Unrecognized PNGs; refusing cleanup')
    plan = [{'path': str(p.relative_to(root)), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in pngs]
    write_json(cleanup_plan, plan)
    print(f'START {root}: {len(sequences)} sequences, {len(pngs)} PNGs', flush=True)
    # Preserve all five image kinds as chronological video, matching existing selection.
    manifest_path = root / 'videos/video_manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    previous_records = {(r['sequence'], r['frame_kind']): r for r in previous.get('videos', [])}
    records = []
    for seq in sequences:
        for kind in KINDS:
            frames = selected[kind][seq]
            fps = fps_for_sequence(seq)
            path = root / 'videos' / seq / f'{kind}.mp4'
            old = previous_records.get((seq, kind), {})
            reusable = (path.exists() and previous.get('selection') == 'first'
                        and old.get('frame_count') == len(frames)
                        and old.get('frame_range') == [frames[0][0], frames[-1][0]])
            if reusable:
                try: validate_video(path, frames, fps)
                except Exception: reusable = False
            if not reusable:
                encode_video(frames, path, fps, 'mp4v', overwrite=path.exists())
            check = validate_video(path, frames, fps)
            records.append({'sequence': seq, 'frame_kind': kind, 'frame_count': len(frames),
                            'frame_range': [frames[0][0], frames[-1][0]], 'fps': fps,
                            'output': str(path), 'reused': reusable, 'verification': check})
            print(f'VIDEO {seq}/{kind}: {len(frames)} frames, reused={reusable}', flush=True)
    write_json(manifest_path, {'eval_root': str(root), 'selection': 'first', 'recursive': True,
                              'fourcc': 'mp4v', 'frame_kinds': KINDS, 'videos': records})
    # Staging uses symlinks only: no PNG copies or extra lossy metric input.
    metric_dir = root / 'metrics_full_reference'
    if not (metric_dir / 'metrics.json').exists():
        stage = Path(tempfile.mkdtemp(prefix='v8_metric_stage_'))
        try:
            for seq in sequences:
                frames = selected['final'][seq]
                c = stage / f'{seq}__{frames[0][0]:06d}-{frames[-1][0]:06d}'
                for kind in ['final', 'gt']:
                    (c / kind).mkdir(parents=True)
                    for frame_id, path, _ in selected[kind][seq]:
                        (c / kind / f'{frame_id:06d}.png').symlink_to(path)
                write_json(c / 'clip_metrics.json', {'video': seq, 'frame_ids': [f[0] for f in frames]})
            write_json(stage / 'summary.json', {'clips': len(sequences)})
            subprocess.run([sys.executable, str(ROOT / 'examples/brushnet/DGAF_VSR_original/postprocess_eval.py'),
                            '--eval_root', str(stage)], check=True)
            if metric_dir.exists(): raise ValueError('Partial metric directory exists')
            shutil.move(str(stage / 'metrics'), str(metric_dir))
        finally:
            shutil.rmtree(stage)
    metrics = json.loads((metric_dir / 'metrics.json').read_text())
    rows = metrics['results']
    if {r['sequence'] for r in rows} != set(sequences) | {'ALL'}: raise ValueError('Incomplete metric sequences')
    import math
    for row in rows:
        for name in METRICS:
            value = row[name]
            if not math.isfinite(value) and not (name == 'PSNR' and value == float('inf')):
                raise ValueError(f'Invalid {name} for {row["sequence"]}')
        expected = sum(len(selected['final'][s]) for s in sequences) if row['sequence'] == 'ALL' else len(selected['final'][row['sequence']])
        if row['frames'] != expected: raise ValueError('Wrong metric frame count')
    write_json(audit / 'selected_sources.json', {kind: {s: [{'frame': i, 'path': str(p.relative_to(root)), 'clip_start': start} for i,p,start in frames] for s,frames in seqs.items()} for kind,seqs in selected.items()})
    # Check source immutability before any deletion.
    for entry in plan:
        p = root / entry['path']; st = p.stat()
        if st.st_size != entry['size'] or st.st_mtime_ns != entry['mtime_ns']: raise ValueError(f'Source changed: {p}')
    write_json(audit / 'ready_to_delete.json', {'root': str(root), 'videos_verified': len(records), 'metric_sequences': len(sequences), 'timestamp': time.time()})
    finish_cleanup(root, audit)


def finish_cleanup(root, audit):
    # Read verified outputs again on resume, before continuing the explicit file list.
    metrics = json.loads((root / 'metrics_full_reference/metrics.json').read_text())
    if not metrics['results']: raise ValueError('Missing metrics')
    manifest = json.loads((root / 'videos/video_manifest.json').read_text())
    for record in manifest['videos']:
        if not Path(record['output']).is_file(): raise ValueError('Missing verified video')
    plan = json.loads((audit / 'cleanup_plan.json').read_text())
    for entry in plan:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.png': raise ValueError('Unsafe cleanup path')
        p = root / relative
        if not p.exists(): continue
        if p.is_symlink(): raise ValueError('Refusing symlink cleanup')
        st = p.stat()
        if st.st_size != entry['size'] or st.st_mtime_ns != entry['mtime_ns']: raise ValueError(f'Source changed: {p}')
        p.unlink()
    remaining = list(root.rglob('*.png'))
    if remaining: raise ValueError(f'{len(remaining)} unremoved PNGs')
    report = {'root': str(root), 'deleted_pngs': len(plan), 'freed_bytes': sum(r['size'] for r in plan),
              'videos_verified': len(manifest['videos']), 'metrics': str(root / 'metrics_full_reference/metrics.csv'), 'timestamp': time.time()}
    write_json(audit / 'complete.json', report)
    print('COMPLETE ' + json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--worker', type=int, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    roots = json.loads(args.inventory.read_text())['roots'][args.worker::args.workers]
    failures = []
    for r in roots:
        try: process(Path(r))
        except Exception as e:
            import traceback
            traceback.print_exc()
            failures.append({'root': r, 'error': repr(e)})
    write_json(args.inventory.parent / f'worker_{args.worker}_result.json', {'roots': roots, 'failures': failures})
    if failures: raise SystemExit(1)

if __name__ == '__main__': main()
