#!/usr/bin/env python3
"""Evaluate flat generated PNG sequences with the established metric definitions."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
METRIC_SCRIPT = ROOT / 'examples/brushnet/DGAF_VSR_original/postprocess_eval.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-root', type=Path, required=True)
    parser.add_argument('--gt-root', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    eval_root = args.eval_root.resolve()
    gt_root = args.gt_root.resolve()
    output = eval_root / 'metrics_full_reference'
    if output.exists():
        raise FileExistsError(output)
    records = []
    for source in sorted(eval_root.iterdir()):
        if not source.is_dir():
            continue
        predictions = {int(p.stem): p for p in source.glob('*.png') if p.stem.isdigit()}
        if not predictions:
            continue
        references = {int(p.stem): p for p in (gt_root / source.name / 'gt').glob('*.png') if p.stem.isdigit()}
        ids = sorted(predictions)
        if not references or set(ids) - set(references):
            raise ValueError(f'Missing GT for {source.name}')
        if len(ids) < 10 or ids != list(range(ids[0], ids[-1] + 1)):
            raise ValueError(f'Need at least ten continuous frames: {source.name}')
        complete = set(ids) == set(references)
        if not complete and not args.allow_partial:
            raise ValueError(f'Incomplete sequence: {source.name}')
        records.append((source.name, ids, predictions, references, complete))
    if not records:
        raise ValueError('No generated sequences found')
    with tempfile.TemporaryDirectory(prefix='flat_sequence_metrics_') as temporary:
        stage = Path(temporary)
        provenance = []
        for name, ids, predictions, references, complete in records:
            sequence = stage / name
            (sequence / 'final').mkdir(parents=True)
            (sequence / 'gt').mkdir()
            for frame_id in ids:
                pred_path = predictions[frame_id]
                with Image.open(pred_path) as image:
                    size = image.size
                (sequence / 'final' / f'{frame_id:06d}.png').symlink_to(pred_path)
                with Image.open(references[frame_id]) as image:
                    image.convert('RGB').resize(size, Image.Resampling.BILINEAR).save(sequence / 'gt' / f'{frame_id:06d}.png')
            (sequence / 'clip_metrics.json').write_text(json.dumps({'video': name, 'frame_ids': ids}))
            provenance.append({'sequence': name, 'evaluated_frames': len(ids),
                               'expected_frames': len(references), 'complete_sequence': complete,
                               'frame_range': [ids[0], ids[-1]],
                               'missing_frame_ids': sorted(set(references) - set(ids))})
            print(f'{name}: {len(ids)}/{len(references)} frames; complete={complete}', flush=True)
        (stage / 'summary.json').write_text(json.dumps({'clips': len(records)}))
        subprocess.run([sys.executable, str(METRIC_SCRIPT), '--eval_root', str(stage)], check=True)
        metrics = stage / 'metrics'
        if not (metrics / 'metrics.csv').is_file():
            raise RuntimeError('Metric evaluator did not complete')
        shutil.move(str(metrics), str(output))
        (output / 'source.json').write_text(json.dumps({
            'eval_root': str(eval_root), 'gt_root': str(gt_root),
            'gt_preprocessing': 'PIL RGB bilinear resize to generated resolution, matching shared_bg_noise_training.py',
            'source_pngs_preserved': True, 'sequences': provenance,
            'aggregate_scope': 'ALL aggregates available frames, including any incomplete sequences'
        }, indent=2) + '\n')
    print(f'Metrics saved: {output}', flush=True)


if __name__ == '__main__':
    main()
