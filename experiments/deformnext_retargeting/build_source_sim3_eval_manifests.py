"""Bind completed file4 M0-aligned renders to the unchanged node16 evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = ('trellis2', 'pixal3d', 'reconviagen', 'mvadapter', 'corradapter', 'shapeup', 'spacecontrol')


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-plan-root', type=Path, required=True)
    parser.add_argument('--scores', type=Path, nargs='+', required=True)
    parser.add_argument('--original-references', type=Path, required=True)
    parser.add_argument('--file4-output-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--expected-samples', type=int, default=502,
                        help='Use 1 only for a mechanical pilot; full evaluation defaults to common502')
    args = parser.parse_args()
    if args.output_root.exists() or not str(args.file4_output_root).startswith('/file4/youngwoo/deformation/experiments/retargeting-v1/source-sim3-test508-20261028/'):
        raise ValueError('new output root and declared file4 experiment required')
    from deformnext_benchmark.io import canonical_hash, digest
    scores = {(row['method'], row['sample_id']): row
              for path in args.scores for row in read_jsonl(path)}
    if len(scores) != args.expected_samples * len(METHODS):
        raise ValueError('seven methods on the declared paired population required')
    ids = sorted({sid for _, sid in scores})
    if len(ids) != args.expected_samples:
        raise ValueError('incomplete declared sample population')
    original_refs = {row['sample_id']: row for row in read_jsonl(args.original_references)}
    jobs = {}
    for chunk in sorted(args.render_plan_root.glob('chunk[0-9][0-9][0-9]')):
        if not (chunk / 'completed.json').is_file():
            raise ValueError('chunk not archived on file4: ' + chunk.name)
        for job in read_jsonl(chunk / 'jobs.jsonl'):
            key = (job['method'], job['sample_id'], job['camera_id'])
            if key in jobs:
                raise ValueError('duplicate aligned render job')
            jobs[key] = (chunk.name, job)
    if len(jobs) != len(scores) * 3:
        raise ValueError('all camera8/9/10 render jobs required')
    refs, preds = [], []
    for sid in ids:
        reference = original_refs[sid]
        refs.append(reference)
        ref_views = {view['camera_id']: view for view in reference['views']}
        if set(ref_views) != {9, 10}:
            raise ValueError('official source-view references must be camera9/10')
        for method in METHODS:
            score = scores[method, sid]
            aligned = [jobs[method, sid, camera] for camera in (8, 9, 10)]
            transforms = [np.asarray(job['to_world']) for _, job in aligned]
            if any(not np.array_equal(matrix, transforms[0]) for matrix in transforms[1:]):
                raise ValueError('camera-specific geometry transform differs')
            views = []
            for camera, (chunk_name, job) in zip((8, 9, 10), aligned):
                png = args.file4_output_root / chunk_name / job['output_rgba']
                receipt_path = png.with_suffix('.receipt.json')
                if not png.is_file() or not receipt_path.is_file():
                    raise FileNotFoundError('aligned render/receipt missing: ' + str(png))
                receipt = json.loads(receipt_path.read_text())
                expected = {'metric_mesh_sha256': score['prediction_mesh_sha256'],
                            'to_world_sha256': canonical_hash(job['to_world']),
                            'output_sha256': digest(png),
                            'protocol_sha256': job['protocol_sha256']}
                if any(receipt.get(key) != value for key, value in expected.items()):
                    raise ValueError('aligned render receipt mismatch: ' + str(png))
                if camera in ref_views:
                    view = ref_views[camera]
                    if (receipt['camera_sha256'] != view['camera_sha256']
                            or receipt['render_sha256'] != view['render_sha256']):
                        raise ValueError('camera/renderer mismatch from original reference')
                    views.append({'camera_id': camera, 'rgba': str(png).lstrip('/'),
                                  'camera_sha256': view['camera_sha256'],
                                  'render_sha256': view['render_sha256']})
            first = aligned[0][1]
            preds.append({'method': method, 'sample_id': sid, 'seed': 0, 'status': 'ok',
                          'mesh': first['metric_mesh'], 'to_world': first['to_world'],
                          'frame_id': first['frame_id'], 'protocol_sha256': first['protocol_sha256'],
                          'views': views})
    args.output_root.mkdir(parents=True)
    (args.output_root / 'references.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in refs))
    (args.output_root / 'predictions.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in preds))
    (args.output_root / 'manifest.json').write_text(json.dumps({
        'status': 'PASS_SOURCE_M0_ALIGNED_EVALUATOR_BINDING', 'samples': len(refs),
        'predictions': len(preds), 'methods': list(METHODS), 'views': [9, 10],
        'target_camera8_rendered_separately': True,
        'reference_root': '/home/seunghyun/editing_reproduction/outputs/deformnext-baselines/2026-10-24-frozen-test/shared/gt-a02',
        'prediction_root': '/',
    }, indent=2) + '\n')
    print(json.dumps({'status': 'PASS_SOURCE_M0_ALIGNED_EVALUATOR_BINDING',
                      'samples': len(refs), 'predictions': len(preds)}))


if __name__ == '__main__':
    main()
