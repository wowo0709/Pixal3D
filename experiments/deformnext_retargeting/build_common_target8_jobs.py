"""Add target-camera 8 renders to the frozen node16 benchmark outputs.

This generates only common-renderer jobs. Existing native predictions and
their camera-9/10 scores remain immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


METHODS = ('trellis2', 'pixal3d', 'reconviagen', 'mvadapter', 'corradapter',
           'shapeup', 'spacecontrol')
OLD_ROOT = '/root/dev/editing_reproduction'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def relative_from_root(path: Path) -> str:
    return str(path.resolve().relative_to('/'))


def remap_meta_path(path: str, benchmark_repo: Path) -> Path:
    if not path.startswith(OLD_ROOT + '/'):
        raise ValueError('unrecognized native result root: ' + path)
    result = benchmark_repo / path[len(OLD_ROOT) + 1:]
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-repo', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--prep-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    output_gt = args.output_root / 'target8-gt-jobs.jsonl'
    output_baseline = args.output_root / 'target8-baseline-jobs.jsonl'
    if output_gt.exists() or output_baseline.exists():
        raise FileExistsError('immutable target8 jobs already exist')
    input_rows = {x['sample_id']: x for x in read_jsonl(args.prep_root / 'inference-inputs-with-meshes.jsonl')}
    if len(input_rows) != 508:
        raise ValueError('preparation does not contain TEST508')
    gt_rows = read_jsonl(args.prep_root / 'GT-render-jobs.jsonl')
    gt9 = {x['sample_id']: x for x in gt_rows if x['camera_id'] == 9}
    if len(gt9) != 508:
        raise ValueError('missing 508 frozen GT render plans')
    gt_jobs = []
    for sid in sorted(input_rows):
        base = gt9[sid]
        job = dict(base)
        job['camera_id'] = 8
        job['camera'] = input_rows[sid]['conditioning_cameras'][0]
        for key in ('trusted_material_dump', 'metric_mesh'):
            job[key] = relative_from_root(args.prep_root / job[key])
        job['output_rgba'] = relative_from_root(args.output_root / 'target8/gt' / (sid + '.png'))
        gt_jobs.append(job)
    baseline_jobs = []
    protocol_hashes = {x['protocol_sha256'] for x in gt_jobs}
    for method in METHODS:
        summary = json.loads((args.baseline_root / 'metrics' / (method + '-full-test') / 'summary.json').read_text())
        protocol_hashes.add(summary['protocol_sha256'])
        expected_hash = summary['prediction_manifest_sha256']
        candidates = list((args.baseline_root / 'launch').glob(method + '-follow*/predictions.jsonl'))
        matching = [p for p in candidates if digest(p) == expected_hash]
        if len(matching) != 1:
            raise ValueError(f'cannot bind scored {method} manifest uniquely: {matching}')
        predictions = read_jsonl(matching[0])
        if len(predictions) != 508 or {x['sample_id'] for x in predictions} != set(input_rows):
            raise ValueError('external method population differs: ' + method)
        for pred in predictions:
            if pred['status'] != 'ok':
                continue
            sid = pred['sample_id']
            view9 = pred['views'][0]
            if view9['camera_id'] != 9:
                raise ValueError('expected frozen source view 9')
            run = args.baseline_root / Path(view9['rgba']).parents[1]
            meta = []
            for path in run.rglob('run_meta.json'):
                item = json.loads(path.read_text())
                if item.get('status') == 'ok' and item.get('render_mesh'):
                    meta.append(item)
            if len(meta) != 1:
                raise ValueError('ambiguous rendered mesh for ' + method + ' ' + sid)
            mesh = remap_meta_path(meta[0]['render_mesh'], args.benchmark_repo)
            metric_mesh = args.baseline_root / pred['mesh']
            if not metric_mesh.is_file():
                raise FileNotFoundError(metric_mesh)
            baseline_jobs.append({
                'schema': 'deformnext-render-job-v1',
                'sample_id': sid, 'method': method, 'camera_id': 8,
                'protocol_sha256': pred['protocol_sha256'],
                'frame_id': pred['frame_id'],
                'camera': input_rows[sid]['conditioning_cameras'][0],
                'mesh': relative_from_root(mesh),
                'metric_mesh': relative_from_root(metric_mesh),
                'to_world': pred['to_world'],
                'trusted_material_dump': None,
                'output_rgba': relative_from_root(args.output_root / 'target8/external' / method / (sid + '.png')),
            })
    if len(protocol_hashes) != 1 or len(gt_jobs) != 508 or len(baseline_jobs) != 3546:
        raise ValueError('target8 protocol/population mismatch')
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_gt.write_text(''.join(json.dumps(x, sort_keys=True) + '\n' for x in gt_jobs))
    output_baseline.write_text(''.join(json.dumps(x, sort_keys=True) + '\n' for x in baseline_jobs))
    print(json.dumps({'status': 'PASS_COMMON_TARGET8_JOB_PLANS',
                      'gt_jobs': len(gt_jobs), 'baseline_jobs': len(baseline_jobs),
                      'protocol_sha256': protocol_hashes.pop(),
                      'gt_output': str(output_gt), 'baseline_output': str(output_baseline)}))


if __name__ == '__main__':
    main()
