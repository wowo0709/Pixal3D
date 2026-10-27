"""Score common Cycles camera-8 renders with the node16 benchmark metrics."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


METHODS = ('trellis2', 'pixal3d', 'reconviagen', 'mvadapter', 'corradapter',
           'shapeup', 'spacecontrol', 'RGB1_CALIBRATED', 'zero-source4',
           'zero-edited4', 'V1', 'RETARGET', 'SELECTED')


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-repo', type=Path, required=True)
    parser.add_argument('--prep-root', type=Path, required=True)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--ours-render-records', type=Path, required=True)
    parser.add_argument('--subset', type=Path, required=True)
    parser.add_argument('--external-only', action='store_true',
                        help='score the seven saved external methods as soon as their target8 renders finish')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('target8 score output is immutable')
    import sys
    sys.path.insert(0, str(args.benchmark_repo.resolve()))
    from deformnext_benchmark.aggregate import summarize
    from deformnext_benchmark.io import canonical_hash, digest, read_rgba
    from deformnext_benchmark.metrics import image_metrics

    protocol = json.loads((args.prep_root / 'protocol.json').read_text())
    ph = canonical_hash(protocol)
    subset = json.loads(args.subset.read_text())
    if (subset['schema'] != 'deformnext-common-success-test508-v1'
            or subset['protocol_sha256'] != ph
            or subset['successful_deformations'] != 502):
        raise ValueError('common success subset mismatch')
    common_ids = set(subset['sample_ids'])
    subset_hash = hashlib.sha256(args.subset.read_bytes()).hexdigest()
    refs = {x['sample_id']: x for x in read_jsonl(args.prep_root / 'references.jsonl')}
    ours_jobs = ([] if args.external_only else
                 sorted(args.ours_render_records.rglob('view08-job.json')))
    if not args.external_only and len(ours_jobs) != 3048:
        raise ValueError('expected six frozen ours rows times TEST508')
    jobs = read_jsonl(args.jobs) + [json.loads(path.read_text()) for path in ours_jobs]
    methods = METHODS[:7] if args.external_only else METHODS
    grouped = defaultdict(dict)
    for job in jobs:
        if job['camera_id'] != 8 or job['protocol_sha256'] != ph:
            raise ValueError('unexpected target8 render job')
        key = 'GT' if job.get('trusted_material_dump') else job['method']
        sid = job['sample_id']
        if sid not in refs or sid in grouped[key]:
            raise ValueError('duplicate/unexpected target8 sample')
        image = Path('/') / job['output_rgba']
        receipt_path = image.with_suffix('.receipt.json')
        if not image.is_file() or not receipt_path.is_file():
            raise FileNotFoundError('incomplete target8 render: ' + str(image))
        receipt = json.loads(receipt_path.read_text())
        expected = {'protocol_sha256': ph, 'camera_sha256': canonical_hash(job['camera']),
                    'render_sha256': canonical_hash(protocol['render']),
                    'output_sha256': digest(image), 'job_sha256': canonical_hash(job),
                    'frame_id': job['frame_id']}
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise ValueError('target8 render receipt mismatch: ' + str(image))
        grouped[key][sid] = image
    if len(grouped['GT']) != 508 or set(grouped) != set(methods) | {'GT'}:
        raise ValueError('missing complete GT or declared method rows')
    results = {}
    for method in methods:
        rows = []
        for sid, pred_path in grouped[method].items():
            ref_path = grouped['GT'][sid]
            prediction, pa = read_rgba(pred_path, resolution=512, background=[1, 1, 1])
            reference, ra = read_rgba(ref_path, resolution=512, background=[1, 1, 1])
            scores = image_metrics(prediction, reference, pa, ra)
            rows.append({**{k: refs[sid][k] for k in ('sample_id', 'source_uid',
                                                      'source_cluster_id', 'category')},
                         'metrics': scores})
        summary = summarize(rows, draws=4000, seed=20261020)
        summary['expected_deformations'] = 508
        summary['conditional_on_success_only'] = len(rows) < 508
        common_rows = [row for row in rows if row['sample_id'] in common_ids]
        if len(common_rows) != 502:
            raise ValueError('common target8 case missing for ' + method)
        summary['common'] = summarize(common_rows, draws=4000, seed=20261020)
        results[method] = summary
    output = {'schema': 'deformnext-common-target8-v1',
              'render_protocol_sha256': ph, 'camera_id': 8,
              'appearance_reference': 'original-material-M1',
              'render_engine': 'Blender 4.2.3 Cycles',
              'expected_deformations': 508, 'common_subset_sha256': subset_hash,
              'common_deformations': 502,
              'mode': 'external-only' if args.external_only else 'all13',
              'methods': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'status': 'PASS_COMMON_TARGET8_SCORES', 'methods': len(results),
                      'protocol_sha256': ph, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
