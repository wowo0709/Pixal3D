"""Exploratory same-case common-renderer QA, never a final TEST comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


EXTERNAL = ('trellis2', 'pixal3d', 'reconviagen', 'mvadapter', 'corradapter',
            'shapeup', 'spacecontrol')
INTERNAL = ('RGB1_CALIBRATED', 'zero-source4', 'zero-edited4', 'V1',
            'RETARGET', 'SELECTED')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-repo', type=Path, required=True)
    parser.add_argument('--prep-root', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--common-root', type=Path, required=True)
    parser.add_argument('--early-root', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.benchmark_repo.resolve()))
    from deformnext_benchmark.aggregate import summarize
    from deformnext_benchmark.io import canonical_hash, read_rgba
    from deformnext_benchmark.metrics import image_metrics

    ids = json.loads((args.early_root / 'sample-ids.json').read_text())
    if len(ids) < 10 or len(ids) != len(set(ids)):
        raise ValueError('expected at least ten unique frozen early QA examples')
    references = {x['sample_id']: x for x in
                  map(json.loads, (args.prep_root / 'references.jsonl').read_text().splitlines())}
    if any(sid not in references for sid in ids):
        raise ValueError('early QA sample outside TEST508')
    methods = {}
    gt_root = args.common_root / 'target8/gt'
    for method in (*EXTERNAL, *INTERNAL):
        directory = (args.baseline_root / 'metrics' / (method + '-full-test')
                     if method in EXTERNAL else args.early_root / 'evaluation')
        stem = canonical_hash(method)[:16]
        result = json.loads((directory / (stem + '-results.json')).read_text())
        rows = [row for row in result['per_deformation_seed'] if row['sample_id'] in ids]
        if len(rows) != len(ids) or {row['sample_id'] for row in rows} != set(ids):
            raise ValueError('paired source/geometry row missing for ' + method)
        source = summarize(rows, draws=4000, seed=20261020)
        target_rows = []
        for sid in ids:
            name = sid + '.png'
            prediction_path = ((args.common_root / 'target8/external' / method / name)
                               if method in EXTERNAL else
                               (args.common_root / 'renders' / method / sid / 'view08.png'))
            pred, pa = read_rgba(prediction_path, resolution=512, background=[1, 1, 1])
            gt, ga = read_rgba(gt_root / name, resolution=512, background=[1, 1, 1])
            target_rows.append({**{k: references[sid][k] for k in
                                   ('sample_id', 'source_uid', 'source_cluster_id', 'category')},
                                'metrics': image_metrics(pred, gt, pa, ga)})
        target = summarize(target_rows, draws=4000, seed=20261020)
        methods[method] = {'source910': source, 'target8': target}
        print(json.dumps({'method': method,
                          'CD': source['deformation_weighted']['CD'],
                          'M_IoU': target['deformation_weighted']['M_IoU']}), flush=True)
    output = {'schema': 'deformnext-exploratory-early-paired-qa-v1',
              'warning': 'Only a few sequential early TEST examples; QA only, not representative or confirmatory.',
              'sample_ids': ids,
              'sample_ids_sha256': hashlib.sha256(json.dumps(ids,sort_keys=True).encode()).hexdigest(),
              'deformations': len(ids),
              'source_clusters': len({references[sid]['source_cluster_id'] for sid in ids}),
              'common_renderer': 'node16 Blender/Cycles 512 protocol',
              'methods': methods}
    path = args.early_root / 'early-paired-scores.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'status': 'PASS_EARLY_PAIRED_QA', 'output': str(path)}))


if __name__ == '__main__':
    main()
