"""Freeze the successful TEST intersection from all seven external feature caches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


METHODS = ('trellis2', 'pixal3d', 'reconviagen', 'mvadapter', 'corradapter',
           'shapeup', 'spacecontrol')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external-metrics-root', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('common success subset already frozen')
    references = {x['sample_id']: x for x in map(json.loads, args.references.read_text().splitlines())}
    if len(references) != 508:
        raise ValueError('TEST reference population differs from 508')
    per_method = {}
    protocols, extractors = set(), set()
    for method in METHODS:
        paths = list((args.external_metrics_root / (method + '-full-test')).glob('*-features.npz'))
        if len(paths) != 1:
            raise ValueError('expected one feature cache for ' + method)
        path = paths[0]
        with np.load(path, allow_pickle=False) as cache:
            keys = [tuple(json.loads(str(x))) for x in cache['row_ids']]
            if len(keys) != len(set(keys)) or cache['prediction'].shape != (len(keys), 2048):
                raise ValueError('duplicate or invalid feature rows for ' + method)
            protocols.add(str(cache['protocol_sha256']))
            extractors.add(str(cache['extractor_id']))
        by_sample = {}
        for sid, seed, cid in keys:
            if sid not in references or seed != 0 or cid not in (9, 10):
                raise ValueError('unexpected sample, seed or camera')
            by_sample.setdefault(sid, set()).add(cid)
        if any(cameras != {9, 10} for cameras in by_sample.values()):
            raise ValueError('partial case in feature cache')
        per_method[method] = {'samples': set(by_sample), 'feature_cache': str(path),
                              'feature_cache_sha256': sha256(path)}
    if len(protocols) != 1 or len(extractors) != 1:
        raise ValueError('feature extractor or protocol differs across methods')
    common = sorted(set.intersection(*(x['samples'] for x in per_method.values())))
    if len(common) != 502:
        raise ValueError('unexpected common successful TEST population')
    doc = {'schema': 'deformnext-common-success-test508-v1',
           'method_ids': list(METHODS), 'protocol_sha256': protocols.pop(),
           'extractor_id': extractors.pop(), 'population': 'TEST508',
           'successful_deformations': len(common), 'views': [9, 10],
           'feature_images': 2 * len(common),
           'source_clusters': len({references[sid]['source_cluster_id'] for sid in common}),
           'sample_ids': common,
           'external': {method: {'successful_deformations': len(info['samples']),
                                 'feature_cache': info['feature_cache'],
                                 'feature_cache_sha256': info['feature_cache_sha256']}
                        for method, info in per_method.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2) + '\n')
    print(json.dumps({'status': 'PASS_FROZEN_COMMON_SUCCESS_SUBSET',
                      'deformations': len(common), 'clusters': doc['source_clusters'],
                      'feature_images': 2 * len(common), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
