"""Score source views and world geometry on the frozen 502-case intersection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


EXTERNAL = ('trellis2', 'pixal3d', 'reconviagen', 'mvadapter', 'corradapter',
            'shapeup', 'spacecontrol')
INTERNAL = ('RGB1_CALIBRATED', 'zero-source4', 'zero-edited4', 'V1', 'RETARGET', 'SELECTED')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def one_method(method: str, directory: Path, subset: dict, protocol: dict, reference_features,
               summarize, frechet_distance, kernel_distance) -> tuple[dict, np.ndarray]:
    stem = hashlib.sha256(json.dumps(method, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]
    results_path = directory / (stem + '-results.json')
    feature_path = directory / (stem + '-features.npz')
    if not results_path.is_file() or not feature_path.is_file():
        raise FileNotFoundError('full result or feature cache missing for ' + method)
    results = json.loads(results_path.read_text())
    if results['method'] != method or results['summary']['successful_deformations'] > 508:
        raise ValueError('method/result mismatch')
    common = set(subset['sample_ids'])
    rows = [r for r in results['per_deformation_seed'] if r['sample_id'] in common]
    if len(rows) != 502 or len({r['sample_id'] for r in rows}) != 502:
        raise ValueError('common success rows missing for ' + method)
    summary = summarize(rows, draws=4000, seed=20261020)
    keys = [(sid, 0, cid) for sid in subset['sample_ids'] for cid in (9, 10)]
    with np.load(feature_path, allow_pickle=False) as features:
        if (str(features['protocol_sha256']) != subset['protocol_sha256']
                or str(features['extractor_id']) != subset['extractor_id']):
            raise ValueError('different feature extractor or protocol for ' + method)
        indices = {tuple(json.loads(str(key))): i for i, key in enumerate(features['row_ids'])}
        if any(key not in indices for key in keys):
            raise ValueError('common feature row missing for ' + method)
        order = [indices[key] for key in keys]
        prediction = features['prediction'][order]
        reference = features['reference'][order]
    if reference_features is not None and not np.array_equal(reference, reference_features):
        raise ValueError('GT Inception features differ between methods')
    dist = protocol['distribution']
    distribution = {'FID': frechet_distance(prediction, reference),
                    **kernel_distance(prediction, reference,
                                      subset_size=dist['kid_subset_size'],
                                      subsets=dist['kid_subsets'], seed=dist['seed']),
                    'image_count': len(prediction), 'extractor': subset['extractor_id'],
                    'subset_sha256': sha256(Path(subset['subset_path']))}
    return {'full_successes': results['summary']['successful_deformations'],
            'common': summary, 'distribution': distribution,
            'results_path': str(results_path), 'features_path': str(feature_path)}, reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-repo', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--subset', type=Path, required=True)
    parser.add_argument('--external-metrics-root', type=Path, required=True)
    parser.add_argument('--mode', choices=('external', 'all'), required=True)
    parser.add_argument('--ours-eval-root', type=Path)
    parser.add_argument('--external-score', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('common source score already exists')
    sys.path.insert(0, str(args.benchmark_repo.resolve()))
    from deformnext_benchmark.aggregate import summarize
    from deformnext_benchmark.io import canonical_hash
    from deformnext_benchmark.metrics import frechet_distance, kernel_distance

    subset = json.loads(args.subset.read_text())
    subset['subset_path'] = str(args.subset)
    protocol = json.loads(args.protocol.read_text())
    if (subset['schema'] != 'deformnext-common-success-test508-v1'
            or subset['successful_deformations'] != 502
            or subset['feature_images'] != 1004
            or subset['protocol_sha256'] != canonical_hash(protocol)):
        raise ValueError('subset/protocol mismatch')
    if args.mode == 'external':
        methods = EXTERNAL
        scores = {}
        reference = None
    else:
        if not args.ours_eval_root or not args.external_score:
            raise ValueError('all-mode needs completed ours evaluation and external common scores')
        external = json.loads(args.external_score.read_text())
        if (external['schema'] != 'deformnext-common-source502-v1'
                or external['subset_sha256'] != sha256(args.subset)
                or set(external['methods']) != set(EXTERNAL)):
            raise ValueError('external common scores do not match frozen subset')
        methods = INTERNAL
        scores = external['methods']
        reference = None
    for method in methods:
        directory = (args.external_metrics_root / (method + '-full-test') if method in EXTERNAL
                     else args.ours_eval_root)
        score, reference = one_method(method, directory, subset, protocol, reference,
                                      summarize, frechet_distance, kernel_distance)
        scores[method] = score
        print(json.dumps({'method': method, 'FID': score['distribution']['FID'],
                          'KID': score['distribution']['KID']}), flush=True)
    result = {'schema': 'deformnext-common-source502-v1',
              'subset_sha256': sha256(args.subset),
              'protocol_sha256': subset['protocol_sha256'],
              'common_deformations': 502, 'distribution_images': 1004,
              'methods': scores}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'PASS_COMMON_SOURCE502_SCORE',
                      'methods': len(scores), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
