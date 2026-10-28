"""Replay frozen raw baseline features on the final ordered common subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from deformnext_benchmark.aggregate import summarize
from deformnext_benchmark.io import canonical_hash, digest, validate_protocol, write_json
from deformnext_benchmark.metrics import frechet_distance, kernel_distance

METHODS = ('trellis2', 'pixal3d', 'reconviagen', 'mvadapter',
           'corradapter', 'shapeup', 'spacecontrol')


def distribution(prediction, reference, protocol):
    settings = protocol['distribution']
    return {'FID': frechet_distance(prediction, reference),
            **kernel_distance(prediction, reference,
                              subset_size=settings['kid_subset_size'],
                              subsets=settings['kid_subsets'], seed=settings['seed'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-metrics-root', type=Path, required=True)
    parser.add_argument('--aligned-evaluation', type=Path, required=True)
    parser.add_argument('--common502-summary', type=Path, required=True)
    parser.add_argument('--common-subset', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('fresh raw replay output required')
    protocol = validate_protocol(json.loads(args.protocol.read_text()))
    ph = canonical_hash(protocol)
    aligned = json.loads((args.aligned_evaluation / 'summary.json').read_text())
    common502 = json.loads(args.common502_summary.read_text())
    failed502 = set(common502['failed_sample_ids'])
    ids = json.loads(args.common_subset.read_text())['sample_ids']
    if len(ids) != 501 or len(set(ids)) != 501 or aligned['protocol_sha256'] != ph:
        raise ValueError('exact unique common501 and frozen protocol required')
    expected_rows = [(sid, 0, camera) for sid in ids for camera in (9, 10)]
    shared = ('schema', 'protocol_sha256', 'implementation_sha256', 'software_versions',
              'learned_software_versions', 'split', 'appearance_reference', 'per_output_alignment')
    output = args.output
    output.mkdir(parents=True)
    reports, provenance = {}, {'subset_sha256': digest(args.common_subset),
                               'common502_summary_sha256': digest(args.common502_summary), 'methods': {}}
    for method in METHODS:
        root = args.raw_metrics_root / (method + '-full-test')
        full = json.loads((root / 'summary.json').read_text())
        if set(full['methods']) != {method} or any(full[k] != aligned[k] for k in shared):
            raise ValueError('raw and aligned evaluator identity differs: ' + method)
        stem = canonical_hash(method)[:16]
        result_file, feature_file = root / (stem + '-results.json'), root / (stem + '-features.npz')
        result = json.loads(result_file.read_text())
        old = full['methods'][method]
        if result['method'] != method or result['summary'] != old:
            raise ValueError('raw results are incomplete: ' + method)
        rows, views = result['per_deformation_seed'], result['per_view']
        view_ids = [(v['sample_id'], v['seed'], v['camera_id']) for v in views]
        with np.load(feature_file, allow_pickle=False) as z:
            cached_ids = [tuple(json.loads(v)) for v in z['row_ids']]
            if (cached_ids != view_ids or str(z['protocol_sha256'].item()) != ph
                    or str(z['reference_manifest_sha256'].item()) != full['reference_manifest_sha256']
                    or str(z['prediction_manifest_sha256'].item()) != full['prediction_manifest_sha256']
                    or str(z['extractor_id'].item()) != aligned['methods'][method]['distribution']['extractor']):
                raise ValueError('raw feature provenance differs: ' + method)
            pred, ref = z['prediction'].copy(), z['reference'].copy()
        if pred.shape != ref.shape or pred.shape != (len(view_ids), 2048):
            raise ValueError('raw feature shape differs: ' + method)
        common502_mask = np.asarray([sid not in failed502 for sid, _, _ in view_ids])
        if int(common502_mask.sum()) != 1004:
            raise ValueError('raw full common502 feature population differs: ' + method)
        original = distribution(pred[common502_mask], ref[common502_mask], protocol)
        prior_distribution = common502['methods'][method]['sample']['distribution']
        replay_error = max(abs(original[k] - prior_distribution[k])
                           for k in ('FID', 'KID', 'KID_subset_std'))
        if replay_error > 1e-8:
            raise ValueError(f'raw full-population replay mismatch: {method} {replay_error}')
        row_index = {r['sample_id']: r for r in rows}
        feature_index = {key: i for i, key in enumerate(view_ids)}
        if len(row_index) != len(rows) or len(feature_index) != len(view_ids):
            raise ValueError('duplicate raw sample or view: ' + method)
        chosen_rows = [row_index[sid] for sid in ids]
        chosen_indices = [feature_index[key] for key in expected_rows]
        with np.load(args.aligned_evaluation / (stem + '-features.npz'), allow_pickle=False) as z:
            aligned_ids = [tuple(json.loads(v)) for v in z['row_ids']]
            aligned_ref = z['reference'].copy()
            if aligned_ids != expected_rows or str(z['extractor_id'].item()) != aligned['methods'][method]['distribution']['extractor']:
                raise ValueError('aligned reference identity differs: ' + method)
        error = float(np.max(np.abs(ref[chosen_indices] - aligned_ref)))
        if error > 1e-6:
            raise ValueError(f'raw and aligned reference features differ: {method} {error}')
        settings = protocol['bootstrap']
        report = summarize(chosen_rows, draws=settings['draws'], seed=settings['seed'])
        report['distribution'] = distribution(pred[chosen_indices], ref[chosen_indices], protocol)
        reports[method] = report
        provenance['methods'][method] = {'summary_sha256': digest(root / 'summary.json'),
            'results_sha256': digest(result_file), 'features_sha256': digest(feature_file),
            'full_distribution_replay_max_error': replay_error,
            'aligned_reference_feature_max_error': error}
        write_json(output / (stem + '-results.json'),
                   {'method': method, 'summary': report, 'per_deformation_seed': chosen_rows,
                    'per_view': [views[feature_index[k]] for k in expected_rows]})
        print(json.dumps({'method': method, 'samples': len(chosen_rows), 'replay_error': replay_error,
                          'reference_error': error}), flush=True)
    write_json(output / 'summary.json', {'status': 'PASS_RAW_EXTERNAL_COMMON501_FEATURE_REPLAY',
               'sample_ids': ids, 'methods': reports, 'provenance': provenance})
    print(json.dumps({'status': 'PASS_RAW_EXTERNAL_COMMON501_FEATURE_REPLAY',
                      'samples': len(ids), 'methods': len(reports)}), flush=True)


if __name__ == '__main__':
    main()
