"""Reaggregate cached external metrics on the final common success set.

The source evaluator has already computed each image pair and saved its exact
Inception vectors. This program verifies those caches against the original
manifests, drops only excluded sample IDs, and uses the benchmark's own
aggregation/FID/KID functions. Official RGB1 rows come from a fresh two-method
evaluation on the same final population.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from deformnext_benchmark.aggregate import summarize
from deformnext_benchmark.io import canonical_hash, digest, read_jsonl, validate_protocol, write_json
from deformnext_benchmark.metrics import frechet_distance, kernel_distance


EXTERNAL = {'trellis2', 'pixal3d', 'reconviagen', 'mvadapter',
            'corradapter', 'shapeup', 'spacecontrol'}
OFFICIAL = {'pixal3d_official_rgb1_raw', 'pixal3d_official_rgb1_source_m0'}
METRICS = ['LPIPS', 'DreamSim', 'CLIP', 'FID', 'KID',
           'PSNR', 'SSIM', 'M_IoU', 'CD', 'FScore']


def result_path(root: Path, method: str) -> Path:
    return root / (canonical_hash(method)[:16] + '-results.json')


def feature_path(root: Path, method: str) -> Path:
    return root / (canonical_hash(method)[:16] + '-features.npz')


def distribution(prediction: np.ndarray, reference: np.ndarray, protocol: dict,
                 extractor: str, independent_uids: int) -> dict:
    settings = protocol['distribution']
    if len(prediction) != len(reference) or len(prediction) < settings['kid_subset_size']:
        raise ValueError('complete matching Inception population required')
    return {'FID': frechet_distance(prediction, reference),
            **kernel_distance(prediction, reference,
                              subset_size=settings['kid_subset_size'],
                              subsets=settings['kid_subsets'], seed=settings['seed']),
            'image_count': len(prediction), 'independent_UIDs': independent_uids,
            'extractor': extractor}


def feature_payload(path: Path, expected_rows: list[tuple[str, int, int]],
                    expected_protocol: str, references: Path,
                    predictions: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as cache:
        ids = [tuple(json.loads(value)) for value in cache['row_ids']]
        if ids != expected_rows:
            raise ValueError('cached feature order and per-view rows differ: ' + str(path))
        if (str(cache['protocol_sha256'].item()) != expected_protocol
                or str(cache['reference_manifest_sha256'].item()) != digest(references)
                or str(cache['prediction_manifest_sha256'].item()) != digest(predictions)):
            raise ValueError('cached features do not bind the frozen full evaluation')
        p = cache['prediction'].copy()
        r = cache['reference'].copy()
        extractor = str(cache['extractor_id'].item())
    if (p.shape != r.shape or p.shape != (len(ids), 2048)
            or not np.isfinite(p).all() or not np.isfinite(r).all()):
        raise ValueError('finite complete Inception-2048 features required')
    return p, r, extractor


def save_features(path: Path, prediction: np.ndarray, reference: np.ndarray,
                  rows: list[tuple[str, int, int]], protocol_sha: str,
                  extractor: str, new_references: Path, new_predictions: Path) -> None:
    np.savez_compressed(path, prediction=prediction, reference=reference,
                        row_ids=np.asarray([json.dumps(list(row)) for row in rows]),
                        protocol_sha256=protocol_sha, extractor_id=extractor,
                        reference_manifest_sha256=digest(new_references),
                        prediction_manifest_sha256=digest(new_predictions))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-evaluation', type=Path, required=True)
    parser.add_argument('--full-binding', type=Path, required=True)
    parser.add_argument('--official-evaluation', type=Path, required=True)
    parser.add_argument('--official-binding', type=Path, required=True)
    parser.add_argument('--nine-binding', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('fresh nine-method evaluation output required')
    protocol = validate_protocol(json.loads(args.protocol.read_text()))
    ph = canonical_hash(protocol)
    full = json.loads((args.full_evaluation / 'summary.json').read_text())
    official = json.loads((args.official_evaluation / 'summary.json').read_text())
    common_metadata = ('schema', 'protocol_sha256', 'implementation_sha256',
                       'software_versions', 'learned_software_versions', 'split',
                       'appearance_reference', 'per_output_alignment')
    if (set(full['methods']) != EXTERNAL or set(official['methods']) != OFFICIAL
            or any(full[key] != official[key] for key in common_metadata)
            or full['protocol_sha256'] != ph or full['split'] != 'test'):
        raise ValueError('same implementation/protocol/metric models required')
    full_refs = args.full_binding / 'references.jsonl'
    full_preds = args.full_binding / 'predictions.jsonl'
    official_refs = args.official_binding / 'references.jsonl'
    official_preds = args.official_binding / 'predictions.jsonl'
    nine_refs = args.nine_binding / 'references.jsonl'
    nine_preds = args.nine_binding / 'predictions.jsonl'
    if (full['reference_manifest_sha256'] != digest(full_refs)
            or full['prediction_manifest_sha256'] != digest(full_preds)
            or official['reference_manifest_sha256'] != digest(official_refs)
            or official['prediction_manifest_sha256'] != digest(official_preds)):
        raise ValueError('source evaluator summaries and manifests differ')
    all_refs = read_jsonl(full_refs)
    final_refs = read_jsonl(nine_refs)
    official_reference_rows = read_jsonl(official_refs)
    full_ref_index = {row['sample_id']: row for row in all_refs}
    final_ids = [row['sample_id'] for row in final_refs]
    final_set = set(final_ids)
    if (len(all_refs) != 502 or len(full_ref_index) != 502
            or not 1 <= len(final_ids) <= 502 or len(final_set) != len(final_ids)
            or final_ids != [row['sample_id'] for row in all_refs
                             if row['sample_id'] in final_set]
            or final_refs != [full_ref_index[sid] for sid in final_ids]
            or official_reference_rows != final_refs):
        raise ValueError('final references must be an ordered exact full502 subset')
    full_prediction_rows = read_jsonl(full_preds)
    final_prediction_rows = read_jsonl(nine_preds)
    official_prediction_rows = read_jsonl(official_preds)
    full_index = {(row['method'], row['sample_id']): row for row in full_prediction_rows}
    final_index = {(row['method'], row['sample_id']): row for row in final_prediction_rows}
    official_index = {(row['method'], row['sample_id']): row for row in official_prediction_rows}
    if (len(full_index) != 502 * len(EXTERNAL)
            or len(full_prediction_rows) != len(full_index)
            or len(final_index) != len(final_ids) * (len(EXTERNAL) + len(OFFICIAL))
            or len(final_prediction_rows) != len(final_index)
            or len(official_index) != len(final_ids) * len(OFFICIAL)
            or len(official_prediction_rows) != len(official_index)
            or {row['method'] for row in final_prediction_rows} != EXTERNAL | OFFICIAL):
        raise ValueError('complete external, official and nine-method bindings required')
    for key, row in final_index.items():
        original = full_index if key[0] in EXTERNAL else official_index
        if row != original.get(key) or row['seed'] != 0 or row['status'] != 'ok':
            raise ValueError('final prediction differs from its source binding: ' + str(key))
    args.output.mkdir(parents=True)
    methods = {}
    provenance = {'status': 'PASS_EXACT_CACHED_FEATURE_SUBSET_REPLAY',
                  'full_samples': 502, 'final_samples': len(final_ids),
                  'excluded_ids': sorted(set(full_ref_index) - final_set),
                  'full_evaluation_summary_sha256': digest(args.full_evaluation / 'summary.json'),
                  'official_evaluation_summary_sha256': digest(args.official_evaluation / 'summary.json'),
                  'full_feature_cache_sha256': {}, 'maximum_full_population_replay_error': 0.0,
                  'reused_external_methods': sorted(EXTERNAL),
                  'fresh_official_methods': sorted(OFFICIAL)}
    settings = protocol['bootstrap']
    for method in sorted(EXTERNAL):
        original_result_path = result_path(args.full_evaluation, method)
        original_result = json.loads(original_result_path.read_text())
        old = full['methods'][method]
        if (original_result['method'] != method or original_result['summary'] != old
                or len(original_result['per_deformation_seed']) != 502
                or len(original_result['per_view']) != 1004
                or original_result['failures'] or not old['full_metric_computation']
                or old['successful_predictions'] != 502):
            raise ValueError('full source method incomplete: ' + method)
        rows = original_result['per_deformation_seed']
        views = original_result['per_view']
        view_ids = [(row['sample_id'], row['seed'], row['camera_id']) for row in views]
        expected_view_ids = [(sid, 0, camera) for sid in full_ref_index for camera in (9, 10)]
        if view_ids != expected_view_ids:
            raise ValueError('full source view order differs: ' + method)
        cache_path = feature_path(args.full_evaluation, method)
        p, r, extractor = feature_payload(cache_path, view_ids, ph, full_refs, full_preds)
        if extractor != old['distribution']['extractor']:
            raise ValueError('full feature extractor identity differs')
        # Reproduce the already published 502 result before changing the set.
        full_summary = summarize(rows, draws=settings['draws'], seed=settings['seed'])
        full_dist = distribution(p, r, protocol, extractor, full_summary['source_uids'])
        error = max(abs(full_dist[key] - old['distribution'][key])
                    for key in ('FID', 'KID', 'KID_subset_std'))
        for group in ('deformation_weighted', 'UID_weighted'):
            error = max(error, *(abs(full_summary[group][key] - old[group][key])
                                  for key in full_summary[group]))
        for metric, bounds in full_summary['deformation_weighted_cluster_CI95'].items():
            error = max(error, *(abs(a-b) for a, b in zip(bounds,
                                                           old['deformation_weighted_cluster_CI95'][metric])))
        if error > 1e-8:
            raise ValueError(f'cannot reproduce full502 metric cache: {method} {error}')
        provenance['maximum_full_population_replay_error'] = max(
            provenance['maximum_full_population_replay_error'], error)
        provenance['full_feature_cache_sha256'][method] = digest(cache_path)
        kept_rows = [row for row in rows if row['sample_id'] in final_set]
        kept_views = [row for row in views if row['sample_id'] in final_set]
        mask = np.asarray([sid in final_set for sid, _, _ in view_ids], dtype=bool)
        kept_view_ids = [row for row, keep in zip(view_ids, mask) if keep]
        if (len(kept_rows) != len(final_ids) or len(kept_views) != len(final_ids) * 2
                or [row['sample_id'] for row in kept_rows] != final_ids):
            raise ValueError('replay rows differ from final ordered set: ' + method)
        report = summarize(kept_rows, draws=settings['draws'], seed=settings['seed'])
        report.update(expected_predictions=len(final_ids),
                      successful_predictions=len(final_ids), failure_count=0,
                      failure_stages={}, conditional_on_success_only=True,
                      complete_population=True, numeric_only=False,
                      full_metric_computation=True)
        report['distribution'] = distribution(p[mask], r[mask], protocol,
                                              extractor, report['source_uids'])
        write_json(result_path(args.output, method),
                   {'method': method, 'summary': report,
                    'per_deformation_seed': kept_rows,
                    'per_view': kept_views, 'failures': []})
        save_features(feature_path(args.output, method), p[mask], r[mask],
                      kept_view_ids, ph, extractor, nine_refs, nine_preds)
        methods[method] = report
        print(json.dumps({'phase': 'REPLAY_EXTERNAL_METHOD', 'method': method,
                          'samples': len(kept_rows), 'full_replay_error': error}), flush=True)
    for method in sorted(OFFICIAL):
        original_result_path = result_path(args.official_evaluation, method)
        result = json.loads(original_result_path.read_text())
        report = official['methods'][method]
        if (result['method'] != method or result['summary'] != report
                or len(result['per_deformation_seed']) != len(final_ids)
                or len(result['per_view']) != len(final_ids) * 2
                or result['failures'] or not report['full_metric_computation']
                or report['successful_predictions'] != len(final_ids)
                or report['distribution']['image_count'] != len(final_ids) * 2):
            raise ValueError('official two-method result incomplete: ' + method)
        view_ids = [(row['sample_id'], row['seed'], row['camera_id'])
                    for row in result['per_view']]
        p, r, extractor = feature_payload(feature_path(args.official_evaluation, method),
                                          view_ids, ph, official_refs, official_preds)
        if extractor != report['distribution']['extractor']:
            raise ValueError('official feature extractor identity differs')
        write_json(result_path(args.output, method), result)
        save_features(feature_path(args.output, method), p, r, view_ids,
                      ph, extractor, nine_refs, nine_preds)
        methods[method] = report
    if len({methods[method]['distribution']['extractor'] for method in methods}) != 1:
        raise ValueError('all methods must use the same Inception extractor')
    summary = {key: full[key] for key in common_metadata}
    summary.update(reference_manifest_sha256=digest(nine_refs),
                   prediction_manifest_sha256=digest(nine_preds),
                   methods={method: methods[method] for method in sorted(methods)})
    write_json(args.output / 'summary.json', summary)
    with (args.output / 'table.csv').open('x', newline='') as stream:
        fields = ['method', 'weighting', 'expected_predictions',
                  'successful_predictions', 'failure_count', 'complete_population', *METRICS]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method in sorted(methods):
            report = methods[method]
            for weighting in ('deformation_weighted', 'UID_weighted'):
                line = {'method': method, 'weighting': weighting,
                        **{key: report[key] for key in fields[2:6]}}
                line.update({key: value for key, value in report[weighting].items()
                             if key in METRICS})
                if weighting == 'deformation_weighted':
                    line.update({key: report['distribution'][key]
                                 for key in ('FID', 'KID')})
                writer.writerow(line)
    write_json(args.output / 'cache-replay-provenance.json', provenance)
    print(json.dumps({'status': provenance['status'], 'samples': len(final_ids),
                      'methods': len(methods),
                      'maximum_full_population_replay_error':
                      provenance['maximum_full_population_replay_error']}), flush=True)


if __name__ == '__main__':
    main()
