"""Prepare exact-image final508 feature plans, then bind completed GPU pools.

This data-only entrypoint neither selects a model nor computes a feature or a
quality score. Both phases require the same completed, development-frozen
eight-metric evaluation. Existing plans and failed outputs are preserved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate_bound_population import pinned, sha
from .extract_official_inception import validate_pool
from .metric_image_identity import scored_image_observation, validate_metric_image_join
from .summarize_frozen_inception_pools import load_frozen_pool, validate_frozen_models


def load_inputs(plan):
    if plan['scope'] != 'final-test508':
        raise ValueError('explicit final-test508 scope required')
    eight = json.loads(pinned(plan['eight_metrics']).read_text())
    if (eight['scope'] != 'final-test508'
            or eight['status'] != 'PASS_ACTUAL_BOUND_CASCADE_EIGHT_METRICS'):
        raise ValueError('completed final-test508 eight-metric inputs required; development outputs cannot be promoted')
    freeze = json.loads(pinned(plan['model_freeze']).read_text())
    ids, clusters, _ = validate_frozen_models(freeze, eight['model_bindings'])
    if (plan['scope'] != 'final-test508' or eight['scope'] != 'final-test508'
            or eight['status'] != 'PASS_ACTUAL_BOUND_CASCADE_EIGHT_METRICS'
            or eight['model_freeze'] != plan['model_freeze']
            or eight['model_bindings'] != freeze['models']
            or eight['manifest']['SHA256'] != freeze['population']['manifest_SHA256']
            or eight['assets'] != 508 or eight['source_clusters'] != 134
            or eight['actual_completed_cases'] != len(freeze['models'])*508
            or eight['actual_perceptual_views'] != len(freeze['models'])*1524
            or not eight['native_stage_weight_hashes_verified']
            or tuple(eight['scored_image_observations']) != tuple(freeze['models'])):
        raise ValueError('completed exact-model final508 eight-metric inputs required')
    for evidence in freeze['development_evidence']:
        pinned(evidence)
    registry = json.loads(pinned(plan['metric_registry']).read_text())
    pinned(plan['weights'])
    if (registry['package'] != 'torch-fidelity0.3.0'
            or registry['weights_SHA256'] != plan['weights']['SHA256']
            or not Path(plan['package_root']).is_dir()):
        raise ValueError('explicit official Inception package and registered weight bytes required')
    return freeze, eight, ids, clusters


def prepare(plan, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('preserve existing feature plans and outputs')
    freeze, eight, ids, clusters = load_inputs(plan)
    batch = plan['batch_size']
    if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
        raise ValueError('explicit positive extraction batch required')
    keys = [(sid, cid) for sid in ids for cid in (8, 9, 10)]
    actual_arrays, references, pools = {}, {}, {}
    for index, (alias, rows) in enumerate(eight['scored_image_observations'].items()):
        observed = {(row['sample_id'], row['camera_id']): row for row in rows}
        if len(rows) != len(observed) or set(observed) != set(keys):
            raise ValueError('complete distinct scored sample/camera population required')
        observations, gt_observations = [], []
        for key in keys:
            row = observed[key]
            if (row['source_cluster_id'] != clusters[key[0]]
                    or eight['input_SHA256'].get(row['archive']) != row['archive_SHA256']):
                raise ValueError('scored source cluster or bound archive differs')
            path = row['archive']
            if path not in actual_arrays:
                pinned(dict(path=path, SHA256=row['archive_SHA256']))
                with np.load(path, allow_pickle=False) as arrays:
                    actual_arrays[path] = scored_image_observation(arrays, sample_id=key[0], camera_id=key[1],
                        source_cluster_id=clusters[key[0]], archive=path, archive_SHA256=row['archive_SHA256'])
            actual = actual_arrays[path]
            if any(actual[field] != value for field, value in row.items()
                   if field not in {'sample_id', 'camera_id', 'source_cluster_id'}):
                raise ValueError('actual scored archive RGB/GT/alpha/conversion identity changed')
            gt_identity = tuple(row[field] for field in ['reference_float_RGB_SHA256',
                'reference_CHW_uint8_SHA256', 'reference_alpha_SHA256'])
            if references.setdefault(key, gt_identity) != gt_identity:
                raise ValueError('all frozen models require the identical scored shared reference')
            common = {name: row[name] for name in ['sample_id', 'camera_id', 'source_cluster_id', 'archive', 'archive_SHA256']}
            observations.append(dict(common, array='prediction',
                original_float_RGB_SHA256=row['prediction_float_RGB_SHA256']))
            gt_observations.append(dict(common, array='GT_shared_renderer',
                original_float_RGB_SHA256=row['reference_float_RGB_SHA256']))
        pools[f'model{index:03d}'] = dict(alias=alias, pool_role='generated-prediction',
                                        model_binding=freeze['models'][alias], observations=observations)
        if alias == freeze['comparison_roles']['vanilla']:
            pools['reference'] = dict(alias=None, pool_role='shared-original-material-GT',
                model_binding=freeze['reference_binding'], observations=gt_observations)
    # Validate every pool before creating any output directory.
    extract_plans = {}
    for name, pool in pools.items():
        extract = dict(scope='final-test508', pool_role=pool['pool_role'], model_binding=pool['model_binding'],
            observations=pool['observations'], expected_sample_ids=ids, weights=plan['weights']['path'],
            weights_SHA256=plan['weights']['SHA256'], package_root=plan['package_root'], batch_size=batch,
            model_freeze=plan['model_freeze'], scored_eight_metric_input=plan['eight_metrics'])
        validate_pool(extract)
        extract_plans[name] = extract
    pinned(plan['model_freeze']); pinned(plan['eight_metrics'])
    output.mkdir(parents=True, exist_ok=False)
    descriptors = {}
    for name, extract in extract_plans.items():
        path = output/(name+'-extract.json')
        path.write_text(json.dumps(extract, indent=2)+'\n')
        descriptors[name] = dict(alias=pools[name]['alias'], extract_plan=dict(path=str(path), SHA256=sha(path)),
                                 output=str(output/(name+'-features')))
    manifest = dict(status='PREPARED_FROZEN_TEST508_OFFICIAL_FEATURE_INPUT_PLANS', scope='final-test508',
        source_inputs=plan, pools=descriptors, models=list(freeze['models']), assets=508, images_per_pool=1524,
        no_neural_feature_extraction_or_quality_score_performed=True, model_selection_unchanged=True,
        actual_scored_archive_RGB_GT_alpha_and_uint8_fingerprints_verified=True, full_goal_complete=False)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def bind_completed_pools(manifest_spec):
    manifest = json.loads(pinned(manifest_spec).read_text())
    if manifest['status'] != 'PREPARED_FROZEN_TEST508_OFFICIAL_FEATURE_INPUT_PLANS':
        raise ValueError('explicit prepared frozen feature input manifest required')
    plan = manifest['source_inputs']
    freeze, eight, ids, clusters = load_inputs(plan)
    predictions, observations, reference, reference_rows = {}, {}, None, None
    for name, descriptor in manifest['pools'].items():
        extract = json.loads(pinned(descriptor['extract_plan']).read_text())
        root = Path(descriptor['output'])
        spec = {key: dict(path=str(root/filename), SHA256=sha(root/filename))
                for key, filename in [('audit', 'audit.json'), ('features', 'features.npz')]}
        alias = descriptor['alias']
        binding = freeze['reference_binding'] if alias is None else freeze['models'][alias]
        role = 'shared-original-material-GT' if alias is None else 'generated-prediction'
        _, audit = load_frozen_pool(spec, role=role, binding=binding, sample_ids=ids, clusters=clusters)
        if (audit['input_plan_SHA256'] != descriptor['extract_plan']['SHA256']
                or audit['weights_SHA256'] != plan['weights']['SHA256']
                or extract['model_freeze'] != plan['model_freeze']
                or extract['scored_eight_metric_input'] != plan['eight_metrics']):
            raise ValueError('completed GPU features must execute the exact prepared frozen extraction plan')
        if alias is None:
            if reference is not None: raise ValueError('one shared reference pool required')
            reference, reference_rows = spec, audit['observations']
        else:
            if alias in predictions: raise ValueError('one feature pool per frozen model required')
            predictions[alias], observations[alias] = spec, audit['observations']
    if tuple(predictions) != tuple(freeze['models']) or reference is None:
        raise ValueError('all frozen model feature pools and the shared reference must finish')
    identity = validate_metric_image_join(eight, dict(feature_image_observations=observations,
        reference_image_observations=reference_rows), sample_ids=ids, clusters=clusters, models=freeze['models'])
    pinned(manifest_spec); pinned(plan['model_freeze']); pinned(plan['eight_metrics'])
    return dict(scope='final-test508', model_freeze=plan['model_freeze'], package_root=plan['package_root'],
        predictions=predictions, reference=reference, prepared_feature_manifest=manifest_spec,
        actual_scored_input_image_identity=identity, no_distribution_score_computed_during_binding=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    first = commands.add_parser('prepare')
    first.add_argument('--plan', type=Path, required=True); first.add_argument('--output', type=Path, required=True)
    second = commands.add_parser('bind')
    second.add_argument('--manifest', type=Path, required=True)
    second.add_argument('--manifest-sha256', required=True); second.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(json.loads(args.plan.read_text()), args.output)
        print(json.dumps(dict(status=result['status'], models=result['models'], assets=508)), flush=True)
    else:
        if args.output.exists(): raise FileExistsError('preserve existing distribution plan')
        result = bind_completed_pools(dict(path=str(args.manifest), SHA256=args.manifest_sha256))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream: json.dump(result, stream, indent=2); stream.write('\n')
        print(json.dumps(dict(status='BOUND_COMPLETED_ACTUAL_FEATURE_POOLS', models=list(result['predictions']))), flush=True)


if __name__ == '__main__':
    main()
