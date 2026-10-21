"""Prepare native SS508 evaluations for every development-frozen composition.

This CPU-only factory neither selects models nor reads their test predictions.
It retains each frozen native argument list except output/panel locations and
population sizes. The existing bound evaluator validates full cache QA, source
disjointness, weight bytes and the native sampler before plans are published.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .evaluate_bound_population import argument, load_binding, pinned, project_dataset, sha
from .summarize_frozen_inception_pools import validate_frozen_models


def population_arguments(original, *, output, panel, seeds):
    """Relocate only explicit population/output arguments; preserve all others."""
    if not isinstance(original, list) or not original or any(not isinstance(x, str) for x in original):
        raise ValueError('explicit frozen native argument list required')
    result = copy.deepcopy(original)
    relocations = {'--output': str(output), '--panel-size': '508',
                   '--panel-file': panel['path'], '--seed-map-file': seeds['path']}
    if argument(original, '--expected-assets', optional=True) is not None:
        relocations['--expected-assets'] = '508'
    for flag, value in relocations.items():
        argument(original, flag)
        result[result.index(flag)+1] = value
    reverse = copy.deepcopy(result)
    for flag in relocations:
        reverse[reverse.index(flag)+1] = argument(original, flag)
    if reverse != original:
        raise ValueError('native argument changes exceed declared population selection')
    return result, relocations


def prepare(plan, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('preserve existing prepared plans and scientific outputs')
    required = {'model_freeze', 'cache_QA', 'cache_QA_plan', 'manifest', 'panel', 'seeds',
                'train_manifest', 'validation_manifest', 'native_entry'}
    if plan.get('scope') != 'final-test508' or not required.issubset(plan):
        raise ValueError('final508 factory requires a development-only freeze and completed native cache QA')
    freeze = json.loads(pinned(plan['model_freeze']).read_text())
    ids, clusters, _ = validate_frozen_models(freeze, freeze['models'])
    for evidence in freeze['development_evidence']:
        pinned(evidence)
    qa_plan = json.loads(pinned(plan['cache_QA_plan']).read_text())
    roots = {key: qa_plan[key] for key in ['match_root', 'projection_root']}
    # Pin common inputs before constructing any scientific execution plans.
    for key in sorted(required-{'model_freeze', 'cache_QA_plan'}):
        pinned(plan[key])
    prepared = {}
    for index, (alias, composition) in enumerate(freeze['models'].items()):
        model = composition['ss64']
        if 'native_arguments' not in model:
            raise ValueError('each frozen SS binding must include its actual native argument template')
        name = f'model{index:03d}'
        destination = output/(name+'-prediction')
        native_args, relocated = population_arguments(model['native_arguments'], output=destination,
                                                       panel=plan['panel'], seeds=plan['seeds'])
        bound = dict(scope='final-test508', model_alias=alias, assets=508, output=str(destination),
                     evaluation_module=model['evaluation_module'], native_arguments=native_args,
                     dataset_roots=roots, **{key: plan[key] for key in sorted(required)})
        rows, actual_ids, _, split, actual_model = load_binding(bound)
        if (split != 'test' or actual_ids != ids or actual_model != model
                or {row['sample_id']:row['source_cluster_id'] for row in rows} != clusters):
            raise ValueError('actual native held-out population/source clusters differ from the model freeze')
        cfg = json.loads(pinned(model['arm_config']).read_text())
        project_dataset(cfg, manifest=plan['manifest']['path'], expected_split=split, **roots)
        prepared[name] = dict(alias=alias, plan=bound, native_argument_relocations=relocated)
    # Do not leave partial plans when any model fails a cache/source/weight guard.
    for key in sorted(required):
        pinned(plan[key])
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    for name, value in prepared.items():
        path = output/(name+'-SS.json')
        path.write_text(json.dumps(value['plan'], indent=2)+'\n')
        records[name] = dict(alias=value['alias'], native_bound_plan=dict(path=str(path), SHA256=sha(path)),
            output=value['plan']['output'], evaluation_module=value['plan']['evaluation_module'],
            native_argument_relocations=value['native_argument_relocations'])
    result = dict(status='PREPARED_ALL_FROZEN_TEST508_NATIVE_SS_POPULATIONS', scope='final-test508',
        source_inputs=plan, model_freeze=plan['model_freeze'], models=list(freeze['models']),
        evaluations=records, assets=508, source_clusters=134, input_RGB_cameras=[8,9,10,11],
        generation_batch=8, sampling_steps=12, guidance=3.,
        frozen_native_source_model_arguments_and_original_training_recipes_preserved=True,
        actual_completed_cache_QA_and_all_population_source_disjointness_checked=True,
        all_frozen_controls_retained=True, no_test_prediction_or_quality_metric_read=True,
        actual_GPU_evaluations_or_queue_installation_not_performed=True,
        downstream_native_cascade_and_ten_metrics_still_required=True, full_goal_complete=False)
    (output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(json.loads(args.plan.read_text()), args.output)
    print(json.dumps(dict(status=result['status'], models=result['models'], assets=result['assets'],
                          no_test_quality_read=True)), flush=True)


if __name__ == '__main__':
    main()
