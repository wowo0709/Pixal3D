"""Bind native generated-support PBR replay to the completed held-out cache.

The frozen constructor, normalization, matching head and sampler stay intact.
Only dataset.args.match_root and dataset.args.path_map may change. This data
preparation happens before model freeze and never executes a model or metric.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .evaluate_bound_population import pinned, sha


SCHEMA = 'native-generated-PBR-test508-dataset-binding-v1'


def project_conditioner_dataset(original, *, file4_cache, match_root, expected_q2c_sha256):
    if (not isinstance(file4_cache, str) or not Path(file4_cache).is_absolute()
            or not isinstance(match_root, str) or not Path(match_root).is_absolute()
            or not isinstance(expected_q2c_sha256, str) or len(expected_q2c_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_q2c_sha256)):
        raise ValueError('explicit absolute data roots and frozen matching-head digest required')
    data = original['dataset']['args']
    if (set(data['path_map']) != {'/file4'}
            or not isinstance(data['path_map']['/file4'], str)
            or not Path(data['path_map']['/file4']).is_absolute()
            or not isinstance(data['match_root'], str)
            or not Path(data['match_root']).is_absolute()
            or data['expected_q2c_sha256'] != expected_q2c_sha256):
        raise ValueError('original native data roots and unchanged frozen matching head required')
    projected = copy.deepcopy(original)
    projected['dataset']['args'].update(path_map={'/file4':file4_cache}, match_root=match_root)
    reverse = copy.deepcopy(projected)
    for key in ['path_map','match_root']:
        reverse['dataset']['args'][key] = data[key]
    if reverse != original:
        raise ValueError('changes outside two explicit generated-PBR data roots')
    return projected


def load_QA(specs):
    qa = json.loads(pinned(specs['cache_QA']).read_text())
    qp = json.loads(pinned(specs['cache_QA_plan']).read_text())
    manifest = pinned(specs['manifest'])
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if (qa['status'] != 'PASS_ACTUAL_TEST508_NATIVE_CACHE_AND_LOADER_QA'
            or qa['assets'] != 508 or not qa['SS_PBR_cache_ready'] or not qa['no_test_asset_dropped']
            or not qa['train_validation_identity_disjoint'] or not qa['all_input_RGB_fingerprints_verified']
            or not qa['no_candidate_model_forward_or_test_quality_metric']
            or qa['matching_head_training_or_calibration_updates'] != 0 or qa['Omega_completed_updates'] != 23597
            or qa['published_manifest_SHA256'] != specs['manifest']['SHA256']
            or Path(qa['published_manifest']).resolve() != manifest.resolve()
            or Path(qp['published_manifest']).resolve() != manifest.resolve()
            or len(rows) != 508 or len({r['sample_id'] for r in rows}) != 508
            or len({r['source_cluster_id'] for r in rows}) != 134
            or any(r['split'] != 'test'
                   or any(not isinstance(r[key],str) or not r[key] for key in ['sample_id','source_cluster_id'])
                   for r in rows)):
        raise ValueError('actual complete native test508/134-source cache and input QA required')
    return qa, qp


def matching_head(original, qa):
    args = original['trainer']['args']['deform_conditioner']
    if args['freeze_q2c'] is not True:
        raise ValueError('the TRAIN matching decoder must remain frozen')
    path = args['q2c_checkpoint']
    if not isinstance(path,str) or not Path(path).is_absolute():
        raise ValueError('explicit absolute frozen TRAIN matching decoder required')
    if path.startswith('/file4/'):
        path = original['dataset']['args']['path_map']['/file4']+path[len('/file4'):]
    spec = dict(path=path, SHA256=qa['frozen_TRAIN_q2c_SHA256'])
    pinned(spec)
    return spec


def validate_binding(binding_spec, *, conditioner_config, cache_QA, cache_QA_plan, manifest):
    record = json.loads(pinned(binding_spec).read_text())
    specs = dict(cache_QA=cache_QA, cache_QA_plan=cache_QA_plan, manifest=manifest)
    if (record['schema'] != SCHEMA or record['scope'] != 'final-test508'
            or record['conditioner_config'] != conditioner_config
            or any(record[key] != value for key, value in specs.items())):
        raise ValueError('executed PBR config and cache binding differ from the frozen composition')
    qa, qp = load_QA(specs)
    original = json.loads(pinned(record['original_conditioner_config']).read_text())
    actual = json.loads(pinned(conditioner_config).read_text())
    expected = project_conditioner_dataset(original, file4_cache=qp['file4_cache'],
        match_root=qp['match_root'], expected_q2c_sha256=qa['frozen_TRAIN_q2c_SHA256'])
    if actual != expected or record['matching_head'] != matching_head(original, qa):
        raise ValueError('frozen PBR constructor/model parameters or bound data roots changed')
    return record


def prepare(plan, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('preserve original and previously bound conditioner configs')
    required = {'original_conditioner_config','cache_QA','cache_QA_plan','manifest'}
    if plan.get('scope') != 'final-test508' or not required.issubset(plan):
        raise ValueError('explicit final-test508 scope and completed cache QA required')
    qa, qp = load_QA(plan)
    original = json.loads(pinned(plan['original_conditioner_config']).read_text())
    derived = project_conditioner_dataset(original, file4_cache=qp['file4_cache'],
        match_root=qp['match_root'], expected_q2c_sha256=qa['frozen_TRAIN_q2c_SHA256'])
    head = matching_head(original, qa)
    for key in sorted(required):
        pinned(plan[key])
    output.mkdir(parents=True, exist_ok=False)
    path = output/'conditioner-config.json'
    path.write_text(json.dumps(derived, indent=2)+'\n')
    result = dict(schema=SCHEMA, scope='final-test508',
        **{key:plan[key] for key in sorted(required)},
        conditioner_config=dict(path=str(path), SHA256=sha(path)), matching_head=head,
        changed_fields=['dataset.args.path_map','dataset.args.match_root'],
        original_models_trainer_conditioner_normalization_and_sampler_preserved=True,
        original_training_config_unchanged=True, no_model_forward_quality_score_or_matching_head_update=True,
        model_selection_not_performed=True, must_be_included_in_future_model_freeze=True,
        full_goal_complete=False)
    (output/'binding.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(json.loads(args.plan.read_text()), args.output)
    print(json.dumps(dict(schema=result['schema'], conditioner_config=result['conditioner_config'],
                          no_model_forward_or_quality_read=True)), flush=True)


if __name__ == '__main__':
    main()
