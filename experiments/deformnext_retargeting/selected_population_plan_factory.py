"""Prepare an own physical SS validation36 or frozen TEST508 plan on the CPU.

The existing population evaluator keeps each family's numerical forward. This
factory reads closed model/input metadata, never constructs a neural model,
and cannot install queue jobs or choose a candidate.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

from .cascade_view_roles import ClosedInputs, file_sha256 as sha
from .collect_module_seed_confirmation import read_file, validate_file
from .pipeline_population_contract_v2 import (
    ENGINEERING, FINAL, MODE, model_binding, required_sources, validate_plan,
)
from .selected_pipeline_model_contract import require, validate_ss_binding, validate_ss_evaluation


SCHEMA = 'selected-physical-SS-population-preparation-inputs-v1'


def check_input(spec, check, aliases):
    """Keep recipe paths while verifying an explicit byte-identical file4 copy."""
    validate_file(spec)
    original = spec['path']; canonical = aliases.get(original, original)
    require(check.check(canonical, spec['receipt']) == spec['SHA256']
        and sha(original) == spec['SHA256'], 'original input and explicit closed copy must have equal bytes')
    return Path(original)


def read_input(spec, check, aliases, *, lines=False):
    value = check_input(spec, check, aliases).read_text()
    return [json.loads(line) for line in value.splitlines() if line.strip()] if lines else json.loads(value)


def sources(arm):
    return sorted(set(required_sources(arm)) | {str(Path(__file__).resolve())})


def make_plan(request, *, bundle, config, rows, panel, seeds, train_rows,
              validation_rows, reference=None, freeze=None, cache_QA=None,
              cache_QA_plan=None, parity=None):
    """Validate every physical model/population gate before publishing a plan."""
    require(request.get('schema') == SCHEMA, 'explicit physical population preparation schema required')
    arm, binding = request['module_arm'], request['model_binding']
    validate_ss_binding(binding, arm)
    hashes = {binding[k]['path']: binding[k]['SHA256'] for k in
              ('native_config', 'checkpoint', 'conditioner_checkpoint') if binding[k] is not None}
    actual = model_binding(bundle, config, hashes, arm=arm)
    require(json.dumps(binding, sort_keys=True, allow_nan=False)
            == json.dumps(actual, sort_keys=True, allow_nan=False),
            'own actual bundle/config must equal the declared physical model')
    scope = request['population_scope']
    require(scope in (ENGINEERING, FINAL), 'validation36 or final TEST508 scope required')
    for key in ['output', 'wrapper_output', 'repo', 'python', 'canonical_local_root']:
        require(isinstance(request.get(key), str) and Path(request[key]).is_absolute(),
                'explicit absolute execution and output paths required')
    require(request['output'] != request['wrapper_output'], 'separate native and wrapper outputs required')
    assigned = request['assigned_GPUs']
    require(isinstance(assigned, list) and assigned and len(assigned) == len(set(assigned))
            and all(isinstance(g, str) and g.isdecimal() for g in assigned),
            'explicit caller-assigned physical GPUs required')
    pinned_sources = request['source_SHA256']
    require(set(sources(arm)) <= set(pinned_sources), 'pin the factory and every physical numerical source')
    for path, digest in pinned_sources.items():
        validate_file(dict(path=path, SHA256=digest), closed=False)
    population = request['population_inputs']
    require(set(population) == {'manifest', 'train_manifest', 'validation_manifest', 'panel', 'seeds'},
            'all exact population manifests, panel and generation seeds required')
    require(isinstance(request.get('frozen_inputs'), list), 'explicit frozen matcher/anchor input list required')
    specs = [request['bundle'], *population.values(), *request['frozen_inputs']]
    require(request['bundle']['receipt'] == str(Path(request['bundle']['path']).parent/'file4-closure.json'),
            'evaluator must use the actual own bundle directory receipt')
    result = {k: deepcopy(request[k]) for k in ['name', 'module_arm', 'model_alias', 'assigned_GPUs',
        'output', 'wrapper_output', 'repo', 'python', 'canonical_local_root', 'source_SHA256',
        'dataset_roots', 'model_binding', 'population_scope', 'population_inputs']}
    result.update(worker_mode='evaluate', mode=MODE, bundle=request['bundle']['path'],
        panel=population['panel']['path'], seeds=population['seeds']['path'],
        expected_assets=36 if scope == ENGINEERING else 508, generation_batch_size=8,
        deterministic_algorithms=True, canonical_input_aliases=deepcopy(request.get('canonical_input_aliases', {})),
        full_goal_complete=False)
    if scope == ENGINEERING:
        require(all(request.get(k) is None for k in
            ['model_freeze', 'cache_QA', 'cache_QA_plan', 'engineering_parity']),
            'validation preparation cannot masquerade as frozen TEST execution')
        require(reference is not None, 'actual own final128 reference required before validation36')
        validate_ss_evaluation(reference, binding, arm)
        require(reference.get('evaluation_mode') == 'final128'
            and reference.get('native_task_checkpoint_bundle') == request['bundle']['path']
            and len(reference['per_asset']) == 128
            and reference['quick_panel_ids'] == [r['sample_id'] for r in reference['per_asset']]
            and panel == reference['quick_panel_ids'][:36]
            and seeds == {r['sample_id']: r['seed'] for r in reference['per_asset'][:36]},
            'same own final128 prefix36 and noise required; no reselected validation assets')
        result['reference_evaluation'] = deepcopy(request['reference_evaluation'])
        specs.append(request['reference_evaluation'])
    else:
        require(freeze is not None and freeze.get('schema') == 'selected-physical-pipeline-development-freeze-v2',
                'actual selected-pipeline DEV freeze required before TEST preparation')
        require(request.get('reference_evaluation') is None,
                'TEST plan cannot retain a validation reference instead of frozen evidence')
        for key in ['model_freeze', 'cache_QA', 'cache_QA_plan', 'engineering_parity']:
            result[key] = deepcopy(request[key]); specs.append(request[key])
    for spec in specs: validate_file(spec)
    if scope == FINAL:
        specs.extend(freeze['development_evidence'])
    closed = {}; inputs = {}
    for spec in specs:
        # Freeze evidence has plain path/SHA entries; its byte pins are checked
        # by the evaluator. The actual freeze itself has a file4 receipt.
        validate_file(spec, closed='receipt' in spec)
        path, digest = spec['path'], spec['SHA256']
        require(path not in inputs or inputs[path] == digest, 'conflicting declared input hashes')
        inputs[path] = digest
        if 'receipt' in spec:
            canonical = result['canonical_input_aliases'].get(path, path)
            require(isinstance(canonical, str) and Path(canonical).is_absolute(), 'absolute explicit closed copy required')
            require(canonical not in closed or closed[canonical] == spec['receipt'], 'conflicting actual input receipts')
            closed[canonical] = spec['receipt']
    result.update(input_SHA256=inputs, closed_inputs=closed)
    validate_plan(result, bundle, config, rows, panel, seeds, train_rows=train_rows,
        validation_rows=validation_rows, binding=binding, freeze=freeze,
        cache_QA=cache_QA, cache_QA_plan=cache_QA_plan, parity=parity)
    return result


def prepare(path, output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'explicit CPU-only plan preparation required')
    path, output = Path(path), Path(output)
    request = json.loads(path.read_text()); check = ClosedInputs(request['canonical_local_root'])
    check.check(path, path.parent/'file4-closure.json')
    for name in ['output', 'wrapper_output']:
        destination = Path(request[name]); destination.relative_to(check.root)
        require(not destination.exists(), 'preserve existing native or wrapper outputs')
    output.relative_to(check.root)
    require(not output.exists(), 'preserve existing prepared plans')
    bundle = read_file(request['bundle'], check); hashes = {}
    for key in ['native_config', 'checkpoint', 'conditioner_checkpoint']:
        if bundle[key] is not None: hashes[bundle[key]] = check.check(bundle[key], request['bundle']['receipt'])
    config = json.loads(Path(bundle['native_config']).read_text())
    own = model_binding(bundle, config, hashes, arm=request['module_arm'])
    require(own == request['model_binding'], 'actual file4 model bytes differ from requested physical binding')
    aliases = request.get('canonical_input_aliases', {})
    for spec in request['frozen_inputs']: check_input(spec, check, aliases)
    values = {key: read_input(spec, check, aliases, lines=key.endswith('manifest'))
              for key, spec in request['population_inputs'].items()}
    kwargs = {}
    if request['population_scope'] == ENGINEERING:
        kwargs['reference'] = read_file(request['reference_evaluation'], check)
    else:
        for name, key in [('freeze', 'model_freeze'), ('cache_QA', 'cache_QA'),
                          ('cache_QA_plan', 'cache_QA_plan'), ('parity', 'engineering_parity')]:
            kwargs[name] = read_file(request[key], check)
    result = make_plan(request, bundle=bundle, config=config, rows=values['manifest'],
        panel=values['panel']['sample_ids'], seeds=values['seeds'], train_rows=values['train_manifest'],
        validation_rows=values['validation_manifest'], **kwargs)
    for source, digest in result['source_SHA256'].items():
        require(sha(source) == digest, 'actual physical population source changed')
    for source, digest in result['input_SHA256'].items():
        require(sha(source) == digest, 'actual closed population input changed')
    for source, digest in check.inputs.items():
        require(sha(source) == digest, 'closed model or evidence changed during preparation')
    output.mkdir(parents=True, exist_ok=False)
    (output/'population-plan.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    audit = dict(status='PREPARED_OWN_PHYSICAL_SS_POPULATION_PLAN', module_arm=result['module_arm'],
        population_scope=result['population_scope'], expected_assets=result['expected_assets'],
        model_binding=own, verified_closed_inputs=check.inputs, source_SHA256=result['source_SHA256'],
        CPU_neural_forward_or_training_updates=0, actual_GPU_execution_or_queue_installation=False,
        actual_file4_closure_of_prepared_plan_still_required=True, full_goal_complete=False)
    (output/'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); value = prepare(args.plan, args.output)
    print(json.dumps({k: value[k] for k in ['status', 'module_arm', 'population_scope', 'expected_assets']}), flush=True)


if __name__ == '__main__': main()
