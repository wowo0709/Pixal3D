"""Run each finite module family unchanged with explicit validation36/TEST508 data.

No architecture is selected here. Final execution still requires actual DEV
freeze, full cache coverage, disjoint identities and two independent validation
loads before any held-out model forward.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from .cascade_view_roles import ClosedInputs, file_sha256
from .evaluate_native_bound_population import dataset_binding, pinned
from .native_population_contract import validate_capture
from .unified_population_contract import (
    ENGINEERING, FINAL, evaluator_binding, model_binding, required_sources, validate_plan,
)


def execute(plan_path):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    check = ClosedInputs(plan['canonical_local_root'])
    check.check(plan_path, plan_path.parent/'file4-closure.json')

    def guard():
        check.check(plan_path, plan_path.parent/'file4-closure.json')
        for path, digest in {**plan['source_SHA256'], **plan.get('input_SHA256', {})}.items():
            if file_sha256(path) != digest:
                raise ValueError('native population source/input changed: '+path)
        for path, receipt in plan['closed_inputs'].items():
            check.check(path, receipt)

    if not set(required_sources(plan['module_arm'])) <= set(plan['source_SHA256']):
        raise ValueError('pin every critical physical-family population entry and original runtime source')
    guard()
    inputs = plan['population_inputs']
    paths = {key: pinned(value) for key, value in inputs.items()}
    rows = [json.loads(line) for line in paths['manifest'].read_text().splitlines()]
    train = [json.loads(line) for line in paths['train_manifest'].read_text().splitlines()]
    validation = [json.loads(line) for line in paths['validation_manifest'].read_text().splitlines()]
    ids = json.loads(paths['panel'].read_text())['sample_ids']
    seeds = json.loads(paths['seeds'].read_text())
    bundle_path = Path(plan['bundle'])
    check.check(bundle_path, bundle_path.parent/'file4-closure.json')
    bundle = json.loads(bundle_path.read_text())
    hashes = {}
    for key in ('native_config', 'checkpoint', 'conditioner_checkpoint'):
        if bundle.get(key):
            hashes[bundle[key]] = check.check(bundle[key], bundle_path.parent/'file4-closure.json')
    cfg = json.loads(Path(bundle['native_config']).read_text())
    own = model_binding(bundle, cfg, hashes, arm=plan['module_arm'])
    extra_specs = []
    if plan['population_scope'] == FINAL:
        extra_specs.extend(plan[k] for k in ('model_freeze', 'cache_QA', 'cache_QA_plan', 'engineering_parity'))
        freeze = json.loads(pinned(plan['model_freeze']).read_text())
        for evidence in freeze['development_evidence']:
            pinned(evidence)
            extra_specs.append(evidence)
        qa = json.loads(pinned(plan['cache_QA']).read_text())
        qa_plan = json.loads(pinned(plan['cache_QA_plan']).read_text())
        parity = json.loads(pinned(plan['engineering_parity']).read_text())
        reference = None
    else:
        extra_specs.append(plan['reference_evaluation'])
        freeze = qa = qa_plan = parity = None
        reference = json.loads(pinned(plan['reference_evaluation']).read_text())
        if ids != reference['quick_panel_ids'][:36]:
            raise ValueError('engineering panel must be the actual reference128 prefix36')
    rebound = validate_plan(plan, bundle, cfg, rows, ids, seeds, train_rows=train,
                            validation_rows=validation, binding=own, freeze=freeze,
                            cache_QA=qa, cache_QA_plan=qa_plan, parity=parity)

    from . import quick_metrics as quick
    from .evaluate_ss_artifact_subset import capture_ss_artifacts, validate_subset_reference

    def select(actual_plan, actual_bundle, actual_cfg):
        if actual_plan != plan or actual_bundle != bundle or actual_cfg != cfg:
            raise ValueError('native evaluator loaded a different actual model or plan')
        return ids, seeds, rebound

    output = Path(plan['output'])
    captured = []
    argv = sys.argv
    try:
        with evaluator_binding(plan['module_arm']) as core:
            sys.argv = [core.__file__, '--plan', str(plan_path)]
            with dataset_binding(core, select), capture_ss_artifacts(quick, output, captured):
                core.main()
    finally:
        sys.argv = argv
    path = output/'evaluation.json'
    result = json.loads(path.read_text())
    validate_capture(result, captured, len(ids))
    if (result['native_variant'] != own['variant']
            or result['actual_source_tensors'] != own['source_tensors']
            or result['actual_LoRA_tensors'] != 360):
        raise ValueError('actual loaded LoRA/source parameters differ from the frozen physical family')
    import numpy as np
    for item in captured:
        artifact = output/item['artifact']
        if file_sha256(artifact) != item['artifact_SHA256']:
            raise ValueError('captured native SS archive changed')
        with np.load(artifact) as arrays:
            for name in ('prediction_latent', 'target_latent'):
                value = arrays[name]
                digest = hashlib.sha256((str(value.dtype)+repr(value.shape)).encode())
                digest.update(value.tobytes())
                item[name+'_SHA256'] = digest.hexdigest()
    prefix = False
    if plan['population_scope'] == ENGINEERING:
        actual = dict(result, per_asset=result['per_asset'][:32], quick_panel_ids=result['quick_panel_ids'][:32])
        validate_subset_reference(actual, reference, captured[:32], count=32)
        prefix = True
    for spec in inputs.values():
        pinned(spec)
    for spec in extra_specs:
        pinned(spec)
    guard()
    result.update(population_scope=plan['population_scope'], diagnostic_only=plan['population_scope'] == ENGINEERING,
                  population_binding_applied_only_to_dataset_selection=True, frozen_native_model_binding=own,
                  module_arm=plan['module_arm'], physical_runtime_family=own['runtime_family'])
    result['native_SS_artifacts'] = dict(status='PASS_ACTUAL_NATIVE_BOUND_SS_POPULATION_ARCHIVE',
        archived_asset_count=len(ids), artifacts=captured, predicted_SS_is_available_for_full_cascade=True,
        GT_SS_is_for_scoring_only=True, full_cascade_generated=False, optimizer_updates=0)
    result['verified_population_inputs'] = check.inputs
    path.write_text(json.dumps(result, indent=2)+'\n')
    audit = dict(status='PASS_ACTUAL_NATIVE_BOUND_SS_POPULATION', population_scope=plan['population_scope'],
        assets=len(ids), model_binding=own, module_arm=plan['module_arm'],
        physical_runtime_family=own['runtime_family'], original_training_config_unchanged=True,
        original_models_trainer_sampler_and_normalization_unchanged=True,
        actual_generation_batch=8, tail_generation_batch=len(ids) % 8, sampling_steps=12, guidance=3.,
        actual_reference_prefix32_records_exact=prefix, actual_saved_SS_occupancy_counts_verified=True,
        loaded_trainable_SHA256=result['actual_loaded_trainable_SHA256'], optimizer_updates=0,
        independent_validation36_repeatability_not_proven_by_one_run=True,
        heldout_scope=plan['population_scope'] == FINAL, full_goal_complete=False)
    (output/'population-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    (output/'completed-bound.json').write_text(json.dumps(dict(status=audit['status'], assets=len(ids),
        population_scope=plan['population_scope'], optimizer_updates=0, full_goal_complete=False), indent=2)+'\n')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.plan)), flush=True)


if __name__ == '__main__':
    main()
