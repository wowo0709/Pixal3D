"""Run the unchanged native SS evaluator with an explicit population binding."""
import argparse
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import json
from pathlib import Path
import sys

from .cascade_view_roles import ClosedInputs, file_sha256
from .native_population_contract import ENGINEERING, FINAL, model_binding, validate_capture, validate_plan


def pinned(spec):
    path = Path(spec['path'])
    if file_sha256(path) != spec['SHA256']:
        raise ValueError('bound population input changed: '+str(path))
    return path


@contextmanager
def dataset_binding(core, select):
    """Only replace panel validation; restore caller dictionaries and function."""
    original = core.validate_panel
    touched = []
    calls = 0

    def bound(plan, bundle, cfg):
        nonlocal calls
        calls += 1
        if calls != 1:
            raise ValueError('one native population selection point required')
        ids, seeds, rebound = select(plan, bundle, cfg)
        if {k: v for k, v in rebound.items() if k != 'validation'} != {
                k: v for k, v in cfg.items() if k != 'validation'}:
            raise ValueError('population binding changed model/trainer/learning fields')
        touched.append((cfg, copy.deepcopy(cfg['validation'])))
        cfg['validation'] = copy.deepcopy(rebound['validation'])
        return ids, seeds

    core.validate_panel = bound
    try:
        yield
        if calls != 1:
            raise ValueError('native evaluator never applied the dataset binding')
    finally:
        for cfg, validation in reversed(touched):
            cfg['validation'] = validation
        core.validate_panel = original


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
    own = model_binding(bundle, cfg, hashes)
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

    from . import evaluate_r_native_checkpoint as core
    from . import quick_metrics as quick
    from .evaluate_r_top1_native_checkpoint import top1_evaluation_binding
    from .evaluate_ss_artifact_subset import capture_ss_artifacts, validate_subset_reference

    def select(actual_plan, actual_bundle, actual_cfg):
        if actual_plan != plan or actual_bundle != bundle or actual_cfg != cfg:
            raise ValueError('native evaluator loaded a different actual model or plan')
        return ids, seeds, rebound

    output = Path(plan['output'])
    captured = []
    argv = sys.argv
    try:
        sys.argv = [core.__file__, '--plan', str(plan_path)]
        top1 = top1_evaluation_binding() if own['variant'] == 'R_top1_only' else nullcontext()
        with top1, dataset_binding(core, select), capture_ss_artifacts(quick, output, captured):
            core.main()
    finally:
        sys.argv = argv
    path = output/'evaluation.json'
    result = json.loads(path.read_text())
    validate_capture(result, captured, len(ids))
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
                  population_binding_applied_only_to_dataset_selection=True, frozen_native_model_binding=own)
    result['native_SS_artifacts'] = dict(status='PASS_ACTUAL_NATIVE_BOUND_SS_POPULATION_ARCHIVE',
        archived_asset_count=len(ids), artifacts=captured, predicted_SS_is_available_for_full_cascade=True,
        GT_SS_is_for_scoring_only=True, full_cascade_generated=False, optimizer_updates=0)
    result['verified_population_inputs'] = check.inputs
    path.write_text(json.dumps(result, indent=2)+'\n')
    audit = dict(status='PASS_ACTUAL_NATIVE_BOUND_SS_POPULATION', population_scope=plan['population_scope'],
        assets=len(ids), model_binding=own, original_training_config_unchanged=True,
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
