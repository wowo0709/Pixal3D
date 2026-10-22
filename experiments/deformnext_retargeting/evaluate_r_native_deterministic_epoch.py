"""Deterministic saved-pair measurement, distinct from stochastic epoch replay.

In-process training used atomic ray reductions. Its old exact-replay failure
remains a failure. This protocol measures the same saved weights under the
already declared deterministic final-evaluation algorithm and can be repeated
exactly; it does not certify equality with the nondeterministic training row.
"""
import argparse
import copy
from contextlib import nullcontext
import json
from pathlib import Path

from . import evaluate_r_native_checkpoint as loader


MODE = 'epoch96-deterministic-saved-pair'
IDENTITY = ('sample_id', 'source_cluster_id', 'seed', 'target_voxels_sha256',
            'target_voxel_shape', 'target_representation')
REPEATED = ('native_variant', 'checkpoint', 'conditioner_checkpoint', 'arm_config', 'step',
            'quick_panel_ids', 'quick_metrics', 'per_asset', 'actual_loaded_trainable_SHA256')


def validate_panel(plan, bundle, cfg, *, original):
    if (plan['mode'] != MODE or plan['deterministic_algorithms'] is not True
            or bundle['step'] != 96):
        raise ValueError('explicit deterministic saved epoch96 measurement required')
    # Preserve the exact original epoch's asset/noise/batch contract. Only the
    # declared reduction algorithm differs; no original metric is replaced.
    contract = copy.deepcopy(plan)
    contract.update(mode='epoch-replay', deterministic_algorithms=False)
    return original(contract, bundle, cfg)


def compare_repeats(actual, reference):
    if (actual.get('evaluation_mode') != MODE or reference.get('evaluation_mode') != MODE
            or not actual.get('deterministic_algorithms') or not reference.get('deterministic_algorithms')):
        raise ValueError('both measurements must use the declared deterministic protocol')
    for key in REPEATED:
        if actual[key] != reference[key]:
            raise ValueError('deterministic repeat changed actual output: '+key)


def check_original_identity(actual, native):
    if (actual['step'] != native['step'] or actual['quick_panel_ids'] != native['quick_panel_ids']
            or len(actual['per_asset']) != len(native['quick_per_asset'])):
        raise ValueError('saved pair must retain the actual epoch panel and step')
    for row, original in zip(actual['per_asset'], native['quick_per_asset']):
        if any(row[k] != original[k] for k in IDENTITY):
            raise ValueError('saved-pair measurement changed GT/noise/asset identity')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    original = loader.validate_panel
    loader.validate_panel = lambda p, b, c: validate_panel(p, b, c, original=original)
    events = dict(matching=[], consumer=[])
    if plan.get('collect_native_signals'):
        from .r_native_signal_probe import native_signal_context
        context = native_signal_context(events)
    else:
        context = nullcontext()
    try:
        with context:
            loader.main()
    finally:
        loader.validate_panel = original
    output = Path(plan['output'])
    actual = json.loads((output/'evaluation.json').read_text())
    bundle = json.loads(Path(plan['bundle']).read_text())
    check_original_identity(actual, bundle['native_validation'])
    actual.update(exact_saved_pair_epoch_metric_replay=False,
        original_inprocess_reduction_mode='nondeterministic_atomic_index_add',
        original_inprocess_metrics_preserved=bundle['native_validation']['quick_metrics'],
        original_inprocess_per_asset_exact=sum(a == b for a, b in zip(
            actual['per_asset'], bundle['native_validation']['quick_per_asset'])),
        same_checkpoint_deterministic_repetition_verified=False)
    if plan.get('collect_native_signals'):
        from .cascade_view_roles import ClosedInputs
        reference_path = Path(plan['signal_reference_evaluation'])
        check = ClosedInputs(plan['canonical_local_root'])
        check.check(reference_path, reference_path.parent/'file4-closure.json')
        reference = json.loads(reference_path.read_text())
        compare_repeats(actual, reference)
        if len(events['matching']) != 4 or len(events['consumer']) != 4:
            raise ValueError('four real batch8 source and consumer observations required')
        actual['same_checkpoint_deterministic_repetition_verified'] = True
        (output/'signal-probe.json').write_text(json.dumps(dict(
            status='PASS_DETERMINISTIC_SAVED_PAIR_REPEAT_AND_INERT_SIGNAL_OBSERVER',
            reference=str(reference_path), actual_same_checkpoint_records_exact=True,
            original_stochastic_epoch_records_certified=False, matching_and_consumer_events=events,
            assets=32, optimizer_updates=0, quality_gain_proven=False, full_goal_complete=False), indent=2)+'\n')
    (output/'evaluation.json').write_text(json.dumps(actual, indent=2)+'\n')


if __name__ == '__main__':
    main()
