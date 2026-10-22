"""Replay native epoch96 with inert signal observers and compare exact normal records."""
import argparse
import json
from pathlib import Path

from .cascade_view_roles import ClosedInputs
from .evaluate_r_native_checkpoint import main as evaluate
from .r_native_signal_probe import native_signal_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan['mode'] != 'epoch-replay' or plan['expected_assets'] != 32:
        raise ValueError('this observer uses the actual first epoch32 replay protocol')
    reference_path = Path(plan['signal_reference_evaluation'])
    check = ClosedInputs(plan['canonical_local_root'])
    check.check(reference_path, reference_path.parent / 'file4-closure.json')
    reference = json.loads(reference_path.read_text())
    if reference['native_variant'] == 'vanilla' or reference['step'] != 96:
        raise ValueError('same actual trained R model native96 reference required')
    events = dict(matching=[], consumer=[])
    with native_signal_context(events):
        evaluate()
    output = Path(plan['output'])
    actual = json.loads((output / 'evaluation.json').read_text())
    for key in ['native_variant', 'checkpoint', 'conditioner_checkpoint', 'arm_config', 'step',
                'quick_panel_ids', 'quick_metrics', 'per_asset', 'actual_loaded_trainable_SHA256']:
        if actual[key] != reference[key]:
            raise ValueError('read-only observer changed actual normal native replay ' + key)
    if len(events['matching']) != 4 or len(events['consumer']) != 4:
        raise ValueError('exactly four actual batch8 inference conditioning passes required')
    record = dict(status='PASS_NATIVE96_SIGNAL_OBSERVER_AND_EXACT_NORMAL_REPLAY',
        native_variant=actual['native_variant'], checkpoint=actual['checkpoint'], step=96,
        assets=32, optimizer_updates=0, actual_native_noisy_sampling_records_exact=True,
        matching_and_consumer_events=events, reference=str(reference_path),
        matching_accuracy_or_causal_quality_gain_not_claimed=True, full_goal_complete=False)
    (output / 'signal-probe.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
