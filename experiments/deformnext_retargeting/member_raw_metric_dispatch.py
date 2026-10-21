"""Select the existing SS metric for the physical RAW-only dataset name.

The real dataset, RAW channels, matching mode, model and sampler are passed
through unchanged. Only a copied metric selector is mapped to its native SS
name. This scoped entry supports versioned pending commands without editing
their frozen original runtime or training recipes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import runpy
import sys


RAW = 'DeformNeXtMemberRawSS64'
SS = 'DeformNeXtSS64'
ENTRIES = {
    'experiments.deformnext_retargeting.member_feature_deformation_runtime',
    'experiments.deformnext_retargeting.evaluate_member_feature_ss_artifact_subset',
    'experiments.deformnext_retargeting.native_source_seed_runtime',
}


@contextmanager
def raw_ss_metric_dispatch(events, *, quick_module=None):
    if quick_module is None:
        from . import quick_metrics as quick_module
    original = quick_module.evaluate_quick

    def selected(trainer, dataset, config):
        if config.get('dataset_name') != RAW:
            return original(trainer, dataset, config)
        selector = dict(config, dataset_name=SS)
        events.append(dict(source_dataset_name=RAW, metric_selector=SS,
                           actual_dataset_class=type(dataset).__name__,
                           actual_dataset_and_trainer_objects_unchanged=True,
                           quick_configuration_object_unchanged=selector.get('quick_metrics') is config.get('quick_metrics')))
        return original(trainer, dataset, selector)

    quick_module.evaluate_quick = selected
    try:
        yield
    finally:
        quick_module.evaluate_quick = original


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--entry-module', choices=sorted(ENTRIES), required=True)
    local, remaining = parser.parse_known_args()
    events = []
    previous = sys.argv
    try:
        sys.argv = [local.entry_module, *remaining]
        with raw_ss_metric_dispatch(events):
            runpy.run_module(local.entry_module, run_name='__main__')
    finally:
        sys.argv = previous
    if not events:
        raise ValueError('the explicit RAW-SS runtime must execute its native SS metric selector')
    if '--output' in remaining:
        output = Path(remaining[remaining.index('--output')+1])
    else:
        config = Path(remaining[remaining.index('--config')+1])
        output = Path(json.loads(config.read_text())['output_dir'])
    report = dict(status='PASS_ACTUAL_RAW_SS_NATIVE_METRIC_DISPATCH', entry=local.entry_module,
                  events=events, original_metric_function_restored=True,
                  real_RAW_dataset_and_original_recipe_preserved=True,
                  no_sampler_optimizer_model_or_input_tensor_change=True,
                  full_goal_complete=False)
    with (output/'raw-SS-metric-dispatch.json').open('x') as stream:
        json.dump(report,stream,indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'],metric_calls=len(events))),flush=True)


if __name__ == '__main__':
    main()
