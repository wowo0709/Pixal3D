"""Use exact original metric workers/reducer with immutable mixed case roots.

Only the CPU preparer/collector joins require an operational entry. All eleven
GPU metric jobs keep their original command, source and numerical plans.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path

from . import collect_selected_final_metrics as collector
from . import prepare_selected_final_metrics as preparer
from .cascade_view_roles import ClosedInputs, file_sha256 as sha
from .operational_selected_case_reuse import loader, load_manifest
from .selected_final_metric_inputs import frozen_inputs
from .selected_pipeline_model_contract import require


MODULE = 'experiments.deformnext_retargeting.operational_reused_metric_entry'


def collector_descriptors(jobs):
    require(len({j['name'] for j in jobs}) == len(jobs), 'unique actual metric descriptors required')
    result = deepcopy(jobs)
    count = 0
    for job in result:
        cmd = job['command']
        if len(cmd) >= 4 and cmd[1:4] == ['-u', '-m', 'experiments.deformnext_retargeting.collect_selected_final_metrics']:
            require(job['resource'] == 'cpu' and len(cmd) == 6 and cmd[4] == '--plan',
                'only the exact original CPU collector command may be redirected')
            job['command'] = [*cmd[:3], MODULE, '--phase', 'collect', *cmd[4:]]
            count += 1
    require(count == 1, 'exactly one original CPU collector descriptor required')
    return result


@contextmanager
def operational_runtime():
    before_prepare = preparer.load_campaign
    before_collect = collector.load_campaign
    before_append = preparer.append_jobs
    def append(context, jobs):
        return before_append(context, collector_descriptors(jobs))
    try:
        preparer.load_campaign = collector.load_campaign = loader
        preparer.append_jobs = append
        yield
    finally:
        preparer.load_campaign, collector.load_campaign = before_prepare, before_collect
        preparer.append_jobs = before_append


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=['prepare', 'collect'])
    p.add_argument('--plan', required=True, type=Path)
    args = p.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'metadata-only CPU metric join required')
    context = json.loads(args.plan.read_text())
    os.sched_setaffinity(0, set(context['CPU_affinity']))
    check = ClosedInputs(context['canonical_local_root'])
    check.check(args.plan, args.plan.parent / 'file4-closure.json')
    parent, controls, models = frozen_inputs(context, check)
    manifest = load_manifest(context, check, parent, models)
    require(manifest['operational_source_SHA256'].get(str(Path(__file__).resolve())) == sha(__file__),
        'actual CPU metric join source must be separately pinned')
    with operational_runtime():
        value = preparer.prepare(args.plan) if args.phase == 'prepare' else collector.collect(args.plan)
    print(json.dumps(dict(status=value['status'], operational_only_case_root_join=True)), flush=True)


if __name__ == '__main__':
    main()
