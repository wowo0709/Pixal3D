"""Operational scheduling adapter; all native/model/metric source files stay frozen.

Retirement runs synchronously while existing GPU cases continue. Thus the
reader ledger cannot change between the original two strict reader checks.
Protected process metadata is waited for, never treated as an empty read.
This entry does not resume or overwrite a previous campaign namespace.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import sys

from . import run_selected_final_campaign as frozen
from .operational_process_access import process_access_waits
from .operational_retirement_wait import verify_operational_sources


MODULE = 'experiments.deformnext_retargeting.operational_campaign_entry'
RETIREMENT = 'experiments.deformnext_retargeting.operational_retirement_wait'


def command(context, path, keeper=None):
    result = [context['python'], '-u', '-m', MODULE, '--plan', str(path)]
    return result if keeper is None else result + ['--keeper', keeper]


class SerializedRetirementCampaign(frozen.Campaign):
    def spawn(self, name, context, module, *, sid, alias, kind, gpu='', role=None):
        if kind == 'retirement':
            if module != 'experiments.deformnext_retargeting.retire_selected_final_raw' or gpu:
                raise ValueError('only the original CPU retirement may use this adapter')
            context = dict(context, operational_source_SHA256=self.context['operational_source_SHA256'],
                operational_campaign_root=str(self.root),
                process_access_event_log=str(self.root / 'workers' / name / 'process-access-events.jsonl'))
            module = RETIREMENT
        elif module != 'experiments.deformnext_retargeting.run_selected_campaign_case':
            raise ValueError('native case entry must remain unchanged')
        super().spawn(name, context, module, sid=sid, alias=alias, kind=kind, gpu=gpu, role=role)
        if kind == 'retirement':
            # No scheduler/reader-ledger update until strict retirement ends.
            # Other case subprocesses continue; no GPU worker receives a signal.
            self.own[name]['child'].wait()
            self.collect()


@contextmanager
def operational_runtime():
    original_command, original_campaign = frozen.command, frozen.Campaign
    try:
        frozen.command, frozen.Campaign = command, SerializedRetirementCampaign
        yield
    finally:
        frozen.command, frozen.Campaign = original_command, original_campaign


def main():
    # The original parser/loader/source guard and all case-plan construction
    # still run. This preliminary read binds only the separate adapter bytes.
    path = Path(sys.argv[sys.argv.index('--plan') + 1])
    context = json.loads(path.read_text())
    verify_operational_sources(context)
    root = Path(context['campaign_root'])
    def emit(event):
        root.mkdir(parents=True, exist_ok=True)
        with (root / 'coordinator-process-access-events.jsonl').open('a') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
    with operational_runtime(), process_access_waits(emit=emit):
        frozen.main()


if __name__ == '__main__':
    main()
