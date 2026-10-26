"""Execute only TEST assets absent from two immutable complete-group producers."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys

from . import run_selected_final_campaign as frozen
from .operational_campaign_entry import SerializedRetirementCampaign
from .operational_process_access import process_access_waits
from .operational_reused_campaign_entry import RemainingCampaign, verify_reuse_sources
from .operational_multi_campaign_reuse import load_manifest
from .selected_pipeline_model_contract import require


MODULE = 'experiments.deformnext_retargeting.operational_multi_campaign_entry'


def command(context, path, keeper=None):
    result = [context['python'], '-u', '-m', MODULE, '--plan', str(path)]
    return result if keeper is None else result + ['--keeper', keeper]


class MultiSourceCampaign(RemainingCampaign):
    def __init__(self, context, path, check, parent, controls, models):
        manifest = load_manifest(context, check, parent, models, verify_groups=True)
        SerializedRetirementCampaign.__init__(self, context, path, check, parent, controls, models)
        self.reused = manifest['groups']
        self.full_ids = list(self.ids)
        self.ids = [sid for sid in self.full_ids if sid not in self.reused]
        require(self.ids and len(self.ids) + len(self.reused) == 508,
            'actual remaining physical assets plus both complete producers required')

    def run(self):
        result = SerializedRetirementCampaign.run(self)
        require(set(result['actual_cases']) == set(self.ids)
            and not set(result['actual_cases']) & set(self.reused),
            'remaining native execution must be complete and disjoint')
        combined = {sid: (self.reused[sid]['cases'] if sid in self.reused
            else result['actual_cases'][sid]) for sid in self.full_ids}
        require(len(combined) == 508 and all(set(v) == set(self.models) for v in combined.values()),
            'all original508 by all five physical models required')
        result.update(assets=508, actual_cases=combined,
            actual_new_assets=len(self.ids), actual_reused_assets=len(self.reused),
            actual_new_native_cases=len(self.ids) * len(self.models),
            actual_reused_native_cases=len(self.reused) * len(self.models),
            immutable_case_reuse_manifest=self.context['source_inputs']['case_reuse_manifest'],
            both_failed_campaigns_and_original_terminals_preserved=True)
        return result


@contextmanager
def operational_runtime():
    old_command, old_campaign = frozen.command, frozen.Campaign
    try:
        frozen.command, frozen.Campaign = command, MultiSourceCampaign
        yield
    finally:
        frozen.command, frozen.Campaign = old_command, old_campaign


def main():
    path = Path(sys.argv[sys.argv.index('--plan') + 1])
    context = json.loads(path.read_text())
    verify_reuse_sources(context)
    root = Path(context['campaign_root'])
    def emit(event):
        root.mkdir(parents=True, exist_ok=True)
        with (root / 'coordinator-process-access-events.jsonl').open('a') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
    with operational_runtime(), process_access_waits(emit=emit):
        frozen.main()


if __name__ == '__main__':
    main()
