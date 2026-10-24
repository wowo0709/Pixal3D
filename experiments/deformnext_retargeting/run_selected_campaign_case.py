"""Run the original selected case with an atomic whole-asset disk credit.

The only replacement is the DiskBudget reservation operation. Native plans,
weights, samplers, RGB inputs, phase entries, scores and closure stay unchanged.
"""
import argparse
import json
import os
from pathlib import Path

from . import run_selected_final_case as native
from .selected_campaign_resources import CampaignCaseBudget, source_guard
from .selected_pipeline_model_contract import require


def execute(path):
    context = json.loads(Path(path).read_text())
    require(context.get('campaign_credit') is not None
        and os.environ.get('SELECTED_FINAL_CAMPAIGN_TOKEN') == context.get('campaign_token'),
        'explicit actual campaign credit and inherited ownership token required')
    own = [str(Path(__file__).resolve()), str(Path(__file__).with_name('selected_campaign_resources.py'))]
    require(set(own) <= set(context['source_SHA256']), 'pin actual credit adapter sources before TEST')
    source_guard(context)
    original = native.DiskBudget
    def budget(*args, **kwargs):
        return CampaignCaseBudget(*args, credit=context['campaign_credit'], **kwargs)
    native.DiskBudget = budget
    try:
        return native.execute(path)
    finally:
        native.DiskBudget = original


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan',type=Path,required=True)
    args = parser.parse_args(); result = execute(args.plan)
    print(json.dumps(dict(status=result['status'],alias=result['alias'],sample_id=result['sample_id'])),flush=True)


if __name__ == '__main__': main()
