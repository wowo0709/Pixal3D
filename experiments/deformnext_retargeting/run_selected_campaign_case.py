"""Use whole-asset disk credit and one render mesh per actual SS geometry.

Model phases and raw archives remain original. Further PBR compositions use
their geometry owner's rendering triangles through a source-bound memory input.
"""
import argparse
import json
import os
from pathlib import Path

from . import run_selected_final_case as native
from .selected_campaign_resources import CampaignCaseBudget, source_guard
from .selected_pipeline_model_contract import require
from .selected_shared_render_geometry import (
    bind_owner_render_geometry, required_sources as render_sources, verify_case_render_geometry,
)


def required_sources():
    own = [str(Path(__file__).resolve()), str(Path(__file__).with_name('selected_campaign_resources.py'))]
    return sorted(set(native.required_sources()) | set(own) | set(render_sources()))


def execute(path):
    context = json.loads(Path(path).read_text())
    require(context.get('campaign_credit') is not None
        and os.environ.get('SELECTED_FINAL_CAMPAIGN_TOKEN') == context.get('campaign_token'),
        'explicit actual campaign credit and inherited ownership token required')
    own = [str(Path(__file__).resolve()), str(Path(__file__).with_name('selected_campaign_resources.py')),
           *render_sources()]
    require(set(own) <= set(context['source_SHA256']), 'pin actual credit adapter sources before TEST')
    source_guard(context)
    original = native.DiskBudget
    original_load, original_validate = native.load_case, native.validate_case_reports
    def budget(*args, **kwargs):
        return CampaignCaseBudget(*args, credit=context['campaign_credit'], **kwargs)
    native.DiskBudget = budget
    state = {}
    def load(native_context, check, pins):
        parent, controls, ss, case, owner = original_load(native_context, check, pins)
        case = bind_owner_render_geometry(case, owner, check, context['source_SHA256'])
        state['check'] = check
        return parent, controls, ss, case, owner
    def validate(case, **kwargs):
        result = original_validate(case, **kwargs)
        proof = verify_case_render_geometry(case, state['check'])
        result['render_geometry_reuse_policy'] = 'canonical-owner-render-mesh-v1'
        if proof is not None:result['fixed_generated_render_geometry'] = proof
        return result
    native.load_case, native.validate_case_reports = load, validate
    try:
        return native.execute(path)
    finally:
        native.DiskBudget = original
        native.load_case, native.validate_case_reports = original_load, original_validate


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--plan',type=Path,required=True)
    args = parser.parse_args(); result = execute(args.plan)
    print(json.dumps(dict(status=result['status'],alias=result['alias'],sample_id=result['sample_id'])),flush=True)


if __name__ == '__main__': main()
