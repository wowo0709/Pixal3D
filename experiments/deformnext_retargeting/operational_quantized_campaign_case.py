"""Route only final rendering through separately pinned integer-pixel GT QA."""
import argparse
import json
import os
from pathlib import Path

from . import run_selected_campaign_case as campaign_case
from . import run_selected_final_case as native
from .cascade_view_roles import file_sha256 as sha
from .selected_pipeline_model_contract import require


ORIGINAL_PHASE = 'experiments.deformnext_retargeting.selected_final_phase'
RENDER_PHASE = 'experiments.deformnext_retargeting.operational_quantized_render_phase'


def execute(path):
    path = Path(path)
    context = json.loads(path.read_text())
    sources = context.get('operational_source_SHA256', {})
    for name in ('operational_quantized_campaign_case.py', 'operational_quantized_render_phase.py'):
        source = Path(__file__).with_name(name)
        require(sources.get(str(source)) == sha(source), 'pin independent final GT QA adapter: ' + name)
    render = Path(__file__).with_name('operational_quantized_render_phase.py')
    old_run_child, old_validate = native.run_child, native.validate_case_reports
    altered = []

    def run_child(child_context, output, label, command):
        if label != 'render':
            return old_run_child(child_context, output, label, command)
        require(command.count(ORIGINAL_PHASE) == 1 and command[3] == ORIGINAL_PHASE,
                'replace only the original final render phase module')
        command = list(command)
        command[3] = RENDER_PHASE
        previous = os.environ.get('SELECTED_OPERATIONAL_RENDER_SHA256')
        os.environ['SELECTED_OPERATIONAL_RENDER_SHA256'] = sources[str(render)]
        try:
            terminal = old_run_child(child_context, output, label, command)
        finally:
            if previous is None:
                os.environ.pop('SELECTED_OPERATIONAL_RENDER_SHA256', None)
            else:
                os.environ['SELECTED_OPERATIONAL_RENDER_SHA256'] = previous
        altered.append(terminal)
        return terminal

    def validate(case, **kwargs):
        result = old_validate(case, **kwargs)
        require(len(altered) == 1 and altered[0]['phase'] == 'render',
                'exactly one pinned final render phase required')
        proof_path = Path(context['output']) / 'render' / 'operational-raster-gate.json'
        proof = json.loads(proof_path.read_text())
        require(proof['status'] == 'PASS_UNIFORM_INTEGER_PIXEL_GT_RASTER_QA'
                and proof['operational_source_SHA256'] == sources[str(render)]
                and len(proof['observations']) == 3,
                'uniform GT QA proof required before case audit')
        result['operational_GT_raster_admission'] = dict(path=str(proof_path), SHA256=sha(proof_path),
            policy=proof['policy'], original_render_source_unchanged=True,
            metric_math_and_GT_arrays_unchanged=True)
        return result

    native.run_child, native.validate_case_reports = run_child, validate
    try:
        return campaign_case.execute(path)
    finally:
        native.run_child, native.validate_case_reports = old_run_child, old_validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.plan)
    print(json.dumps(dict(status=result['status'], alias=result['alias'],
                          sample_id=result['sample_id'])), flush=True)


if __name__ == '__main__':
    main()
