"""Consume completed single-view TEST geometry and score it on shared GT pixels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assignment',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if gpu not in {'6','7'} or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','):
        raise ValueError('one assigned render GPU6/7 required')
    cases=json.loads(args.assignment.read_text())
    if not cases or len({x['sample_id'] for x in cases})!=len(cases):
        raise ValueError('nonempty unique render assignment required')
    repo=Path(__file__).resolve().parents[2]
    failures=[]
    for index,case in enumerate(cases):
        sid=case['sample_id'];root=args.output_root/sid
        generation=root/'geometry/generation.json'
        failure=root/'generation-failure.json'
        while not generation.is_file() and not failure.is_file():time.sleep(8)
        if failure.is_file():
            failures.append(sid);continue
        source=json.loads(Path(case['reference_case_plan']).read_text())
        phases={p['name']:p['plan'] for p in source['phases']}
        try:
            report=json.loads(generation.read_text())
            if report.get('status')!='PASS_ORIGINAL_PIXAL3D_SINGLE_VIEW_GENERATION' or report['sample_id']!=sid:
                raise ValueError('actual single-view generation required')
            render=dict(phases['render']);render.update(generation_root=str(root/'geometry'),output=str(root/'render'))
            shared=dict(phases['shared']);shared.update(prediction_render_root=str(root/'render'),output=str(root/'shared'))
            for phase,plan in [('render',render),('shared',shared)]:
                document=root/(phase+'-plan.json')
                if document.exists() and json.loads(document.read_text())!=plan:
                    raise ValueError('previous phase plan differs: '+phase)
                document.write_text(json.dumps(plan,indent=2)+'\n')
                result=root/phase/'render-evaluation.json'
                if result.is_file():continue
                if (root/phase).exists():raise FileExistsError('incomplete phase output: '+phase)
                command=[sys.executable,'-u','-m','experiments.deformnext_retargeting.singleview_adapted_eval_entry',
                    '--phase',phase,'--plan',str(document)]
                with (root/(phase+'.log')).open('w') as log:
                    done=subprocess.run(command,cwd=repo,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                if done.returncode:raise RuntimeError(f'{phase} exited {done.returncode}')
            result=dict(status='PASS_PIXAL3D_SINGLE_VIEW_RENDER_AND_SHARED_GT',sample_id=sid,
                original_single_view_input=True,GT_not_used_for_generation=True,
                physical_GPU=gpu,geometry_metrics=report['metrics'],
                target_and_source_scores=json.loads((root/'shared/render-evaluation.json').read_text())['per_view'])
            (root/'case-audit.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(status=result['status'],sample_id=sid,completed=index+1,total=len(cases))),flush=True)
        except Exception as exc:
            failures.append(sid)
            (root/'render-failure.json').write_text(json.dumps(dict(sample_id=sid,error=repr(exc),traceback=traceback.format_exc()),indent=2)+'\n')
            print(json.dumps(dict(status='FAILED_SINGLE_VIEW_RENDER',sample_id=sid,error=repr(exc))),flush=True)
    if failures:raise RuntimeError(f'{len(failures)} single-view render cases failed')


if __name__=='__main__':main()
