"""Execute one immutable SS/perceptual/Inception stage on one assigned GPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import time

import psutil

from .evaluate_bound_population import pinned, sha
from .execute_frozen_cascade_case import close
from .cascade_view_roles import require

PACKAGE='experiments.deformnext_retargeting.'


def command_for(plan, execution):
    kind=plan['kind']; native=plan['native_plan']
    modules={'SS':'evaluate_bound_population','perceptual':'bound_official_perceptual','Inception':'extract_official_inception'}
    require(kind in modules,'explicit frozen SS/perceptual/Inception GPU stage required')
    source=Path(__file__).with_name(modules[kind]+'.py')
    require(pinned(plan['entry_source']).resolve()==source.resolve(),'compiled frozen stage entrypoint differs')
    args=['--plan',native['path']]
    if kind=='perceptual':args+=['--plan-sha256',native['SHA256']]
    if kind=='Inception':args+=['--output',plan['output']]
    return [execution['python'],'-u','-m',PACKAGE+modules[kind],*args]


def execute(plan_spec, execution):
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    require(gpu.isdigit() and gpu in execution['allowed_physical_GPUs']
            and gpu in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','),
            'one explicitly assigned physical GPU required')
    require(pinned(execution['stage_executor_source']).resolve()==Path(__file__).resolve()
            and Path(execution['source_root']).resolve()==Path(__file__).resolve().parents[2],
            'explicit compiled frozen stage executor/source checkout required')
    plan=json.loads(pinned(plan_spec).read_text());pinned(plan['native_plan']);pinned(plan['model_freeze'])
    require(plan['scope']=='final-test508','frozen full-test508 stage required')
    output=Path(plan['output']).resolve()
    require(output.is_relative_to(Path(execution['canonical_local_root']).resolve()),'owned canonical stage outputs required')
    if output.exists():raise FileExistsError('preserve previous GPU stage and failed outputs')
    if 'CPU_affinity_by_GPU' in execution:os.sched_setaffinity(0,set(execution['CPU_affinity_by_GPU'][gpu]))
    command=command_for(plan,execution)
    started=time.monotonic()
    # A sibling log does not create the native evaluator's fresh output folder.
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_suffix('.native.log').open('xb') as stream:
        child=subprocess.Popen(command,cwd=execution['source_root'],env=dict(os.environ,
            PYTHONPATH=execution['source_root'],CUBLAS_WORKSPACE_CONFIG=':4096:8',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'),
            stdout=stream,stderr=subprocess.STDOUT)
        birth=psutil.Process(child.pid).create_time();code=child.wait()
    output.mkdir(parents=True,exist_ok=True)
    shutil.move(str(output.with_suffix('.native.log')),str(output/'native.log'))
    terminal=dict(native_PID=child.pid,birth=birth,returncode=code,physical_GPU=gpu,command=command,
                  plan=plan_spec,elapsed_seconds=time.monotonic()-started)
    (output/'terminal.json').write_text(json.dumps(terminal,indent=2)+'\n')
    if code:raise RuntimeError('native frozen GPU stage failed; outputs preserved')
    pinned(plan_spec);pinned(plan['native_plan']);pinned(plan['entry_source']);pinned(execution['stage_executor_source'])
    result=json.loads((output/plan['result_file']).read_text())
    require(result['status']==plan['result_status'],'native GPU stage status differs')
    if plan['kind']=='SS':
        require(result['bound_population']['scope']=='final-test508'
                and result['bound_population']['model_freeze']==plan['model_freeze']
                and result['native_SS_artifacts']['archived_asset_count']==508
                and result['optimizer_updates_during_evaluation']==0,'completed full frozen SS archive required')
    elif plan['kind']=='perceptual':
        require(result['full_panel'] and len(result['per_view'])==1524
                and (output/'binding-audit.json').is_file(),'completed bound official1524-view perceptual scores required')
    else:
        require(result['scope']=='final-test508' and result['images']==1524 and result['assets']==508
                and result['optimizer_updates']==0,'completed native1524-image official feature bank required')
    close(output,execution=execution,tag=plan['kind'])
    return terminal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--execution',type=Path,required=True);parser.add_argument('--execution-sha256',required=True)
    args=parser.parse_args()
    execution=json.loads(pinned(dict(path=str(args.execution),SHA256=args.execution_sha256)).read_text())
    print(json.dumps(execute(dict(path=str(args.plan),SHA256=args.plan_sha256),execution)),flush=True)


if __name__=='__main__':main()
