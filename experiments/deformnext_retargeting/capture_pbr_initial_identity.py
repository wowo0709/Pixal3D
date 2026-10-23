"""Capture a native DDP initial4 identity failure before any optimizer update.

The original worker and evaluator run unchanged. The identity checker records
its actual inputs/results and always stops before permitting training. Preserve
the deliberate child exception separately from scientific-training outcomes.
"""
import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.multiprocessing as mp

from . import train_r_mediated_pbr_native as native


def differences(actual, reference, fields):
    lookup={r['sample_id']:r for r in reference['per_asset']}
    result=[]
    for row in actual['quick_per_asset']:
        prior=lookup[row['sample_id']]
        mismatches={k:dict(actual=row[k],reference=prior[k]) for k in fields if row[k]!=prior[k]}
        metric={k:dict(actual=v,reference=prior['metrics'][k],delta=v-prior['metrics'][k])
            for k,v in row['metrics'].items() if v!=prior['metrics'][k]}
        result.append(dict(sample_id=row['sample_id'],identity_mismatches=mismatches,metric_mismatches=metric))
    return result


def worker(rank,cfg,plan):
    from . import r_mediated_pbr_science_metrics as metrics
    from .evaluate_r_mediated_pbr_initial import IDENTITY_FIELDS
    original=metrics.validate_initial4
    def capture(actual,reference):
        assert rank==0 and actual['step']==0
        error=None
        try:original(actual,reference)
        except ValueError as exc:error=str(exc)
        report=dict(status='ACTUAL_NATIVE_INITIAL4_IDENTITY_CAPTURED_BEFORE_UPDATE',
            original_identity_check_passed=error is None,original_error=error,
            actual=actual,reference=reference,differences=differences(actual,reference,IDENTITY_FIELDS),
            actual_optimizer_updates=0,original_evaluator_model_and_optimizer_guard_already_passed=True,
            original_identity_guard_not_bypassed=True,training_not_admitted=True,full_goal_complete=False)
        (Path(cfg['output_dir'])/'initial-identity-capture.json').write_text(json.dumps(report,indent=2)+'\n')
        raise RuntimeError('INTENTIONAL_STOP_AFTER_NATIVE_INITIAL4_CAPTURE_BEFORE_UPDATE')
    metrics.validate_initial4=capture
    try:native.worker(rank,cfg,plan)
    finally:metrics.validate_initial4=original


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    cfg,_=native.guard_plan(args.plan,plan)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    allowed=set(os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','))
    if len(visible)!=4 or len(set(visible))!=4 or not set(visible)<=allowed or set(visible)!={'0','1','2','3'}:
        raise ValueError('original four assigned GPUs and DDP topology required')
    output=Path(cfg['output_dir']);output.mkdir(parents=True,exist_ok=False)
    original=copy.deepcopy(cfg);cfg['config']=str(plan['config'])
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    error=None
    try:mp.spawn(worker,args=(cfg,plan),nprocs=4,join=True)
    except mp.ProcessRaisedException as exc:
        if 'INTENTIONAL_STOP_AFTER_NATIVE_INITIAL4_CAPTURE_BEFORE_UPDATE' not in str(exc):raise
        error=str(exc)
    path=output/'initial-identity-capture.json'
    if not path.exists()or error is None or any(output.glob('runtime_batch*.jsonl')):
        raise ValueError('actual initial capture and intentional pre-update stop required')
    native.guard_plan(args.plan,plan)
    (output/'diagnostic-completed.json').write_text(json.dumps(dict(
        status='PASS_ACTUAL_NATIVE_DDP4_INITIAL_IDENTITY_DIAGNOSTIC',
        intentional_child_stop=error,actual_optimizer_updates=0,scientific_training_complete=False,
        source_model_training_configuration_unchanged=True,full_goal_complete=False),indent=2)+'\n')
    print(json.dumps(dict(status='PASS_ACTUAL_NATIVE_DDP4_INITIAL_IDENTITY_DIAGNOSTIC',
        original_identity_check_passed=json.loads(path.read_text())['original_identity_check_passed'])),flush=True)


if __name__=='__main__':main()
