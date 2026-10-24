"""Prepare and install a selected508 campaign only from actual frozen inputs.

Both existing queues receive source-bound descriptors before the one common
activation receipt is published. A partial append cannot start either keeper
or the coordinator. This tool does not select a model or execute a NN.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import psutil

from .cascade_view_roles import ClosedInputs,file_sha256 as sha
from .native_run_disk_budget import DiskBudget
from . import run_selected_final_campaign as campaign
from .selected_campaign_resources import live,source_guard
from .selected_pipeline_model_contract import require


SCHEMA='selected-physical-final-campaign-preparation-v1'


def descriptors(context,path,activation_receipt):
    """Every cross-dispatcher job waits for the same final activation receipt."""
    root=Path(context['campaign_root']);required=[str(Path(path).parent/'file4-closure.json'),str(activation_receipt)]
    jobs={}
    for name,group in context['keeper_groups'].items():
        output=root/'ownership'/name
        jobs[group['job_name']]=dict(name=group['job_name'],resource='exclusive',priority=-100,gpus=group['assigned_GPUs'],
            requires=required,outputs=[str(output/'terminal.json'),str(output/'file4-closure.json')],
            command=campaign.command(context,path,name))
    group=context['coordinator_queue'];jobs[group['job_name']]=dict(name=group['job_name'],resource='cpu',priority=-100,
        requires=required,outputs=[str(root/'result/file4-closure.json'),str(root/'release-closure/file4-closure.json')],
        command=campaign.command(context,path))
    require(len(jobs)==3,'distinct two keepers and one coordinator required')
    return jobs


def append_queues(context,jobs):
    """Append actual jobs, never modify dispatcher state or prior descriptors."""
    targets={}
    for group in [*context['keeper_groups'].values(),context['coordinator_queue']]:
        targets.setdefault(group['jobs'],[]).append(jobs[group['job_name']])
    prepared={}
    for path,new in targets.items():
        raw=Path(path).read_bytes();old=json.loads(raw);names=[j['name'] for j in old]
        require(len(names)==len(set(names)) and not (set(names)&{j['name'] for j in new}), 'preserve all old jobs and refuse duplicate campaign names')
        prepared[path]=(raw,old,new)
    for path,(before,old,new) in prepared.items():
        require(Path(path).read_bytes()==before,'queue changed during preparation; no activation is authorized')
        campaign.atomic(path,old+new)
    for path,(_,old,new) in prepared.items():
        actual=json.loads(Path(path).read_text());expected={j['name']:j for j in [*old,*new]}
        require(len(actual)==len({j['name'] for j in actual})
            and all(expected[n]==j for j in actual if (n:=j['name']) in expected)
            and set(expected)<={j['name'] for j in actual},'both actual queue descriptors must retain every prior and new job')
    return {path:dict(before_SHA256=hashlib.sha256(before).hexdigest(),after_SHA256=sha(path),
        actual_added_names=[j['name'] for j in new],old_jobs=len(old)) for path,(before,old,new) in prepared.items()}


def prepare(path):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','explicit CPU-only final campaign preparation required')
    path=Path(path);request=json.loads(path.read_text())
    require(request.get('schema')==SCHEMA,'explicit closed selected campaign preparation request required')
    context=request['campaign'];campaign.validate(context);source_guard(context)
    os.sched_setaffinity(0,set(context['CPU_affinity']));os.nice(10)
    check=ClosedInputs(context['canonical_local_root']);check.check(path,path.parent/'file4-closure.json')
    parent,controls,models=campaign.load_population(context,check)
    base=Path(context['canonical_local_root']);output=Path(request['prepared_root']);root=Path(context['campaign_root'])
    require(output.is_absolute() and output!=base and output.is_relative_to(base) and output.resolve()==output
        and not output.exists() and not root.exists() and not output.is_relative_to(root) and not root.is_relative_to(output),
        'new separate prepared and runtime output roots required; no overwrite or automatic restart')
    for group in [*context['keeper_groups'].values(),context['coordinator_queue']]:
        require(live(group['dispatcher']),'actual existing dispatcher required')
        proc=psutil.Process(group['dispatcher']['PID']);cmd=proc.cmdline()
        require('experiments.deformnext_retargeting.matching_v2_queue' in cmd and '--work-root' in cmd
            and str(Path(cmd[cmd.index('--work-root')+1])/'queue-jobs.json')==group['jobs']
            and str(Path(group['jobs']).parent/'queue-state.json')==group['states']
            and set(group['assigned_GPUs'])<=set(cmd[cmd.index('--gpus')+1].split(','))
            and proc.cwd()==context['repo'], 'actual dispatcher command, checkout, ledgers and assigned GPU scope required')
    allowance=2*sum(context['case_budget_bytes'].values())
    admission=DiskBudget(context['disk_reservations'],workspace_bytes=context['workspace_bytes']).inspect(allowance)
    require(admission['admitted'],'actual disk must admit at least one complete asset group before reserving GPUs')
    output.mkdir(parents=True,exist_ok=False);pre=output/'prelaunch-v1';pre.mkdir()
    context_path=pre/'context.json';campaign.write(context_path,context);campaign.close(pre,context,'campaign-context')
    activation=output/'activation-v1';jobs=descriptors(context,context_path,activation/'file4-closure.json')
    report=dict(status='PASS_ACTUAL_FROZEN_SELECTED_CAMPAIGN_PREPARED_NOT_YET_ACTIVATED',
        actual_models=list(models),assets=len(parent['population']['sample_ids']),actual_whole_asset_disk_admission=admission,
        actual_context=campaign.closed_spec(context_path),jobs=jobs,actual_GPU_forward=False,full_goal_complete=False)
    report_folder=output/'preparation-v1';report_folder.mkdir();campaign.write(report_folder/'audit.json',report)
    campaign.close(report_folder,context,'campaign-preparation')
    try:
        source_guard(context);queues=append_queues(context,jobs)
        activation.mkdir()
        campaign.write(activation/'audit.json',dict(status='PASS_ACTUAL_BOTH_DISPATCHER_QUEUES_INSTALLED_FOR_FROZEN_CAMPAIGN',
            actual_queues=queues,actual_context=campaign.closed_spec(context_path),models=list(models),assets=508,
            actual_GPU_forward_performed_by_preparer=False,TEST_quality_used_for_selection=False,verified_inputs=check.inputs,full_goal_complete=False))
        campaign.close(activation,context,'campaign-activation')
    except BaseException as exc:
        campaign.write(output/'failure.json',dict(status='FAILED_CAMPAIGN_PREPARATION_PRESERVED',error=repr(exc),automatic_retry=False,
            activation_receipt_created=(activation/'file4-closure.json').exists()))
        raise
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.plan)),flush=True)


if __name__=='__main__':main()
