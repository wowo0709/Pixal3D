"""Capture actual V1 inputs and prepare a chosen model's seed202 admission.

CPU metadata IO only. No model selection, neural execution or GPU scheduling.
New outputs must subsequently receive an actual file4 receipt; the numerical
runner refuses an unclosed admission plan.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import psutil

from .cascade_view_roles import ClosedInputs, file_sha256
from .collect_module_seed_confirmation import read_file, require, verify_terminal
from .pipeline_confirmation_recipe_v2 import validate_arm
from .selected_confirmation_runtime import (
    OBSERVER_FILES, validate_observer_values, validate_recipe,
)


def write_new(path,value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')


def first_five(path):
    """Read only complete prefix records, while the remaining log may grow."""
    with Path(path).open('rb') as stream:lines=[stream.readline() for _ in range(5)]
    require(all(line.endswith(b'\n') for line in lines),'five fully written actual V1 records required')
    raw=b''.join(lines);rows=[json.loads(line) for line in lines]
    require([row['step'] for row in rows]==[1,2,3,4,5], 'actual updates1 through5 required')
    for row in rows:
        require(set(row)=={'step','asset_ids','x_t_SHA256','t_SHA256','global_SHA256'}
            and len(row['asset_ids'])==1 and len(row['asset_ids'][0])==96,
            'original fullbatch96 asset/noise/timestep/global trace required')
    return raw


def actual_owner(wrapper,recipe):
    active=json.loads((wrapper/'active.json').read_text())
    command=active['command']
    require('--config' in command and command[command.index('--config')+1]==str(recipe),
        'actual reference process must use the declared closed recipe')
    terminal=wrapper/'terminal.json'
    if terminal.exists():
        record=json.loads(terminal.read_text())
        require(record['PID']==active['PID'] and record['birth']==active['birth'],
            'actual terminal must belong to the same launched reference process')
        return dict(kind='completed_actual_reference',**verify_terminal(record))
    process=psutil.Process(active['PID'])
    require(process.create_time()==active['birth'] and process.cmdline()==command
        and process.status()!=psutil.STATUS_ZOMBIE,'actual reference child must be confirmed live now')
    return dict(kind='live_actual_reference_observed',PID=process.pid,birth=process.create_time(),
        command=command,full_training_completion_not_claimed=True)


def capture_reference_prefix(plan,check):
    require(plan.get('schema')=='actual-V1-seed202-prefix-capture-inputs-v1',
        'explicit actual seed202 V1 prefix-capture inputs required')
    cfg=read_file(plan['reference_recipe'],check)
    validate_arm(cfg,'V1','science')
    require(type(cfg['training_seed']) is int and cfg['training_seed']==202,
        'actual native V1 training seed202 required')
    native=Path(cfg['output_dir']);wrapper=Path(plan['reference_wrapper'])
    owner=actual_owner(wrapper,Path(plan['reference_recipe']['path']))
    paths=[native/'config.json',native/'startup.json']
    before={str(p):file_sha256(p) for p in paths}
    actual=json.loads(paths[0].read_text());startup=json.loads(paths[1].read_text())
    require(json.dumps(actual,sort_keys=True,allow_nan=False)==json.dumps(cfg,sort_keys=True,allow_nan=False),
        'actual saved native config must exactly equal the closed seed202 V1 recipe')
    require(startup.get('status')=='PASS_ACTUAL_COLD_NATIVE_TASK_SS_STARTUP'
        and startup.get('variant')=='vanilla' and startup.get('source_tensors')==0
        and startup.get('LoRA_tensors')==360
        and (startup.get('global_batch'),startup.get('microbatch'),startup.get('accumulation'),
            startup.get('actual_optimizer_states'))==(96,96,1,0),
        'actual original cold V1 startup required')
    trace=native/'training-noise-trace.jsonl';raw=first_five(trace)
    # The log may append later updates; exactly the captured first five must
    # remain stable. Its whole-file digest would mislabel normal appends.
    require(first_five(trace)==raw and before=={str(p):file_sha256(p) for p in paths},
        'actual reference prefix/config/startup changed during capture')
    output=Path(plan['output']);output.relative_to(check.root);output.mkdir(parents=True,exist_ok=False)
    for p in paths:
        data=p.read_bytes()
        require(file_sha256(p)==before[str(p)],'actual reference input changed before copy')
        (output/p.name).write_bytes(data)
        require(file_sha256(output/p.name)==before[str(p)],'reference snapshot bytes differ')
    (output/'training-noise-trace.jsonl').write_bytes(raw)
    write_new(output/'audit.json',dict(status='PASS_ACTUAL_V1_SEED202_FIVE_UPDATE_PREFIX_CAPTURE',
        actual_reference_owner=owner,training_seed=202,steps=[1,2,3,4,5],
        original_native_root=str(native),original_trace=str(trace),
        original_config_startup_SHA256=before,verified_closed_inputs=check.inputs,
        scientific_weights_copied=False,full_training_completion_claimed=False,
        candidate_selected=False,quality_gain_proven=False,actual_file4_closure_still_required=True,
        full_goal_complete=False))
    return output


def prepare_admission(plan,check):
    require(plan.get('schema')=='actual-selected-SS202-admission-inputs-v1',
        'actual selected-model admission inputs required')
    arm=plan['module_arm']
    original=read_file(plan['original_science_config'],check)
    cfg=read_file(plan['own_science_config'],check)
    validate_recipe(original,cfg,arm=arm,phase='science')
    values={key:read_file(plan[key],check,lines=key.endswith('_trace')) for key in OBSERVER_FILES}
    admission=dict(schema='actual-selected-SS202-paired-prefix-admission-v1',module_arm=arm,
        training_seed=202,canonical_local_root=str(check.root),
        **{key:deepcopy(plan[key]) for key in OBSERVER_FILES})
    validate_observer_values(admission,values,cfg,arm)
    output=Path(plan['output']);output.relative_to(check.root);output.mkdir(parents=True,exist_ok=False)
    write_new(output/'observer-plan.json',admission)
    write_new(output/'audit.json',dict(status='PASS_ACTUAL_OWN_SEED202_ENGINEERING_AND_V1_PREFIX_ADMISSION',
        module_arm=arm,training_seed=202,all_three_actual_capacity_inputs_paired=True,
        own_original_proof_capacity_weights_equal=True,own_initial32_matches_original_native=True,
        actual_reference_five_training_inputs_captured=True,verified_inputs=check.inputs,
        actual_selected_science_still_required=True,candidate_selected_by_this_tool=False,
        new_training_queued=False,heldout_quality_read=False,actual_file4_closure_still_required=True,
        full_goal_complete=False))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--mode',choices=['reference-prefix','admission'],required=True)
    args=parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES','')=='','CPU-only metadata preparation required')
    plan=json.loads(args.plan.read_text());check=ClosedInputs(plan['canonical_local_root'])
    check.check(args.plan,args.plan.parent/'file4-closure.json')
    for path,digest in plan['source_SHA256'].items():require(file_sha256(path)==digest,'pinned preparation source changed')
    output=(capture_reference_prefix if args.mode=='reference-prefix' else prepare_admission)(plan,check)
    print(json.dumps(dict(output=str(output),actual_file4_closure_still_required=True,full_goal_complete=False)))


if __name__=='__main__':main()
