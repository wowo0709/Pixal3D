"""CPU-only retirement of three canonically archived completed DEV duplicates.

Case generation, rendering, metrics and their jobs are never modified or
rerun here. Future case paths are inert until the real case and job finish.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

from .cascade_view_roles import ClosedInputs,file_sha256 as sha
from .collect_module_seed_confirmation import read_file,verify_terminal
from .dev_cascade_cache_retirement import ARMS,candidates,validate_candidate,validate_completed_case
from .retire_selected_final_raw import identity,reader_snapshot,remote_hashes,required_sources as prior_sources
from .selected_pipeline_model_contract import require


SCHEMA='completed-final-Fixed-DEV-local-duplicates-v1'


def required_sources():
    root=Path(__file__).resolve().parent
    return sorted(set(prior_sources()+[str(root/(s+'.py')) for s in [
        'retire_dev_cascade_cache','dev_cascade_cache_retirement',
        'final_fixed_ablation_cascade','shared_ablation_pipeline_contract',
        'fixed_correspondence_pipeline_contract']]))


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def close(folder,context,label):
    folder=Path(folder);base=Path(context['canonical_local_root'])
    manifest=Path(context['manifest_directory'])/(context['name']+'-'+label+'-files.json')
    write(manifest,[str(p.relative_to(base)) for p in sorted(folder.rglob('*'))
        if p.is_file() and not p.is_symlink() and p!=folder/'file4-closure.json'])
    subprocess.run([*context['closure_command'],'--manifest',str(manifest),
        '--receipt',str((folder/'file4-closure.json').relative_to(base)),
        '--scope','Completed DEV case three local raw duplicates only; canonical originals, geometry predictions, guides, scored pixels and checkpoints retained'],check=True)


def validate_context(plan,policy):
    require(plan.get('schema')==policy.get('schema')==SCHEMA,'explicit DEV cache policy required')
    require(policy['repo']==str(Path(__file__).resolve().parents[2]),'this compiled checkout required')
    base,owned,outputs=(Path(policy[k]) for k in ['canonical_local_root','owned_outputs_root','output_root'])
    require(all(p.is_absolute() and p.resolve()==p for p in [base,owned,outputs])
        and owned!=base and owned.is_relative_to(base) and outputs.is_relative_to(base)
        and not outputs.is_relative_to(owned) and not owned.is_relative_to(outputs),
        'separate canonical owned cases and retirement journals required')
    arm=plan['arm'];require(arm in ARMS,'one of the five explicit controls required')
    token=plan['case_token'];require(len(token)==16 and all(c in '0123456789abcdef' for c in token),'exact case token required')
    job=policy['case_job_prefixes'][arm]+'-'+token
    require(plan['case_job']==job and plan['name']=='retire-'+job+'-cache-CPU'
        and Path(plan['case_root'])==owned/arm/'cases'/token
        and Path(plan['case_plan'])==Path(policy['case_plan_roots'][arm])/(job+'-case.json')
        and Path(plan['output'])==outputs/arm/token,
        'exact own case/plan/job/journal identity required')
    require(set(required_sources())|set(policy['case_workers'].values())<=set(policy['source_SHA256'])
        and policy['closure_command'][0]==policy['python']
        and policy['closure_command'][1] in policy['source_SHA256'],
        'pin the complete cleanup and actual closure sources')
    require(isinstance(policy['CPU_affinity'],list) and policy['CPU_affinity']
        and all(type(c) is int and c>=0 for c in policy['CPU_affinity'])
        and Path(policy['reader_lock']).is_absolute()
        and Path(policy['manifest_directory']).is_absolute(), 'explicit metadata resources required')
    require(policy['reader_queues'] and all(set(q)=={'jobs','states'}
        and all(Path(p).is_absolute() for p in q.values()) for q in policy['reader_queues']),
        'both actual job and state ledgers required')
    require(policy['ssh_argv'][0]=='ssh' and Path(policy['canonical_remote_root']).is_absolute(),
        'explicit canonical remote connection required')


def completed_job(plan,policy,*,wait_seconds=30):
    """A closure can precede the scheduler's terminal write by a few seconds."""
    deadline=time.monotonic()+wait_seconds
    while True:
        found=[]
        for queue in policy['reader_queues']:
            jobs=json.loads(Path(queue['jobs']).read_text());states=json.loads(Path(queue['states']).read_text())
            found.extend((j,states.get(j['name'],{})) for j in jobs if j['name']==plan['case_job'])
        require(len(found)==1,'one actual original case job required')
        job,state=found[0];command=job['command']
        require(job['resource']=='exclusive' and '--plan' in command
            and command[command.index('--plan')+1]==plan['case_plan']
            and command[2]==policy['case_workers'][plan['arm']], 'unchanged original case worker required')
        if state.get('status')=='completed':
            require(state.get('returncode')==0,'original case must finish successfully')
            return dict(job=job,state=state)
        require(state.get('status')=='running' and time.monotonic()<deadline,
            'preserve raw outputs until actual original case-job completion')
        time.sleep(1)


def execute(path):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','explicit CPU-only execution required')
    path=Path(path);plan=json.loads(path.read_text());policy=json.loads(Path(plan['policy']['path']).read_text())
    validate_context(plan,policy);os.sched_setaffinity(0,set(policy['CPU_affinity']));os.nice(10)
    check=ClosedInputs(policy['canonical_local_root']);check.check(path,path.parent/'file4-closure.json')
    read_file(plan['policy'],check);context=dict(policy,name=plan['name'])
    def guard(removed=()):
        for p,d in policy['source_SHA256'].items():require(sha(p)==d,'cleanup source changed')
        for p,d in check.inputs.items():
            if p not in removed:require(sha(p)==d,'closed cleanup input changed')
    guard();job=completed_job(plan,policy);root=Path(plan['case_root']);base=Path(policy['canonical_local_root'])
    def read(p):
        p=Path(p);check.check(p,p.parent/'file4-closure.json');return json.loads(p.read_text())
    cp=read(plan['case_plan']);case=read(root/'audit.json')
    require(cp['name']==plan['case_job'] and cp['SS_arm']==case['SS_arm']==plan['arm']
        and cp['sample_id']==case['sample_id']==plan['sample_id']
        and cp['output']==str(root)
        and cp['SS_model_binding']==case['actual_SS_model_binding']
        and cp['PBR_model_binding']==case['actual_PBR_model_binding'],
        'actual unchanged case metadata and physical models required')
    geometry=read(root/'geometry/generation.json');replay=read(root/'PBR/generation.json')
    validate_completed_case(case_root=root,owned_root=policy['owned_outputs_root'],case=case,geometry=geometry,replay=replay)
    for terminal in case['actual_owned_terminals']:verify_terminal(terminal)
    shared=read(root/'shared/render-evaluation.json')
    require(shared['same_reference_pixels_for_every_model'] and shared['sample_id']==plan['sample_id']
        and {r['camera_id'] for r in shared['per_view']}=={8,9,10}, 'three actually scored camera views required')
    # Retain and verify all late metric inputs before considering any raw removal.
    for p in [root/'geometry/prediction.npz',root/'geometry/native-shape-guides.pt',
              *[root/'shared'/f'camera{c:02d}.npz' for c in [8,9,10]]]:
        check.check(p,p.parent/'file4-closure.json')
    output=Path(plan['output']);require(not output.exists(),'preserve prior cleanup intents and partial journals')
    with Path(policy['reader_lock']).open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        readers=[];queues={}
        for folder in [root/'geometry',root/'PBR']:
            own,snapshot=reader_snapshot(context,folder);readers.extend(own)
            require(not queues or queues==snapshot,'job descriptors changed during reader inspection');queues=snapshot
        tokens=[];identities={}
        for p in candidates(root):
            before=identity(p);digest=check.check(p,p.parent/'file4-closure.json')
            require(identity(p)==before,'raw file changed during hashing')
            receipt=json.loads((p.parent/'file4-closure.json').read_text())
            entry=next(r for r in receipt['files'] if r['relative_path']==str(p.relative_to(base)))
            tokens.append(validate_candidate(p,case_root=root,owned_root=policy['owned_outputs_root'],
                case=case,geometry=geometry,replay=replay,canonical_entry=entry,actual_local_SHA256=digest,readers=readers))
            identities[str(p)]=before
        rows=[dict(relative_path=str(Path(t['path']).relative_to(base)),sha256=t['sha256'],bytes=t['bytes']) for t in tokens]
        actual=remote_hashes(context,rows)
        require(set(actual)=={r['relative_path'] for r in rows}
            and all(actual[r['relative_path']]=={k:r[k] for k in ['sha256','bytes']} for r in rows),
            'fresh remote SHA256/size must equal all three local archives')
        output.mkdir(parents=True,exist_ok=False);intent=output/'before-retirement';intent.mkdir()
        write(intent/'audit.json',dict(status='PASS_FRESH_CANONICAL_DEV_RAW_BEFORE_LOCAL_RETIREMENT',
            selected=tokens,actual_remote=actual,actual_completed_original_job=job,
            local_inode_identities=identities,reader_queue_SHA256=queues,verified_inputs=check.inputs,full_goal_complete=False))
        close(intent,context,'before');guard()
        require(all(sha(p)==d for p,d in queues.items()),'a new reader requires a new scan')
        for folder in [root/'geometry',root/'PBR']:
            readers,latest=reader_snapshot(context,folder)
            require(not readers and latest==queues,'a reader appeared after canonical verification')
        require(all(identity(t['path'])==identities[t['path']] for t in tokens),'raw inode changed before removal')
        removed=[]
        for token in tokens:
            target=Path(token['path']);require(identity(target)==identities[str(target)],'raw inode changed')
            target.unlink();removed.append(str(target))
            with (output/'retired-local-cache.jsonl').open('a') as stream:
                stream.write(json.dumps(token)+'\n');stream.flush();os.fsync(stream.fileno())
        guard(removed)
        result=dict(status='PASS_COMPLETED_DEV_THREE_LOCAL_DUPLICATES_RETIRED',retired=tokens,
            bytes_retired=sum(t['bytes'] for t in tokens),canonical_file4_retained=True,
            geometry_predictions_native_guides_scored_pixels_metadata_checkpoints_retained=True,
            no_NN_training_or_case_rerun=True,full_goal_complete=False)
        write(output/'retirement.json',result);close(output,context,'result')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(execute(args.plan)),flush=True)


if __name__=='__main__':main()
