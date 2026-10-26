"""Offload only canonical guides of immutable, fully consumed reused groups.

This separate metadata lifecycle keeps canonical NAS guides and all scored
pixels. It does not modify a live campaign, its context, budget or scientific
source, and never selects current/partial groups or data/model checkpoints.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

from .cascade_view_roles import ClosedInputs, file_sha256 as sha
from .collect_module_seed_confirmation import read_file, verify_terminal
from .operational_dev_geometry_retirement import actual_queue_jobs, write
from .operational_process_access import process_access_waits
from .operational_selected_case_reuse import load_manifest
from .retire_selected_final_raw import identity, remote_hashes
from .run_selected_final_campaign import load_population
from .selected_final_case_integrity import CASE_STATUS
from .selected_final_metric_inputs import read_pixels
from .selected_pipeline_freeze_v2 import exact
from .selected_pipeline_model_contract import require


SCHEMA = 'immutable-reused-TEST-guide-local-cache-retirement-v1'
TERMINAL = {'completed','failed','cancelled','canceled','terminated','skipped'}


def close(folder,policy,label):
    folder,base=Path(folder),Path(policy['canonical_local_root'])
    manifest=Path(policy['manifest_directory'])/(policy['name']+'-'+label+'-files.json')
    write(manifest,[str(p.relative_to(base)) for p in sorted(folder.rglob('*'))
        if p.is_file() and p!=folder/'file4-closure.json'])
    subprocess.run([*policy['closure_command'],'--manifest',str(manifest),'--receipt',
        str((folder/'file4-closure.json').relative_to(base)),
        '--scope','Immutable fully consumed reused TEST guides: local duplicates offloaded only after fresh NAS hashes; canonical guides and all scored pixels, audits, terminals, data and checkpoints retained'],check=True)


def pending_guide_reads(jobs, states, paths):
    targets = {str(Path(p)) for p in paths} | {str(Path(p).parent) for p in paths}
    readers = []
    def contains(value):
        if isinstance(value,str): return value in targets
        if isinstance(value,list): return any(contains(v) for v in value)
        if isinstance(value,dict): return any(contains(k) or contains(v) for k,v in value.items())
        return False
    for job in jobs:
        if states.get(job['name'],{}).get('status','pending').lower() in TERMINAL: continue
        documents = [job]; command = job.get('command',[])
        if '--plan' in command:
            p = Path(command[command.index('--plan')+1])
            require(p.is_file(),'unreadable unfinished guide reader plan: '+job['name'])
            documents.append(json.loads(p.read_text()))
        if any(contains(d) for d in documents): readers.append(job['name'])
    return sorted(set(readers))


def reader_snapshot(policy, paths):
    jobs,states,queues = actual_queue_jobs(policy)
    readers = pending_guide_reads(jobs,states,paths)
    names = {str(Path(p)) for p in paths}
    for p in psutil.process_iter():
        try:
            if any(f.path in names for f in p.open_files()): readers.append('actual open process '+str(p.pid))
        except (psutil.NoSuchProcess,psutil.ZombieProcess): pass
    return sorted(set(readers)),queues


def validate_candidate(path, *, row, manifest, group, case, terminal, canonical_entry, digest, readers):
    sid,alias = row['sample_id'],row['alias']
    require(sid in manifest['groups'] and alias in manifest['models'], 'one explicitly reused frozen group/model required')
    record = manifest['groups'][sid]
    case_root = Path(record['cases'][alias]['audit']['path']).parent
    path = Path(path)
    require(group.get('status')=='PASS_ACTUAL_COMPLETE_ASSET_GROUP_NATIVE_CASES_AND_RAW_RETIREMENT'
        and group.get('sample_id')==sid and exact(group.get('cases'),record['cases'])
        and set(group['cases'])==set(manifest['models']) and len(group['retirements'])==8,
        'all five actual case consumers and all eight raw retirements must be complete')
    require(case.get('status')==CASE_STATUS and case.get('actual_completed_native_case') is True
        and case.get('GT_not_used_in_generation') is True and case.get('optimizer_updates')==0
        and case.get('sample_id')==sid and case.get('alias')==alias
        and case.get('case_output')==str(case_root) and case.get('geometry_source_root')==str(case_root/'geometry')
        and exact(case.get('frozen_model_binding'),manifest['models'][alias])
        and terminal.get('returncode')==0 and terminal.get('actual_session_drained') is True
        and terminal.get('case_output')==str(case_root), 'own immutable fully consumed guide and native model required')
    require(path==case_root/'geometry/native-shape-guides.pt' and path.is_file() and not path.is_symlink()
        and path.resolve()==path and not readers, 'only exact unread local reused-case guides may offload')
    require(digest==case['actual_native_guide_archive_SHA256']==canonical_entry.get('sha256')
        and path.stat().st_size==canonical_entry.get('bytes'), 'actual guide bytes must match native producer and canonical NAS')
    return dict(path=str(path),sha256=digest,bytes=path.stat().st_size,
        local_duplicate_only=True,canonical_file4_retained=True,sample_id=sid,alias=alias)


def retained_case_files(guide):
    guide=Path(guide);root=guide.parent.parent
    files=[guide.parent/'generation.json',root/'audit.json',root/'shared/render-evaluation.json',
        *[root/'shared'/f'camera{cid:02d}.npz' for cid in [8,9,10]]]
    require(all(p.is_file() and not p.is_symlink() for p in files),
        'retain all actual case/generation/scoring metadata and every scored archive')
    return files


def execute(path):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU metadata-only guide lifecycle required')
    os.sched_setaffinity(0,{92,93});os.nice(10)
    path=Path(path);policy=json.loads(path.read_text())
    require(policy.get('schema')==SCHEMA and policy['repo']==str(Path(__file__).resolve().parents[2]),
        'explicit new guide lifecycle policy and actual checkout required')
    base=Path(policy['canonical_local_root']);output=Path(policy['output']);check=ClosedInputs(base)
    require(base.resolve()==base and output.is_absolute() and output.resolve()==output and output.is_relative_to(base)
        and output!=base and not output.exists()
        and str(Path(__file__).resolve()) in policy['operational_source_SHA256'],
        'fresh separate output and explicit guide lifecycle source pin required')
    check.check(path,path.parent/'file4-closure.json')
    context=read_file(policy['campaign_context'],check)
    require(len(context['source_SHA256'])==505 and context['source_SHA256']==policy['scientific_source_SHA256'],
        'same original505 numerical source required')
    def guard():
        for key in ['scientific_source_SHA256','operational_source_SHA256']:
            require(all(sha(p)==d for p,d in policy[key].items()),'pinned scientific or guide lifecycle source changed')
    guard();parent,controls,models=load_population(context,check)
    manifest=load_manifest(context,check,parent,models,verify_groups=True)
    rows=read_file(policy['case_index'],check)
    require(len(rows)==645 and len(manifest['groups'])==129 and {(r['sample_id'],r['alias']) for r in rows}
        =={(sid,a) for sid in manifest['groups'] for a in models}, 'exact original129 reused groups by all five models required')
    old=Path(manifest['original_campaign_root'])
    require(output!=old and not output.is_relative_to(old) and not old.is_relative_to(output)
        and Path(context['campaign_root'])!=old, 'only old immutable groups, never live successor files')
    require(type(policy['batch_size']) is int and 0<policy['batch_size']<=16,'bounded actual guide batches required')
    output.mkdir(parents=True,exist_ok=False);retired=[]
    def emit(event):
        with (output/'process-access-events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    with process_access_waits(emit=emit):
        for offset in range(0,len(rows),policy['batch_size']):
            batch=rows[offset:offset+policy['batch_size']];records=[]
            for row in batch:
                record=manifest['groups'][row['sample_id']];group=read_file(record['group_audit'],check)
                descriptor=record['cases'][row['alias']];case=read_file(descriptor['audit'],check)
                terminal=read_file(descriptor['worker_terminal'],check);verify_terminal(terminal)
                for phase in case['terminals']:verify_terminal(phase)
                target=Path(descriptor['audit']['path']).parent/'geometry/native-shape-guides.pt'
                retained_case_files(target)
                require(row['path']==str(target),'explicit original guide index required')
                before=identity(target);digest=check.check(target,target.parent/'file4-closure.json')
                require(identity(target)==before,'guide changed while hashing')
                receipt=json.loads((target.parent/'file4-closure.json').read_text())
                entry=next(e for e in receipt['files'] if e['relative_path']==str(target.relative_to(base)))
                token=validate_candidate(target,row=row,manifest=manifest,group=group,case=case,terminal=terminal,
                    canonical_entry=entry,digest=digest,readers=[])
                records.append((token,before))
            attempt=0
            while True:
                paths=[r[0]['path'] for r in records];readers,queues=reader_snapshot(policy,paths)
                require(not readers,'unfinished or actual readers still consume a reused guide')
                require(all(identity(t['path'])==before for t,before in records),'guide inode changed before canonical verification')
                remote_rows=[dict(relative_path=str(Path(t['path']).relative_to(base)),sha256=t['sha256'],bytes=t['bytes']) for t,_ in records]
                actual=remote_hashes(policy,remote_rows)
                require(set(actual)=={r['relative_path'] for r in remote_rows}
                    and all(actual[r['relative_path']]=={k:r[k] for k in ['sha256','bytes']} for r in remote_rows),
                    'fresh real NAS guide hashes and sizes required')
                folder=output/f'batch-{offset:04d}-attempt-{attempt:04d}';folder.mkdir()
                intent=folder/'before-retirement';intent.mkdir()
                write(intent/'audit.json',dict(status='PASS_FRESH_CANONICAL_IMMUTABLE_REUSED_GUIDES',
                    selected=[t for t,_ in records],actual_remote=actual,reader_queue_SHA256=queues,
                    original_case_and_pixel_artifacts_retained=True,full_goal_complete=False))
                close(intent,policy,f'batch-{offset:04d}-attempt-{attempt:04d}-intent')
                guard();readers2,queues2=reader_snapshot(policy,paths)
                require(not readers2,'a new actual guide reader prevents local offload')
                if queues2!=queues:
                    write(folder/'audit.json',dict(status='DEFERRED_READER_DESCRIPTOR_CHANGE_NO_GUIDE_UNLINKED',
                        old_queues=queues,new_queues=queues2,actual_deletions=0,full_goal_complete=False))
                    close(folder,policy,f'batch-{offset:04d}-attempt-{attempt:04d}-deferred')
                    attempt+=1;time.sleep(1);continue
                require(all(identity(t['path'])==before for t,before in records),'guide inode changed immediately before unlink')
                for token,before in records:
                    target=Path(token['path']);require(identity(target)==before,'guide inode changed before own unlink')
                    target.unlink()
                    with (folder/'retired-local-cache.jsonl').open('a') as f:
                        f.write(json.dumps(token)+'\n');f.flush();os.fsync(f.fileno())
                write(folder/'audit.json',dict(status='PASS_IMMUTABLE_REUSED_GUIDE_LOCAL_DUPLICATES_OFFLOADED',
                    retired=[t for t,_ in records],canonical_NAS_guides_retained=True,
                    scored_pixels_and_all_native_case_audits_retained=True,full_goal_complete=False))
                close(folder,policy,f'batch-{offset:04d}-attempt-{attempt:04d}-result')
                retired.extend(t for t,_ in records)
                print(json.dumps(dict(retired=len(retired),total=645,GiB_retired=sum(t['bytes'] for t in retired)/2**30)),flush=True)
                break
    # Prove the frozen actual scored-pixel consumer still sees every original
    # producer observation after all local guides are absent. No metric NN or
    # numerical metric value is computed or substituted here.
    verified_pixels=0
    for row in rows:
        descriptor=manifest['groups'][row['sample_id']]['cases'][row['alias']]
        case=read_file(descriptor['audit'],check)
        require(not Path(row['path']).exists(),'an offloaded local guide reappeared')
        for cid,expected in zip([8,9,10],case['scored_image_observations']):
            pixel=read_pixels(Path(case['case_output'])/'shared'/f'camera{cid:02d}.npz',alias=row['alias'],
                sid=row['sample_id'],cluster=case['source_cluster_id'],check=check)
            require(exact(pixel['pixel_observation'],expected),'original scored RGB/GT/alpha bytes changed after guide offload')
            verified_pixels+=1
    require(verified_pixels==1935,'all original1935 scored archives must remain usable')
    guard();result=dict(status='PASS_ACTUAL645_IMMUTABLE_REUSED_GUIDE_LOCAL_DUPLICATES_OFFLOADED',
        files=len(retired),bytes_retired=sum(t['bytes'] for t in retired),actual_fresh_NAS_hash_verified=True,
        actual_original_scored_archives_exact_with_guides_absent=verified_pixels,
        scored_pixels_native_case_audits_and_terminals_retained=True,canonical_NAS_guides_retained=True,
        original505_unchanged=True,live_successor_context_and_workers_not_modified=True,CPU_NN_forward_or_training=0,
        full_goal_complete=False)
    write(output/'audit.json',result);close(output,policy,'complete');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True,type=Path)
    print(json.dumps(execute(p.parse_args().plan)),flush=True)


if __name__=='__main__':main()
