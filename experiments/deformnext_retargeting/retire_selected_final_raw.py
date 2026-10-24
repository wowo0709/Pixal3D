"""Retire verified selected-case local duplicates after all real consumers.

The remote data, guides, scores, images and checkpoints remain untouched.
Canonical remote hashes are reread immediately before a closed intent, then
local inode/readers/queue descriptors are checked again before unlinking.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

import numpy as np
import psutil

from .cascade_view_roles import ClosedInputs,file_sha256 as sha
from .collect_module_seed_confirmation import read_file,verify_terminal
from .generated_artifact_cache_retirement import RAW_NAMES,pending_raw_reads
from .metric_image_identity import scored_image_observation
from .native_run_disk_budget import DiskBudget
from .selected_final_raw_retirement import required_consumers,validate_raw
from .selected_pipeline_freeze_v2 import exact
from .selected_pipeline_model_contract import require


SCHEMA = 'selected-physical-final-raw-retirement-v1'


def required_sources():
    root = Path(__file__).resolve().parent
    return [str(root/(name+'.py')) for name in ['retire_selected_final_raw','selected_final_raw_retirement',
        'selected_final_population','selected_final_case_integrity','selected_pipeline_model_contract',
        'selected_pipeline_freeze_v2','generated_artifact_cache_retirement','native_run_disk_budget',
        'checkpoint_cache_retirement','collect_module_seed_confirmation','cascade_view_roles',
        'metric_image_identity','extract_official_inception','selected_pipeline_joint_summary_v2',
        'original_zero_population','pbr_replay_artifacts','native_final_freeze']]


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def identity(path):
    value = Path(path).stat()
    return value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns


def validate_context(context):
    require(context.get('schema') == SCHEMA,'explicit selected-case raw retirement context required')
    require(context.get('repo') == str(Path(__file__).resolve().parents[2]), 'use this actual compiled source checkout')
    root,owned,output = (Path(context[k]) for k in ['canonical_local_root','owned_outputs_root','output'])
    require(all(p.is_absolute() and p.resolve() == p for p in [root,owned,output])
        and owned != root and owned.is_relative_to(root) and output.is_relative_to(root)
        and not output.is_relative_to(owned) and not owned.is_relative_to(output),
        'separate owned case namespace and immutable retirement output below the canonical root required')
    require(context['role'] in ('geometry','PBR') and set(required_sources()) <= set(context['source_SHA256']),
            'pin the complete actual raw retirement contract')
    cpus = context['CPU_affinity']
    require(isinstance(cpus,list) and cpus and len(cpus) == len(set(cpus))
        and all(type(c) is int and c >= 0 for c in cpus), 'explicit metadata CPU assignment required')
    closure = context['closure_command']
    require(isinstance(closure,list) and len(closure) == 2 and closure[0] == context['python']
        and Path(closure[1]).is_absolute() and closure[1] in context['source_SHA256'], 'actual pinned file4 closure helper required')
    require(isinstance(context['python'],str) and Path(context['python']).is_absolute()
        and all(Path(context[k]).is_absolute() for k in ['manifest_directory','disk_reservations'])
        and type(context.get('workspace_bytes')) is int and context['workspace_bytes'] > 0,
        'explicit absolute runtime and disk reservation paths required')
    require(isinstance(context['ssh_argv'],list) and context['ssh_argv'] and context['ssh_argv'][0] == 'ssh'
        and all(isinstance(x,str) and x for x in context['ssh_argv'])
        and Path(context['canonical_remote_root']).is_absolute(), 'explicit canonical remote store connection required')
    require(isinstance(context['reader_queues'],list) and context['reader_queues']
        and all(set(q) == {'jobs','states'} and all(Path(p).is_absolute() for p in q.values())
            for q in context['reader_queues']), 'explicit current reader ledgers required')


def reader_snapshot(context,folder):
    readers = []; snapshots = {}
    for queue in context['reader_queues']:
        payload = Path(queue['jobs']).read_bytes(); jobs = json.loads(payload)
        require(len({j['name'] for j in jobs}) == len(jobs),'unique actual reader jobs required')
        states = json.loads(Path(queue['states']).read_text())
        readers.extend(pending_raw_reads([j for j in jobs if j['name'] != context['name']],states,folder))
        snapshots[queue['jobs']] = sha(queue['jobs'])
        require(Path(queue['jobs']).read_bytes() == payload,'reader plans changed during inspection')
    files = {str(Path(folder)/name) for name in RAW_NAMES}
    for process in psutil.process_iter():
        try:
            if any(f.path in files for f in process.open_files()):readers.append('actual open process '+str(process.pid))
        except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
        # AccessDenied is deliberately not ignored: an unknown reader cannot
        # authorize deletion. No process is interrupted to make the check pass.
    return sorted(set(readers)),snapshots


def remote_hashes(context,rows):
    source = '''import hashlib,json,pathlib,sys,os
os.nice(10)
root=pathlib.Path(sys.argv[1]);result={}
for row in json.load(sys.stdin):
 p=root/row['relative_path'];before=p.stat();h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(8388608),b''):h.update(block)
 after=p.stat();assert(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 result[row['relative_path']]=dict(sha256=h.hexdigest(),bytes=after.st_size)
print(json.dumps(result))
'''
    command = shlex.join(['python3','-c',source,context['canonical_remote_root']])
    result = subprocess.run([*context['ssh_argv'],command],input=json.dumps(rows),text=True,capture_output=True,check=True)
    return json.loads(result.stdout)


def close(folder,context,label):
    folder = Path(folder); base = Path(context['canonical_local_root'])
    manifest = Path(context['manifest_directory'])/(context['name']+'-'+label+'-files.json')
    write(manifest,[str(p.relative_to(base)) for p in sorted(folder.rglob('*'))
        if p.is_file() and p != folder/'file4-closure.json'])
    subprocess.run([*context['closure_command'],'--manifest',str(manifest),'--receipt',str((folder/'file4-closure.json').relative_to(base)),
        '--scope','Actual selected-case local raw duplicate retirement only; canonical file4 arrays, guides, reports, scored pixels and checkpoints retained'],check=True)


def load_inputs(context,check):
    inputs = context['source_inputs']; parent = read_file(inputs['parent_freeze'],check); controls = read_file(inputs['control_freeze'],check)
    names = required_consumers(context['alias'],context['role'],parent=parent,controls=controls)
    require(set(inputs['case_audits']) == set(inputs['worker_terminals']) == set(names), 'all frozen raw consumers and their actual workers required')
    cases = {name:read_file(inputs['case_audits'][name],check) for name in names}
    workers = {name:read_file(inputs['worker_terminals'][name],check) for name in names}
    for name,case in cases.items():
        require(case.get('sample_id') in parent['population']['sample_ids']
            and case.get('source_cluster_id') == parent['population']['source_clusters_by_sample'][case['sample_id']]
            and case.get('alias') == name and workers[name].get('case_output') == case['case_output']
            and workers[name].get('actual_session_drained') is True, 'actual finished case worker and own output required')
        verify_terminal(workers[name])
        for terminal in case['terminals']:verify_terminal(terminal)
        observed = []
        for cid in [8,9,10]:
            path = Path(case['case_output'])/'shared'/f'camera{cid:02d}.npz'
            digest = check.check(path,path.parent/'file4-closure.json')
            with np.load(path,allow_pickle=False) as archive:
                pixels = {k:archive[k].copy() for k in ['prediction','GT_shared_renderer','GT_alpha']}
            observed.append(scored_image_observation(pixels,sample_id=case['sample_id'],camera_id=cid,
                source_cluster_id=case['source_cluster_id'],archive=str(path),archive_SHA256=digest))
        require(exact(observed,case['scored_image_observations']), 'actual scored RGB/reference bytes differ from the completed case')
    case = cases[context['alias']]; root = Path(case['case_output'])
    require(root.is_absolute() and root.is_relative_to(Path(context['owned_outputs_root'])), 'one own completed native case required')
    for p,d in context['source_SHA256'].items():
        if p != context['closure_command'][1]:
            require(parent['evaluation_protocol']['source_SHA256'].get(p) == d,'DEV freeze did not pin this actual retirement source')
    reservation = json.loads(Path(context['disk_reservations']).read_text())['runs'][context['disk_reservation_name']]
    require(reservation.get('released') is False and reservation['native_output'] == reservation['wrapper_output'] == str(root)
        and reservation['owner']['PID'] == workers[context['alias']]['PID']
        and reservation['owner']['birth'] == workers[context['alias']]['birth'], 'actual own case disk reservation and completed owner required')
    folder = root/context['role']; check.check(folder/'generation.json',folder/'file4-closure.json')
    generation = json.loads((folder/'generation.json').read_text())
    receipt = json.loads((folder/'file4-closure.json').read_text())
    entries = {r['relative_path']:r for r in receipt['files']}
    readers,queues = reader_snapshot(context,folder); tokens = []; identities = {}
    for name in sorted(RAW_NAMES):
        path = folder/name; digest = check.check(path,folder/'file4-closure.json')
        tokens.append(validate_raw(path,case_root=root,owned_root=context['owned_outputs_root'],case=case,generation=generation,
            consumers=cases,parent=parent,controls=controls,parent_freeze={k:inputs['parent_freeze'][k] for k in ['path','SHA256']},
            control_freeze={k:inputs['control_freeze'][k] for k in ['path','SHA256']},
            canonical_entry=entries[str(path.relative_to(check.root))],actual_SHA256=digest,readers=readers))
        identities[str(path)] = identity(path)
    return root,folder,tokens,identities,queues


def execute(path):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'explicit CPU-only raw retirement required')
    path = Path(path); context = json.loads(path.read_text()); validate_context(context)
    os.sched_setaffinity(0,set(context['CPU_affinity'])); check = ClosedInputs(context['canonical_local_root'])
    check.check(path,path.parent/'file4-closure.json')
    def guard(removed=()):
        check.check(path,path.parent/'file4-closure.json')
        for p,d in context['source_SHA256'].items():require(sha(p) == d,'actual retirement source changed')
        for p,d in check.inputs.items():
            if p not in removed:require(sha(p) == d,'actual closed retirement input changed')
    guard(); root,folder,tokens,identities,queues = load_inputs(context,check)
    output = Path(context['output']);require(not output.exists(),'preserve existing retirement intents and journals')
    rows = [dict(relative_path=str(Path(t['path']).relative_to(check.root)),sha256=t['sha256'],bytes=t['bytes']) for t in tokens]
    actual = remote_hashes(context,rows)
    require(set(actual) == {r['relative_path'] for r in rows}
        and all(actual[r['relative_path']] == {k:r[k] for k in ['sha256','bytes']} for r in rows),
        'fresh canonical remote SHA256/size must match every actual local raw byte')
    output.mkdir(parents=True,exist_ok=False); intent = output/'before-retirement';intent.mkdir()
    write(intent/'audit.json',dict(status='PASS_FRESH_CANONICAL_SELECTED_RAW_BEFORE_LOCAL_RETIREMENT',
        selected=tokens,actual_remote=actual,local_inode_identities=identities,reader_queue_SHA256=queues,
        verified_inputs=check.inputs,original_data_and_checkpoints_not_selected=True,full_goal_complete=False))
    close(intent,context,'intent');guard()
    require(all(sha(p) == d for p,d in queues.items()), 'a new reader plan requires a new retirement scan')
    readers,latest = reader_snapshot(context,folder)
    require(not readers and latest == queues,'raw files acquired another actual or pending reader')
    # All selected files must be unchanged BEFORE the first unlink.
    require(all(identity(t['path']) == identities[t['path']] for t in tokens),'local raw inode changed after canonical verification')
    removed = []
    for token in tokens:
        target = Path(token['path']);require(identity(target) == identities[str(target)],'local raw inode changed before unlink')
        target.unlink();removed.append(str(target))
        with (output/'retired-local-cache.jsonl').open('a') as stream:
            stream.write(json.dumps(token)+'\n');stream.flush();os.fsync(stream.fileno())
    guard(removed)
    remaining = [str(root/role/name) for role in ['geometry','PBR'] for name in sorted(RAW_NAMES) if (root/role/name).exists()]
    released = False
    if not remaining:
        DiskBudget(context['disk_reservations'],workspace_bytes=context['workspace_bytes']).release(
            context['disk_reservation_name'],actual_wrapper_receipt=root/'file4-closure.json')
        released = True
    record = dict(status='PASS_ACTUAL_SELECTED_CASE_LOCAL_RAW_DUPLICATES_RETIRED',retired=tokens,
        fresh_actual_remote_hash_verified=True,canonical_remote_files_retained=True,
        guides_scores_images_reports_and_checkpoints_retained=True,remaining_own_raw_files=remaining,
        actual_own_disk_reservation_released=released,bytes_retired=sum(t['bytes'] for t in tokens),full_goal_complete=False)
    write(output/'audit.json',record);close(output,context,'result')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(execute(args.plan)),flush=True)


if __name__ == '__main__':main()
