"""Bounded actual508 final cascade, with whole-asset credits and two keepers.

Run --keeper once as an exclusive group job in each existing dispatcher and
the coordinator as a CPU job. No dispatcher is restarted. Activation requires
the actual DEV freeze and every original saved-model/cache/sampler gate.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil

from .cascade_view_roles import ClosedInputs,file_sha256 as sha
from .collect_module_seed_confirmation import read_file,verify_terminal
from .native_run_disk_budget import DiskBudget
from .process_session_drain import drain_owned_session
from .run_selected_final_case import SCHEMA as CASE_SCHEMA, required_sources as case_sources
from .retire_selected_final_raw import SCHEMA as RETIRE_SCHEMA, required_sources as retirement_sources
from .selected_campaign_resources import (geometry_owners,live,process_identity,ready_aliases,
    reserve_asset_group,source_guard,token_processes,verify_queue_owner)
from .selected_final_case_integrity import CASE_STATUS
from .selected_final_population import admit_all_ss,compose_models
from .selected_pipeline_freeze_v2 import exact
from .selected_pipeline_model_contract import require


SCHEMA = 'selected-physical-final-campaign-v1'
MODULE = 'experiments.deformnext_retargeting.run_selected_final_campaign'


def required_sources():
    root=Path(__file__).resolve().parent
    return sorted(set(case_sources()) | set(retirement_sources()) | {
        str(root/(n+'.py')) for n in ['run_selected_final_campaign','run_selected_campaign_case','selected_campaign_resources',
            'prepare_selected_final_campaign']})


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def atomic(path,value):
    path=Path(path); temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(temporary,path)


def closed_spec(path):
    return dict(path=str(path),SHA256=sha(path),receipt=str(Path(path).parent/'file4-closure.json'))


def close(folder,context,label):
    folder=Path(folder); base=Path(context['canonical_local_root'])
    manifest=Path(context['manifest_directory'])/(context['name']+'-'+label+'-files.json')
    write(manifest,[str(p.relative_to(base)) for p in sorted(folder.rglob('*'))
        if p.is_file() and p.name not in ('file4-closure.json','failure-file4-closure.json')])
    subprocess.run([*context['closure_command'],'--manifest',str(manifest),'--receipt',
        str((folder/'file4-closure.json').relative_to(base)),
        '--scope','Actual selected final campaign ownership, native case or canonical raw retirement; no new model selection'],check=True)


def command(context,path,keeper=None):
    result=[context['python'],'-u','-m',MODULE,'--plan',str(path)]
    return result if keeper is None else result+['--keeper',keeper]


def validate(context):
    require(context.get('schema') == SCHEMA and context['repo'] == str(Path(__file__).resolve().parents[2]),
        'actual selected final campaign context and source checkout required')
    keys=['canonical_local_root','campaign_root','manifest_directory','disk_reservations','python']
    require(all(isinstance(context[k],str) and Path(context[k]).is_absolute() for k in keys), 'explicit absolute campaign paths required')
    base=Path(context['canonical_local_root']); root=Path(context['campaign_root'])
    require(root != base and root.is_relative_to(base) and root.resolve() == root and base.resolve() == base,
        'own resolved campaign root below canonical root required')
    devices=context['assigned_GPUs']; groups=context['keeper_groups']
    require(isinstance(devices,list) and devices and len(set(devices)) == len(devices)
        and all(isinstance(x,str) and x.isdigit() for x in devices), 'explicit distinct assigned physical devices required')
    owned=[g for row in groups.values() for g in row['assigned_GPUs']]
    require(len(groups) == 2 and len(owned) == len(set(owned)) and set(owned) == set(devices)
        and all(row['assigned_GPUs'] for row in groups.values()), 'two disjoint full GPU group reservations required')
    require(type(context['max_asset_groups']) is int and 0 < context['max_asset_groups'] <= len(devices),
        'explicit bounded complete asset groups required')
    require(type(context['workspace_bytes']) is int and context['workspace_bytes'] > 0
        and set(required_sources()) <= set(context['source_SHA256']), 'positive workspace and full campaign source pins required')
    require(context['closure_command'][0] == context['python'] and len(context['closure_command']) == 2
        and context['closure_command'][1] in context['source_SHA256'], 'actual pinned closure helper required')
    require(context['coordinator_queue']['assigned_GPUs'] == []
        and context['coordinator_queue']['jobs'] in {g['jobs'] for g in groups.values()}, 'coordinator is a CPU task in one actual dispatcher')
    require(context['CPU_affinity'] and all(type(c) is int for c in context['CPU_affinity']), 'explicit metadata CPU assignment required')
    require(context['reader_queues'] and all(set(q) == {'jobs','states'} for q in context['reader_queues']), 'actual current reader ledgers required')
    require({(g['jobs'],g['states']) for g in groups.values()} <= {(q['jobs'],q['states']) for q in context['reader_queues']},
        'both actual dispatcher reader ledgers required')
    require(set(context['case_defaults']['CPU_affinity_per_GPU']) == set(devices)
        and type(context['case_defaults']['minimum_free_GPU_bytes']) is int
        and context['case_defaults']['minimum_free_GPU_bytes'] > 0, 'explicit per-GPU CPU and measured memory admission required')


def load(path):
    path=Path(path);context=json.loads(path.read_text());validate(context);source_guard(context)
    check=ClosedInputs(context['canonical_local_root']);check.check(path,path.parent/'file4-closure.json')
    return context,check


def keeper_ready(context,path,name,ready):
    require(ready.get('plan_SHA256') == sha(path) and ready.get('keeper') == name
        and ready.get('assigned_GPUs') == context['keeper_groups'][name]['assigned_GPUs'], 'actual own closed campaign keeper required')
    return verify_queue_owner(context['keeper_groups'][name],identity=ready['identity'],command=command(context,path,name))


def keeper(context,path,name):
    require(name in context['keeper_groups'], 'explicit one owned dispatcher group required')
    group=context['keeper_groups'][name];identity=process_identity();root=Path(context['campaign_root'])/'ownership'/name
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == ','.join(group['assigned_GPUs']), 'keeper actual visible GPU group required')
    os.sched_setaffinity(0,set(context['CPU_affinity']));os.nice(10)
    root.mkdir(parents=True,exist_ok=False)
    # The queue publishes running state after Popen. Wait for that exact child,
    # never infer ownership from an old file or an observation deadline.
    while True:
        state=json.loads(Path(group['states']).read_text()).get(group['job_name'],{})
        require(live(group['dispatcher']), 'actual owning dispatcher disappeared; preserve reservation evidence')
        if state.get('pid') == identity['PID'] and state.get('status') == 'running':break
        time.sleep(.5)
    verify_queue_owner(group,identity=identity,command=command(context,path,name))
    write(root/'ready.json',dict(keeper=name,identity=identity,assigned_GPUs=group['assigned_GPUs'],plan_SHA256=sha(path)))
    release=Path(context['campaign_root'])/'release.json'
    stopped=[False];previous={}
    def stop(reason):
        stopped[0]=True
        if not (root/'stop-request.json').exists():write(root/'stop-request.json',dict(reason=str(reason),identity=identity))
    def interrupted(signum,frame):stop('keeper interrupted '+str(signum))
    for sig in [signal.SIGINT,signal.SIGTERM]:previous[sig]=signal.signal(sig,interrupted)
    try:
        while True:
            try:
                if not stopped[0]:
                    source_guard(context);keeper_ready(context,path,name,json.loads((root/'ready.json').read_text()))
                if release.exists() and (release.parent/'release-closure/file4-closure.json').exists():
                    check=ClosedInputs(context['canonical_local_root']);check.check(release,release.parent/'release-closure/file4-closure.json')
                    value=json.loads(release.read_text())
                    require(value.get('plan_SHA256') == sha(path) and value.get('all_actual_campaign_children_drained') is True
                        and not token_processes(sha(path)), 'actual complete descendant drain required before releasing any GPUs')
                    actual=json.loads((release.parent/'coordinator.json').read_text())
                    require(value.get('coordinator') == actual['identity'], 'actual coordinator release identity required')
                    break
            except Exception as exc:
                # A source/IO/ownership error requests a coordinator stop. It
                # cannot release this reservation while NN children may live.
                stop(repr(exc))
            time.sleep(2)
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
    write(root/'terminal.json',dict(status='PASS_ACTUAL_CAMPAIGN_GPU_RESERVATION_RELEASED',identity=identity,
        plan_SHA256=sha(path),actual_owned_GPU_descendants_absent=True,campaign_status=value['status']))
    close(root,context,'keeper-'+name)


def load_population(context,check):
    inputs=context['source_inputs'];parent=read_file(inputs['parent_freeze'],check);controls=read_file(inputs['control_freeze'],check)
    models=compose_models(parent,controls)
    require(set(required_sources()) <= set(parent['evaluation_protocol']['source_SHA256'])
        and all(parent['evaluation_protocol']['source_SHA256'].get(p) == d for p,d in context['source_SHA256'].items()
            if p != context['closure_command'][1]), 'freeze actual complete campaign sources before any TEST prediction')
    records={name:read_file(value,check) for name,value in inputs['SS_evaluations'].items()};admit_all_ss(records,parent=parent,controls=controls)
    require(set(context['case_budget_bytes']) == set(models)
        and all(type(n) is int and n > 0 for n in context['case_budget_bytes'].values()), 'measured whole-case budgets for actual deduplicated models required')
    modes={m['PBR_mode'] for m in models.values() if m['kind'] == 'selected-trained'}
    require(set(context['PBR_by_mode']) == modes, 'actual own selected and strong-control PBR inputs required')
    for data in context['PBR_by_mode'].values():
        require(set(data['input_SHA256_by_sample']) == set(data['closed_inputs_by_sample']) == set(parent['population']['sample_ids']),
            'all508 actual PBR input-image cache bindings required')
    return parent,controls,models


class Campaign:
    def __init__(self,context,path,check,parent,controls,models):
        self.context=context;self.path=Path(path);self.check=check;self.parent=parent;self.controls=controls;self.models=models
        self.root=Path(context['campaign_root']);self.identity=process_identity();self.token=sha(path)
        self.budget=DiskBudget(context['disk_reservations'],workspace_bytes=context['workspace_bytes'])
        self.ids=list(parent['population']['sample_ids']);self.next=0;self.groups={};self.own={};self.done={};self.retired={};self.error=None
        self.last_disk_admission=None
        self.jobs=[];self.states={};self.ledger=self.root/'reader-ledger';self.ledger.mkdir()
        self.update_ledger()

    def update_ledger(self):
        atomic(self.ledger/'jobs.json',self.jobs);atomic(self.ledger/'states.json',self.states)

    def group_path(self,sid):
        import hashlib
        return hashlib.sha256(sid.encode()).hexdigest()[:24]

    def case_name(self,sid,alias):
        require(alias and all(c.isalnum() or c in '-_' for c in alias), 'safe actual frozen model alias required')
        return self.context['name']+'-'+self.group_path(sid)+'-'+alias

    def admit(self):
        if self.next == len(self.ids) or len(self.groups) >= self.context['max_asset_groups']:return False
        sid=self.ids[self.next];key=self.group_path(sid);output=self.root/'groups'/key
        cases={self.case_name(sid,a):dict(output=str(self.root/'cases'/key/a),bytes=self.context['case_budget_bytes'][a]) for a in self.models}
        name=self.context['name']+'-asset-'+key
        admission=reserve_asset_group(self.budget,name,cases=cases,output=output,owner=self.identity)
        self.last_disk_admission=admission
        if not admission['admitted']:return False
        output.mkdir(parents=True,exist_ok=False);write(output/'admission.json',admission)
        self.groups[sid]=dict(name=name,key=key,output=output,completed={},retirements={},retirement_tasks=[])
        self.next+=1;return True

    def context_for_case(self,sid,alias):
        group=self.groups[sid];c=deepcopy(self.context['case_defaults']);name=self.case_name(sid,alias)
        c.update(schema=CASE_SCHEMA,name=name,sample_id=sid,alias=alias,repo=self.context['repo'],python=self.context['python'],
            canonical_local_root=self.context['canonical_local_root'],manifest_directory=self.context['manifest_directory'],
            disk_reservations=self.context['disk_reservations'],workspace_bytes=self.context['workspace_bytes'],
            output=str(self.root/'cases'/group['key']/alias),assigned_GPUs=self.context['assigned_GPUs'],
            source_SHA256=self.context['source_SHA256'],closure_command=self.context['closure_command'],
            raw_budget_bytes=self.context['case_budget_bytes'][alias],source_inputs=deepcopy(self.context['source_inputs']),
            campaign_token=self.token,campaign_credit=dict(group_name=group['name'],case_name=name,coordinator=self.identity))
        reference=self.root/'cases'/group['key']/'zero-source4'
        c.update(reference_shared_root=str(reference/'shared'),reference_render_root=str(reference/'render'))
        if alias != 'zero-source4':c['source_inputs']['material_reference']=group['completed']['zero-source4']['audit']
        owner=geometry_owners(self.models)[alias]
        if owner != alias:c['source_inputs']['geometry_owner']=group['completed'][owner]['audit']
        model=self.models[alias]
        if model['kind'] == 'selected-trained':
            data=self.context['PBR_by_mode'][model['PBR_mode']];c['source_inputs']['PBR_template']=data['template']
            c['PBR_input_SHA256']=data['input_SHA256_by_sample'][sid];c['PBR_closed_inputs']=data['closed_inputs_by_sample'][sid]
        return c

    def spawn(self,name,context,module,*,sid,alias,kind,gpu='',role=None):
        directory=self.root/'plans'/name;directory.mkdir(parents=True,exist_ok=False);path=directory/'context.json'
        write(path,context);close(directory,self.context,'plan-'+name)
        worker=self.root/'workers'/name;worker.mkdir(parents=True,exist_ok=False)
        cmd=[self.context['python'],'-u','-m',module,'--plan',str(path)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,CUDA_DEVICE_ORDER='PCI_BUS_ID',MATCHING_V2_ALLOWED_GPUS=','.join(self.context['assigned_GPUs']),
            SELECTED_FINAL_CAMPAIGN_TOKEN=self.token,SELECTED_FINAL_WORKER_TOKEN=self.token+':'+name,PYTHONPATH=self.context['repo'],
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',OPENCV_IO_ENABLE_OPENEXR='1')
        job=dict(name=name,command=cmd,resource='exclusive' if gpu else 'cpu',outputs=[])
        self.jobs.append(job);self.states[name]=dict(status='pending');self.update_ledger()
        with (worker/'worker.log').open('xb') as stream:
            child=subprocess.Popen(cmd,cwd=self.context['repo'],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        identity=process_identity(child.pid)
        value=dict(child=child,identity=identity,name=name,worker=worker,sid=sid,alias=alias,kind=kind,gpu=gpu,role=role,context=context)
        self.own[name]=value;write(worker/'active.json',dict(**identity,command=cmd,kind=kind,physical_GPU=gpu,context=closed_spec(path)))
        self.states[name]=dict(status='running',pid=child.pid,gpu=gpu,command=cmd);self.update_ledger()

    def start_cases(self):
        occupied={v['gpu'] for v in self.own.values() if v['kind'] == 'case'}
        for gpu in self.context['assigned_GPUs']:
            if gpu in occupied:continue
            for sid,group in self.groups.items():
                running={v['alias'] for v in self.own.values() if v['kind'] == 'case' and v['sid'] == sid}
                ready=ready_aliases(self.models,group['completed'],running)
                if ready:
                    alias=ready[0];self.spawn(self.case_name(sid,alias),self.context_for_case(sid,alias),
                        'experiments.deformnext_retargeting.run_selected_campaign_case',sid=sid,alias=alias,kind='case',gpu=gpu)
                    break

    def collect(self):
        for name,entry in list(self.own.items()):
            child=entry['child'];code=child.poll()
            if code is None:continue
            child.wait();drain_owned_session(child.pid,entry['identity']['birth'])
            require(not token_processes(self.token+':'+name,field='SELECTED_FINAL_WORKER_TOKEN'),
                'worker died while an actual separately-sessioned phase remains; preserve reservation and drain')
            terminal=dict(**entry['identity'],session_ID=child.pid,returncode=code,actual_session_drained=True,
                case_output=entry['context']['output'],command=self.states[name]['command'])
            write(entry['worker']/'terminal.json',terminal);close(entry['worker'],self.context,'worker-'+name)
            self.states[name].update(status='completed' if code == 0 else 'FAILED',returncode=code);self.update_ledger()
            del self.own[name]
            require(code == 0,'actual final '+entry['kind']+' failed; no automatic retry: '+name)
            verify_terminal(terminal);audit_path=Path(entry['context']['output'])/'audit.json'
            audit=read_file(closed_spec(audit_path),self.check);group=self.groups[entry['sid']]
            value=dict(audit=closed_spec(audit_path),worker_terminal=closed_spec(entry['worker']/'terminal.json'))
            if entry['kind'] == 'case':
                require(audit.get('status') == CASE_STATUS and audit.get('alias') == entry['alias']
                    and audit.get('sample_id') == entry['sid'] and exact(audit.get('frozen_model_binding'),self.models[entry['alias']]),
                    'own actual frozen native case required')
                group['completed'][entry['alias']]=value;self.done.setdefault(entry['sid'],{})[entry['alias']]=value
            else:
                require(audit.get('status') == 'PASS_ACTUAL_SELECTED_CASE_LOCAL_RAW_DUPLICATES_RETIRED'
                    and audit.get('fresh_actual_remote_hash_verified') is True, 'actual canonical retirement result required')
                group['retirements'][name]=value;self.retired[name]=value

    def retirement_context(self,sid,alias,role):
        group=self.groups[sid];name=self.case_name(sid,alias)+'-retire-'+role
        names=[a for a,m in self.models.items() if m['geometry_group'] == self.models[alias]['geometry_group']] if role == 'geometry' else [alias]
        c=dict(schema=RETIRE_SCHEMA,name=name,repo=self.context['repo'],python=self.context['python'],
            canonical_local_root=self.context['canonical_local_root'],owned_outputs_root=str(self.root/'cases'),
            output=str(self.root/'retirements'/name),alias=alias,role=role,CPU_affinity=self.context['CPU_affinity'],
            source_SHA256=self.context['source_SHA256'],closure_command=self.context['closure_command'],
            manifest_directory=self.context['manifest_directory'],disk_reservations=self.context['disk_reservations'],
            disk_reservation_name=self.case_name(sid,alias),workspace_bytes=self.context['workspace_bytes'],
            ssh_argv=self.context['ssh_argv'],canonical_remote_root=self.context['canonical_remote_root'],
            reader_queues=[*self.context['reader_queues'],dict(jobs=str(self.ledger/'jobs.json'),states=str(self.ledger/'states.json'))],
            source_inputs={k:self.context['source_inputs'][k] for k in ['parent_freeze','control_freeze']})
        c['source_inputs'].update(case_audits={a:group['completed'][a]['audit'] for a in names},
            worker_terminals={a:group['completed'][a]['worker_terminal'] for a in names})
        return c

    def retire_groups(self):
        busy=any(v['kind'] == 'retirement' for v in self.own.values())
        owners=geometry_owners(self.models)
        for sid,group in list(self.groups.items()):
            if set(group['completed']) != set(self.models):continue
            tasks=[(a,'PBR') for a,m in self.models.items() if m['kind'] == 'selected-trained']
            tasks += [(a,'geometry') for a in self.models if owners[a] == a]
            pending=[(a,r) for a,r in tasks if self.case_name(sid,a)+'-retire-'+r not in group['retirements']]
            if pending:
                if not busy:
                    alias,role=pending[0];c=self.retirement_context(sid,alias,role)
                    self.spawn(c['name'],c,'experiments.deformnext_retargeting.retire_selected_final_raw',
                        sid=sid,alias=alias,role=role,kind='retirement');busy=True
                continue
            with self.budget._locked() as value:
                entry=value['runs'][group['name']]
                require(entry['checkpoint_budget_bytes'] == 0 and all(c['claimed'] for c in entry['cases'].values())
                    and all(value['runs'][name].get('released') is True for name in entry['cases']),
                    'all actual raw case reservations must retire before the asset group can release')
            write(group['output']/'audit.json',dict(status='PASS_ACTUAL_COMPLETE_ASSET_GROUP_NATIVE_CASES_AND_RAW_RETIREMENT',
                sample_id=sid,cases=group['completed'],retirements=group['retirements'],raw_duplicates_only=True))
            close(group['output'],self.context,'group-'+group['key'])
            self.budget.release(group['name'],actual_wrapper_receipt=group['output']/'file4-closure.json')
            del self.groups[sid]

    def check_owned_sizes(self):
        for sid,group in self.groups.items():
            for alias in self.models:
                root=self.root/'cases'/group['key']/alias
                size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
                require(size <= self.context['case_budget_bytes'][alias], 'actual case exceeded its measured whole-case budget; preserve data and stop launches')

    def run(self):
        while self.next < len(self.ids) or self.groups or self.own:
            source_guard(self.context);self.check.check(self.path,self.path.parent/'file4-closure.json')
            for name in self.context['keeper_groups']:
                require(not (self.root/'ownership'/name/'stop-request.json').exists(), 'GPU keeper requested a controlled campaign stop')
                ready=json.loads((self.root/'ownership'/name/'ready.json').read_text());keeper_ready(self.context,self.path,name,ready)
            self.collect();self.check_owned_sizes();self.retire_groups()
            while self.admit():pass
            self.start_cases()
            atomic(self.root/'progress.json',dict(status='RUNNING',admitted_assets=self.next,total_assets=len(self.ids),
                completed_cases=sum(map(len,self.done.values())),total_cases=len(self.ids)*len(self.models),
                open_asset_groups=len(self.groups),last_actual_disk_admission=self.last_disk_admission,
                actual_workers=[dict(name=n,**v['identity'],GPU=v['gpu'],kind=v['kind']) for n,v in self.own.items()]))
            time.sleep(2)
        return dict(status='PASS_ACTUAL_SELECTED_FINAL_CASE_CAMPAIGN',assets=len(self.ids),models=list(self.models),
            actual_cases=self.done,actual_raw_retirements=self.retired,all_actual_campaign_children_drained=True,
            no_new_model_selection=True,all_ten_metrics_collected=False,full_goal_complete=False)

    def drain(self):
        # Signal the source-bound worker; its original handler drains every
        # separately-sessioned NN phase. If it died first, signal only matching
        # inherited ownership tokens, never an unrelated GPU process.
        for entry in self.own.values():
            if live(entry['identity']):
                try:entry['child'].send_signal(signal.SIGINT)
                except ProcessLookupError:pass
        signalled=set()
        while True:
            rows=token_processes(self.token)
            if not rows:break
            for row in rows:
                try:
                    proc=psutil.Process(row['PID']);worker_token=proc.environ().get('SELECTED_FINAL_WORKER_TOKEN')
                except (psutil.NoSuchProcess,psutil.ZombieProcess):continue
                owner=next((v for name,v in self.own.items() if worker_token == self.token+':'+name),None)
                # A live worker already handles and drains its own phases.
                # Intervene directly only when that actual worker has died.
                if owner is not None and live(owner['identity']):continue
                key=(row['PID'],row['birth'])
                if key not in signalled and live(row):
                    try:os.kill(row['PID'],signal.SIGINT)
                    except ProcessLookupError:pass
                    signalled.add(key)
            time.sleep(1)
        for entry in self.own.values():
            entry['child'].wait();drain_owned_session(entry['child'].pid,entry['identity']['birth'])


def coordinator(context,path,check):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'explicit CPU-only final campaign coordinator required')
    os.sched_setaffinity(0,set(context['CPU_affinity']));os.nice(10)
    root=Path(context['campaign_root']);root.mkdir(parents=True,exist_ok=True)
    require(not (root/'coordinator.json').exists(), 'preserve any previous campaign, including failures; no automatic restart')
    write(root/'coordinator.json',dict(identity=process_identity(),plan_SHA256=sha(path)))
    campaign=None;error=None;result=None
    previous={}
    def interrupted(signum,frame):raise KeyboardInterrupt('selected final campaign interrupted '+str(signum))
    for sig in [signal.SIGINT,signal.SIGTERM]:previous[sig]=signal.signal(sig,interrupted)
    try:
        parent,controls,models=load_population(context,check)
        # CPU queue state, like keeper state, is published just after Popen.
        while True:
            group=context['coordinator_queue'];require(live(group['dispatcher']), 'actual coordinator dispatcher required')
            state=json.loads(Path(group['states']).read_text()).get(group['job_name'],{})
            if state.get('pid') == os.getpid() and state.get('status') == 'running':break
            time.sleep(.5)
        verify_queue_owner(group,identity=process_identity(),command=command(context,path))
        while True:
            require(live(group['dispatcher']), 'actual coordinator dispatcher disappeared')
            for name in context['keeper_groups']:
                require(not (root/'ownership'/name/'stop-request.json').exists(), 'actual keeper requested stop before activation')
            files=[root/'ownership'/name/'ready.json' for name in context['keeper_groups']]
            if all(f.exists() for f in files):
                for name in context['keeper_groups']:keeper_ready(context,path,name,json.loads((root/'ownership'/name/'ready.json').read_text()))
                break
            time.sleep(2)
        campaign=Campaign(context,path,check,parent,controls,models)
        result=campaign.run()
    except BaseException as exc:
        error=exc
    finally:
        # Further ordinary interrupts cannot release keepers during drain.
        for sig in previous:signal.signal(sig,signal.SIG_IGN)
        if campaign is not None:campaign.drain()
        require(not token_processes(sha(path)), 'no keeper release while any actual campaign descendant remains')
        result=result or dict(status='FAILED_SELECTED_FINAL_CAMPAIGN_PRESERVED',error=repr(error),automatic_retry=False,full_goal_complete=False)
        folder=root/'result';folder.mkdir();write(folder/'audit.json',result);close(folder,context,'result')
        release=dict(status=result['status'],plan_SHA256=sha(path),coordinator=process_identity(),all_actual_campaign_children_drained=True,
            actual_result=closed_spec(folder/'audit.json'))
        # Close a separate immutable folder, then publish a link-free release
        # file whose actual bytes are included in that same remote receipt.
        release_folder=root/'release-closure';release_folder.mkdir();write(root/'release.json',release)
        manifest=Path(context['manifest_directory'])/(context['name']+'-release-files.json')
        write(manifest,[str((root/'release.json').relative_to(Path(context['canonical_local_root'])))])
        subprocess.run([*context['closure_command'],'--manifest',str(manifest),'--receipt',
            str((release_folder/'file4-closure.json').relative_to(Path(context['canonical_local_root']))),
            '--scope','Actual complete owned descendant drain before releasing two dispatcher GPU groups'],check=True)
        for sig,handler in previous.items():signal.signal(sig,handler)
    if error is not None:raise error
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--keeper')
    args=parser.parse_args();context,check=load(args.plan)
    if args.keeper:keeper(context,args.plan,args.keeper)
    else:print(json.dumps(coordinator(context,args.plan,check)),flush=True)


if __name__ == '__main__':main()
