import ast
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting import selected_final_metric_inputs as original
from experiments.deformnext_retargeting.operational_selected_case_reuse import (
    STATUS, adapt_case_root, physical_case_root, validate_manifest,
)
from experiments.deformnext_retargeting.operational_reused_campaign_entry import RemainingCampaign,base_operational_sources,verify_reuse_sources
from experiments.deformnext_retargeting.operational_campaign_entry import SerializedRetirementCampaign
from experiments.deformnext_retargeting.operational_reused_metric_entry import collector_descriptors


def fixture(tmp_path):
    ids = [f'asset-{i:03d}' for i in range(508)]
    models = {a: dict(actual_model=a) for a in ['zero-source4','zero-edited4','V1','RETARGET','SELECTED']}
    inputs = {k: dict(path=str(tmp_path/k), SHA256='a'*64, receipt=str(tmp_path/'receipt'))
        for k in ['parent_freeze','control_freeze']}
    context = dict(canonical_local_root=str(tmp_path), campaign_root=str(tmp_path/'new'),
        source_SHA256={str(tmp_path/f'source-{i}.py'):'a'*64 for i in range(505)}, source_inputs=inputs)
    old = tmp_path/'old'
    groups = {}
    for sid in ids[:129]:
        token = hashlib.sha256(sid.encode()).hexdigest()[:24]
        groups[sid] = dict(group_audit=dict(path=str(old/'groups'/token/'audit.json')),
            cases={a:dict(audit=dict(path=str(old/'cases'/token/a/'audit.json')),
                          worker_terminal=dict(path=str(old/'workers'/token/a/'terminal.json'))) for a in models})
    manifest = dict(status=STATUS, reused_sample_ids=ids[:129], groups=groups,models=models,
        original_campaign_root=str(old), original_source_SHA256=deepcopy(context['source_SHA256']),
        actual_same_scored_archive_bytes_verified=True,no_case_or_terminal_relocated_or_synthesized=True,**inputs)
    manifest['models'] = deepcopy(models)
    return manifest,context,dict(population=dict(sample_ids=ids)),models


@pytest.mark.parametrize('problem',[None,'changed-source','same-root','partial','different-model',
    'outcome-reordered','unknown-id','pixels-unverified','relocated-case','relocated-group'])
def test_exact_original_complete_groups_only(tmp_path,problem):
    m,c,p,models=deepcopy(fixture(tmp_path))
    sid=p['population']['sample_ids'][0]
    if problem=='changed-source': m['original_source_SHA256']={}
    elif problem=='same-root': m['original_campaign_root']=c['campaign_root']
    elif problem=='partial': m['groups'][sid]['cases'].pop('SELECTED')
    elif problem=='different-model': m['models']['SELECTED']={}
    elif problem=='outcome-reordered': m['reused_sample_ids'].reverse()
    elif problem=='unknown-id': m['reused_sample_ids'].append('not-in-frozen-population')
    elif problem=='pixels-unverified': m['actual_same_scored_archive_bytes_verified']=False
    elif problem=='relocated-case': m['groups'][sid]['cases']['SELECTED']['audit']['path']=str(tmp_path/'new/audit.json')
    elif problem=='relocated-group': m['groups'][sid]['group_audit']['path']=str(tmp_path/'new/audit.json')
    if problem is None:
        assert validate_manifest(m,context=c,parent=p,models=models)==p['population']['sample_ids'][:129]
    else:
        with pytest.raises(ValueError):validate_manifest(m,context=c,parent=p,models=models)


def test_root_join_keeps_both_physical_producer_paths(tmp_path):
    m,c,p,_=fixture(tmp_path)
    for index,expected in [(0,'old'),(129,'new')]:
        sid=p['population']['sample_ids'][index]
        token=hashlib.sha256(sid.encode()).hexdigest()[:24]
        assert physical_case_root(c,m,sid,'SELECTED')==tmp_path/expected/'cases'/token/'SELECTED'


def test_actual_frozen_loader_ast_changes_only_one_metadata_expression():
    source=inspect.getsource(original.load_campaign)
    tree,proof=adapt_case_root(source)
    assert proof['changed_case_root_expressions']==1 and proof['all_other_AST_statements_unchanged']
    target=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
        and n.func.id=='_actual_case_root')
    original_expression=ast.parse("Path(runtime['campaign_root'])/'cases'/hashlib.sha256(sid.encode()).hexdigest()[:24]/alias",mode='eval').body
    for n in ast.walk(tree):
        if isinstance(n,ast.Assign) and n.value is target:n.value=original_expression
    assert ast.dump(tree,include_attributes=False)==ast.dump(ast.parse(source),include_attributes=False)
    assert inspect.getsource(original.load_campaign)==source


@pytest.mark.parametrize('change',['missing','duplicate'])
def test_unexpected_loader_structure_fails_closed(change):
    source=inspect.getsource(original.load_campaign)
    if change=='missing':source=source.replace("runtime['campaign_root']","runtime['other']")
    else:source+='\ndef duplicate():\n    root=Path(runtime[\'campaign_root\'])/\'cases\'/hashlib.sha256(sid.encode()).hexdigest()[:24]/alias\n'
    with pytest.raises(ValueError):adapt_case_root(source)


def test_all_original_gpu_jobs_and_collector_options_preserved():
    gpu=[dict(name=f'gpu-{i}',resource='exclusive',command=['python','-u','-m',
        'experiments.deformnext_retargeting.run_selected_final_metric','--plan',f'/gpu-{i}.json']) for i in range(11)]
    cpu=dict(name='collector',resource='cpu',requires=['actual-11-worker-closures'],
        command=['python','-u','-m','experiments.deformnext_retargeting.collect_selected_final_metrics','--plan','/collect.json'])
    jobs=deepcopy(gpu+[cpu]);result=collector_descriptors(jobs)
    assert result[:11]==jobs[:11] and result[-1]['requires']==cpu['requires']
    assert result[-1]['command']==['python','-u','-m','experiments.deformnext_retargeting.operational_reused_metric_entry',
        '--phase','collect','--plan','/collect.json']
    assert jobs==gpu+[cpu]
    with pytest.raises(ValueError):collector_descriptors(gpu)
    with pytest.raises(ValueError):collector_descriptors([cpu,cpu])
    cpu['resource']='exclusive'
    with pytest.raises(ValueError):collector_descriptors([cpu])


@pytest.mark.parametrize('problem',[None,'partial-new','overlap','missing-model'])
def test_success_requires_actual_full508_union_without_fabricating_terminals(tmp_path,monkeypatch,problem):
    m,c,p,models=fixture(tmp_path)
    obj=RemainingCampaign.__new__(RemainingCampaign)
    obj.context=dict(source_inputs=dict(case_reuse_manifest=dict(path='/closed-real-reuse.json')))
    obj.reused=m['groups'];obj.full_ids=p['population']['sample_ids'];obj.ids=obj.full_ids[129:];obj.models=models
    new={sid:{a:dict(actual_terminal=f'real-new-{sid}-{a}') for a in models} for sid in obj.ids}
    if problem=='partial-new':new.pop(obj.ids[-1])
    elif problem=='overlap':new[obj.full_ids[0]]=deepcopy(new[obj.ids[0]])
    elif problem=='missing-model':new[obj.ids[0]].pop('SELECTED')
    monkeypatch.setattr(SerializedRetirementCampaign,'run',lambda self:dict(actual_cases=new,assets=379))
    if problem is None:
        result=obj.run()
        assert result['assets']==508 and result['actual_reused_native_cases']==645 and result['actual_new_native_cases']==1895
        assert result['actual_cases'][obj.full_ids[0]] is m['groups'][obj.full_ids[0]]['cases']
        assert result['actual_cases'][obj.ids[0]] is new[obj.ids[0]]
    else:
        with pytest.raises(ValueError):obj.run()


@pytest.mark.parametrize('admitted',[False,True])
def test_disk_pressure_defers_new_launch_without_waiving_atomic_case_guard(monkeypatch,admitted):
    obj=RemainingCampaign.__new__(RemainingCampaign);calls=[]
    obj.budget=SimpleNamespace(inspect=lambda size:dict(admitted=admitted))
    monkeypatch.setattr(SerializedRetirementCampaign,'start_cases',lambda self:calls.append('original-start'))
    obj.start_cases()
    assert calls==(['original-start'] if admitted else [])


def test_actual_serialized_cpu_workers_finish_without_retirement_size_scan_race(tmp_path,monkeypatch):
    # Reuse the existing real CPU process/mirror fixture. No scientific loader
    # or GPU native phase is claimed verified by these tiny fixture workers.
    from test_selected_final_campaign_execution import prepare
    from experiments.deformnext_retargeting import run_selected_final_campaign as frozen
    from experiments.deformnext_retargeting.selected_campaign_resources import token_processes
    engine,calls,mirror=prepare(tmp_path,monkeypatch)
    engine.__class__=SerializedRetirementCampaign
    engine.context['operational_source_SHA256']={}
    fixture_popen=frozen.subprocess.Popen
    def launch_fixture(cmd,**kw):
        if cmd[3]=='experiments.deformnext_retargeting.operational_retirement_wait':
            cmd=[*cmd[:3],'experiments.deformnext_retargeting.retire_selected_final_raw',*cmd[4:]]
        return fixture_popen(cmd,**kw)
    monkeypatch.setattr(frozen.subprocess,'Popen',launch_fixture)
    try:result=engine.run()
    finally:engine.drain()
    assert result['assets']==2 and sum(map(len,result['actual_cases'].values()))==10
    assert not engine.groups and not engine.own and not token_processes(engine.token)
    assert len([k for k,c in calls if k=='case'])==10
    assert len([k for k,c in calls if k=='retirement'])==14
    assert list(mirror.rglob('prediction.npz')) and not list((engine.root/'cases').rglob('prediction.npz'))


def test_reuse_pins_preserve_original_strict_three_source_retirement_contract(tmp_path,monkeypatch):
    from experiments.deformnext_retargeting import operational_reused_campaign_entry as module
    from experiments.deformnext_retargeting import run_selected_final_campaign as frozen
    from experiments.deformnext_retargeting.cascade_view_roles import file_sha256 as sha
    root=Path(module.__file__).resolve().parent
    names=['operational_process_access.py','operational_retirement_wait.py','operational_campaign_entry.py',
        'operational_reused_campaign_entry.py','operational_reused_metric_entry.py','operational_selected_case_reuse.py']
    context=dict(operational_source_SHA256={str(root/n):sha(root/n) for n in names})
    verify_reuse_sources(context)
    assert len(base_operational_sources(context))==3
    seen=[];waited=[];collected=[]
    def spawn(self,name,c,entry,**kw):
        seen.append((c,entry));self.own[name]=dict(child=SimpleNamespace(wait=lambda:waited.append(True)))
    monkeypatch.setattr(frozen.Campaign,'spawn',spawn)
    obj=RemainingCampaign.__new__(RemainingCampaign);obj.context=context;obj.root=tmp_path;obj.own={}
    obj.collect=lambda:collected.append(True)
    obj.spawn('retire',{},'experiments.deformnext_retargeting.retire_selected_final_raw',
        sid='fixture',alias='SELECTED',kind='retirement')
    assert obj.context is context and len(context['operational_source_SHA256'])==6
    assert seen[0][0]['operational_source_SHA256']==base_operational_sources(context)
    assert waited==collected==[True]
    context['operational_source_SHA256'][str(root/'operational_selected_case_reuse.py')]='a'*64
    with pytest.raises(ValueError):verify_reuse_sources(context)
