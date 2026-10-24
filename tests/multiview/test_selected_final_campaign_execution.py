"""Campaign scheduling with real CPU fixture workers and a local mirror.

The scientific loader/native phase modules are deliberately not replaced or
claimed verified here. Tiny synthetic cases exercise the operational engine;
the public CLI must still admit the real closed508 freeze before using it.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from experiments.deformnext_retargeting import run_selected_final_campaign as module
from experiments.deformnext_retargeting.cascade_view_roles import ClosedInputs,file_sha256 as sha
from experiments.deformnext_retargeting.selected_campaign_resources import process_identity,token_processes


WORKER = '''
import json,os,pathlib,shutil,sys,time,subprocess
from experiments.deformnext_retargeting.selected_campaign_resources import CampaignCaseBudget,process_identity
from experiments.deformnext_retargeting.native_run_disk_budget import DiskBudget
from experiments.deformnext_retargeting.cascade_view_roles import file_sha256 as sha
from experiments.deformnext_retargeting.collect_module_seed_confirmation import verify_terminal
from experiments.deformnext_retargeting.selected_final_case_integrity import CASE_STATUS
c=json.load(open(sys.argv[1]));mode=sys.argv[2];base=pathlib.Path(sys.argv[3]);mirror=pathlib.Path(sys.argv[4]);root=pathlib.Path(c['output'])
def close(folder):
 files=[p for p in folder.rglob('*') if p.is_file() and p.name!='file4-closure.json'];rows=[]
 for p in files:
  rows.append(dict(relative_path=str(p.relative_to(base)),sha256=sha(p),bytes=p.stat().st_size))
  dest=mirror/p.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 (folder/'file4-closure.json').write_text(json.dumps(dict(status='PASS',local_equals_actual_file4_SHA256=True,files=rows)))
if mode=='case':
 b=CampaignCaseBudget(c['disk_reservations'],workspace_bytes=c['workspace_bytes'],credit=c['campaign_credit'])
 b.reserve(c['name'],checkpoint_budget_bytes=c['raw_budget_bytes'],native_output=str(root),wrapper_output=str(root),owner=process_identity())
 root.mkdir(parents=True)
 if c['alias']!='zero-source4':
  ref=json.load(open(c['source_inputs']['material_reference']['path']));assert ref['alias']=='zero-source4' and ref['sample_id']==c['sample_id']
 if 'geometry_owner' in c['source_inputs']:
  ref=json.load(open(c['source_inputs']['geometry_owner']['path']));assert ref['alias']=='V1' and ref['sample_id']==c['sample_id']
 if c['fixture_failure']==c['alias']:
  orphan=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
  (root/'owned-orphan.json').write_text(json.dumps(process_identity(orphan.pid)))
  sys.exit(3)
 model=c['fixture_models'][c['alias']];roles=[] if 'geometry_owner' in c['source_inputs'] else ['geometry']
 if model['kind']=='selected-trained':roles.append('PBR')
 for role in roles:
  d=root/role;d.mkdir()
  for name in ['prediction.npz','geometry-evaluation.npz']:(d/name).write_bytes(b'CPU-fixture-raw')
  (d/'native-shape-guides.pt').write_bytes(b'CPU-fixture-retained-guides')
 value=dict(status=CASE_STATUS,alias=c['alias'],sample_id=c['sample_id'],frozen_model_binding=model,fixture_only=True)
 time.sleep(.04);(root/'audit.json').write_text(json.dumps(value));close(root)
else:
 for name,spec in c['source_inputs']['worker_terminals'].items():verify_terminal(json.load(open(spec['path'])))
 audits=[json.load(open(v['path'])) for v in c['source_inputs']['case_audits'].values()]
 own=next(a for a in audits if a['alias']==c['alias']);assert all(a['sample_id']==own['sample_id'] for a in audits)
 ledger=json.load(open(c['disk_reservations']));case=pathlib.Path(ledger['runs'][c['disk_reservation_name']]['native_output'])
 for name in ['prediction.npz','geometry-evaluation.npz']:
  p=case/c['role']/name;assert p.read_bytes()==(mirror/p.relative_to(base)).read_bytes();p.unlink()
 remaining=[case/r/n for r in ['geometry','PBR'] for n in ['prediction.npz','geometry-evaluation.npz'] if (case/r/n).exists()]
 if not remaining:DiskBudget(c['disk_reservations'],workspace_bytes=c['workspace_bytes']).release(c['disk_reservation_name'],actual_wrapper_receipt=case/'file4-closure.json')
 root.mkdir(parents=True);(root/'audit.json').write_text(json.dumps(dict(status='PASS_ACTUAL_SELECTED_CASE_LOCAL_RAW_DUPLICATES_RETIRED',fresh_actual_remote_hash_verified=True,fixture_only=True)))
 close(root)
'''


def prepare(tmp_path,monkeypatch,failure=None):
    base=tmp_path/'local';base.mkdir();mirror=tmp_path/'mirror';mirror.mkdir();root=base/'campaign';root.mkdir()
    script=tmp_path/'fixture_worker.py';script.write_text(WORKER)
    models={'zero-source4':dict(geometry_group='zero-source4',kind='original-zero'),
        'zero-edited4':dict(geometry_group='zero-edited4',kind='original-zero'),
        'V1':dict(geometry_group='V1',kind='selected-trained',PBR_mode='vanilla'),
        'C_DECODER':dict(geometry_group='C_DECODER',kind='selected-trained',PBR_mode='retarget'),
        'SELECTED':dict(geometry_group='V1',kind='selected-trained',PBR_mode='raw')}
    path=base/'plan/context.json';path.parent.mkdir();path.write_text('{"test_fixture_only":true}')
    manifest=base/'manifests';manifest.mkdir()
    def close(folder,context=None,label=None):
        folder=Path(folder);files=[p for p in folder.rglob('*') if p.is_file() and p.name!='file4-closure.json'];entries=[]
        for p in files:
            entries.append(dict(relative_path=str(p.relative_to(base)),sha256=sha(p),bytes=p.stat().st_size))
            dst=mirror/p.relative_to(base);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
        (folder/'file4-closure.json').write_text(json.dumps(dict(status='PASS',local_equals_actual_file4_SHA256=True,files=entries)))
    close(path.parent)
    inputs={k:dict(path=str(base/k),SHA256='a'*64,receipt=str(base/'receipt.json')) for k in ['parent_freeze','control_freeze']}
    context=dict(name='fixture',repo=str(Path(__file__).resolve().parents[2]),python=sys.executable,canonical_local_root=str(base),
        campaign_root=str(root),manifest_directory=str(manifest),disk_reservations=str(base/'reservations.json'),workspace_bytes=1,
        assigned_GPUs=['0','6'],CPU_affinity=sorted(os.sched_getaffinity(0)),source_SHA256={},max_asset_groups=1,
        keeper_groups={},reader_queues=[],ssh_argv=['ssh','fixture-never-contacted'],canonical_remote_root='/fixture-not-contacted',
        closure_command=[sys.executable,str(base/'unused-helper')],source_inputs=inputs,case_budget_bytes={a:100000 for a in models},
        case_defaults=dict(fixture_models=models,fixture_failure=failure,CPU_affinity_per_GPU={'0':[92],'6':[93]},minimum_free_GPU_bytes=1),
        PBR_by_mode={m:dict(template={},input_SHA256_by_sample={sid:{} for sid in ['s1','s2']},closed_inputs_by_sample={sid:{} for sid in ['s1','s2']}) for m in ['vanilla','retarget','raw']})
    parent=dict(population=dict(sample_ids=['s1','s2']));engine=module.Campaign(context,path,ClosedInputs(base),parent,{},models)
    original=subprocess.Popen;calls=[]
    def popen(cmd,**kw):
        assert cmd[2]=='-m';kind='retirement' if cmd[3].endswith('retire_selected_final_raw') else 'case'
        calls.append((kind,json.load(open(cmd[-1]))))
        return original([sys.executable,str(script),cmd[-1],kind,str(base),str(mirror)],**kw)
    monkeypatch.setattr(module,'close',close);monkeypatch.setattr(module.subprocess,'Popen',popen)
    actual_sleep=time.sleep;monkeypatch.setattr(module.time,'sleep',lambda n:actual_sleep(min(n,.02)))
    return engine,calls,mirror


def test_two_assets_five_models_finish_with_a_one_asset_window_and_real_workers(tmp_path,monkeypatch):
    engine,calls,mirror=prepare(tmp_path,monkeypatch)
    try:result=engine.run()
    finally:engine.drain()
    assert result['assets']==2 and len(result['actual_cases'])==2 and all(len(v)==5 for v in result['actual_cases'].values())
    assert not engine.groups and not engine.own and not token_processes(engine.token)
    assert len([k for k,c in calls if k=='case'])==10
    assert len([k for k,c in calls if k=='retirement'])==14
    ledger=json.load(open(engine.context['disk_reservations']))['runs'];assert len(ledger)==12
    assert all(v['released'] is True for v in ledger.values())
    # Only canonical local duplicates of the tiny fixtures retire. The mirror,
    # native guides and case audits remain, including the shared-owner case.
    assert list(mirror.rglob('prediction.npz')) and not list((engine.root/'cases').rglob('prediction.npz'))
    assert list((engine.root/'cases').rglob('native-shape-guides.pt')) and len(list((engine.root/'cases').rglob('audit.json')))==10
    assert not result['all_ten_metrics_collected'] and not result['full_goal_complete']
    names=[(k,c['alias'],c.get('role')) for k,c in calls]
    for sid in ['s1','s2']:
        own=[c['alias'] for k,c in calls if k=='case' and c['sample_id']==sid]
        assert own[0]=='zero-source4' and own.index('V1') < own.index('SELECTED')


def test_failed_worker_with_new_session_orphan_keeps_raw_credit_until_real_drain(tmp_path,monkeypatch):
    engine,calls,mirror=prepare(tmp_path,monkeypatch,failure='zero-source4')
    try:
        with pytest.raises(ValueError,match='separately-sessioned phase'):engine.run()
        assert token_processes(engine.token)
    finally:engine.drain()
    assert not token_processes(engine.token)
    ledger=json.load(open(engine.context['disk_reservations']))['runs']
    assert all(v['released'] is False for v in ledger.values())
    assert list((engine.root/'cases').rglob('owned-orphan.json'))
    assert not list((engine.root/'groups').rglob('audit.json'))


def test_cli_context_rejects_a_fixture_instead_of_fabricating_a_508_freeze():
    with pytest.raises(ValueError):module.validate(dict(schema='test-fixture-only'))
