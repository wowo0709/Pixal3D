"""Synthetic closed metadata and process handles only; no neural execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting import prepare_selected_confirmation as prepare
from experiments.deformnext_retargeting.cascade_view_roles import ClosedInputs
from experiments.deformnext_retargeting.selected_confirmation_runtime import OBSERVER_FILES,load_observer
from test_module_confirmation_recipe import inputs as configs
from test_selected_confirmation_runtime import reports


class Files:
    def __init__(self,root):
        self.root=root;self.receipt=root/'file4-closure.json';self.entries=[]
    def add(self,path,value,lines=False):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(row)+'\n' for row in value) if lines else json.dumps(value))
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        self.entries.append(dict(relative_path=str(path.relative_to(self.root)),sha256=digest,bytes=path.stat().st_size))
        return dict(path=str(path),SHA256=digest,receipt=str(self.receipt))
    def close(self):
        self.receipt.write_text(json.dumps(dict(status='PASS',local_equals_actual_file4_SHA256=True,files=self.entries)))


def trace():
    return [dict(step=i+1,asset_ids=[['train']*96],x_t_SHA256='noise'+str(i),t_SHA256='t',global_SHA256='global')
        for i in range(7)]


def test_reads_stable_first_five_without_hashing_or_reusing_appending_tail(tmp_path):
    p=tmp_path/'trace.jsonl';rows=trace()
    p.write_text(''.join(json.dumps(row)+'\n' for row in rows[:5])+'{"step":6')
    first=prepare.first_five(p)
    p.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    assert prepare.first_five(p)==first
    assert [r['step'] for r in map(json.loads,first.splitlines())]==[1,2,3,4,5]


@pytest.mark.parametrize('damage',['four-records','partial-fifth','wrong-order','micro48','missing-noise'])
def test_never_relabels_incomplete_or_different_trace_as_native_five_updates(tmp_path,damage):
    rows=trace()[:5]
    if damage=='four-records':rows.pop()
    elif damage=='wrong-order':rows[1]['step']=1
    elif damage=='micro48':rows[0]['asset_ids'][0]=['train']*48
    elif damage=='missing-noise':rows[0].pop('x_t_SHA256')
    text=''.join(json.dumps(row)+'\n' for row in rows)
    if damage=='partial-fifth':text=text.rstrip('\n')
    p=tmp_path/'trace.jsonl';p.write_text(text)
    with pytest.raises(ValueError):prepare.first_five(p)


@pytest.mark.parametrize('damage',[None,'birth','command','zombie','config-path'])
def test_live_reference_requires_actual_pid_birth_and_command(tmp_path,monkeypatch,damage):
    recipe=tmp_path/'recipe.json';command=['python','-m','original.runtime','--config',str(recipe)]
    active=dict(PID=1234,birth=123.5,command=command)
    (tmp_path/'active.json').write_text(json.dumps(active))
    actual=SimpleNamespace(pid=1234,create_time=lambda:124. if damage=='birth' else 123.5,
        cmdline=lambda:['other'] if damage=='command' else command,
        status=lambda:'zombie' if damage=='zombie' else 'running')
    monkeypatch.setattr(prepare.psutil,'Process',lambda pid:actual)
    if damage=='config-path':recipe=tmp_path/'other.json'
    if damage:
        with pytest.raises(ValueError):prepare.actual_owner(tmp_path,recipe)
    else:
        result=prepare.actual_owner(tmp_path,recipe)
        assert result['kind']=='live_actual_reference_observed' and result['full_training_completion_not_claimed']


def prefix_inputs(tmp_path):
    _,cfg=configs(tmp_path,'V1')
    native=tmp_path/'native';native.mkdir();cfg['output_dir']=str(native)
    files=Files(tmp_path);recipe=files.add(tmp_path/'recipe.json',cfg);files.close()
    (native/'config.json').write_text(json.dumps(cfg))
    startup,_=reports('V1');(native/'startup.json').write_text(json.dumps(startup))
    (native/'training-noise-trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace()))
    plan=dict(schema='actual-V1-seed202-prefix-capture-inputs-v1',reference_recipe=recipe,
        reference_wrapper=str(tmp_path/'wrapper'),output=str(tmp_path/'prefix'))
    return plan,native,files


def test_capture_copies_actual_config_startup_and_prefix_without_weights(tmp_path,monkeypatch):
    plan,native,files=prefix_inputs(tmp_path)
    monkeypatch.setattr(prepare,'actual_owner',lambda *args:dict(kind='live_actual_reference_observed'))
    out=prepare.capture_reference_prefix(plan,ClosedInputs(tmp_path))
    assert (out/'startup.json').read_bytes()==(native/'startup.json').read_bytes()
    assert (out/'config.json').read_bytes()==(native/'config.json').read_bytes()
    assert (out/'training-noise-trace.jsonl').read_bytes()==prepare.first_five(native/'training-noise-trace.jsonl')
    report=json.loads((out/'audit.json').read_text())
    assert report['training_seed']==202 and report['actual_file4_closure_still_required']
    assert not report['scientific_weights_copied'] and not report['full_training_completion_claimed']
    with pytest.raises(FileExistsError):prepare.capture_reference_prefix(plan,ClosedInputs(tmp_path))


@pytest.mark.parametrize('damage',['seed','config','warm-start','short-trace','owner-missing'])
def test_capture_rejects_wrong_reference_before_writing_a_snapshot(tmp_path,monkeypatch,damage):
    plan,native,files=prefix_inputs(tmp_path)
    def owner(*args):
        if damage=='owner-missing':raise ProcessLookupError('not live')
        return dict(kind='live_actual_reference_observed')
    monkeypatch.setattr(prepare,'actual_owner',owner)
    if damage=='seed':
        cfg=json.loads((tmp_path/'recipe.json').read_text());cfg['training_seed']=101
        plan['reference_recipe']=files.add(tmp_path/'recipe101.json',cfg);files.close()
    elif damage=='config':
        cfg=json.loads((native/'config.json').read_text());cfg['trainer']['args']['ffn_chunk_assets']=12
        (native/'config.json').write_text(json.dumps(cfg))
    elif damage=='warm-start':
        startup=json.loads((native/'startup.json').read_text());startup['actual_optimizer_states']=360
        (native/'startup.json').write_text(json.dumps(startup))
    elif damage=='short-trace':(native/'training-noise-trace.jsonl').write_text('')
    with pytest.raises((ValueError,ProcessLookupError)):prepare.capture_reference_prefix(plan,ClosedInputs(tmp_path))
    assert not Path(plan['output']).exists()


def admission_inputs(tmp_path):
    original,cfg=configs(tmp_path,'FULL')
    cfg['experiment'].update(assigned_GPUs=['6','7'],CPU_affinity_per_GPU={'6':[76,77],'7':[78,79]})
    reference_folder=tmp_path/'reference';reference_folder.mkdir()
    _,ref_cfg=configs(reference_folder,'V1')
    _,values=reports('FULL');values['reference_config']=ref_cfg
    files=Files(tmp_path)
    plan=dict(schema='actual-selected-SS202-admission-inputs-v1',module_arm='FULL',
        canonical_local_root=str(tmp_path),output=str(tmp_path/'admission'))
    plan['original_science_config']=files.add(tmp_path/'original.json',original)
    plan['own_science_config']=files.add(tmp_path/'own202.json',cfg)
    for key in OBSERVER_FILES:plan[key]=files.add(tmp_path/(key+('.jsonl' if key.endswith('_trace') else '.json')),
        values[key],lines=key.endswith('_trace'))
    files.close()
    return plan,cfg,files


def test_real_byte_checked_inputs_produce_runner_readable_admission_after_closure(tmp_path):
    plan,cfg,files=admission_inputs(tmp_path)
    output=prepare.prepare_admission(plan,ClosedInputs(tmp_path))
    observer=output/'observer-plan.json'
    # The new artifact has no actual receipt yet; the numerical runner refuses it.
    with pytest.raises(FileNotFoundError):load_observer(observer,cfg,'FULL')
    value=json.loads(observer.read_text());files.add(observer,value);files.close()
    (output/'file4-closure.json').write_bytes(files.receipt.read_bytes())
    values,check=load_observer(observer,cfg,'FULL')
    assert len(values['reference_trace'])==5 and len(check.inputs)==len(OBSERVER_FILES)+1
    audit=json.loads((output/'audit.json').read_text())
    assert audit['actual_selected_science_still_required'] and not audit['new_training_queued']


@pytest.mark.parametrize('damage',['reference-noise','source-weights','receipt-bytes','missing-proof','different-recipe'])
def test_bad_evidence_does_not_create_an_admission(tmp_path,damage):
    plan,cfg,files=admission_inputs(tmp_path)
    if damage=='missing-proof':Path(plan['own_proof']['path']).unlink()
    else:
        key={'reference-noise':'reference_capacity_trace','source-weights':'own_capacity',
            'receipt-bytes':'own_proof','different-recipe':'own_science_config'}[damage]
        path=Path(plan[key]['path'])
        if damage=='receipt-bytes':path.write_text('{}')
        elif damage=='reference-noise':
            rows=[json.loads(line) for line in path.read_text().splitlines()];rows[0]['t_SHA256']='other'
            plan[key]=files.add(tmp_path/'new-trace.jsonl',rows,lines=True);files.close()
        else:
            value=json.loads(path.read_text())
            if damage=='source-weights':value['initial_source_SHA256']='warm'
            else:value['trainer']['args']['ffn_chunk_assets']=12
            plan[key]=files.add(tmp_path/'new-value.json',value);files.close()
    with pytest.raises((ValueError,FileNotFoundError)):prepare.prepare_admission(plan,ClosedInputs(tmp_path))
    assert not Path(plan['output']).exists()
