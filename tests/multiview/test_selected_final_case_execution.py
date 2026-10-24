"""Closed synthetic metadata IO and real CPU child ownership; no neural forward."""
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting.cascade_view_roles import ClosedInputs,file_sha256 as sha
from experiments.deformnext_retargeting.frozen_cascade_metrics import PinnedFiles
from experiments.deformnext_retargeting.run_selected_final_case import SCHEMA,load_case,required_sources,run_child,validate_execution
from experiments.deformnext_retargeting.selected_final_phase import SCHEMA as PHASE_SCHEMA,ENTRIES,run_original,validate_binding
from experiments.deformnext_retargeting.selected_final_population import compose_models
from test_selected_final_case_plans import inputs
from test_selected_final_population import evaluation


REPO=Path(__file__).resolve().parents[2]


def context(tmp_path):
    helper=tmp_path/'close.py';helper.write_text('# synthetic closure helper: not executed\n')
    return dict(schema=SCHEMA,name='synthetic-final-case',repo=str(REPO),python=sys.executable,
        canonical_local_root=str(tmp_path),output=str(tmp_path/'results/SELECTED'),assigned_GPUs=['0','6'],
        CPU_affinity_per_GPU={'0':[84,85],'6':[76,77]},manifest_directory=str(tmp_path/'manifests'),
        disk_reservations=str(tmp_path/'reservations.json'),minimum_free_GPU_bytes=1,raw_budget_bytes=100,workspace_bytes=100,
        closure_command=[sys.executable,str(helper)],source_SHA256={p:sha(p) for p in [*required_sources(),str(helper)]})


def prepare(tmp_path,arm='E_LEARNED_RAW',mode='top1-only',alias=None):
    # Frozen model bindings are synthetic and intentionally keep their fixture
    # weight paths. RecordingPins substitutes ONLY those missing model files;
    # all emitted metadata, receipt hashes and SS archive bytes are read for real.
    ctx=context(tmp_path);args=inputs(tmp_path,arm,mode,alias=alias);parent=args['parent'];controls=args['controls']
    models=compose_models(parent,controls);alias=args['alias'];ctx.update(alias=alias,sample_id='test0')
    out=tmp_path/'results'/alias;ctx['output']=str(out)
    reference=tmp_path/'results/zero-source4';ctx.update(reference_shared_root=str(reference/'shared'),reference_render_root=str(reference/'render'))
    receipt=tmp_path/'file4-closure.json'
    def emit(path,value,lines=False):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in value) if lines else json.dumps(value))
        return dict(path=str(path),SHA256=sha(path),receipt=str(receipt))
    ids=parent['population']['sample_ids'];clusters=parent['population']['source_clusters_by_sample']
    rows=[dict(sample_id=sid,source_cluster_id=clusters[sid],split='test') for sid in ids];rows[0]=args['row']
    manifest=emit(tmp_path/'data/test.jsonl',rows,lines=True);parent['population']['manifest']={k:manifest[k] for k in ['path','SHA256']}
    qa=dict(status='PASS_ACTUAL_TEST508_NATIVE_CACHE_AND_LOADER_QA',assets=508,SS_PBR_cache_ready=True,no_test_asset_dropped=True,
        train_validation_identity_disjoint=True,all_input_RGB_fingerprints_verified=True,no_candidate_model_forward_or_test_quality_metric=True,
        matching_head_training_or_calibration_updates=0,Omega_completed_updates=23597,published_manifest=manifest['path'],published_manifest_SHA256=manifest['SHA256'])
    qa_spec=emit(tmp_path/'data/QA.json',qa);parent['development_evidence_roles']['cache_QA']={k:qa_spec[k] for k in ['path','SHA256']}
    parent['development_evidence']=list(parent['development_evidence_roles'].values())
    qa_plan=emit(tmp_path/'data/QA-plan.json',dict(published_manifest=manifest['path']))
    assets=[dict(sample_id=sid) for sid in ids];assets[0]=args['asset']
    packet=emit(tmp_path/'data/assets.json',dict(schema='native-final-cascade-asset-inputs-v1',assets=508,
        source_clusters=134,records=assets,file4_cache_root='/cache'))
    template_specs={k:emit(tmp_path/'templates'/(k+'.json'),args[k+'_template']) for k in ['geometry','render','shared']}
    frozen=parent['evaluation_protocol'];frozen['source_SHA256'].update(ctx['source_SHA256'])
    frozen['generation_template']={k:template_specs['geometry'][k] for k in ['path','SHA256']}
    frozen['final_case_templates']={k:{a:template_specs[k][a] for a in ['path','SHA256']} for k in ['render','shared']}
    frozen['final_case_templates']['PBR_by_mode']={}
    inputs_map=dict(manifest=manifest,cache_QA=qa_spec,cache_QA_plan=qa_plan,asset_packet=packet,
        **{k+'_template':v for k,v in template_specs.items()})
    pins=PinnedFiles();overrides={};fake_checked=[]
    if models[alias]['kind']=='selected-trained':
        pbr_spec=emit(tmp_path/'templates/PBR.json',args['PBR_template']);inputs_map['PBR_template']=pbr_spec
        frozen['final_case_templates']['PBR_by_mode'][mode]={k:pbr_spec[k] for k in ['path','SHA256']}
        cfg=tmp_path/'synthetic-PBR-config.json';cfg.write_text(json.dumps(args['PBR_config']))
        overrides[models[alias]['pbr1024']['native_config']['path']]=cfg
        ctx.update(PBR_input_SHA256=args['PBR_inputs'],PBR_closed_inputs=args['PBR_closed_inputs'])
    class RecordingPins:
        def check(self,value):
            p=Path(value['path'])
            if str(p) in overrides:return overrides[str(p)]
            if p.exists():return pins.check(value)
            fake_checked.append(value);return p
    ss_specs={};groups={}
    for name,model in models.items():
        group=model['geometry_group']
        if group not in groups:
            folder=tmp_path/'SS'/group;folder.mkdir(parents=True)
            art=folder/'ss-artifacts/0.npz';art.parent.mkdir();art.write_bytes(b'synthetic SS artifact, not a GPU result')
            record=evaluation(model,parent,name);record['native_SS_artifacts']['artifacts'][0].update(artifact_SHA256=sha(art),prediction_voxels_SHA256='b'*64)
            groups[group]=emit(folder/'evaluation.json',record)
        ss_specs[name]=groups[group]
    inputs_map['SS_evaluations']=ss_specs
    parent_spec=emit(tmp_path/'frozen/parent.json',parent);control_spec=emit(tmp_path/'frozen/controls.json',controls)
    inputs_map.update(parent_freeze=parent_spec,control_freeze=control_spec)
    freeze_fields=dict(parent_freeze={k:parent_spec[k] for k in ['path','SHA256']},control_freeze={k:control_spec[k] for k in ['path','SHA256']})
    if alias!='zero-source4':
        material=dict(status='PASS_ACTUAL_SELECTED_PHYSICAL_TEST508_CASE',alias='zero-source4',sample_id='test0',
            source_cluster_id=clusters['test0'],actual_completed_native_case=True,case_output=str(reference),terminals=[],**freeze_fields)
        inputs_map['material_reference']=emit(reference/'audit.json',material)
        for role in ['shared','render']:
            emit(reference/role/'render-evaluation.json',dict(synthetic=True))
            for cid in [8,9,10]:(reference/role/f'camera{cid:02d}.npz').write_bytes(b'synthetic camera archive')
    first=next(name for name,m in models.items() if m['geometry_group']==models[alias]['geometry_group'])
    if first!=alias:
        owner=dict(status='PASS_ACTUAL_SELECTED_PHYSICAL_TEST508_CASE',alias=first,sample_id='test0',actual_completed_native_case=True,
            geometry_source_root=str(tmp_path/'results'/first/'geometry'),**freeze_fields)
        inputs_map['geometry_owner']=emit(tmp_path/'results'/first/'audit.json',owner)
    files=[p for p in tmp_path.rglob('*') if p.is_file()]
    payload=dict(status='PASS',local_equals_actual_file4_SHA256=True,
        files=[dict(relative_path=str(p.relative_to(tmp_path)),sha256=sha(p),bytes=p.stat().st_size) for p in files])
    receipt.write_text(json.dumps(payload))
    if alias!='zero-source4':
        for role in ['shared','render']:(reference/role/'file4-closure.json').write_text(json.dumps(payload))
    ctx['source_inputs']=inputs_map
    return ctx,ClosedInputs(tmp_path),RecordingPins(),fake_checked


@pytest.mark.parametrize('arm,mode,alias',[
    ('E_LEARNED_RAW','top1-only',None),('FULL_QK_ONLY','vanilla',None),('V1','top1-only',None),
    ('E_LEARNED_RAW','top1-only','zero-source4'),('E_LEARNED_RAW','top1-only','zero-edited4')])
def test_real_closed_metadata_derives_own_full_population_and_native_plans(tmp_path,arm,mode,alias):
    ctx,check,pins,fake=prepare(tmp_path,arm,mode,alias)
    parent,controls,result,case,owner=load_case(ctx,check,pins)
    assert len(result['per_asset'])==508 and len(parent['population']['sample_ids'])==508
    assert case['alias']==ctx['alias'] and case['SS_evaluation']['path']==ctx['source_inputs']['SS_evaluations'][ctx['alias']]['path']
    assert bool(case['geometry_reference'])==(owner is not None)
    assert all(p['plan']['assigned_GPUs']==ctx['assigned_GPUs'] for p in case['phases'])
    assert fake  # This test does not pretend to verify actual neural weights.


@pytest.mark.parametrize('damage',['SS-bytes','template-bytes','metadata-bytes','unfrozen-source','wrong-model','other-reference','other-owner'])
def test_actual_metadata_or_artifact_changes_fail_before_any_native_child(tmp_path,damage):
    ctx,check,pins,_=prepare(tmp_path,'V1','top1-only' if damage=='other-owner' else 'vanilla')
    if damage=='SS-bytes':
        path=Path(ctx['source_inputs']['SS_evaluations'][ctx['alias']]['path']).parent/'ss-artifacts/0.npz';path.write_bytes(b'changed')
    elif damage=='template-bytes':Path(ctx['source_inputs']['render_template']['path']).write_text('{}')
    elif damage=='metadata-bytes':Path(ctx['source_inputs']['parent_freeze']['path']).write_text('{}')
    elif damage=='unfrozen-source':ctx['source_SHA256'][str(REPO/'new-not-frozen.py')]='a'*64
    elif damage=='wrong-model':ctx['alias']='not-the-selected-model'
    elif damage=='other-reference':ctx['reference_shared_root']=str(tmp_path/'wrong-reference')
    else:
        # Use a valid receipt for the foreign owner to test its content guard,
        # rather than only demonstrating a corrupt-byte rejection.
        foreign=Path(ctx['source_inputs']['geometry_owner']['path']);data=json.loads(foreign.read_text());data['sample_id']='test1'
        foreign.write_text(json.dumps(data));ctx['source_inputs']['geometry_owner']['SHA256']=sha(foreign)
        receipt=Path(ctx['source_inputs']['geometry_owner']['receipt']);payload=json.loads(receipt.read_text())
        row=next(x for x in payload['files'] if x['relative_path']==str(foreign.relative_to(tmp_path)));row.update(sha256=sha(foreign),bytes=foreign.stat().st_size);receipt.write_text(json.dumps(payload))
    with pytest.raises((ValueError,KeyError)):load_case(ctx,check,pins)


@pytest.mark.parametrize('damage',[None,'unassigned-GPU','wrong-repo','output-outside','missing-source','bad-closure','relative-path','no-memory','duplicate-GPU','bad-CPU'])
def test_explicit_source_device_disk_and_memory_admission(tmp_path,damage):
    ctx=context(tmp_path);gpu='6'
    if damage=='unassigned-GPU':gpu='7'
    elif damage=='wrong-repo':ctx['repo']='/elsewhere'
    elif damage=='output-outside':ctx['output']='/elsewhere/output'
    elif damage=='missing-source':ctx['source_SHA256'].pop(required_sources()[0])
    elif damage=='bad-closure':ctx['closure_command'][0]='/other/python'
    elif damage=='relative-path':ctx['disk_reservations']='relative.json'
    elif damage=='no-memory':ctx['minimum_free_GPU_bytes']=0
    elif damage=='duplicate-GPU':ctx['assigned_GPUs']=['6','6']
    elif damage=='bad-CPU':ctx['CPU_affinity_per_GPU']['6']=[76,76]
    if damage:
        with pytest.raises(ValueError):validate_execution(ctx,gpu=gpu,allowed='0,6,7')
    else:validate_execution(ctx,gpu=gpu,allowed='0,6,7')


@pytest.mark.parametrize('kind',list(ENTRIES))
def test_each_phase_is_bound_to_exact_existing_numerical_entry(kind):
    path=REPO/'experiments/deformnext_retargeting'/ENTRIES[kind];path=path.with_suffix('.py')
    adapter=path.with_name('selected_final_phase.py');native=dict(source_SHA256={str(path):sha(path),str(adapter):sha(adapter)})
    binding=dict(schema=PHASE_SCHEMA,kind=kind,native_source=dict(path=str(path),SHA256=sha(path)),
        native_plan=dict(path='/actual/native.json',SHA256='a'*64),phase_source_SHA256=sha(adapter),
        parent_freeze={},control_freeze={},SS_evaluation={})
    assert validate_binding(binding,native)==(kind,path)
    binding['native_source']['SHA256']='f'*64
    with pytest.raises(ValueError):validate_binding(binding,native)


@pytest.mark.parametrize('fails',[False,True])
def test_legacy_render_adapter_restores_argv_and_changes_only_GPU_literal(tmp_path,monkeypatch,fails):
    from experiments.deformnext_retargeting import selected_final_phase as module
    source=tmp_path/'original.py';source.write_text("def main():\n    gpu='6'\n    if gpu not in {'0','1','2','3'}:\n        raise ValueError('unassigned')\n    events.append('original-body')\n"+("    raise RuntimeError('original-failure')\n" if fails else ''))
    events=[];fake=SimpleNamespace(events=events);monkeypatch.setattr(module.importlib,'import_module',lambda _:fake)
    before=source.read_bytes();argv=sys.argv
    binding=dict(native_source=dict(path=str(source)))
    if fails:
        with pytest.raises(RuntimeError):run_original('assigned-native-render',source,dict(assigned_GPUs=['6']),binding)
    else:
        proof=run_original('assigned-native-render',source,dict(assigned_GPUs=['6']),binding)
        assert proof['numerical_entry_AST_device_literal_only']['all_other_AST_statements_unchanged']
    assert sys.argv is argv and source.read_bytes()==before and events==['original-body']


@pytest.mark.parametrize('kind',['native-guide-capture','native-guide-replay'])
def test_existing_guide_wrapper_receives_exact_capture_or_replay_arguments(tmp_path,monkeypatch,kind):
    from experiments.deformnext_retargeting import native_guide_entry
    calls=[];monkeypatch.setattr(native_guide_entry,'main',lambda:calls.append(sys.argv.copy()))
    argv=sys.argv;path=tmp_path/'native.json';run_original(kind,path,{},{});
    assert sys.argv is argv and calls==[['native_guide_entry','--kind','capture' if kind.endswith('capture') else 'replay','--plan',str(path)]]


def test_owned_child_does_not_release_before_its_real_grandchild_exits(tmp_path):
    ctx=dict(repo=str(REPO));started=time.monotonic()
    code="import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import time;time.sleep(.7)']);time.sleep(.1)"
    terminal=run_child(ctx,tmp_path,'geometry',[sys.executable,'-c',code])
    assert time.monotonic()-started>=.7 and terminal['actual_session_drained'] and terminal['owner_birth_verified']
    assert terminal['returncode']==0 and (tmp_path/'geometry-terminal.json').is_file()


def test_failed_actual_child_is_preserved_without_automatic_retry(tmp_path):
    before=signal.getsignal(signal.SIGTERM)
    with pytest.raises(ValueError):run_child(dict(repo=str(REPO)),tmp_path,'PBR',
        [sys.executable,'-c','import time;print("single actual attempt",flush=True);time.sleep(.1);raise SystemExit(7)'])
    terminal=json.loads((tmp_path/'PBR-terminal.json').read_text())
    assert terminal['returncode']==7 and terminal['actual_session_drained']
    assert (tmp_path/'PBR.log').read_text().count('single actual attempt')==1
    assert signal.getsignal(signal.SIGTERM) is before
