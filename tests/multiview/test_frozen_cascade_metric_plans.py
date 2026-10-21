"""Temporary metadata/pixels only: no canonical result or GPU score is made."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.deformnext_retargeting import prepare_frozen_cascade_metrics as factory
from experiments.deformnext_retargeting import bound_official_perceptual as bound
from experiments.deformnext_retargeting.cascade_view_roles import ClosedInputs
from experiments.deformnext_retargeting.evaluate_bound_population import sha


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    return factory.dump(path, value)


def close_unit(folder, base):
    # A unit-only simulation of the receipt schema; never written to file4.
    files=[dict(relative_path=str(p.relative_to(base)), bytes=p.stat().st_size, sha256=sha(p))
           for p in folder.rglob('*') if p.is_file() and p.name!='file4-closure.json']
    put(folder/'file4-closure.json',dict(status='PASS',local_equals_actual_file4_SHA256=True,files=files))


@pytest.fixture(params=['native-inline','native-replay'])
def completed_case(tmp_path, request):
    replay=request.param=='native-replay'
    source=put(tmp_path/'frozen-source.py',dict(synthetic_only=True))
    freeze=put(tmp_path/'freeze.json',dict(synthetic_only=True))
    ss=put(tmp_path/'SS'/'evaluation.json',dict(synthetic_only=True)); close_unit(tmp_path/'SS',tmp_path)
    guards=dict(model_freeze=freeze,SS_evaluations={'candidate':ss})
    root=tmp_path/'actual'; root.mkdir()
    folder=tmp_path/'plans'; folder.mkdir()
    phases=[]; terminals=[]
    names=['generation','PBR','render','shared'] if replay else ['generation','render','shared']
    modules={'generation':'cascade_from_ss_artifact','PBR':'replay_pbr_on_generated_shape',
             'render':'render_cascade_artifact_v2','shared':'score_cascade_fixed_reference'}
    for name in names:
        output=root/name; output.mkdir()
        native=put(folder/(name+'.json'),dict(output=str(output),SS_evaluation=ss['path']))
        remote={'generation':'generation','PBR':'PBR-replay','render':'render'}.get(name)
        phase=dict(name=name,module=factory.PACKAGE+modules[name],remote_stage=remote,
                   native_plan=native,native_source=source,execution_source=source,
                   result='generation.json' if name in ['generation','PBR'] else 'render-evaluation.json')
        phases.append(phase)
        if remote:
            IO=put(root/(name+'-IO.json'),dict(native_plan=native['path'],native_plan_SHA256=native['SHA256']))
            module='pbr_remote_results' if name=='PBR' else 'cascade_remote_results'
            actual_plan=IO['path']
        else:module=modules[name];actual_plan=native['path']
        terminal=dict(phase=name,native_source=source,native_plan=native,physical_GPU='0',returncode=0,
            command=['python','-m',factory.PACKAGE+module,'--plan',actual_plan])
        terminals.append(terminal);put(output/'terminal.json',terminal)
        put(output/phase['result'],dict(status='PASS_UNIT_ONLY_NOT_A_REAL_NATIVE_RESULT'))
        if name=='shared':
            arrays=dict(prediction=np.full((3,512,512),.2,np.float32),GT_shared_renderer=np.full((3,512,512),.8,np.float32),
                        prediction_alpha=np.ones((512,512),np.float32),GT_alpha=np.ones((512,512),np.float32))
            for cid in (8,9,10):np.savez_compressed(output/f'camera{cid:02d}.npz',**arrays)
        close_unit(output,tmp_path)
    put(root/'case-completed.json',dict(status='PASS_ACTUAL_FROZEN_NATIVE_CASCADE_CASE',alias='candidate',
        sample_id='unit-asset',source_cluster_id='unit-source',model_freeze=freeze,terminals=terminals))
    close_unit(root,tmp_path)
    case=put(folder/'case.json',dict(alias='candidate',sample_id='unit-asset',source_cluster_id='unit-source',
        model_freeze=freeze,source_inputs=guards,phases=phases))
    close_unit(folder,tmp_path)
    kwargs=dict(alias='candidate',model={'texture_protocol':request.param},source_inputs=guards,
        prepared_receipt=str(folder/'file4-closure.json'),SS_receipt=str(tmp_path/'SS'/'file4-closure.json'),closed=ClosedInputs(tmp_path),
        surface_references=SimpleNamespace(join=lambda *a,**k:{'unit_metadata_GT_join':True}))
    return case,kwargs,root,tmp_path


def test_completed_native_and_replay_cases_bind_actual_outputs_and_closed512_pixels(completed_case):
    spec,kwargs,root,base=completed_case
    row,images=factory.bind_case(spec,**kwargs)
    assert [v['camera_id'] for v in images]==[8,9,10]
    assert all(v['shared_npz_SHA256']==sha(v['shared_npz']) for v in images)
    assert all(v['image_audit']['original_float_prediction_SHA256']!=v['image_audit']['original_float_GT_SHA256'] for v in images)
    assert row['generation']==str(root/('PBR' if kwargs['model']['texture_protocol']=='native-replay' else 'generation')/'generation.json')
    if kwargs['model']['texture_protocol']=='native-replay':
        assert row['shape_generation']==str(root/'generation'/'generation.json')
        assert row['PBR_IO_binding']==str(root/'PBR-IO.json')
    for path,receipt in row['closed_inputs'].items():ClosedInputs(base).check(path,receipt)


@pytest.mark.parametrize('problem',['unfinished','bad-terminal','swapped-plan','changed-pixels','wrong-source','wrong-SS','missing-camera'])
def test_unfinished_or_mixed_native_case_cannot_become_metric_input(completed_case,problem):
    spec,kwargs,root,base=completed_case
    if problem=='unfinished':(root/'case-completed.json').unlink()
    elif problem=='changed-pixels':(root/'shared'/'camera08.npz').write_bytes(b'changed image')
    elif problem=='missing-camera':(root/'shared'/'camera10.npz').unlink()
    elif problem in ['bad-terminal','swapped-plan']:
        path=root/'render'/'terminal.json';v=json.loads(path.read_text())
        if problem=='bad-terminal':v['returncode']=1
        else:v['command'][-1]='/different-plan'
        put(path,v);close_unit(path.parent,base)
        completed=json.loads((root/'case-completed.json').read_text())
        completed['terminals'][-2]=v
        put(root/'case-completed.json',completed);close_unit(root,base)
    elif problem=='wrong-source':kwargs['source_inputs']=dict(model_freeze={'path':'different'},SS_evaluations={})
    else:
        path=Path(kwargs['source_inputs']['SS_evaluations']['candidate']['path']);put(path,dict(changed_SS=True));close_unit(path.parent,base)
    with pytest.raises((ValueError,FileNotFoundError)):factory.bind_case(spec,**kwargs)


def test_final_planner_rejects_development_before_loading_or_writing_outputs(tmp_path):
    with pytest.raises(ValueError,match='final-test508'):factory.prepare({'scope':'engineering-validation32'},tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_unassigned_perceptual_GPU_rejected_before_inputs_or_model_load(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','7');monkeypatch.setenv('MATCHING_V2_ALLOWED_GPUS','0,1,2,3')
    with pytest.raises(ValueError,match='assigned'):bound.run({'path':'/must-not-read'})


@pytest.fixture
def result_binding(tmp_path):
    row=dict(arm='candidate',sample_id='unit-asset',source_cluster_id='unit-source',camera_id=8,
             shared_npz_SHA256='unit-pixels',image_audit={'unit':True})
    spec={'path':'/unit-plan','SHA256':'unit-plan'}
    plan=dict(output=str(tmp_path),model_freeze={'unit':True},frozen_alias='candidate',evaluator_source={'unit-evaluator':True},
              wrapper_source={'unit-wrapper':True},metric_registry={'SHA256':'unit-registry'},inputs=[copy.deepcopy(row) for _ in range(1524)])
    report=dict(status='PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION',scope='final-test508',full_panel=True,
        fullframe_primary=True,metric_registry_SHA256='unit-registry',model_precision='float32,TF32off,noAMP',
        GPU_sanity={'status':'PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY'},per_view=[copy.deepcopy(row) for _ in range(1524)])
    report_spec=put(tmp_path/'evaluation.json',report)
    binding=dict(status='PASS_ACTUAL_BOUND_OFFICIAL_TEST508_PERCEPTUAL_METRICS',executed_plan=spec,
        model_freeze=plan['model_freeze'],alias='candidate',evaluator_source=plan['evaluator_source'],wrapper_source=plan['wrapper_source'],
        views=1524,physical_GPU='0',evaluation_SHA256=report_spec['SHA256'])
    return plan,spec,report,binding


@pytest.mark.parametrize('problem',[None,'subset','camera','model','pixels','crop','registry','precision','executed-plan','report-bytes'])
def test_completed_official_metrics_bind_every_model_camera_pixel_and_execution(result_binding,problem):
    plan,spec,report,binding=result_binding
    if problem=='subset':report['per_view'].pop()
    elif problem=='camera':report['per_view'][0]['camera_id']=9
    elif problem=='model':report['per_view'][0]['arm']='other'
    elif problem=='pixels':report['per_view'][0]['shared_npz_SHA256']='other'
    elif problem=='crop':report['per_view'][0]['image_audit']={'other':True}
    elif problem=='registry':report['metric_registry_SHA256']='other'
    elif problem=='precision':report['model_precision']='fp16'
    elif problem=='executed-plan':binding['executed_plan']={'path':'other'}
    elif problem=='report-bytes':binding['evaluation_SHA256']='other'
    if problem is None:bound.verify_result(plan,spec,report,binding)
    else:
        with pytest.raises(ValueError):bound.verify_result(plan,spec,report,binding)


@pytest.mark.parametrize('problem',[None,'subset','duplicate-image','camera-order','wrong-source','wrong-model'])
def test_full508_official_input_population_retains_every_ordered_model_camera(problem):
    ids=[f'unit-asset{i}' for i in range(508)]
    clusters={sid:f'unit-source{i%134}' for i,sid in enumerate(ids)}
    rows=[dict(arm='candidate',sample_id=sid,source_cluster_id=clusters[sid],camera_id=cid,
               shared_npz=f'/unit-only/{sid}/camera{cid}.npz') for sid in ids for cid in (8,9,10)]
    if problem=='subset':rows.pop()
    elif problem=='duplicate-image':rows[0]['shared_npz']=rows[1]['shared_npz']
    elif problem=='camera-order':rows[0],rows[1]=rows[1],rows[0]
    elif problem=='wrong-source':rows[0]['source_cluster_id']='different-source'
    elif problem=='wrong-model':rows[0]['arm']='different-model'
    plan=dict(frozen_alias='candidate',inputs=rows)
    if problem is None:bound.validate_observations(plan,ids=ids,clusters=clusters)
    else:
        with pytest.raises(ValueError):bound.validate_observations(plan,ids=ids,clusters=clusters)


@pytest.mark.parametrize('fail',[False,True])
def test_wrapper_calls_unchanged_official_entrypoint_and_preserves_failures(tmp_path,monkeypatch,fail):
    # Mocked control flow only; no CUDA or neural metric is executed.
    source=put(tmp_path/'synthetic-source',{'unit-source':True})
    plan=dict(output=str(tmp_path/'actual'),evaluator_source=source,wrapper_source=source,metric_registry=source,
              model_freeze={'unit':True},frozen_alias='candidate')
    spec=put(tmp_path/'plan.json',plan)
    calls=[]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0');monkeypatch.setenv('MATCHING_V2_ALLOWED_GPUS','0,1,2,3')
    monkeypatch.setattr(bound,'validate_inputs',lambda actual:calls.append(('validate',actual)))
    monkeypatch.setattr(bound,'verify_result',lambda *args:calls.append(('verify',args)))
    previous=bound.sys.argv
    def control(module,run_name):
        assert module=='experiments.deformnext_retargeting.evaluate_official_perceptual'
        assert run_name=='__main__' and bound.sys.argv==[source['path'],'--plan',spec['path']]
        calls.append(('native',module))
        output=Path(plan['output']);output.mkdir()
        if fail:
            (output/'preserved-failure.log').write_text('unit simulated native failure')
            raise RuntimeError('unit native failure')
        put(output/'evaluation.json',{'status':'UNIT_ONLY_NOT_REAL_METRICS'})
    monkeypatch.setattr(bound.runpy,'run_module',control)
    if fail:
        with pytest.raises(RuntimeError,match='unit native failure'):bound.run(spec)
        assert (Path(plan['output'])/'preserved-failure.log').exists()
        assert not (Path(plan['output'])/'binding-audit.json').exists()
        assert [c[0] for c in calls]==['validate','native']
    else:
        actual=bound.run(spec)
        assert actual['executed_plan']==spec
        assert [c[0] for c in calls]==['validate','native','validate','verify']
    assert bound.sys.argv is previous


@pytest.mark.parametrize('problem',[None,'missing-case','different-reference'])
def test_full508_metadata_assembly_and_completed_metric_binding_keep_both_zero_controls(tmp_path,monkeypatch,problem):
    # Lower case/pixel binding is tested above. This harness exercises full
    # population aggregation with unit-only lower-boundary observations.
    from test_frozen_ss_population_plans import fixture
    from experiments.deformnext_retargeting.perceptual_image_inputs import paired_images
    source=fixture(tmp_path)
    source['file4_cache_root']='/unit-cache/file4'
    freeze=json.loads(Path(source['model_freeze']['path']).read_text())
    ids=freeze['population']['sample_ids'];clusters=freeze['population']['source_clusters_by_sample']
    case_map={alias:[dict(sample_id=sid,plan=dict(path='/unit-descriptor/'+sid,SHA256='unit'))
                    for sid in ids] for alias in freeze['models']}
    if problem=='missing-case':case_map['candidate'].pop()
    compiled=dict(status='PREPARED_ALL_FROZEN_TEST508_NATIVE_CASCADE_CASES',scope='final-test508',
        source_inputs=source,model_freeze=source['model_freeze'],models=list(freeze['models']),cases=case_map,
        assets=508,source_clusters=134,total_cases=2540)
    compiled_root=tmp_path/'compiled';compiled_root.mkdir()
    compiled_spec=put(compiled_root/'manifest.json',compiled);close_unit(compiled_root,tmp_path)
    registry=put(tmp_path/'metric-models'/'registry.json',
        dict(status='PASS_OFFICIAL_PERCEPTUAL_WEIGHT_SOURCE_CACHE_AND_CPU_LOAD',files=[]))
    plan=dict(scope='final-test508',case_manifest=compiled_spec,prepared_cases_receipt=str(compiled_root/'file4-closure.json'),
        SS_evaluation_receipts={alias:'/unit-unused-SS-receipt' for alias in freeze['models']},
        canonical_local_root=str(tmp_path),metric_registry=registry,
        GT_surface_reference={'unit_reference':True},GT_surface_reference_receipt='/unit-unused-reference-receipt')
    _,audit=paired_images(np.zeros((3,512,512),np.float32),np.ones((3,512,512),np.float32),np.ones((512,512),np.float32))
    def lower(spec,*,alias,**kwargs):
        sid=spec['path'].removeprefix('/unit-descriptor/')
        row=dict(sample_id=sid,source_cluster_id=clusters[sid],unit_metadata_only=True)
        images=[dict(arm=alias,sample_id=sid,source_cluster_id=clusters[sid],camera_id=cid,
                     shared_npz=f'/unit-only/{alias}/{sid}/camera{cid}.npz',shared_npz_SHA256='unit-pixels',
                     image_audit=copy.deepcopy(audit)) for cid in (8,9,10)]
        if problem=='different-reference' and alias=='candidate':
            images[0]['image_audit']['original_float_GT_SHA256']='different-reference'
        return row,images
    monkeypatch.setattr(factory,'bind_case',lower)
    monkeypatch.setattr(factory,'SurfaceReferences',lambda *a,**k:SimpleNamespace(unit_only=True))
    out=tmp_path/'metric-prepared'
    if problem is not None:
        with pytest.raises(ValueError):factory.prepare(plan,out)
        assert not out.exists()
        return
    actual=factory.prepare(plan,out)
    assert actual['total_cases']==2540 and actual['total_perceptual_views']==7620
    assert list(actual['models'])==['v1','retarget','candidate','zero-source4','zero-edited']
    for alias,descriptor in actual['perceptuals'].items():
        metric=json.loads(Path(descriptor['plan']['path']).read_text())
        bound.validate_observations(metric,ids=ids,clusters=clusters)
        root=Path(descriptor['output']);root.mkdir()
        report=dict(status='PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION',scope='final-test508',full_panel=True,
            fullframe_primary=True,metric_registry_SHA256=registry['SHA256'],model_precision='float32,TF32off,noAMP',
            GPU_sanity={'status':'PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY'},
            per_view=[{k:v for k,v in row.items() if k!='shared_npz'} for row in metric['inputs']])
        report_spec=put(root/'evaluation.json',report)
        put(root/'binding-audit.json',dict(status='PASS_ACTUAL_BOUND_OFFICIAL_TEST508_PERCEPTUAL_METRICS',
            executed_plan=descriptor['plan'],model_freeze=source['model_freeze'],alias=alias,views=1524,physical_GPU='0',
            evaluator_source=metric['evaluator_source'],wrapper_source=metric['wrapper_source'],
            evaluation_SHA256=report_spec['SHA256']))
        close_unit(root,tmp_path)
    result=factory.bind_completed(dict(path=str(out/'manifest.json'),SHA256=sha(out/'manifest.json')))
    assert tuple(result['models'])==tuple(freeze['models']) and len(result['perceptuals'])==5
    assert result['cache_QA']==source['cache_QA'] and result['cache_QA_plan']==source['cache_QA_plan']
    assert result['GT_surface_reference']==plan['GT_surface_reference']


