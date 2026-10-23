"""Per-asset plan IO invariants; no model construction or neural execution."""
from copy import deepcopy
import hashlib

import pytest

from experiments.deformnext_retargeting.archived_native_guides import POLICY
from experiments.deformnext_retargeting.pipeline_population_contract_v2 import ARMS
from experiments.deformnext_retargeting.common_geometry_pbr_contract import MODES
from experiments.deformnext_retargeting.selected_pipeline_plan_factory import geometry_case,replay_case
from test_common_pipeline_contract import pbr
from test_selected_pipeline_execution import binding,index


def digest(value):return hashlib.sha256(str(value).encode()).hexdigest()
def spec(path):return dict(path=path,SHA256=digest(path),receipt='/closed/file4-closure.json')


def geometry_inputs(arm='E_LEARNED_RAW'):
    source={'/original/entry.py':digest('entry')}
    ready,_=pbr()
    case=dict(name='old',sample_id='0',source_cluster_id='cluster0',SS_arm='V1',output='/old/case',
        SS_evaluation='/old/V1/evaluation.json',SS_terminal='/old/V1/terminal.json',reference_SS='/old/V1/evaluation.json',
        geometry_plan='/old/geometry-plan.json',render_plan='/old/render.json',shared_plan='/old/shared.json',
        source_SHA256=source,input_SHA256={'/rgb.png':digest('rgb')},
        closed_inputs={p:'/old/file4-closure.json' for p in ['/old/V1/evaluation.json','/old/V1/terminal.json','/rgb.png']})
    geometry=dict(input_states=['M1','M0','M0','M0'],RGB_paths=['T','S1','S2','S3'],
        row=dict(sample_id='0',conditioning_camera_ids=[8,9,10,11],split='validation'),seed=42,surface_seed=101,
        source_geometry='/source.npz',target_vertices='/gt.npy',decoders={'shape':'original-decoder'},
        pipeline_config='/pipeline.json',DINO_local_path='/DINO',allocator_fraction=.4,
        texture_support_policy=POLICY,SS_evaluation=case['SS_evaluation'],output='/old/case/generation',
        stages=dict(shape512=dict(config='/shape512/config.json',checkpoint='/shape512/ckpts/denoiser_step0004210.pt'),
            shape1024=dict(config='/shape1024/config.json',checkpoint='/shape1024/ckpts/denoiser_step0016850.pt'),
            pbr1024=dict(config=ready['config'],checkpoint=ready['checkpoint'])),
        source_SHA256=deepcopy(source),inputs_SHA256={'/rgb.png':digest('rgb')},closed_inputs=deepcopy(case['closed_inputs']))
    render=dict(row={'sample_id':'0'},views=[8,9,10],AO_seed=101,generation_root='/old/case/generation',
        output='/old/case/render',prediction_render_face_budget=450000)
    shared=dict(sample_id='0',views=[8,9,10],prediction_render_root='/old/case/render',output='/old/case/shared',
        fixed_GT_shared_root='/GT/shared',fixed_GT_render_root='/GT/render')
    kw=dict(arm=arm,binding=binding(arm),evaluation=spec('/own/'+arm+'/evaluation.json'),
        terminal=spec('/own/'+arm+'/terminal.json'),artifact=spec('/own/'+arm+'/prediction.npz'),
        name='new',output='/new/case',paths={k:'/new/'+k+'.json' for k in ['case','geometry','render','shared']},
        ready=ready,source_SHA256=dict(source,**{'/new/guard.py':digest('guard')}))
    return case,geometry,render,shared,kw


@pytest.mark.parametrize('arm',ARMS)
def test_own_ss_geometry_rebind_preserves_all_native_inputs_and_stage_math(arm):
    c,g,r,s,kw=geometry_inputs(arm);before=deepcopy([c,g,r,s]);out=geometry_case(c,g,r,s,**kw)
    assert [c,g,r,s]==before
    for key in ['row','RGB_paths','input_states','seed','surface_seed','stages','decoders','allocator_fraction',
                'source_geometry','target_vertices','pipeline_config','DINO_local_path']:
        assert out['geometry'][key]==g[key]
    assert out['case']['SS_model_binding']==kw['binding']
    assert out['geometry']['SS_evaluation']==kw['evaluation']['path']
    assert out['case']['geometry_template']==c['geometry_plan']
    assert '/old/V1/terminal.json' not in out['case']['closed_inputs']
    assert c['reference_SS'] in out['case']['closed_inputs']
    for key in ['evaluation','terminal','artifact']:
        sp=kw[key];assert out['geometry']['inputs_SHA256'][sp['path']]==sp['SHA256']
    for key in ['views','AO_seed','prediction_render_face_budget']:assert out['render'][key]==r[key]
    assert out['shared']['fixed_GT_shared_root']==s['fixed_GT_shared_root']


@pytest.mark.parametrize('damage',['source-pin','same-output','relative-path','wrong-shape','wrong-camera','wrong-pbr',
    'other-sample','other-binding','missing-receipt','duplicate-plan'])
def test_changed_source_geometry_or_protocol_is_rejected(damage):
    c,g,r,s,kw=geometry_inputs()
    if damage=='source-pin':kw['source_SHA256']['/original/entry.py']=digest('changed')
    elif damage=='same-output':kw['output']=c['output']
    elif damage=='relative-path':kw['paths']['geometry']='relative.json'
    elif damage=='wrong-shape':g['stages']['shape512']['checkpoint']='/different-step.pt'
    elif damage=='wrong-camera':g['row']['conditioning_camera_ids']=[9,10,11]
    elif damage=='wrong-pbr':kw['ready']['checkpoint']='/other/ckpts/denoiser_step0001900.pt'
    elif damage=='other-sample':g['row']['sample_id']='other'
    elif damage=='other-binding':kw['binding']['module_arm']='FULL'
    elif damage=='missing-receipt':kw['artifact'].pop('receipt')
    else:kw['paths']['shared']=kw['paths']['render']
    with pytest.raises(ValueError):geometry_case(c,g,r,s,**kw)


def replay_inputs(arm='E_LEARNED_RAW',mode='raw-topk-plus-feature'):
    value=index((arm,mode));alias=value['prospective_selected_alias'];pb=value['compositions'][alias]['pbr1024']
    source={'/original/replay.py':digest('replay')};old_geometry='/old/V1/0/generation'
    case=dict(name='old',sample_id='0',source_cluster_id='cluster0',SS_arm='V1',mode=mode,
        stage_root='/PBR/'+mode,output='/old/replay',common_case='/old/V1/0',source_SHA256=source,
        input_SHA256={old_geometry+'/prediction.npz':digest('prediction'),pb['checkpoint']['path']:pb['checkpoint']['SHA256']},
        closed_inputs={old_geometry+'/'+f:old_geometry+'/file4-closure.json' for f in
            ['generation.json','prediction.npz','geometry-evaluation.npz','native-shape-guides.pt']})
    case['closed_inputs'].update({'/old/V1/0/audit.json':'/old/V1/0/file4-closure.json',
        '/fixedGT.npz':'/fixedGT/file4-closure.json',pb['checkpoint']['path']:'/PBR/file4-closure.json'})
    replay=dict(name='old-replay',mode=mode,checkpoint_kind='trained-native1900',engineering_only=False,
        row={'sample_id':'0'},input_states=['M1','M0','M0','M0'],RGB_paths=['T','S1','S2','S3'],
        PBR_replay_seed=20261020,allocator_fraction=.75,texture_support_policy=POLICY,
        PBR_config=pb['native_config']['path'],PBR_checkpoint=pb['checkpoint']['path'],
        conditioner_checkpoint=None if pb['conditioner_checkpoint'] is None else pb['conditioner_checkpoint']['path'],
        generated_shape_root=old_geometry,native_guides=old_geometry+'/native-shape-guides.pt',
        output='/old/replay/generation',source_SHA256=deepcopy(source),
        input_SHA256=deepcopy(case['input_SHA256']),closed_inputs=deepcopy(case['closed_inputs']),
        sampler_params=dict(steps=12,guidance=3.))
    render=dict(generation_root='/old/replay/generation',output='/old/replay/render',AO_seed=101,views=[8,9,10])
    shared=dict(sample_id='0',prediction_render_root='/old/replay/render',output='/old/replay/shared',
        fixed_GT_shared_root='/fixedGT/shared',fixed_GT_render_root='/fixedGT/render')
    kw=dict(arm=arm,ss_binding=binding(arm),pbr_binding=pb,evaluation=spec('/SS/'+arm+'/evaluation.json'),
        reference=spec('/SS/V1/evaluation.json'),base_case=dict(sample_id='0',source_cluster_id='cluster0',
            case_output='/base/'+arm+'/0',generation='/base/'+arm+'/0/generation'),name='new-replay',
        output='/new/replay',paths={k:'/new/'+k+'.json' for k in ['case','replay','render','shared']},
        source_SHA256=dict(source,**{'/new/guard.py':digest('guard')}))
    return case,replay,render,shared,kw


@pytest.mark.parametrize('arm',ARMS)
@pytest.mark.parametrize('mode',MODES)
def test_every_physical_pair_reuses_only_its_own_geometry_and_original_sampler(arm,mode):
    c,p,r,s,kw=replay_inputs(arm,mode);before=deepcopy([c,p,r,s]);out=replay_case(c,p,r,s,**kw)
    assert [c,p,r,s]==before
    for key in ['PBR_config','PBR_checkpoint','conditioner_checkpoint','RGB_paths','row','input_states',
        'PBR_replay_seed','sampler_params','allocator_fraction','texture_support_policy']:
        assert out['replay'][key]==p[key]
    assert out['replay']['generated_shape_root']==kw['base_case']['generation']
    assert out['case']['SS_model_binding']==kw['ss_binding'] and out['case']['PBR_model_binding']==kw['pbr_binding']
    assert '/old/V1/0/audit.json' not in out['case']['closed_inputs']
    assert '/old/V1/0/generation/prediction.npz' not in out['replay']['input_SHA256']
    assert '/fixedGT.npz' in out['case']['closed_inputs']
    assert out['shared']['fixed_GT_shared_root']==s['fixed_GT_shared_root']


@pytest.mark.parametrize('damage',['source-pin','other-PBR','other-source','old-seed','other-mode',
    'GT-geometry','other-sample','other-cluster','same-output','other-SS','untrained'])
def test_wrong_model_geometry_or_native_protocol_cannot_create_replay_plans(damage):
    c,p,r,s,kw=replay_inputs()
    if damage=='source-pin':kw['source_SHA256'].pop('/original/replay.py')
    elif damage=='other-PBR':p['PBR_checkpoint']='/other/ckpts/denoiser_step0001900.pt'
    elif damage=='other-source':p['conditioner_checkpoint']='/other/source.pt'
    elif damage=='old-seed':p['PBR_replay_seed']=101
    elif damage=='other-mode':c['mode']='vanilla'
    elif damage=='GT-geometry':kw['base_case']['generation']='/GT/generation'
    elif damage=='other-sample':kw['base_case']['sample_id']='other'
    elif damage=='other-cluster':kw['base_case']['source_cluster_id']='other'
    elif damage=='same-output':kw['output']=c['output']
    elif damage=='other-SS':kw['ss_binding']['source_tensors']=19
    else:p['engineering_only']=True
    with pytest.raises(ValueError):replay_case(c,p,r,s,**kw)
