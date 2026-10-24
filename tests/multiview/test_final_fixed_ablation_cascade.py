"""Own model admission, same native templates and paired camera-role metrics."""
from copy import deepcopy

import pytest

from experiments.deformnext_retargeting import final_fixed_ablation_cascade as module
from test_fixed_correspondence_pipeline_contract import inputs as matching_inputs
from test_shared_ablation_pipeline_contract import fixture as delivery_inputs


def own_binding(tmp_path, arm):
    return (matching_inputs(tmp_path, arm)[0] if arm.startswith('M_')
            else delivery_inputs(arm)[0])


def pbr():
    root = '/PBR/native'
    result = dict(schema='native-cold-PBR1900-frozen-binding-v1', mode='top1-only',
        step=1900, training_seed=101, source_tensors=10)
    for key, suffix in [('native_config','config.json'),('checkpoint','ckpts/denoiser_step0001900.pt'),
                        ('conditioner_checkpoint','ckpts/deform_conditioner_step0001900.pt')]:
        result[key] = dict(path=root+'/'+suffix, SHA256='d'*64)
    return result


def inputs(tmp_path, arm='A_FIXED_UNIFORM'):
    source = {'/source/native.py':'a'*64}
    row = dict(sample_id='s0', source_cluster_id='c0', split='validation',
        conditioning_camera_ids=[8,9,10,11], target_anchor_kind='rgb',
        target_anchor='/data/8.png', target_rgb_input='/data/8.png', source_images=['s9','s10','s11'])
    paths = ['/data/8.png','/data/9.png','/data/10.png','/data/11.png']
    asset = dict(sample_id='s0', source_cluster_id='c0', row=row, RGB_paths=paths,
        inputs={str(i):dict(path=p,SHA256='b'*64) for i,p in enumerate(paths)},
        render_views=[dict(camera_id=i,GT_RGBA=paths[i-8]) for i in [8,9,10]])
    asset['inputs'].update(source_geometry=dict(path='/data/source.npz',SHA256='b'*64),
                           target_vertices=dict(path='/data/target.npy',SHA256='b'*64))
    geometry = dict(RGB_paths=['/old/input.png'], SS_evaluation='/old/SS/evaluation.json',
        source_geometry='/old/source.npz',target_vertices='/old/target.npy',
        input_states=['M1','M0','M0','M0'], stages={'shape512':'original512','shape1024':'original1024'},
        decoders={'shape':'original'}, pipeline_config='/models/pipeline.json',
        DINO_local_path='/models/DINO',surface_seed=101, allocator_fraction=.85,
        texture_support_policy='original',source_SHA256=source,
        inputs_SHA256={'/old/SS/one.npz':'old','/old/input.png':'old','/models/pipeline.json':'c'*64},
        closed_inputs={'/old/SS/one.npz':'/old/receipt','/models/pipeline.json':'/models/receipt'})
    model = dict(ss64=own_binding(tmp_path, arm),pbr1024=pbr())
    replay = dict(mode='top1-only',checkpoint_kind='trained-native1900',engineering_only=False,
        source_SHA256=source, generated_shape_root='/old/geometry',row={'sample_id':'old'},
        input_SHA256={'/models/decoder':'c'*64},closed_inputs={'/models/decoder':'/models/receipt',
            '/old/geometry/native-shape-guides.pt':'/old/geometry/receipt',
            '/old/common/audit.json':'/old/common/receipt'},sampler_params={'steps':12},
        PBR_replay_seed=20261020)
    for key, field in [('native_config','PBR_config'),('checkpoint','PBR_checkpoint'),
                       ('conditioner_checkpoint','conditioner_checkpoint')]:
        path=model['pbr1024'][key]['path'];replay[field]=path;replay['closed_inputs'][path]='/PBR/receipt'
    render = dict(environment_map='/models/env.hdr',input_SHA256={'/models/env.hdr':'c'*64},
                  AO_seed=0,source_SHA256=source)
    return dict(arm=arm,model=model,asset=asset,
        evaluation=dict(path='/own/evaluation.json',SHA256='e'*64,receipt='/own/receipt'),
        artifact=dict(path='/own/ss.npz',SHA256='e'*64,receipt='/own/receipt',sample_id='s0',seed=17),
        terminal=dict(path='/own/terminal.json',SHA256='e'*64,receipt='/own/receipt'),
        generation_seed=17, geometry_template=geometry,replay_template=replay,
        replay_common_case='/old/common',render_template=render,sources=source,
        output='/results/s0',reference_root='/reference/shared',reference_render_root='/reference/render',
        devices=['0','1','2','3'])


@pytest.mark.parametrize('arm',module.ARMS)
def test_actual_own_binding_retains_original_numerical_settings(tmp_path,arm):
    values=inputs(tmp_path,arm);before=deepcopy(values);case=module.case_plans(**values)
    assert values==before and case['actual_GPU_execution'] is False
    g,r=case['geometry'],case['replay']
    for key in ['stages','decoders','pipeline_config','DINO_local_path','surface_seed',
                'allocator_fraction','texture_support_policy','input_states']:
        assert g[key]==values['geometry_template'][key]
    assert g['seed']==17 and r['PBR_replay_seed']==20261020
    assert not any(p.startswith('/old/') for p in g['inputs_SHA256'])
    assert not any(p.startswith('/old/') for p in r['closed_inputs'])
    assert r['native_guides']=='/results/s0/geometry/native-shape-guides.pt'
    for key in ['native_config','checkpoint','conditioner_checkpoint']:
        spec=values['model']['pbr1024'][key]
        assert r['input_SHA256'][spec['path']]==spec['SHA256']
    assert case['shared']['fixed_GT_shared_root']=='/reference/shared'


@pytest.mark.parametrize('damage',['depth','test','noise','foreign-PBR','changed-source','unknown-arm','seed202'])
def test_wrong_population_model_or_native_template_is_rejected(tmp_path,damage):
    values=inputs(tmp_path)
    if damage=='depth':values['asset']['row']['target_anchor_kind']='depth'
    elif damage=='test':values['asset']['row']['split']='test'
    elif damage=='noise':values['generation_seed']+=1
    elif damage=='foreign-PBR':values['replay_template']['PBR_checkpoint']='/wrong/model.pt'
    elif damage=='changed-source':values['sources']={'/source/native.py':'f'*64}
    elif damage=='unknown-arm':values['arm']='E_FIXED_RAW'
    else:values['model']['ss64']['training_seed']=202
    with pytest.raises(ValueError):module.case_plans(**values)


def test_matching_support_semantics_and_physical_controls_remain_distinct(tmp_path):
    for arm in module.ARMS:
        values=inputs(tmp_path,arm);b=values['model']['ss64']
        assert module.validate_models({arm:values['model']})==arm
        assert b['variant']==arm
        if arm=='M_FIXED_SPATIAL':assert b['actual_candidate_budget']==3
        if arm=='A_FIXED_NO_R':assert b['retargeter_present'] is False
    with pytest.raises(ValueError):module.own_contract('A_LEARNED_NO_R')


def population():
    from experiments.deformnext_retargeting.cascade_view_roles import IMAGE_METRICS
    ids=list(map(str,range(128)));clusters={sid:str(i%93) for i,sid in enumerate(ids)}
    records={}
    for j,arm in enumerate(module.ARMS+module.BASELINES):
        records[arm]=[dict(sample_id=sid,source_cluster_id=clusters[sid],CD=.01+j*.001,FScore=.8,
            views={cid:{m:float(i+j+cid) for m in IMAGE_METRICS} for cid in [8,9,10]})
            for i,sid in enumerate(ids)]
    return ids,clusters,records


def test_seven_models_use_adjacent_matching_and_shared_path_contrasts():
    ids,clusters,records=population();result=module.summarize(records,sample_ids=ids,clusters=clusters,draws=20)
    assert result['assets']==128 and result['source_clusters']==93 and result['TEST'] is False
    assert set(result['roles']['all8_9_10']['contrasts'])=={a+'_minus_'+b for a,b in module.CONTRASTS}
    assert ('M_FIXED_RAW','M_FIXED_SPATIAL') in module.CONTRASTS
    assert ('E_FIXED_RAW','M_FIXED_SPATIAL') not in module.CONTRASTS


@pytest.mark.parametrize('damage',['missing-control','suffix-source','missing-asset'])
def test_final_statistics_cannot_omit_an_unfavorable_arm_or_suffix(damage):
    ids,clusters,records=population()
    if damage=='missing-control':records.pop(module.ARMS[0])
    elif damage=='suffix-source':records[module.ARMS[0]][-1]['source_cluster_id']='wrong'
    else:records[module.ARMS[0]].pop()
    with pytest.raises(ValueError):module.summarize(records,sample_ids=ids,clusters=clusters,draws=20)


@pytest.mark.parametrize('damage',[None,'TEST','parent-alias','suffix-order','suffix-source','missing-archive'])
def test_own_index_never_uses_a_baseline_archive_or_incomplete_panel(tmp_path,damage):
    ids,clusters,records=population();arm='A_FIXED_UNIFORM'
    index=dict(schema=module.SCHEMA,TEST=False,sample_ids=ids,source_clusters_by_sample=clusters,
        bindings={arm:inputs(tmp_path)['model']},models={arm:records[arm]},SS_archives={arm:{}})
    if damage=='TEST':index['TEST']=True
    elif damage=='parent-alias':index['bindings']={'E_FIXED_RAW':index['bindings'][arm]}
    elif damage=='suffix-order':index['models'][arm][-1],index['models'][arm][-2]=index['models'][arm][-2:]
    elif damage=='suffix-source':index['models'][arm][-1]['source_cluster_id']='wrong'
    elif damage=='missing-archive':index['SS_archives']={}
    if damage is None:assert module.validate_index(index)==arm
    else:
        with pytest.raises(ValueError):module.validate_index(index)
