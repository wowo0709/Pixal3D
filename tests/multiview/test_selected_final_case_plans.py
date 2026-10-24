"""Synthetic sealed asset metadata and plans; no image/model/GT-array reads."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from experiments.deformnext_retargeting.archived_native_guides import POLICY
from experiments.deformnext_retargeting.common_geometry_pbr_contract import MODES as PBR_MODES
from experiments.deformnext_retargeting.native_final_case_integrity import paired_seed
from experiments.deformnext_retargeting.native_final_case_plans import IMAGE_FIELDS
from experiments.deformnext_retargeting.pipeline_population_contract_v2 import ARMS
from experiments.deformnext_retargeting.prepare_frozen_cascade_cases import CORE_FIELDS,core_digest
from experiments.deformnext_retargeting.selected_final_case_plans import case_plans,required_sources
from experiments.deformnext_retargeting.selected_final_population import compose_models
from test_selected_final_population import parents


def inputs(tmp_path,arm='E_LEARNED_RAW',mode='vanilla',alias=None):
    parent,controls=parents(tmp_path,arm,mode)
    parent['evaluation_protocol']['source_SHA256'].update({p:'a'*64 for p in required_sources()})
    parent['evaluation_protocol']['source_SHA256'].update({'/original/cascade.py':'a'*64,'/original/shared.py':'a'*64})
    models=compose_models(parent,controls)
    alias=parent['selected_alias'] if alias is None else alias;model=models[alias]
    row=dict(sample_id='test0',source_uid='testuid0',source_cluster_id='testcluster0',split='test',
        conditioning_input_schema='deformnext-edited-rgb-input-v1',conditioning_input_sha256='a'*64,
        target_anchor_kind='rgb',target_anchor='/file4/rgb/target8.png',target_rgb_input='/file4/rgb/target8.png',
        source_anchor_rgb='/file4/rgb/source8.png',source_images=[f'/file4/rgb/source{i}.png' for i in [9,10,11]],
        evaluation_target_images=[f'/file4/rgb/GT{i}.png' for i in [8,9,10]],conditioning_camera_ids=[8,9,10,11],
        camera_angle_x=[1.]*4,transform_matrix=[np.eye(4).tolist()]*4,mesh_scale=1.,
        source_anchor_transform_matrix=np.eye(4).tolist(),source_anchor_camera_angle_x=1.,
        source_geometry='/file4/mesh/source.npz',target_vertices='/file4/mesh/target.npy',
        transport_cache='/file4/descriptors/SS.npz',transport_cache_pbr='/file4/descriptors/PBR.npz')
    paths=dict(target_rgb='/cache/rgb/target8.png',source_anchor_rgb='/cache/rgb/source8.png',
        source_geometry='/cache/mesh/source.npz',target_vertices='/cache/mesh/target.npy',
        pbr_dump='/cache/material.pickle',GT_rgb_9='/cache/rgb/GT9.png',GT_rgb_10='/cache/rgb/GT10.png')
    paths.update({f'source_rgb_{i}':f'/cache/rgb/source{i}.png' for i in [9,10,11]})
    asset=dict(sample_id=row['sample_id'],source_cluster_id=row['source_cluster_id'],
        immutable_asset_fields={k:row[k] for k in CORE_FIELDS},inputs={k:dict(path=v,SHA256='a'*64) for k,v in paths.items()},
        edited_RGB_paths=[paths['target_rgb'],*[paths[f'source_rgb_{i}'] for i in [9,10,11]]],
        source_only_RGB_paths=[paths['source_anchor_rgb'],*[paths[f'source_rgb_{i}'] for i in [9,10,11]]],
        render_views=[dict(camera_id=i,input_camera_index=i-8,GT_RGBA=paths['target_rgb' if i==8 else f'GT_rgb_{i}']) for i in [8,9,10]])
    asset['inputs']['pbr_dump']['canonical_path']='/file4/material.pickle'
    asset['immutable_asset_fields_SHA256']=core_digest(asset['immutable_asset_fields'])
    geometry=dict(RGB_paths=['/old/input.png'],SS_evaluation='/old/SS/evaluation.json',
        source_geometry='/old/source.npz',target_vertices='/old/target.npy',texture_support_policy=POLICY,
        allocator_fraction=.85,decoders=dict(shape_slat_decoder='/models/shape',tex_slat_decoder='/models/texture'),
        DINO_local_path='/models/DINO',pipeline_config='/models/pipeline.json',
        stages={k:dict(config='/old/'+k+'/config',checkpoint='/old/'+k+'/checkpoint') for k in ['shape512','shape1024','pbr1024']},
        inputs_SHA256={'/models/pipeline.json':'a'*64,'/old/input.png':'old','/old/SS/0.npz':'old'},
        closed_inputs={'/old/SS/0.npz':'/old/file4-closure.json'},source_SHA256={'/original/cascade.py':'a'*64},
        sampler_parameters=dict(steps=12,guidance=3.))
    render=dict(environment_map='/models/environment.hdr',AO_seed=0,input_SHA256={'/models/environment.hdr':'a'*64})
    shared=dict(source_SHA256={'/original/shared.py':'a'*64},environment_map=render['environment_map'])
    group=model['geometry_group'];SS=dict(path='/SS/'+group+'/evaluation.json',SHA256='a'*64)
    artifact=dict(sample_id='test0',seed=parent['population']['generation_seeds']['test0'],
        path='/SS/'+group+'/ss-artifacts/0.npz',SHA256='a'*64,prediction_voxels_SHA256='b'*64)
    values=dict(parent=parent,controls=controls,alias=alias,row=row,asset=asset,file4_cache_root='/cache',
        assignment=dict(assigned_GPUs=['0','6'],canonical_local_root='/results'),
        SS_evaluation=SS,SS_artifact=artifact,geometry_template=geometry,
        geometry_template_spec=parent['evaluation_protocol']['generation_template'],render_template=render,
        shared_template=shared,output='/results/'+alias,reference_root='/results/zero-source4/shared',
        reference_render_root='/results/zero-source4/render')
    if model['kind']=='selected-trained':
        pbr=model['pbr1024'];native=Path(pbr['native_config']['path']).parent
        values['PBR_config']=dict(output_dir=str(native),training_seed=101,experiment={'PBR_source_mode':model['PBR_mode']},
            dataset={'args':{'path_map':{'/file4':'/cache'}}})
        values['PBR_template']=dict(mode=model['PBR_mode'],checkpoint_kind='trained-native1900',engineering_only=False,
            texture_support_policy=POLICY,PBR_config=pbr['native_config']['path'],PBR_checkpoint=pbr['checkpoint']['path'],
            conditioner_checkpoint=None if pbr['conditioner_checkpoint'] is None else pbr['conditioner_checkpoint']['path'],
            input_states=['M1','M0','M0','M0'],sampler_params=dict(steps=12,guidance=3.),
            input_SHA256={'/old/wrongasset':'old'},closed_inputs={'/old/wrongasset':'old'},cold_reference='/old/reference')
        pins={p['path']:p['SHA256'] for k,p in pbr.items() if k in ['native_config','checkpoint','conditioner_checkpoint'] and p is not None}
        pins.update({p:'a'*64 for p in asset['edited_RGB_paths']})
        pins.update({'/cache/descriptors/SS.npz':'c'*64,'/cache/descriptors/PBR.npz':'d'*64})
        values['PBR_inputs']=pins;values['PBR_closed_inputs']={p:'/cache/file4-closure.json' for p in pins}
        first=next(name for name,m in models.items() if m['geometry_group']==group)
        if first!=alias:
            values['geometry_reference']=dict(alias=first,sample_id='test0',generation_root='/results/'+first+'/geometry',
                SS_evaluation=SS,SS_prediction_voxels_SHA256=artifact['prediction_voxels_SHA256'])
    return values


@pytest.mark.parametrize('arm',ARMS)
@pytest.mark.parametrize('mode',PBR_MODES)
def test_every_physical_pair_uses_own_SS_original_shapes_captured_guides_and_same_texture_seed(tmp_path,arm,mode):
    values=inputs(tmp_path,arm,mode);before=deepcopy(values);case=case_plans(**values)
    assert values==before and case['model']['SS_arm']==arm and case['model']['PBR_mode']==mode
    g=case['phases'][0]['plan'];p=case['phases'][1]['plan']
    assert g['seed']==paired_seed('test0','cascade') and p['PBR_replay_seed']==paired_seed('test0','texture')
    assert g['surface_seed']==101 and p['texture_support_policy']==POLICY and len(p['RGB_paths'])==4
    assert p['generated_shape_root']==case['geometry_source_root']
    assert p['native_guides']==case['geometry_source_root']+'/native-shape-guides.pt'
    assert set(p['row'])==set(IMAGE_FIELDS) and 'target_vertices' not in p['row']
    assert p['input_SHA256']==values['PBR_inputs'] and '/old/wrongasset' not in p['closed_inputs']
    assert not any(k.startswith('/old/') for k in g['inputs_SHA256'])
    assert not any(k.startswith('/old/') for k in g['closed_inputs'])
    for key in ['sampler_parameters','allocator_fraction','decoders','DINO_local_path','pipeline_config']:
        assert g[key]==values['geometry_template'][key]
    assert case['phases'][1]['module']=='native_guide_entry' and case['phases'][1]['arguments']==['--kind','replay']
    assert all(p['plan']['assigned_GPUs']==['0','6'] for p in case['phases'])


@pytest.mark.parametrize('alias',['zero-source4','zero-edited4'])
def test_both_original_controls_keep_four_RGB_inputs_and_all_original_downstream_weights(tmp_path,alias):
    args=inputs(tmp_path,alias=alias);case=case_plans(**args);g=case['phases'][0]['plan']
    assert len(case['phases'])==3 and case['phases'][0]['module']=='selected_original_zero_cascade_population'
    assert g['input_states']==(['M0']*4 if alias=='zero-source4' else ['M1','M0','M0','M0'])
    assert len(g['RGB_paths'])==4 and g['RGB_paths'][0].endswith('source8.png' if alias=='zero-source4' else 'target8.png')
    for name in ['shape512','shape1024','pbr1024']:
        assert g['stages'][name]['checkpoint']==args['parent']['original_stages'][name]['checkpoint']['path']


@pytest.mark.parametrize('damage',['wrong-split','wrong-source','depth','SS-noise','wrong-template','wrong-PBR',
    'wrong-config','wrong-source-checkpoint','unpinned-RGB','unpinned-descriptors','missing-inputs','guide-policy',
    'reference','repeat-geometry','wrong-reuse-SS','reuse-outside-assignment','missing-source-pin','different-resource'])
def test_changed_population_model_inputs_or_native_guide_policy_are_rejected(tmp_path,damage):
    args=inputs(tmp_path,mode='top1-only')
    if damage=='wrong-split':args['row']['split']='validation'
    elif damage=='wrong-source':args['row']['source_cluster_id']='other'
    elif damage=='depth':args['row']['target_anchor_kind']='depth'
    elif damage=='SS-noise':args['SS_artifact']['seed']+=1
    elif damage=='wrong-template':args['geometry_template_spec']=dict(path='/other/template.json',SHA256='a'*64)
    elif damage=='wrong-PBR':args['PBR_template']['PBR_checkpoint']='/other.pt'
    elif damage=='wrong-config':args['PBR_config']['training_seed']=202
    elif damage=='wrong-source-checkpoint':args['PBR_template']['conditioner_checkpoint']=None
    elif damage=='unpinned-RGB':args['PBR_inputs'].pop(args['asset']['edited_RGB_paths'][0])
    elif damage=='unpinned-descriptors':args['PBR_inputs'].pop('/cache/descriptors/SS.npz')
    elif damage=='missing-inputs':args['PBR_inputs']=None
    elif damage=='guide-policy':args['geometry_template']['texture_support_policy']='rebuilt-from-output-mesh'
    elif damage=='reference':args=inputs(tmp_path,alias='zero-source4');args['reference_root']='/other/reference'
    elif damage=='repeat-geometry':args=inputs(tmp_path,'V1','top1-only');args.pop('geometry_reference')
    elif damage=='wrong-reuse-SS':args=inputs(tmp_path,'V1','top1-only');args['geometry_reference']['SS_prediction_voxels_SHA256']='c'*64
    elif damage=='reuse-outside-assignment':args=inputs(tmp_path,'V1','top1-only');args['geometry_reference']['generation_root']='/other/geometry'
    elif damage=='missing-source-pin':args['parent']['evaluation_protocol']['source_SHA256'].pop(required_sources()[0])
    else:args['assignment']['assigned_GPUs']=['6','6']
    with pytest.raises(ValueError):case_plans(**args)
