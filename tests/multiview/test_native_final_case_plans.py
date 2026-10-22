from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.native_final_case_plans import ALIASES, IMAGE_FIELDS, case_plans, compose_models
from experiments.deformnext_retargeting.native_final_freeze import make_freeze
from test_native_final_freeze import inputs as freeze_inputs
from test_original_zero_cascade_population import cascade_inputs


def models():
    parent=make_freeze(**freeze_inputs())
    _,_,kw=cascade_inputs();controls=kw['controls']
    controls['downstream_original_stages']={k:deepcopy(parent['models']['vanilla'][k])for k in ['shape512','shape1024']}
    controls['downstream_original_stages']['pbr1024']=deepcopy(parent['original_PBR_for_SS_isolation'])
    return parent,controls,compose_models(parent,controls)


def inputs(alias):
    _,_,all_models=models();model=all_models[alias]
    row=dict(sample_id='test0',source_cluster_id='cluster0',split='test',conditioning_camera_ids=[8,9,10,11],
        target_anchor='/file4/target.png',source_anchor_rgb='/file4/source8.png',
        source_images=['/file4/source9.png','/file4/source10.png','/file4/source11.png'],
        camera_angle_x=[1.]*4,transform_matrix=['calibration']*4,mesh_scale=1.,
        conditioning_input_schema='RGB4',conditioning_input_sha256='RGB',target_anchor_kind='rgb',
        transport_cache='/cache/SS-descriptors.npz',transport_cache_pbr='/cache/PBR-descriptors.npz',
        target_vertices='/GT/target.npy')
    asset=dict(sample_id='test0',source_cluster_id='cluster0',
        edited_RGB_paths=['/cache/target.png','/cache/source9.png','/cache/source10.png','/cache/source11.png'],
        source_only_RGB_paths=['/cache/source8.png','/cache/source9.png','/cache/source10.png','/cache/source11.png'])
    files={'target_rgb':'/cache/target.png','source_anchor_rgb':'/cache/source8.png',
           'GT_rgb_9':'/cache/GT9.png','GT_rgb_10':'/cache/GT10.png',
           'source_geometry':'/GT/source.npz','target_vertices':'/GT/target.npy','pbr_dump':'/GT/material.pickle'}
    files.update({'source'+str(c):'/cache/source'+str(c)+'.png' for c in [9,10,11]})
    asset['inputs']={k:dict(path=v,SHA256='a'*64)for k,v in files.items()}
    asset['render_views']=[dict(camera_id=c,GT_RGBA=files['target_rgb'if c==8 else'GT_rgb_'+str(c)])for c in [8,9,10]]
    geometry=dict(RGB_paths=['/old/input.png'],SS_evaluation='/old/SS/evaluation.json',
        source_geometry='/old/source.npz',target_vertices='/old/target.npy',
        stages={'pbr1024':dict(config='/old/config',checkpoint='/old/checkpoint')},
        inputs_SHA256={'/pipeline.json':'pipeline','/old/SS/prediction.npz':'old','/old/input.png':'old',
                      '/old/source.npz':'old','/old/target.npy':'old','/old/config':'old','/old/checkpoint':'old'})
    render=dict(environment_map='/env.hdr',input_SHA256={'/env.hdr':'env','/old/GT.png':'old'})
    shared=dict(source_SHA256={'/source/shared.py':'code'},environment_map='/env.hdr')
    root='/out/'+alias
    own=dict(alias=alias,model=model,row=row,asset=asset,SS_evaluation='/new/'+alias+'/SS.json',
        SS_evaluation_SHA256='report',SS_artifact=dict(sample_id='test0',path='/new/'+alias+'/SS.npz',SHA256='archive'),
        geometry_template=geometry,render_template=render,shared_template=shared,output=root,
        reference_root='/out/zero-source4/shared',reference_render_root='/out/zero-source4/render',
        cascade_seed=12,texture_seed=34)
    if model['kind']=='native-trained':
        pbr=model['pbr1024']
        own['PBR_template']=dict(mode=pbr['mode'],PBR_checkpoint=pbr['checkpoint']['path'],
            PBR_config=pbr['native_config']['path'],conditioner_checkpoint=None if pbr['conditioner_checkpoint']is None else pbr['conditioner_checkpoint']['path'],
            checkpoint_kind='trained-native1900',engineering_only=False,
            input_SHA256={'/old/wrongasset.npz':'old'},closed_inputs={'/old/wrongasset.npz':'oldreceipt'},cold_reference='old')
        own['PBR_inputs']={v['path']:v['SHA256']for k,v in pbr.items()if k in ['native_config','checkpoint','conditioner_checkpoint']and v is not None}
        own['PBR_inputs'].update({row[k]:'descriptors'for k in ['transport_cache','transport_cache_pbr']})
        own['PBR_closed_inputs']={'/new/stage-completed.json':'/new/file4-closure.json'}
    return own


def test_all_six_models_preserve_original_stages_and_distinct_control_meaning():
    _,_,m=models()
    assert tuple(m)==ALIASES
    assert m['zero-source4']['ss64']['checkpoint']==m['zero-edited4']['ss64']['checkpoint']
    assert m['zero-source4']['ss64']['conditioning_mode']!=m['zero-edited4']['ss64']['conditioning_mode']
    assert m['R_top1_only']['pbr1024']['mode']=='top1-only'


@pytest.mark.parametrize('alias',ALIASES)
def test_final_case_keeps_all_inputs_noise_geometry_and_shared_reference_explicit(alias):
    args=inputs(alias);before=deepcopy(args);result=case_plans(**args)
    assert args==before
    phases=result['phases'];g=phases[0]['plan']
    assert g['surface_seed']==101 and g['seed']==12 and len(g['RGB_paths'])==4
    assert g['input_states']==(['M0']*4 if alias=='zero-source4'else['M1','M0','M0','M0'])
    assert not any(p.startswith('/old/')for p in g['inputs_SHA256'])
    assert g['SS_evaluation']=='/new/'+alias+'/SS.json'
    assert phases[-1]['module']==('score_cascade_shared_material'if alias=='zero-source4'else'score_cascade_fixed_reference')
    if alias.startswith('zero-'):
        assert len(phases)==3 and phases[0]['kind']=='zero-bound-entry'
        assert result['final_generation_root'].endswith('/geometry')
    else:
        assert len(phases)==4 and phases[1]['module']=='replay_r_pbr_on_generated_shape'
        p=phases[1]['plan'];assert set(p['row'])==set(IMAGE_FIELDS) and 'target_vertices'not in p['row']
        assert p['PBR_replay_seed']==34 and not any(x.startswith('/old/')for x in p['input_SHA256'])
        assert p['input_SHA256']==args['PBR_inputs']and p['closed_inputs']==args['PBR_closed_inputs']
        assert 'cold_reference'not in p and result['final_generation_root'].endswith('/PBR')


@pytest.mark.parametrize('damage',['GT-test-swap','archive-other-asset','RGB3','negative-seed','wrong-PBR','wrong-config','wrong-source','old-data-hashes','wrong-reference'])
def test_case_factory_rejects_changed_population_models_or_unpinned_replay(damage):
    a=inputs('R_top1_only')
    if damage=='GT-test-swap':a['row']['split']='validation'
    elif damage=='archive-other-asset':a['SS_artifact']['sample_id']='other'
    elif damage=='RGB3':a['asset']['edited_RGB_paths'].pop()
    elif damage=='negative-seed':a['cascade_seed']=-1
    elif damage=='wrong-PBR':a['PBR_template']['PBR_checkpoint']='/other.pt'
    elif damage=='wrong-config':a['PBR_template']['PBR_config']='/other.json'
    elif damage=='wrong-source':a['PBR_template']['conditioner_checkpoint']=None
    elif damage=='old-data-hashes':a['PBR_inputs']={'/old/asset':'old'}
    elif damage=='wrong-reference':
        a=inputs('zero-source4');a['reference_root']='/out/arbitrary/shared'
    with pytest.raises(ValueError):case_plans(**a)
