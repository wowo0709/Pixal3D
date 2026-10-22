from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.deformnext_retargeting.original_zero_cascade_population import metadata_validator, validate
from experiments.deformnext_retargeting.zero_shot_cascade_inputs import zero_shot_inputs
from test_original_zero_population import inputs


def cascade_inputs(alias='zero-source4'):
    p,_,binding,rows,ids,seeds,kw=inputs(alias=alias)
    parent,controls=kw['parent'],kw['controls']
    stages={name:dict(step=step,config=dict(path='/'+name+'/config.json',SHA256='a'*64),
        checkpoint=dict(path='/'+name+f'/denoiser_step{step:07d}.pt',SHA256='b'*64))
        for name,step in [('shape512',4210),('shape1024',16850),('pbr1024',16850)]}
    controls['downstream_original_stages']=deepcopy(stages)
    parent['models']['vanilla']={k:v for k,v in stages.items()if k!='pbr1024'}
    parent['original_PBR_for_SS_isolation']=stages['pbr1024']
    row=dict(rows[0],conditioning_camera_ids=[8,9,10,11],source_images=['s9.png','s10.png','s11.png'],
        target_anchor='edited8.png',source_anchor_rgb='source8.png',target_anchor_kind='rgb',
        transform_matrix=[np.eye(4).tolist()]*4,source_anchor_transform_matrix=np.eye(4).tolist(),
        camera_angle_x=[1.]*4,source_anchor_camera_angle_x=1.)
    paths={k:'/cache/'+k for k in [row['target_anchor'],row['source_anchor_rgb'],*row['source_images']]}
    protocol=zero_shot_inputs(row,binding['conditioning_mode'],paths)
    plan=dict(protocol,row=row,RGB_path_map=paths,original_control_alias=alias,
        original_SS_checkpoint=binding['checkpoint']['path'],
        stages={name:dict(config=stage['config']['path'],checkpoint=stage['checkpoint']['path'],
                         original_multiview_step=stage['step'])for name,stage in stages.items()},
        original_stage_checkpoints={name:stage['checkpoint']['path']for name,stage in stages.items()})
    evaluation=dict(protocol,population_scope='final-test508',status='PASS',stage='ss64',step=0,
        original_multiview_SS_step=4210,deformation_finetuning_updates=0,frozen_native_model_binding=binding,
        checkpoint=binding['checkpoint']['path'],conditioner_checkpoint=None,optimizer_updates_during_evaluation=0,
        diagnostic_only=False,quick_panel_ids=ids,
        actual_original_full_base_and_LoRA_validation=dict(original_full_base_and_LoRA_exact_before_evaluation=True),
        native_SS_artifacts=dict(status='PASS_ACTUAL_ORIGINAL_ZERO_SS_BOUND_POPULATION_ARCHIVE',
            archived_asset_count=508,artifacts=[dict(sample_id=s)for s in ids]),
        per_asset=[dict(sample_id=s,seed=seeds[s])for s in ids])
    return plan,evaluation,dict(parent=parent,controls=controls)


@pytest.mark.parametrize('alias',['zero-source4','zero-edited4'])
def test_zero_cascade_uses_genuine508_archive_and_exact_original_four_stage_weights(alias):
    p,e,kw=cascade_inputs(alias)
    protocol=validate(p,e,**kw)
    assert protocol['input_camera_ids']==[8,9,10,11]
    assert protocol['input_states']==(['M0']*4 if alias=='zero-source4' else ['M1','M0','M0','M0'])


@pytest.mark.parametrize('damage',[
    'fake-prefix32','renamed-step','missing-asset','wrong-seed','wrong-weight','wrong-stage','wrong-cluster',
    'GT-depth','source3','changed-anchor','wrong-loaded-values','validation-data','different-parent-shape',
])
def test_new_population_guard_rejects_fake_replay_or_changed_original_protocol(damage):
    p,e,kw=cascade_inputs()
    if damage=='fake-prefix32':e['native_SS_artifacts']['status']='PASS_ACTUAL_NATIVE_SS_ARCHIVE_AND_EXACT_REFERENCE_SUBSET_REPLAY'
    elif damage=='renamed-step':e['step']=4210
    elif damage=='missing-asset':e['native_SS_artifacts']['artifacts'].pop()
    elif damage=='wrong-seed':e['per_asset'][0]['seed']+=1
    elif damage=='wrong-weight':e['checkpoint']='/new/denoiser_step0000480.pt'
    elif damage=='wrong-stage':p['stages']['pbr1024']['checkpoint']='/deformation762.pt'
    elif damage=='wrong-cluster':p['row']['source_cluster_id']='other'
    elif damage=='GT-depth':p['row']['target_anchor_kind']='depth'
    elif damage=='source3':p['row']['conditioning_camera_ids']=[9,10,11]
    elif damage=='changed-anchor':p['RGB_paths'][0]='/cache/edited8.png'
    elif damage=='wrong-loaded-values':e['actual_original_full_base_and_LoRA_validation']['original_full_base_and_LoRA_exact_before_evaluation']=False
    elif damage=='validation-data':p['row']['split']='validation'
    elif damage=='different-parent-shape':kw['parent']['models']['vanilla']['shape512']['checkpoint']['SHA256']='c'*64
    with pytest.raises(ValueError):validate(p,e,**kw)


def test_only_metadata_callback_is_scoped_and_restored_after_failure():
    original=lambda *_:'old';core=SimpleNamespace(validate_zero_shot_cascade=original)
    p,e,kw=cascade_inputs()
    with pytest.raises(RuntimeError,match='owned'):
        with metadata_validator(core,lambda p,r:validate(p,r,**kw)):
            assert core.validate_zero_shot_cascade(p,e)['conditioning_mode']=='source-only'
            raise RuntimeError('owned numerical failure')
    assert core.validate_zero_shot_cascade is original
    with pytest.raises(ValueError,match='did not use'):
        with metadata_validator(core,lambda *_:None):pass
    assert core.validate_zero_shot_cascade is original
