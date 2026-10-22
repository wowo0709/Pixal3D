from copy import deepcopy

import numpy as np
import pytest

from experiments.deformnext_retargeting.generated_r_mediated_pbr import IMAGE_ROW_FIELDS
from experiments.deformnext_retargeting.pbr_replay_artifacts import PREDICTION_FIELDS
from experiments.deformnext_retargeting.r_pbr_replay_contract import (
    validate_lineage, validate_generated_shape, validate_cold_comparison,
)


def fixture(mode='top1-plus-feature', trained=True):
    row={key:'synthetic fixture, not actual evidence'for key in IMAGE_ROW_FIELDS}
    row.update(camera_angle_x=[.8]*4,transform_matrix=[
        [[1.,0,0,0],[0,0,-1.,-2.],[0,1.,0,0],[0,0,0,1.]]]*4,mesh_scale=1.)
    source=None if mode=='vanilla' else {'appearance_mode':mode}
    cfg=dict(output_dir='/owned/native',training_seed=101,
        experiment=dict(PBR_source_mode=mode,source_multiview_step=16850),
        trainer={'args':dict(continuation_ckpt={'denoiser':'/original/16850.pt'},deform_conditioner=source)})
    plan=dict(mode=mode,checkpoint_kind='trained-native1900'if trained else'original16850-engineering',
        engineering_only=not trained,row=row,input_states=['M1','M0','M0','M0'],
        RGB_paths=['synthetic0','synthetic1','synthetic2','synthetic3'],PBR_replay_seed=42,allocator_fraction=.9,
        PBR_checkpoint='/owned/native/ckpts/denoiser_step0001900.pt'if trained else'/original/16850.pt',
        conditioner_checkpoint='/owned/native/ckpts/deform_conditioner_step0001900.pt'if trained and mode!='vanilla'else None)
    completion=dict(status='PASS_ACTUAL_ORIGINAL_PBR16850_FRESH_NATIVE1900_STAGE',mode=mode,
        actual_optimizer_updates=1900,training_seed=101,source_multiview_step=16850,
        source_tensors=0 if mode=='vanilla'else 10 if mode=='top1-only'else 19,
        stage_training_complete=True,frozen_base_matcher_and_anchors_exact=True,post_training_DDP_parameters_exact=True)
    return plan,cfg,completion if trained else None


@pytest.mark.parametrize('mode',['vanilla','top1-only','top1-plus-feature','raw-topk-plus-feature'])
@pytest.mark.parametrize('trained',[False,True])
def test_distinguishes_original_engineering_from_completed_own_scientific_pairs(mode,trained):
    plan,cfg,completion=fixture(mode,trained)
    assert validate_lineage(plan,cfg,completion)==(mode,plan['checkpoint_kind'])


@pytest.mark.parametrize('problem',['wrong-nn-step','wrong-source-step','wrong-mode','no-completion',
    'temporary-three-updates','failed-DDP','GT-row-field','wrong-camera-input','wrong-seed'])
def test_rejects_ambiguous_weights_engineering_and_GT_input_routes(problem):
    plan,cfg,completion=fixture()
    if problem=='wrong-nn-step':plan['PBR_checkpoint']=plan['PBR_checkpoint'].replace('1900','0380')
    elif problem=='wrong-source-step':plan['conditioner_checkpoint']=plan['conditioner_checkpoint'].replace('1900','0380')
    elif problem=='wrong-mode':cfg['trainer']['args']['deform_conditioner']['appearance_mode']='top1-only'
    elif problem=='no-completion':completion=None
    elif problem=='temporary-three-updates':completion['actual_optimizer_updates']=3
    elif problem=='failed-DDP':completion['post_training_DDP_parameters_exact']=False
    elif problem=='GT-row-field':plan['row']['pbr_latent']='/forbidden/GT.npz'
    elif problem=='wrong-camera-input':plan['input_states'][0]='M0'
    elif problem=='wrong-seed':plan['PBR_replay_seed']=-1
    with pytest.raises(ValueError):validate_lineage(plan,cfg,completion)


def test_cold_source_cannot_be_used_as_trained_weight_or_load_temporary_engineering_source():
    plan,cfg,_=fixture(trained=False)
    plan['engineering_only']=False
    with pytest.raises(ValueError):validate_lineage(plan,cfg)
    plan['engineering_only']=True;plan['conditioner_checkpoint']='/temporary/3update.pt'
    with pytest.raises(ValueError):validate_lineage(plan,cfg)


def test_validates_actual_prediction_support_and_disallows_GT_report():
    plan,_,_=fixture()
    prediction={key:np.zeros(1)for key in PREDICTION_FIELDS}
    prediction.update(shape_coords=np.array([[0,0,0,0],[0,63,63,63]],np.int32),
                      shape_feats=np.ones((2,32),np.float32),texture_feats=np.zeros((2,32),np.float32))
    report=dict(sample_id=plan['row']['sample_id'],actual_resolution=1024,
        GT_geometry_or_coords_used_in_generation=False,per_output_alignment=False,actual_HR_tokens=2)
    validate_generated_shape(plan,report,prediction)
    with pytest.raises(ValueError):
        validate_generated_shape(plan,dict(report,GT_geometry_or_coords_used_in_generation=True),prediction)
    prediction['shape_coords'][1]=prediction['shape_coords'][0]
    with pytest.raises(ValueError):validate_generated_shape(plan,report,prediction)


def test_cold_parity_requires_actual_noise_latent_and_decoded_array_identity():
    reference=dict(checkpoint_kind='original16850-engineering',PBR_checkpoint_SHA256='same NN',
        original_prediction_SHA256='same predicted shape',sample_id='same asset',PBR_arm='vanilla',
        actual_initial_texture_noise_SHA256='noise',actual_texture_latent_SHA256='latent',
        actual_decoded_coordinates_SHA256='coords',actual_decoded_attributes_SHA256='attrs',
        PBR_replay_seed=42,sampler_class='native',sampler_params={'steps':12},geometry_array_byte_identity={'vertices':True})
    actual=dict(reference,PBR_arm='top1-plus-feature')
    validate_cold_comparison(actual,reference)
    for key in ['actual_initial_texture_noise_SHA256','actual_texture_latent_SHA256',
                'actual_decoded_coordinates_SHA256','actual_decoded_attributes_SHA256']:
        changed=deepcopy(actual);changed[key]='different actual bytes'
        with pytest.raises(ValueError,match=key):validate_cold_comparison(changed,reference)
