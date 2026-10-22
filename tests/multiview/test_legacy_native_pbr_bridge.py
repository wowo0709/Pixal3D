import copy

import pytest

from experiments.deformnext_retargeting.legacy_native_pbr_bridge import (
    CASE_STATUS, validate_case, validate_geometry, validate_pair,
)


def geometry_fixture():
    metrics = dict(threshold=.02, prediction_samples=20000, target_samples=20000,
                   per_output_alignment=False, chamfer_L1=.01, surface_FScore=.8)
    g = dict(status='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION', sample_id='s',
             actual_resolution=1024, GT_geometry_or_coords_used_in_generation=False,
             GT_loaded_only_after_generation=True, fixed_shape_models=True,
             source_defined_normalization=True, per_output_alignment=False,
             original_stage_configs_and_weights_unchanged=True, SS_source_evaluation='/archive/evaluation.json',
             SS_prediction_SHA256='SS', generated_mesh='/old/prediction.npz',
             PBR_checkpoint='/old/denoiser_step0000762.pt', metrics=metrics)
    archive = dict(step=480, optimizer_updates_during_evaluation=0, target_modality='rgb',
                   input_states=['M1','M0','M0','M0'], input_camera_ids=[8,9,10,11],
                   native_SS_artifacts=dict(artifacts=[dict(sample_id='s',prediction_voxels_SHA256='SS')]))
    shapes = {name: dict(stage=dict(config='/'+name+'.json', checkpoint=f'/{name}/denoiser_step{step:07d}.pt'),
                        SHA256={'/'+name+'.json':'cfg', f'/{name}/denoiser_step{step:07d}.pt':'nn'})
              for name,step in [('shape512',4210),('shape1024',16850)]}
    p = dict(row=dict(sample_id='s'), output='/old', SS_evaluation='/archive/evaluation.json',
             input_states=['M1','M0','M0','M0'], RGB_paths=['t','a','b','c'],
             GT_loaded_only_after_generation=True, stages={k:v['stage'] for k,v in shapes.items()},
             inputs_SHA256={k:v for s in shapes.values() for k,v in s['SHA256'].items()})
    p['stages']['pbr1024'] = dict(checkpoint=g['PBR_checkpoint'])
    return g,archive,p,dict(sample_id='s',archive_path='/archive/evaluation.json',expected_shapes=shapes,RGB_paths=['t','a','b','c'])


def pair_fixture():
    geometry=geometry_fixture()[0]
    g=dict(status='PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE',sample_id='s',PBR_arm='vanilla',
           conditioner_checkpoint=None,source_weights_unchanged=True,GT_geometry_or_coords_used_in_generation=False,
           conditioning_preserves_CPU_and_CUDA_RNG=True,raw_surface_metrics_unchanged=True,
           geometry_array_byte_identity=dict(vertices=True,faces=True,shape_coords=True),
           original_prediction_SHA256='mesh',SS_prediction_SHA256='SS',SS_source_evaluation=geometry['SS_source_evaluation'],
           metrics=geometry['metrics'],PBR_replay_seed=20261020,actual_initial_texture_noise_SHA256='noise',
           sampler_class='sampler',sampler_params=dict(steps=12),geometry_evaluation_SHA256='surface')
    prior=dict(g,checkpoint_kind='original16850-engineering',engineering_only=True,PBR_checkpoint_SHA256='original')
    matched=dict(g,checkpoint_kind='trained-native1900',engineering_only=False,PBR_checkpoint_SHA256='trained',
                 PBR_checkpoint='/native/ckpts/denoiser_step0001900.pt')
    return prior,matched,geometry,dict(sample_id='s',original_PBR_SHA256='original',trained_PBR_SHA256='trained',original_prediction_SHA256='mesh')


def test_accepts_recorded_legacy_geometry_and_native_texture_pair():
    g,a,p,k=geometry_fixture();validate_geometry(g,a,p,**k)
    prior,matched,g,k=pair_fixture();validate_pair(prior,matched,g,**k)


@pytest.mark.parametrize('mutation', ['GT','shape','SS','RGB','PBR_provenance'])
def test_rejects_changed_geometry_provenance(mutation):
    g,a,p,k=geometry_fixture();p=copy.deepcopy(p)
    if mutation=='GT':g['GT_geometry_or_coords_used_in_generation']=True
    elif mutation=='shape':p['inputs_SHA256']['/shape512/denoiser_step0004210.pt']='other'
    elif mutation=='SS':g['SS_prediction_SHA256']='other'
    elif mutation=='RGB':p['RGB_paths'][0]='depth'
    else:g['PBR_checkpoint']='/native/ckpts/denoiser_step0001900.pt'
    with pytest.raises(ValueError):validate_geometry(g,a,p,**k)


@pytest.mark.parametrize('field,value', [('actual_initial_texture_noise_SHA256','other'),
    ('PBR_checkpoint_SHA256','other'),('conditioner_checkpoint','R.pt'),
    ('geometry_array_byte_identity',{}),('geometry_array_byte_identity',dict(vertices=False)),
    ('PBR_checkpoint','/ckpts/denoiser_step0000762.pt'),('PBR_replay_seed',42)])
def test_rejects_unmatched_texture_and_checkpoint(field,value):
    prior,matched,g,k=pair_fixture();matched[field]=value
    with pytest.raises(ValueError):validate_pair(prior,matched,g,**k)


def test_live_or_failed_children_cannot_pass_case_gate():
    case=dict(status=CASE_STATUS,SS_variant='feature-spatial',PBR_mode='vanilla',
              existing_geometry_reused_read_only=True,new_SS_or_Shape_inferences=0,
              original_Shape5124210_Shape102416850=True,paired_original_and_trained_texture_initial_noise_exact=True,
              paired_raw_geometry_and_native_predicted_LOD_identical=True,
              actual_prior_and_matched_three_camera_original_material_M1_shared_scores=True,
              original_generated_geometry_not_aligned_or_replaced=True,optimizer_updates=0,
              terminals=[dict(returncode=0,actual_session_drained=True) for _ in range(6)])
    validate_case(case)
    case['terminals'][0]['actual_session_drained']=False
    with pytest.raises(ValueError):validate_case(case)
