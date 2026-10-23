import copy

import pytest

from experiments.deformnext_retargeting.common_pipeline_contract import validate_pbr_ready,validate_archive,validate_geometry_plan
from experiments.deformnext_retargeting.archived_native_guides import POLICY


def pbr():
    cfg=dict(experiment=dict(PBR_source_mode='vanilla',engineering_only=False,
        source_multiview_step=16850,target_modality='rgb',training_assets=9139,epochs=5,
        world_size=4,global_batch=24,microbatch_per_gpu=6,accumulation=1),
        trainer=dict(args=dict(lora=dict(rank=16,alpha=16,dropout=0.),max_steps=1900)))
    ranks={str(i):dict(rank=i,mode='vanilla',actual_optimizer_updates=1900,source_tensors=0,
        actual_optimizer_states=360,stage_training_complete=True,initial4_identity_before_first_update=True,
        frozen_base_matcher_and_anchors_exact=True,post_training_DDP_parameters_exact=True) for i in range(4)}
    ready=dict(status='PASS_ACTUAL_TRAINED_NATIVE_PBR1900_MODEL_READY',mode='vanilla',
        conditioner_checkpoint=None,checkpoint='/native/ckpts/denoiser_step0001900.pt',config='/native/config.json',
        checkpoint_SHA256='actualSHA',actual_four_rank_completion=ranks)
    return ready,cfg


@pytest.mark.parametrize('damage',['none','wrong-mode','not-trained','missing-rank','changed-base','extra-conditioner','wrong-base'])
def test_common_PBR_requires_actual_trained_vanilla_lineage(damage):
    ready,cfg=pbr()
    if damage=='none':assert validate_pbr_ready(ready,cfg)=='actualSHA';return
    if damage=='wrong-mode':ready['mode']='top1-only'
    elif damage=='not-trained':ready['actual_four_rank_completion']['0']['actual_optimizer_updates']=0
    elif damage=='missing-rank':ready['actual_four_rank_completion'].pop('3')
    elif damage=='changed-base':ready['actual_four_rank_completion']['1']['frozen_base_matcher_and_anchors_exact']=False
    elif damage=='extra-conditioner':cfg['trainer']['args']['deform_conditioner']={}
    else:cfg['experiment']['source_multiview_step']=762
    with pytest.raises(ValueError):validate_pbr_ready(ready,cfg)


def archive(count=32,variant='vanilla'):
    rows=[dict(sample_id=str(i),source_cluster_id=str(i//2),seed=i,target_voxels_sha256='GT'+str(i),tp=2,fp=1,fn=1) for i in range(count)]
    artifacts=[dict(sample_id=r['sample_id'],seed=r['seed'],target_voxels_SHA256=r['target_voxels_sha256'],
        occupancy_counts={k:r[k] for k in ['tp','fp','fn']}) for r in rows]
    return dict(status='PASS',stage='ss64',step=480,native_variant=variant,
        input_states=['M1','M0','M0','M0'],input_camera_ids=[8,9,10,11],target_modality='rgb',
        optimizer_updates_during_evaluation=0,deterministic_algorithms=True,per_asset=rows,
        native_SS_artifacts=dict(predicted_SS_is_available_for_full_cascade=True,archived_asset_count=count,
            artifacts=artifacts,original_prefix32_all_records_and_occupancy_counts_exact=count==32,
            same_forward_prediction_and_target_capture=count==128))


def test_new128_capture_and_old32_archive_share_exact_original_cascade_prefix():
    ref=archive();new=archive(128,'C_TOP1_V')
    assert validate_archive(new,'C_TOP1_V',ref)==[str(i) for i in range(32)]
    for field,value in [('seed',999),('target_voxels_sha256','otherGT'),('source_cluster_id','other')]:
        changed=copy.deepcopy(new);changed['per_asset'][0][field]=value
        with pytest.raises(ValueError):validate_archive(changed,'C_TOP1_V',ref)
    changed=copy.deepcopy(new);changed['native_SS_artifacts']['artifacts'][0]['occupancy_counts']['tp']=1
    with pytest.raises(ValueError):validate_archive(changed,'C_TOP1_V',ref)


def test_cascade_changes_only_own_SS_and_common_trained_PBR():
    ready,_=pbr()
    template=dict(input_states=['M1','M0','M0','M0'],RGB_paths=['T','S1','S2','S3'],
        row=dict(conditioning_camera_ids=[8,9,10,11],split='validation'),seed=1,surface_seed=2,
        source_geometry='source',target_vertices='target',decoders={},pipeline_config='pipeline',
        DINO_local_path='dino',allocator_fraction=.7,
        stages=dict(shape512=dict(config='shape512cfg',checkpoint='/shape512/ckpts/denoiser_step0004210.pt'),
            shape1024=dict(config='shape1024cfg',checkpoint='/shape1024/ckpts/denoiser_step0016850.pt'),
            pbr1024=dict(config='oldcfg',checkpoint='oldweights')))
    plan=copy.deepcopy(template);plan.update(texture_support_policy=POLICY,output='new',SS_evaluation='actual-own-SS')
    plan['stages']['pbr1024']=dict(config=ready['config'],checkpoint=ready['checkpoint'])
    assert validate_geometry_plan(plan,template,ready)
    for damage in ['shape','camera','RGB','PBR','GT-support']:
        changed=copy.deepcopy(plan)
        if damage=='shape':changed['stages']['shape512']['checkpoint']='other'
        elif damage=='camera':changed['row']['conditioning_camera_ids']=[0,1,2,3]
        elif damage=='RGB':changed['RGB_paths'][0]='depth'
        elif damage=='PBR':changed['stages']['pbr1024']['checkpoint']='other'
        else:changed['source_geometry']='different-GT'
        with pytest.raises(ValueError):validate_geometry_plan(changed,template,ready)
