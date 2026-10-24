"""Own model bytes and complete128 archive must survive downstream binding."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.deformnext_retargeting import fixed_correspondence_pipeline_contract as contract
from experiments.deformnext_retargeting.fixed_correspondence_native_recipe import VARIANTS
from test_fixed_correspondence_native_runtime import fixture,bundle


def inputs(tmp_path,arm):
    cfg,_=fixture(tmp_path,arm);b=bundle(cfg)
    hashes={b[k]:'a'*64 for k in ('native_config','checkpoint','conditioner_checkpoint')}
    binding=contract.model_binding(b,cfg,hashes)
    record=dict(status='PASS',stage='ss64',step=480,weights='raw',native_variant=arm,
        actual_source_tensors=10,actual_LoRA_tensors=360,optimizer_updates_during_evaluation=0,
        target_modality='rgb',input_states=['M1','M0','M0','M0'],input_camera_ids=[8,9,10,11],
        generation_batch_size=8,sampling_steps=12,guidance=3.,deterministic_algorithms=True,
        evaluation_mode='final128',verified_inputs=hashes)
    for k in ['training_seed','conditioning_schema','correspondence_study_schema','correspondence_parent_variant',
              'correspondence_mode','actual_candidate_budget','matching_decoder_physically_absent',
              'matching_scalar_score_physically_absent','evidence_mode','retargeter_present',
              'shared_carrier_use_retarget_evidence','learned_projection_tensors','learned_attention_head_tensors',
              'correspondence_source_SHA256']:
        record[k]=deepcopy(binding[k])
    for k,f in [('native_config','arm_config'),('checkpoint','checkpoint'),('conditioner_checkpoint','conditioner_checkpoint')]:
        record[f]=b[k]
    record['per_asset']=[dict(sample_id=str(i),source_cluster_id=str(i%93),seed=i,
        target_voxels_sha256='b'*64,tp=1,fp=2,fn=3) for i in range(128)]
    record['native_SS_artifacts']=dict(status='PASS_ACTUAL_ORIGINAL_FIXED_CORRESPONDENCE_SS128_CAPTURE',
        archived_asset_count=128,same_forward_prediction_and_target_capture=True,
        predicted_SS_is_available_for_full_cascade=True,GT_SS_is_for_scoring_only=True,
        artifacts=[dict(sample_id=r['sample_id'],seed=r['seed'],target_voxels_SHA256=r['target_voxels_sha256'],
            occupancy_counts={k:r[k] for k in ['tp','fp','fn']}) for r in record['per_asset']])
    return binding,record,deepcopy(record)


@pytest.mark.parametrize('arm',VARIANTS)
def test_own_physical_correspondence_archive_is_admitted(tmp_path,arm):
    binding,r,ref=inputs(tmp_path,arm)
    assert contract.validate_archive128(r,arm,ref,binding=binding)==list(map(str,range(128)))


@pytest.mark.parametrize('field,value',[('variant','C_RAW'),('source_tensors',19),
    ('actual_candidate_budget',64),('matching_decoder_physically_absent',False),
    ('evidence_mode','learned-anchored'),('shared_carrier_use_retarget_evidence',False)])
def test_old_top1_or_another_physical_model_cannot_be_relabelled(tmp_path,field,value):
    b,_,_=inputs(tmp_path,'M_FIXED_SPATIAL');b[field]=value
    with pytest.raises(ValueError):contract.validate_ss_binding(b,'M_FIXED_SPATIAL')


@pytest.mark.parametrize('damage',['foreign-weight','source-mode','depth','noise','GT-generation','last-count'])
def test_real_evaluation_identity_and_suffix_cannot_be_bypassed(tmp_path,damage):
    b,r,ref=inputs(tmp_path,'M_FIXED_RAW')
    if damage=='foreign-weight':r['verified_inputs'][r['checkpoint']]='c'*64
    elif damage=='source-mode':r['correspondence_mode']='D1-cosine'
    elif damage=='depth':r['target_modality']='depth'
    elif damage=='noise':r['per_asset'][127]['seed']+=1
    elif damage=='GT-generation':r['native_SS_artifacts']['GT_SS_is_for_scoring_only']=False
    else:r['native_SS_artifacts']['artifacts'][127]['occupancy_counts']['tp']+=1
    with pytest.raises(ValueError):contract.validate_archive128(r,'M_FIXED_RAW',ref,binding=b)
