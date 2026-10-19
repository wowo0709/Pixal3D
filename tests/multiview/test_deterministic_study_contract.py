import copy

import pytest

from experiments.deformnext_retargeting.deterministic_study_contract import (
    legacy_epoch_comparison, verify_replay,
)


def evaluation(count):
    return dict(status='PASS',stage='ss64',step=480,weights='raw',sampling_steps=12,
        guidance=3.,generation_batch_size=8,optimizer_updates=0,optimizer_updates_during_evaluation=0,
        target_modality='rgb',conditioning_mode='edited-rgb',input_view_count=4,
        input_camera_ids=[8,9,10,11],input_states=['M1','M0','M0','M0'],diagnostic_only=False,
        transport_control='normal',r9_path_control='normal',task_value_control='normal',
        checkpoint='/weights/denoiser480.pt',conditioner_checkpoint='/weights/conditioner480.pt',
        arm_config='/recipes/ray-full.json',manifest_sha256='f'*64,arm='ray-full',
        deterministic_evaluation=dict(protocol='ss-deterministic-v1',actual_deterministic_algorithms=True,
            CUBLAS_WORKSPACE_CONFIG=':4096:8',model_data_trainer_sources_modified=False,
            optimizer_updates=0,entry_source_SHA256='a'*64,adapter_SHA256='b'*64),
        per_asset=[dict(sample_id=str(i),source_cluster_id='source-'+str(i),seed=i+42,
            tp=8,fp=2,fn=2,target_voxels_sha256='c'*64,target_voxel_shape=[64,64,64]) for i in range(count)])


def test_new_replay_keeps_the_original_exact_count_gate():
    small,full=evaluation(32),evaluation(128)
    assert verify_replay(small,full)['original_exact_replay_assertions_unchanged']
    full['per_asset'][0]['fp']+=1
    with pytest.raises(ValueError,match='does not replay'):
        verify_replay(small,full)


def test_matching_counts_with_different_gt_are_rejected():
    small,full=evaluation(32),evaluation(128)
    full['per_asset'][0]['target_voxels_sha256']='d'*64
    with pytest.raises(ValueError,match='target voxels differ'):
        verify_replay(small,full)


def test_legacy_results_cannot_be_presented_as_deterministic():
    small,full=evaluation(32),evaluation(128)
    small.pop('deterministic_evaluation')
    with pytest.raises(ValueError,match='provenance'):
        verify_replay(small,full)


def test_a_different_checkpoint_or_evaluator_is_rejected():
    small,full=evaluation(32),evaluation(128)
    full['checkpoint']='/weights/other.pt'
    with pytest.raises(ValueError,match='model/data/config'):
        verify_replay(small,full)
    full=evaluation(128)
    full['deterministic_evaluation']['entry_source_SHA256']='d'*64
    with pytest.raises(ValueError,match='source differs'):
        verify_replay(small,full)


def test_legacy_generated_drift_is_preserved_as_drift():
    small=evaluation(32)
    legacy={'quick_per_asset':copy.deepcopy(small['per_asset'])}
    legacy['quick_per_asset'][0]['fp']+=1
    report=legacy_epoch_comparison(legacy,small)
    assert report['GT_source_and_noise_exact']
    assert report['generated_counts_exact'] is False
    assert report['different_generated_cases']==1
