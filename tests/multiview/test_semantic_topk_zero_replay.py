import copy

import pytest

from experiments.deformnext_retargeting.semantic_topk_zero_replay import validate_zero,exact_legacy_replay,zero_paths


def record(mode='source-only'):
    return {'status':'PASS','stage':'ss64','step':0,'weights':'raw','sampling_steps':12,'guidance':3.,
        'generation_batch_size':8,'optimizer_updates_during_evaluation':0,'conditioning_mode':mode,
        'target_modality':'rgb','input_camera_ids':[8,9,10,11],
        'input_states':['M0']*4 if mode=='source-only' else ['M1','M0','M0','M0'],
        'per_asset':[{'sample_id':str(i),'source_cluster_id':str(i),'seed':i+10,'tp':2,'fp':1,'fn':3,
            'target_voxels_sha256':'a'*64} for i in range(128)]}


@pytest.mark.parametrize('mode',['source-only','edited-rgb'])
def test_zero_shot_requires_four_original_RGB_views_and_zero_new_updates(mode):
    e=record(mode);validate_zero(e,mode)
    for key,bad in [('input_camera_ids',[9,9,10,11]),('target_modality','depth'),('step',480),('optimizer_updates_during_evaluation',1)]:
        changed=copy.deepcopy(e);changed[key]=bad
        with pytest.raises(ValueError):validate_zero(changed,mode)


def test_fresh_zero_replay_cannot_fill_missing_legacy_hash_or_change_counts():
    fresh=record();legacy=copy.deepcopy(fresh)
    for r in legacy['per_asset']:r.pop('target_voxels_sha256')
    assert exact_legacy_replay(fresh,legacy)
    assert 'target_voxels_sha256' not in legacy['per_asset'][0]
    legacy['per_asset'][0]['tp']+=1
    with pytest.raises(ValueError,match='replay legacy'):exact_legacy_replay(fresh,legacy)


def test_replay_paths_preserve_previous_noise_repeats_and_write_new_noise0(tmp_path):
    paths=zero_paths(tmp_path/'new',tmp_path/'old','zero-source-only')
    assert paths[0]==tmp_path/'new/evaluations/zero-shot-replay-v1/source-only/evaluation.json'
    assert all(p.is_relative_to(tmp_path/'old') for p in paths[1:])
