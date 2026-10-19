"""An unchanged aggregate metric cannot hide changed GT or DDP execution."""
import copy

import pytest

from experiments.deformnext_retargeting.audit_ray_group_startup import verify_initial, verify_ranks


def initial():
    ids=[str(i) for i in range(32)]
    return dict(step=0,quick_panel_ids=ids,quick_metrics={'voxel_iou':.4},
        quick_per_asset=[dict(sample_id=sid,seed=i,tp=3,fp=2,fn=1,target_voxels_sha256='hash'+sid)
                         for i,sid in enumerate(ids)])


def test_native_initial_check_catches_changed_GT_noise_and_order_with_unchanged_IoU():
    reference=initial()
    verify_initial(copy.deepcopy(reference),reference)
    for field,value in [('seed',77),('target_voxels_sha256','other'),('tp',4)]:
        candidate=copy.deepcopy(reference)
        candidate['quick_per_asset'][0][field]=value
        with pytest.raises(ValueError,match='predictions/GT/noise'):
            verify_initial(candidate,reference)
    candidate=copy.deepcopy(reference)
    candidate['quick_panel_ids'].reverse()
    with pytest.raises(ValueError,match='order'):
        verify_initial(candidate,reference)


def ranks():
    return [[dict(step=s,rank=r,world_size=2,effective_batch_size=96,accumulation_packs=1,
                  microbatch_assets=[48],K=4,loss={'loss':.1},peak_allocated_gib=51.,compute_seconds=18.)
             for s in [1,2,3]] for r in [0,1]]


def test_actual_startup_rejects_accumulation_missing_rank_and_nonfinite_measurements():
    good=ranks()
    verify_ranks(good,first_steps=[1,2,3])
    changed=copy.deepcopy(good)
    changed[1][0].update(accumulation_packs=2,microbatch_assets=[24,24])
    with pytest.raises(ValueError,match='DDP/microbatch'):
        verify_ranks(changed,first_steps=[1,2,3])
    with pytest.raises(ValueError,match='two actual rank'):
        verify_ranks(good[:1],first_steps=[1,2,3])
    for field in ['peak_allocated_gib','compute_seconds']:
        changed=copy.deepcopy(good)
        changed[1][0][field]=float('nan')
        with pytest.raises(ValueError,match='DDP/microbatch'):
            verify_ranks(changed,first_steps=[1,2,3])
