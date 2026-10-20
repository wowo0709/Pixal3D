import copy

import pytest
import torch

from experiments.deformnext_retargeting.retarget_only_pbr_contract import (
    ARM,build_retarget_only,validate_profile)


def test_control_clones_full_R_and_physically_removes_extra_channels():
    torch.manual_seed(101)
    config=dict(arm='c2',key_dim=1024,value_dim=2048,query_dim=1024,
        deformation_dim=2048,implementation='r9-native-v5',candidate_aggregation='top1',
        projection_mode='nearest',hidden_dim=8,topk=64,g_max=.25)
    module,audit=build_retarget_only(config)
    assert module.arm=='c1' and module.retargeter.net[0].in_features==3072
    assert not hasattr(module.retargeter,'deformation_projection')
    assert not hasattr(module,'ray_readout')
    assert audit['common_R_exact_to_Full_constructor'] and audit['common_R_state_tensors']==30
    assert len([p for p in module.parameters() if p.requires_grad])==6
    assert all(not p.requires_grad for p in module.q2c.parameters())
    with pytest.raises(ValueError):
        build_retarget_only(dict(config,value_dim=1024))


def test_R_only_capacity_cannot_reuse_full_or_accumulated_profile():
    report=dict(status='PASS_ACTUAL_RETARGET_ONLY_PBR_DDP4_CAPACITY',arm=ARM,
        initial_optimizer_states=360,actual_optimizer_states=366,
        changed_trainable_tensors_by_group=dict(default=360,retargeter=6),
        final_optimizer_steps=dict(default=[766],retargeter=[4]),
        all_frozen_NN_and_matcher_states_exact=True,common_R_exact_to_Full_constructor=True,
        physical_R_input_channels=3072,attention_and_DQ_physically_absent=True,
        world_size=4,microbatch=6,global_batch=24,accumulation=1,
        post_update_DDP_equal=True,post_validation_full_micro_update_passed=True,
        initial_native_prediction_exact_to_V1_capacity=True,first_loss_fields_exact_to_V1_capacity=True)
    validate_profile(report,capacity=True)
    for change in [dict(actual_optimizer_states=376),dict(accumulation=2),
                   dict(initial_native_prediction_exact_to_V1_capacity=False)]:
        broken=copy.deepcopy(report);broken.update(change)
        with pytest.raises(ValueError):validate_profile(broken,capacity=True)
