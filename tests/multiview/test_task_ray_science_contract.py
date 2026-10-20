import copy
import pytest
from experiments.deformnext_retargeting.task_ray_science_contract import validate_capacity,validate_initial32


def capacity():
    return dict(status='PASS_ACTUAL_TASK_RAY369_DDP2_CAPACITY',world_size=2,microbatch=48,
        global_batch=96,accumulation=1,initial_optimizer_states=366,actual_optimizer_states=369,
        changed_trainable_tensors=dict(default=360,deform_conditioner=6,deformation_query=1,task_ray_attention=2),
        optimizer_steps=dict(default=[99],deform_conditioner=[99],deformation_query=[3],task_ray_attention=[3]),
        all_frozen_parameters_and_buffers_exact=True,post_update_DDP_equal=True,first97_common_loss_fields_exact=True)


def test_old_capacity_and_partial_attention_update_cannot_admit_science():
    valid=capacity();validate_capacity(valid)
    old=copy.deepcopy(valid);old['actual_optimizer_states']=367
    with pytest.raises(ValueError):validate_capacity(old)
    dead=copy.deepcopy(valid);dead['changed_trainable_tensors']['task_ray_attention']=1
    with pytest.raises(ValueError):validate_capacity(dead)
    accumulated=copy.deepcopy(valid);accumulated.update(microbatch=24,accumulation=2)
    with pytest.raises(ValueError):validate_capacity(accumulated)


def test_generated_identity_is_distinct_from_micro2_loss_identity():
    record=dict(status='PASS_ACTUAL_NATIVE_TASK_RAY_INITIAL32_PARITY',assets=32,generation_batch=8,
        reference_step=96,optimizer_updates=0,all32_baseline_record_fields_and_metrics_exact=True,conditioner_state_tensors=46)
    validate_initial32(record)
    record['assets']=2
    with pytest.raises(ValueError):validate_initial32(record)
