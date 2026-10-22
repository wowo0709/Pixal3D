import copy

import pytest

from experiments.deformnext_retargeting.evaluate_r_native_readout_intervention import (
    MODE, compare_reference, validate_protocol,
)


def protocol():
    return (dict(mode=MODE, expected_assets=32, intervention='normal',
                 deterministic_algorithms=True, generation_batch_size=8),
            dict(variant='R_raw_topk', step=480, source_teacher_updates=0))


@pytest.mark.parametrize('field,value', [('step',112), ('source_teacher_updates',112),
                                        ('variant','R_feature'), ('variant','R_top1_only')])
def test_teacher_or_other_native_arm_cannot_be_substituted(field,value):
    plan,bundle=protocol()
    bundle[field]=value
    with pytest.raises(ValueError):validate_protocol(plan,bundle)


@pytest.mark.parametrize('field,value', [('expected_assets',128), ('deterministic_algorithms',False),
                                        ('generation_batch_size',1), ('intervention','uniform')])
def test_diagnostic_retains_native_panel_algorithm_and_explicit_control(field,value):
    plan,bundle=protocol();validate_protocol(plan,bundle)
    plan[field]=value
    with pytest.raises(ValueError):validate_protocol(plan,bundle)


def reference():
    return dict(native_variant='R_raw_topk',checkpoint='NN',conditioner_checkpoint='source',
        arm_config='config',step=480,actual_loaded_trainable_SHA256='actual-hash',
        source_teacher_updates=0,deterministic_algorithms=True,quick_metrics={'IoU':.5},
        quick_panel_ids=list(range(32)),per_asset=[dict(sample_id=i,source_cluster_id=i,
            seed=i,target_voxels_sha256=str(i),target_voxel_shape=[64]*3,
            target_representation='same',voxel_iou=.5)for i in range(32)])


def test_normal_requires_exact_outputs_but_intervention_can_change_quality():
    original=reference();actual=copy.deepcopy(original)
    compare_reference(actual,original,exact=True)
    actual['per_asset'][0]['voxel_iou']=.6
    compare_reference(actual,original,exact=False)
    with pytest.raises(ValueError):compare_reference(actual,original,exact=True)


@pytest.mark.parametrize('field,value', [('seed',999), ('target_voxels_sha256','other'),
                                        ('source_cluster_id','other')])
def test_same_saved_weights_do_not_allow_GT_noise_or_cluster_change(field,value):
    original=reference();actual=copy.deepcopy(original)
    actual['per_asset'][0][field]=value
    with pytest.raises(ValueError):compare_reference(actual,original,exact=False)


def test_wrong_loaded_model_is_rejected_even_with_matching_quality():
    original=reference();actual=copy.deepcopy(original)
    actual['actual_loaded_trainable_SHA256']='other-model'
    with pytest.raises(ValueError):compare_reference(actual,original,exact=False)
