from copy import deepcopy

import numpy as np
import pytest

from experiments.deformnext_retargeting.ray_evidence_illustration import first_recoverable_relation,patch_box
from test_ray_attention_utilization import example


def inputs():
    data=example();keys=['positive_indices','positive_visible','target_foreground']
    return {k:v for k,v in data.items() if k not in keys},{k:data[k] for k in keys}


def test_saved_relation_is_not_reweighted_or_confused_with_entire_ray():
    observed,truth=inputs();before=deepcopy(observed);result=first_recoverable_relation(observed,truth)
    assert result['source_ray']==result['target_patch']==0
    assert result['observed_member_sites']==[0,1,2]
    assert result['observed_target_member_patches']==[0,1]
    assert result['GT_in_K_ranks']==[2] and result['top1_candidate']==0
    assert result['GT_mass_prior']==.25 and result['GT_mass_actual']==pytest.approx(.6)
    assert result['GT_mass_uniform']==1/64
    assert result['observed_members_not_entire_native_ray']
    for key in observed:np.testing.assert_array_equal(observed[key],before[key])


def test_no_eligible_relation_is_not_presented_as_a_topK_success():
    observed,truth=inputs();truth['positive_indices'][0,0]=100
    assert first_recoverable_relation(observed,truth) is None
    observed,truth=inputs();truth['positive_visible'][:]=False
    assert first_recoverable_relation(observed,truth) is None


def test_prior_top1_and_actual_attention_are_not_interchanged():
    observed,truth=inputs();observed['weights'][0,0]=.05;observed['weights'][0,1]=.95
    result=first_recoverable_relation(observed,truth)
    assert result['top1_candidate']==0 and result['GT_mass_actual']>.9


def test_native_patch_mapping_uses_saved_camera_address_not_image_content():
    assert patch_box(33,512,512)==(16,16,16,16)
    assert patch_box(1023,1024,1024)==(992,992,32,32)
    for args in [(1024,512,512),(33,513,512),(33.,512,512)]:
        with pytest.raises(ValueError):patch_box(*args)
