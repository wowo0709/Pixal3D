import copy

import pytest

from experiments.deformnext_retargeting.projected_module_summary import summarize_projected_modules
from test_r_top1_native_summary import four


def five():
    value=four();value.pop('R_feature')
    for name in ['C_TOP1_V','C_FIXED_TOPK']:
        value[name]=copy.deepcopy(value['R_top1_only'])
        for result in value[name]:
            result['native_variant']=name;result['actual_source_tensors']=12
    return value


def test_projected_controls_keep_future_module_family_and_paired_noise_scope():
    result=summarize_projected_modules(five(),draws=100)
    assert len(result['comparisons'])==5
    assert result['comparisons']['C_FIXED_TOPK_minus_C_TOP1_V']['mean_asset_delta']['voxel_iou']==0.
    assert result['comparisons']['R_raw_topk_minus_C_FIXED_TOPK']['prespecified_IoU_family_size']==9
    assert result['comparisons']['R_raw_topk_minus_vanilla']['prespecified_IoU_family_size']==2
    assert result['independent_training_seeds']==1 and not result['automatic_model_adoption']


@pytest.mark.parametrize('damage',['noise','GT','source-count','arm'])
def test_other_model_or_generation_inputs_cannot_be_used_as_module_effect(damage):
    value=five();r=value['C_FIXED_TOPK'][0]
    if damage=='noise':r['per_asset'][0]['seed']+=1
    elif damage=='GT':r['per_asset'][0]['target_voxels_sha256']='wrong'
    elif damage=='source-count':r['actual_source_tensors']=19
    else:r['native_variant']='R_raw_topk'
    with pytest.raises(ValueError):summarize_projected_modules(value,draws=10)
