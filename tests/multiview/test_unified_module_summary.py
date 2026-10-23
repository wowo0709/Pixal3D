import copy

import numpy as np
import pytest

from experiments.deformnext_retargeting.unified_module_summary import (
    ALIASES,SOURCE_COUNTS,METRICS,summarize_unified_modules,
)


def records():
    results={}
    for arm,count in SOURCE_COUNTS.items():
        value=.4 if arm=='V1' else .5
        results[arm]=[dict(stage='ss64',step=480,native_variant=ALIASES.get(arm,arm),
            evaluation_mode='final128',optimizer_updates_during_evaluation=0,
            actual_source_tensors=count,quick_metrics=dict(voxel_iou=.3),
            per_asset=[dict(sample_id=str(i),source_cluster_id=str(i//2),seed=i*3+j,
                target_voxels_sha256='GT'+str(i),**{m:value for m in METRICS}) for i in range(128)])
            for j in range(3)]
    return results


def test_all_causal_module_and_pipeline_contrasts_use_the_complete_planned_families():
    result=summarize_unified_modules(records(),draws=100)
    assert len(result['comparisons'])==12
    assert result['per_model_mean_asset']['V1']['voxel_iou']==pytest.approx(.4)
    assert result['per_model_native_global_occupancy_metrics']['V1'][0]['voxel_iou']==.3
    for name,c in result['comparisons'].items():
        assert c['prespecified_IoU_family_size']==(2 if name.startswith('pipeline_') else 10)
    assert result['comparisons']['pipeline_V1']['mean_asset_delta']['voxel_iou']==pytest.approx(.1)
    assert result['comparisons']['matching_decoder']['candidate']=='C_D1_COS'
    assert result['comparisons']['matching_scalar_score']['reference']=='C_D1_COS'
    assert not result['automatic_model_adoption'] and not result['full_cascade_gain_proven']


@pytest.mark.parametrize('damage',['missing-model','wrong-source-count','wrong-variant','wrong-GT',
    'wrong-seed','wrong-cluster','row-order','trained-during-eval','nonfinite','same-noise'])
def test_incomplete_or_unpaired_results_cannot_be_published_as_module_effects(damage):
    r=records();target=r['FULL_QK_ONLY'][0]
    if damage=='missing-model':r.pop('C_D1_COS')
    elif damage=='wrong-source-count':target['actual_source_tensors']=12
    elif damage=='wrong-variant':target['native_variant']='R_raw_topk'
    elif damage=='wrong-GT':target['per_asset'][0]['target_voxels_sha256']='other'
    elif damage=='wrong-seed':target['per_asset'][0]['seed']+=999
    elif damage=='wrong-cluster':target['per_asset'][0]['source_cluster_id']='other'
    elif damage=='row-order':target['per_asset'].reverse()
    elif damage=='trained-during-eval':target['optimizer_updates_during_evaluation']=1
    elif damage=='nonfinite':target['per_asset'][0]['voxel_iou']=np.nan
    else:
        for noises in r.values():noises[2]['per_asset'][0]['seed']=noises[1]['per_asset'][0]['seed']
    with pytest.raises(ValueError):summarize_unified_modules(r,draws=100)
