import pytest

from experiments.deformnext_retargeting.projected_module_summary import summarize_projected_modules as old_summary
from experiments.deformnext_retargeting.projected_module_summary_v2 import summarize_projected_modules
from test_projected_module_summary import five


def test_scalar_score_split_preserves_measurements_and_uses_full_ten_comparison_family():
    records=five();old=old_summary(records,draws=100);new=summarize_projected_modules(records,draws=100)
    assert new['per_model_mean_asset']==old['per_model_mean_asset']
    for key,current in new['comparisons'].items():
        original=old['comparisons'][key]
        assert current['mean_asset_delta']==original['mean_asset_delta']
        assert current['source_cluster_CI95']==original['source_cluster_CI95']
        if key.startswith(('C_TOP1_V','C_FIXED_TOPK')) or key=='R_raw_topk_minus_C_FIXED_TOPK':
            assert current['prespecified_IoU_family_size']==10
            assert current['primary_IoU_family_CI95'][0]<=original['primary_IoU_family_CI95'][0]
            assert current['primary_IoU_family_CI95'][1]>=original['primary_IoU_family_CI95'][1]
        else:assert current['prespecified_IoU_family_size']==2
