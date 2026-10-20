import copy

import pytest

from experiments.deformnext_retargeting.render_panel_statistics import paired_panel_statistics


def panel():
    groups = ['A', 'A', 'A', 'B']
    rows = [dict(sample_id=str(i), source_cluster_id=group, metric=0.)
            for i, group in enumerate(groups)]
    result = {model: copy.deepcopy(rows) for model in ['v1', 'retargeting', 'direct']}
    for i, value in enumerate([1., 1., 1., -3.]):
        result['direct'][i]['metric'] = value
    return result


def test_equal_source_and_equal_deformation_estimands_remain_distinct():
    result = paired_panel_statistics(panel(), ['metric'], draws=100)
    assert result['assets'] == 4 and result['source_clusters'] == 2
    contrast = result['contrasts']['direct_minus_v1']
    assert contrast['deformation_weighted_delta']['metric'] == 0
    assert contrast['source_weighted_delta']['metric'] == -1
    # Resampling the two complete sources yields only -3, -1 or1 for
    # equal-source means. The four deformations are never resampled separately.
    assert contrast['source_weighted_cluster_CI95']['metric'] == [-3, 1]
    identical = result['contrasts']['retargeting_minus_v1']
    assert identical['deformation_weighted_cluster_CI95']['metric'] == [0, 0]


@pytest.mark.parametrize('problem', ['cluster', 'sample', 'nonfinite'])
def test_comparison_rejects_unpaired_or_nonfinite_records(problem):
    records = panel()
    if problem == 'cluster':
        records['direct'][0]['source_cluster_id'] = 'C'
    elif problem == 'sample':
        records['direct'][0]['sample_id'] = 'other'
    else:
        records['direct'][0]['metric'] = float('nan')
    with pytest.raises(ValueError):
        paired_panel_statistics(records, ['metric'])
