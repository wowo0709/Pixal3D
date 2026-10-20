import copy

import pytest

from experiments.deformnext_retargeting.render_view_role_statistics import view_role_statistics


def observations():
    base = [dict(sample_id=str(i), source_cluster_id=cluster, camera_id=camera, metric=0.)
            for i, cluster in enumerate(['A', 'A', 'B']) for camera in (8, 9, 10)]
    rows = {model:copy.deepcopy(base) for model in ['v1', 'retargeting', 'direct']}
    for row in rows['direct']:
        row['metric'] = .3 if row['camera_id'] == 8 else -.15
    return rows


def test_target_source_tradeoff_is_visible_when_three_view_mean_cancels():
    result = view_role_statistics(observations(), ['metric'], draws=100)
    roles = result['roles']
    contrast = 'direct_minus_v1'
    assert roles['all_three_cameras']['contrasts'][contrast]['source_weighted_delta']['metric'] == 0
    assert roles['edited_target_camera8']['contrasts'][contrast]['source_weighted_delta']['metric'] == pytest.approx(.3)
    assert roles['source_cameras9_10_M1_reference']['contrasts'][contrast]['source_weighted_delta']['metric'] == pytest.approx(-.15)
    interaction = roles['target_minus_source_camera_interaction']['contrasts'][contrast]
    assert interaction['source_weighted_cluster_CI95']['metric'] == pytest.approx([.45, .45])
    assert roles['all_three_cameras']['assets'] == 3
    assert roles['all_three_cameras']['source_clusters'] == 2


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'cluster', 'nonfinite'])
def test_roles_reject_missing_duplicate_or_unpaired_camera_observations(problem):
    rows = observations()
    if problem == 'missing':
        rows['direct'].pop()
    elif problem == 'duplicate':
        rows['direct'].append(rows['direct'][0])
    elif problem == 'cluster':
        rows['direct'][0]['source_cluster_id'] = 'C'
    else:
        rows['direct'][0]['metric'] = float('nan')
    with pytest.raises(ValueError):
        view_role_statistics(rows, ['metric'])
