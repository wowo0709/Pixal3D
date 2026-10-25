"""Protect actual scored-image identity when reusing expensive GPU endpoints."""
from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.common_geometry_pbr_contract import MODES
from experiments.deformnext_retargeting.selected_pbr_official_reuse import merge_official_reports


def fixture():
    rows = [dict(arm=mode, sample_id=f'asset{i}', source_cluster_id=f'cluster{i//2}',
                 camera_id=cid, shared_npz_SHA256=f'{m*100+i*3+cid:064x}',
                 scores={'DreamSim_fullframe': .01*(i+cid+m)},
                 image_audit={'converted_image_SHA256': f'image{m}/{i}/{cid}'})
            for m, mode in enumerate(MODES) for i in range(32) for cid in (8, 9, 10)]
    meta = dict(status='PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION',
                metric_registry_SHA256='a'*64, model_precision='float32,TF32off,noAMP',
                full_panel=True, no_model_training=True, global_environment_changed=False,
                GPU_sanity=dict(status='PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY',
                                CPU_model_forward_or_training_updates=0),
                model_weights={'DreamSim': 'b'*64, 'CLIP': 'c'*64})
    full = dict(meta, per_view=deepcopy(rows))
    fresh = dict(meta, per_view=deepcopy(rows[192:]))
    reuse = []
    for m, mode in enumerate(MODES[:2]):
        old = dict(meta, per_view=deepcopy(rows[m*96:(m+1)*96]))
        for row in old['per_view']:
            row['arm'] = 'SELECTED'
        reuse.append(dict(mode=mode, source_arm='SELECTED', record=old,
                          spec=dict(path=f'/results/{mode}/evaluation.json', SHA256=f'{m+1:064x}')))
    return full, fresh, reuse, rows


def merge(fresh, reuse, rows):
    return merge_official_reports(fresh, reuse, expected_inputs=rows,
                                  metric_registry_SHA256='a'*64)


def test_reuse_preserves_every_score_and_image_audit_without_mutating_inputs():
    full, fresh, reuse, rows = fixture()
    before = deepcopy((fresh, reuse, rows))
    result = merge(fresh, reuse, rows)
    assert result['per_view'] == full['per_view']
    assert result['actual_current_GPU_views'] == result['reused_closed_GPU_views'] == 192
    assert not result['all384_rows_recomputed_this_job']
    assert (fresh, reuse, rows) == before
    result['per_view'][0]['scores']['DreamSim_fullframe'] = -1
    assert reuse[0]['record']['per_view'][0]['scores']['DreamSim_fullframe'] >= 0


def test_fresh_four_mode_run_also_retains_exact_results():
    full, _, _, rows = fixture()
    result = merge(full, [], rows)
    assert result['per_view'] == full['per_view']
    assert result['actual_current_GPU_views'] == 384 and result['reused_closed_GPU_views'] == 0


@pytest.mark.parametrize('where', ['fresh', 'reused'])
@pytest.mark.parametrize('field,value', [('shared_npz_SHA256', 'f'*64),
                                       ('source_cluster_id', 'another-source')])
def test_wrong_geometry_render_pixels_or_source_are_rejected(where, field, value):
    _, fresh, reuse, rows = fixture()
    report = fresh if where == 'fresh' else reuse[0]['record']
    report['per_view'][0][field] = value
    with pytest.raises(ValueError, match='another source or scored pixel'):
        merge(fresh, reuse, rows)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'overlap', 'weights', 'registry',
                                   'CPU', 'camera', 'expected-cluster', 'missing-expected'])
def test_incomplete_ambiguous_or_different_protocol_reuse_is_rejected(change):
    _, fresh, reuse, rows = fixture()
    if change == 'missing': reuse[0]['record']['per_view'].pop()
    elif change == 'duplicate': reuse[0]['record']['per_view'][1] = deepcopy(reuse[0]['record']['per_view'][0])
    elif change == 'overlap': reuse[0]['mode'] = MODES[2]
    elif change == 'weights':
        reuse[0]['record']['model_weights'] = {'DreamSim': 'd'*64, 'CLIP': 'c'*64}
    elif change == 'registry': reuse[0]['record']['metric_registry_SHA256'] = 'e'*64
    elif change == 'CPU':
        reuse[0]['record']['GPU_sanity'] = dict(status='PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY',
                                               CPU_model_forward_or_training_updates=1)
    elif change == 'camera': fresh['per_view'][0]['camera_id'] = '8'
    elif change == 'expected-cluster': rows[96]['source_cluster_id'] = 'wrong'
    elif change == 'missing-expected': rows.pop()
    with pytest.raises(ValueError):
        merge(fresh, reuse, rows)
