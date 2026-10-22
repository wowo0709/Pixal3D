import copy

import pytest

from experiments.deformnext_retargeting.evaluate_r_native_deterministic_epoch import (
    MODE, REPEATED, compare_repeats, validate_panel, check_original_identity,
)


def test_repeat_requires_exact_output_and_own_declared_protocol():
    row = {key: 'same' for key in REPEATED}
    row.update(evaluation_mode=MODE, deterministic_algorithms=True, quick_metrics={'IoU': .43})
    compare_repeats(row, copy.deepcopy(row))
    changed = copy.deepcopy(row); changed['quick_metrics']['IoU'] += 1e-12
    with pytest.raises(ValueError, match='quick_metrics'):
        compare_repeats(changed, row)
    with pytest.raises(ValueError, match='declared deterministic'):
        compare_repeats(dict(row, deterministic_algorithms=False), row)


def test_new_protocol_retains_original_panel_checks_and_cannot_claim_epoch_replay():
    seen = []
    def original(p, b, c):
        seen.append(p)
        return ['asset'], {'asset': 7}
    plan = dict(mode=MODE, deterministic_algorithms=True, panel='same32', generation_batch_size=8)
    assert validate_panel(plan, {'step':96}, {}, original=original) == (['asset'], {'asset':7})
    assert seen[0]['mode'] == 'epoch-replay' and not seen[0]['deterministic_algorithms']
    assert seen[0]['panel'] == plan['panel'] and plan['deterministic_algorithms']
    with pytest.raises(ValueError):
        validate_panel(dict(plan, mode='epoch-replay'), {'step':96}, {}, original=original)


def test_original_GT_and_noise_must_match_even_when_stochastic_metrics_differ():
    row = dict(sample_id='a', source_cluster_id='c', seed=9, target_voxels_sha256='GT',
               target_voxel_shape=[64,64,64], target_representation='decoded', voxel_iou=.4)
    actual = dict(step=96, quick_panel_ids=['a'], per_asset=[row])
    native = dict(step=96, quick_panel_ids=['a'], quick_per_asset=[dict(row, voxel_iou=.41)])
    check_original_identity(actual, native)
    native['quick_per_asset'][0]['seed'] = 10
    with pytest.raises(ValueError, match='GT/noise'):
        check_original_identity(actual, native)
