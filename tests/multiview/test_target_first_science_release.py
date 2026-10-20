"""Initial quality releases must be paired and cannot release a regressed carrier."""
import copy
import pytest

from experiments.deformnext_retargeting.train_target_first_observed import initial_carrier_release


def report(delta):
    rows = [dict(sample_id=str(i), seed=i, target_voxels_sha256=str(i)) for i in range(32)]
    return dict(status='PASS_ACTUAL_TARGET_FIRST_INITIAL32_CARRIER_COMPARISON',
                panels={'target-carrier': dict(metrics={'voxel_iou': .5 + delta}, per_asset=copy.deepcopy(rows)),
                        'native-four-view-V1': dict(metrics={'voxel_iou': .5}, per_asset=copy.deepcopy(rows))},
                target_minus_native_V1_micro_IoU=(.5 + delta) - .5)


def test_regression_holds_only_own_ss():
    result = initial_carrier_release(report(-.01))
    assert result['status'] == 'HOLD_INITIAL_SS_CARRIER_REGRESSION'
    assert 'independent PBR science is unaffected' in result['scope']


def test_equal_carrier_can_release_without_claiming_v1_identity():
    result = initial_carrier_release(report(0))
    assert result['status'] == 'PASS'
    assert result['target_carrier_is_native_V1'] is False
    assert result['quality_gain_proven'] is False


@pytest.mark.parametrize('key', ['sample_id', 'seed', 'target_voxels_sha256'])
def test_unpaired_carriers_cannot_release(key):
    value = report(.01)
    value['panels']['target-carrier']['per_asset'][0][key] = 'wrong'
    with pytest.raises(ValueError, match='paired assets'):
        initial_carrier_release(value)


def test_inconsistent_or_nonfinite_difference_cannot_release():
    for difference in (.02, float('nan')):
        value = report(.01)
        value['target_minus_native_V1_micro_IoU'] = difference
        with pytest.raises(ValueError, match='inconsistent'):
            initial_carrier_release(value)
