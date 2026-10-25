"""Exercise the producer JSON/consumer statistics boundary, including pairing."""
from copy import deepcopy
import json

import pytest

from experiments.deformnext_retargeting.cascade_record_json import decode_cascade_records
from experiments.deformnext_retargeting.cascade_view_roles import IMAGE_METRICS, summarize_view_roles


def records():
    return {
        model: [dict(sample_id=f'asset{i}', source_cluster_id=f'source{i // 2}',
            views={camera: {metric: .1 + .01 * (i + camera + j + model_index)
                            for j, metric in enumerate(IMAGE_METRICS)}
                   for camera in (8, 9, 10)}, CD=.01 + .001 * i,
            FScore=.7 + .01 * (i + model_index), metadata={'8': 'unchanged'})
            for i in range(4)]
        for model_index, model in enumerate(('reference', 'candidate'))
    }


def test_json_roundtrip_preserves_actual_reducer_means_pairing_and_intervals():
    original = records()
    serialized = json.loads(json.dumps(original))
    before = deepcopy(serialized)
    with pytest.raises(ValueError, match='Exactly cameras'):
        summarize_view_roles(serialized, [('candidate', 'reference')], draws=100, seed=17)
    decoded = decode_cascade_records(serialized)
    assert decoded == original
    assert serialized == before
    assert summarize_view_roles(decoded, [('candidate', 'reference')], draws=100, seed=17) == (
        summarize_view_roles(original, [('candidate', 'reference')], draws=100, seed=17))


def test_internal_integer_records_remain_equal_and_do_not_alias_input():
    original = records()
    decoded = decode_cascade_records(original)
    assert decoded == original
    decoded['reference'][0]['views'][8]['PSNR'] = -1
    assert original['reference'][0]['views'][8]['PSNR'] != -1
    assert decoded['reference'][0]['metadata'] == {'8': 'unchanged'}


@pytest.mark.parametrize('keys', [
    ('8', 9, 10), ('08', '9', '10'), ('8', '9', '11'), ('8', '9'),
    (8., 9., 10.), (8, 9, 10, '8'), (),
])
def test_ambiguous_missing_or_unexpected_cameras_are_rejected(keys):
    value = records()
    value['candidate'][0]['views'] = {key: {} for key in keys}
    with pytest.raises(ValueError, match='homogeneous cameras'):
        decode_cascade_records(value)


@pytest.mark.parametrize('value', [{}, [], {'model': []}, {'model': [None]}, {'model': [{}]}])
def test_incomplete_records_are_not_filled(value):
    with pytest.raises(ValueError):
        decode_cascade_records(value)
