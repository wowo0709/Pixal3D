import copy

import numpy as np
import pytest

from experiments.deformnext_retargeting.extract_official_inception import render_uint8, validate_pool


def pool():
    ids = [str(i) for i in range(508)]
    return dict(scope='final-test508',pool_role='generated-prediction',expected_sample_ids=ids,
        model_binding={'fixture_only':'not a scientific model'},
        observations=[dict(sample_id=sid,camera_id=cid,source_cluster_id='source'+sid,array='prediction')
                      for sid in ids for cid in (8,9,10)])


def test_complete_final_pool_keeps_every_camera_and_source():
    plan = pool()
    assert validate_pool(plan) is plan['observations']


@pytest.mark.parametrize('problem',['missing','duplicate','mixed-reference','cluster'])
def test_final_pool_rejects_silent_dropping_and_reference_contamination(problem):
    plan = pool()
    if problem == 'missing':
        plan['observations'].pop()
    elif problem == 'duplicate':
        plan['observations'][-1] = copy.deepcopy(plan['observations'][0])
    elif problem == 'mixed-reference':
        plan['observations'][0]['array'] = 'GT_shared_renderer'
    else:
        plan['observations'][0]['source_cluster_id'] = 'other'
    with pytest.raises(ValueError):
        validate_pool(plan)


def test_native_rgb_conversion_preserves_channel_order_and_official_uint8_domain():
    image = np.empty((3,512,512),dtype=np.float32)
    image[0],image[1],image[2] = .5,-1.,2.
    pixels = render_uint8(image)
    assert pixels.dtype == np.uint8
    assert pixels[:,0,0].tolist() == [128,0,255]
