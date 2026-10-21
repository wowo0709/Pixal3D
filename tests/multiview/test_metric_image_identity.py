import hashlib

import numpy as np
import pytest

from experiments.deformnext_retargeting.metric_image_identity import scored_image_observation


def test_actual_float_pixels_and_official_CHW_rounding_are_bound_separately():
    rgb = np.zeros((3, 512, 512), dtype=np.float32)
    rgb[0, 0, :4] = [-.1, .5, 1.1, .5001]
    gt = rgb.copy(); gt[1, 0, 0] = .25
    arrays = dict(prediction=rgb, GT_shared_renderer=gt, GT_alpha=np.ones((512, 512), dtype=np.float32))
    kwargs = dict(sample_id='synthetic-only', camera_id=8, source_cluster_id='source',
                  archive='/synthetic-only.npz', archive_SHA256='archive')
    before = scored_image_observation(arrays, **kwargs)
    assert before['prediction_float_RGB_SHA256'] == hashlib.sha256(rgb.tobytes()).hexdigest()
    converted = (np.clip(rgb, 0, 1)*255).round().astype(np.uint8)
    assert before['prediction_CHW_uint8_SHA256'] == hashlib.sha256(converted.tobytes()).hexdigest()
    # Two distinct native floats can quantize to the same official input. The
    # join must retain both identities rather than only the rounded image.
    arrays['prediction'][0, 0, 3] = .5002
    after = scored_image_observation(arrays, **kwargs)
    assert after['prediction_float_RGB_SHA256'] != before['prediction_float_RGB_SHA256']
    assert after['prediction_CHW_uint8_SHA256'] == before['prediction_CHW_uint8_SHA256']
    assert after['reference_float_RGB_SHA256'] == before['reference_float_RGB_SHA256']


@pytest.mark.parametrize('problem', ['HWC', 'double', 'NaN', 'alpha-shape'])
def test_scored_image_fingerprints_require_actual_native_FP32_fullframes(problem):
    arrays = dict(prediction=np.zeros((3, 512, 512), dtype=np.float32),
        GT_shared_renderer=np.ones((3, 512, 512), dtype=np.float32),
        GT_alpha=np.ones((512, 512), dtype=np.float32))
    if problem == 'HWC': arrays['prediction'] = arrays['prediction'].transpose(1, 2, 0)
    elif problem == 'double': arrays['GT_shared_renderer'] = arrays['GT_shared_renderer'].astype(np.float64)
    elif problem == 'NaN': arrays['GT_shared_renderer'][0, 0, 0] = np.nan
    else: arrays['GT_alpha'] = arrays['GT_alpha'][None]
    with pytest.raises(ValueError):
        scored_image_observation(arrays, sample_id='synthetic-only', camera_id=8, source_cluster_id='source',
                                 archive='/synthetic-only.npz', archive_SHA256='archive')
