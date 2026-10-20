import numpy as np
import pytest

from experiments.deformnext_retargeting.perceptual_image_inputs import paired_images


def test_crop_reference_never_depends_on_prediction_foreground():
    gt = np.zeros((3, 20, 30), dtype=np.float32)
    alpha = np.zeros((20, 30), dtype=np.float32)
    gt[:, 8:12, 14:18] = .5
    alpha[8:12, 14:18] = 1
    pred = gt.copy()
    images, audit = paired_images(pred, gt, alpha)
    pred[:, :3, :3] = 1
    changed, changed_audit = paired_images(pred, gt, alpha)
    assert audit['GT_only_crop_box'] == [13, 7, 19, 13]
    assert changed_audit['GT_only_crop_box'] == audit['GT_only_crop_box']
    assert not np.array_equal(np.asarray(images['prediction_full']),np.asarray(changed['prediction_full']))
    for name in ['reference_full','reference_GTcrop','prediction_GTcrop']:
        assert np.array_equal(np.asarray(images[name]),np.asarray(changed[name]))


def test_bad_rgb_and_empty_gt_foreground_are_rejected():
    rgb = np.zeros((3, 4, 4), dtype=np.float32)
    alpha = np.ones((4, 4), dtype=np.float32)
    bad = rgb.copy()
    bad[0,0,0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        paired_images(bad,rgb,alpha)
    with pytest.raises(ValueError, match='nonempty GT'):
        paired_images(rgb,rgb,alpha*0)
