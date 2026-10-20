"""Shared, prediction-independent RGB conversion and GT foreground crops."""
from __future__ import annotations

import hashlib
import numpy as np
from PIL import Image


def paired_images(prediction, reference, reference_alpha):
    if (prediction.ndim != 3 or prediction.shape[0] != 3
            or reference.shape != prediction.shape
            or reference_alpha.shape != prediction.shape[1:]
            or not all(np.isfinite(x).all() for x in (prediction, reference, reference_alpha))):
        raise ValueError('finite paired CHW RGB and GT HW alpha required')
    y, x = np.nonzero(reference_alpha > .5)
    if not len(x):
        raise ValueError('nonempty GT foreground required')
    height, width = reference_alpha.shape
    margin = int(np.ceil(.1 * max(x.max()-x.min()+1, y.max()-y.min()+1)))
    box = (max(0, int(x.min())-margin), max(0, int(y.min())-margin),
           min(width, int(x.max())+margin+1), min(height, int(y.max())+margin+1))

    def image(array):
        rgb8 = np.rint(np.clip(array.transpose(1, 2, 0), 0, 1)*255).astype(np.uint8)
        return Image.fromarray(rgb8)

    pred, gt = image(prediction), image(reference)
    images = dict(prediction_full=pred, reference_full=gt,
                  prediction_GTcrop=pred.crop(box), reference_GTcrop=gt.crop(box))
    audit = dict(GT_only_crop_box=list(box), crop_padding_fraction=.1,
        conversion='clamp[0,1],round-to-nearest uint8 RGB;identical for prediction and GT',
        original_float_prediction_SHA256=hashlib.sha256(prediction.tobytes()).hexdigest(),
        original_float_GT_SHA256=hashlib.sha256(reference.tobytes()).hexdigest(),
        original_GT_alpha_SHA256=hashlib.sha256(reference_alpha.tobytes()).hexdigest(),
        converted_image_SHA256={name:hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
            for name,value in images.items()}, per_output_alignment=False)
    return images, audit
