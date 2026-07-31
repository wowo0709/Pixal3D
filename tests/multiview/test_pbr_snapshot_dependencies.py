import os
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_bundled_forest_hdri_decodes_as_finite_float_rgb_image():
    hdri_path = REPO_ROOT / "assets" / "hdri" / "forest.exr"

    assert hdri_path.is_file()
    bgr = cv2.imread(str(hdri_path), cv2.IMREAD_UNCHANGED)

    assert bgr is not None
    assert bgr.ndim == 3
    assert bgr.shape[2] == 3
    assert np.issubdtype(bgr.dtype, np.floating)
    assert np.isfinite(bgr).all()

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    assert rgb.shape == bgr.shape
    assert rgb.dtype == bgr.dtype
    assert np.isfinite(rgb).all()
