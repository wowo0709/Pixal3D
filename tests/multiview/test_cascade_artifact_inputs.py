import numpy as np
from PIL import Image
import pytest
from experiments.deformnext_retargeting.cascade_artifact_inputs import native_shape_support,prepared_views


def test_native_SS64_support_is_pooled_into_shape512_grid32():
    occupancy=np.zeros((64,64,64),dtype=bool)
    occupancy[2,4,6]=True;occupancy[3,5,7]=True;occupancy[63,63,63]=True
    assert native_shape_support(occupancy).tolist()==[[0,1,2,3],[0,31,31,31]]
    with pytest.raises(ValueError):native_shape_support(np.zeros_like(occupancy))


def test_prepared_RGBA_black_compositing_preserves_camera_frame():
    pixels=np.zeros((8,8,4),dtype=np.uint8);pixels[2:6,1:4]=[255,0,0,128]
    value=prepared_views([Image.fromarray(pixels)],8,'cpu')
    assert value.shape==(1,1,3,8,8)
    assert value[0,0,0,2,1].item()==pytest.approx(128/255)
    assert value[0,0,:,0,0].count_nonzero()==0
