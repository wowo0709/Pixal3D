import numpy as np
import pytest

from experiments.deformnext_retargeting.prepare_surface_reference import surface_reference
from experiments.deformnext_retargeting.surface_metrics import sample_surface


def fixture():
    source = np.asarray([[5., -4., 1.], [9., -4., 1.], [5., 0., 1.], [5., -4., 5.]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    target = source + np.asarray([1., .25, -.5], dtype=np.float32)
    return source, faces, target


def test_reference_is_bitwise_original_cascade_sampler_in_source_defined_frame():
    source, faces, target = fixture()
    result = surface_reference(source, faces, target, faces)
    center = (source.min(0) + source.max(0)) / 2
    extent = float(np.ptp(source, axis=0).max())
    legacy = sample_surface((target-center)/extent, faces, seed=101)
    assert np.array_equal(result['target_points'], legacy)
    assert result['target_points'].shape == (20000, 3)
    assert result['target_points'].dtype == np.float64
    assert np.array_equal(result['source_center'], center)
    assert result['source_extent'] == extent


def test_reference_retains_target_translation_instead_of_fitting_it_away():
    source, faces, target = fixture()
    native = surface_reference(source, faces, source, faces, count=100)
    moved = surface_reference(source, faces, target, faces, count=100)
    np.testing.assert_allclose(moved['target_points']-native['target_points'],
        np.broadcast_to(np.asarray([1., .25, -.5])/4, (100, 3)), rtol=0, atol=1e-15)


def test_source_topology_mismatch_is_rejected_without_guessing():
    source, faces, target = fixture()
    with pytest.raises(ValueError, match='exact target faces'):
        surface_reference(source, faces, target, faces[:, ::-1])


@pytest.mark.parametrize('invalid', ['nonfinite', 'zero_extent', 'target_shape'])
def test_invalid_geometry_cannot_become_a_cached_reference(invalid):
    source, faces, target = fixture()
    if invalid == 'nonfinite':
        target[0, 0] = np.nan
    elif invalid == 'zero_extent':
        source[:] = 0
    else:
        target = target[:-1]
    with pytest.raises(ValueError):
        surface_reference(source, faces, target, faces)
