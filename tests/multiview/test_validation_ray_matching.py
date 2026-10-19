import numpy as np
import pytest

from experiments.deformnext_retargeting.validation_ray_matching import positive_sets


def test_visible_source_relations_do_not_cross_view_or_include_background():
    ids = np.tile(np.arange(64), (2, 1))
    ids[0, :4] = [33, 1024+33, 2048+34, 2048+99]
    teacher = dict(positive_indices=np.array([[33, 1024+33, 2048+33], [0, 1024, 2048]]),
        positive_visible=np.array([[True, True, True], [True, True, True]]),
        target_foreground=np.array([True, False]))
    relation, exact, near = positive_sets(ids, np.arange(2), teacher)
    assert relation.tolist() == [0, 0, 0]
    assert exact[0, 0] and exact[1, 1] and not exact[2].any()
    assert near[2, 2] and not near[2, 3]
    assert not near[0, 1] and not near[1, 0]
    assert exact.shape == near.shape == (3, 64)


def test_empty_rays_and_invalid_source_addresses():
    teacher = dict(positive_indices=np.empty((0, 3), dtype=np.int64),
        positive_visible=np.empty((0, 3), dtype=bool), target_foreground=np.empty(0, dtype=bool))
    relation, exact, near = positive_sets(np.empty((0, 64), dtype=np.int64), np.empty(0, dtype=np.int64), teacher)
    assert relation.shape == (0,) and exact.shape == near.shape == (0, 64)
    with pytest.raises(ValueError, match='native64'):
        positive_sets(np.full((1, 64), 3072), np.array([0]), teacher)
