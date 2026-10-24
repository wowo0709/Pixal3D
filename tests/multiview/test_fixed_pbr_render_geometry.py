from copy import deepcopy

import numpy as np
import pytest

from experiments.deformnext_retargeting.fixed_pbr_render_geometry import (
    FACE_BUDGET, render_only_prediction, verify_render_only_prediction,
)


def fixture():
    mesh = dict(vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
                faces=np.array([[0, 1, 2]], np.int32))
    common = dict(**deepcopy(mesh), coords=np.array([[1, 2, 3]], np.int32),
                  origin=np.zeros(3, np.float32), voxel_size=np.array(.01, np.float32),
                  voxel_shape=np.array([1024] * 3, np.int32),
                  shape_coords=np.array([[1, 2, 3]], np.int32),
                  shape_feats=np.ones((1, 32), np.float32),
                  attrs=np.ones((1, 6), np.float32), texture_feats=np.ones((1, 32), np.float32))
    model = deepcopy(common)
    model['attrs'] *= .7
    model['texture_feats'] *= .3
    return model, common, mesh


def test_material_variants_keep_own_material_on_one_frozen_render_mesh():
    model, common, frozen = fixture()
    before = deepcopy(model)
    derived = render_only_prediction(model, common, frozen)
    proof = verify_render_only_prediction(derived, model, frozen, deepcopy(frozen))
    assert proof['rendered_vertices_faces_exact']
    assert proof['new_generation_model_forwards'] == 0
    assert all(np.array_equal(model[k], before[k]) for k in model)
    assert np.array_equal(derived['attrs'], model['attrs'])


def test_large_raw_mesh_uses_previously_simplified_vertices_without_touching_sample():
    model, common, frozen = fixture()
    common['faces'] = np.tile(common['faces'], (FACE_BUDGET + 1, 1))
    model['faces'] = common['faces'].copy()
    frozen['vertices'][0, 0] = .01
    derived = render_only_prediction(model, common, frozen)
    verify_render_only_prediction(derived, model, frozen, frozen)
    assert len(derived['faces']) == 1
    assert len(model['faces']) == FACE_BUDGET + 1
    assert model['vertices'][0, 0] == 0


@pytest.mark.parametrize('field', ['vertices', 'faces', 'coords', 'shape_feats'])
def test_different_raw_geometry_or_sparse_support_cannot_be_called_pbr_only(field):
    model, common, frozen = fixture()
    model[field].flat[0] += 1
    with pytest.raises(ValueError, match='raw geometry/support'):
        render_only_prediction(model, common, frozen)


@pytest.mark.parametrize('damage', ['nonfinite', 'index', 'dtype', 'over_budget', 'changed_bounded'])
def test_frozen_mesh_must_be_valid_and_must_skip_another_simplification(damage):
    model, common, frozen = fixture()
    if damage == 'nonfinite':frozen['vertices'][0, 0] = np.nan
    elif damage == 'index':frozen['faces'][0, 0] = 3
    elif damage == 'dtype':frozen['vertices'] = frozen['vertices'].astype(np.float64)
    elif damage == 'over_budget':frozen['faces'] = np.tile(frozen['faces'], (FACE_BUDGET + 1, 1))
    else:frozen['vertices'][0, 0] += .01
    with pytest.raises(ValueError):render_only_prediction(model, common, frozen)


@pytest.mark.parametrize('damage', ['renderer_mesh', 'derivative_material', 'derivative_support'])
def test_actual_renderer_output_and_material_are_checked_not_only_face_count(damage):
    model, common, frozen = fixture()
    derived = deepcopy(render_only_prediction(model, common, frozen))
    rendered = deepcopy(frozen)
    if damage == 'renderer_mesh':rendered['vertices'][0, 0] += .01
    elif damage == 'derivative_material':derived['attrs'][0, 0] += .01
    else:derived['coords'][0, 0] += 1
    with pytest.raises(ValueError):verify_render_only_prediction(derived, model, frozen, rendered)
