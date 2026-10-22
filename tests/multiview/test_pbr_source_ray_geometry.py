import pytest
import torch

from pixal3d.modules.pbr_source_ray_geometry import (
    pool_sparse_source_ray_queries, project_sparse_pbr_source_delta,
    sparse_source_ray_membership,
)
from pixal3d.modules.ray_group_query import native_ray_membership, pool_ray_queries
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    ProjGrid, compute_multiview_projection_matrices,
)


def fixture():
    grid = ProjGrid(4, 512)
    transforms = grid.front_view_transform_matrix.repeat(2, 4, 1, 1)
    transforms[1, :, 1, 3] = -2.4
    transforms[:, 2, 0, 3] = .12
    camera = dict(transform_matrix=transforms, distance=torch.tensor([[2.] * 4, [2.4] * 4]),
        camera_angle_x=torch.tensor([[.8, .9, 1., 1.1], [1., .8, 1.1, .9]]),
        mesh_scale=torch.tensor([.85, 1.2]))
    # Deliberately interleave batches and use unequal numbers of rows.
    coords = torch.tensor([[1, 3, 1, 2], [0, 0, 0, 0], [1, 0, 2, 1],
        [0, 3, 3, 3], [0, 1, 2, 3]], dtype=torch.int32)
    return grid, coords, camera


def flat_indices(coords, resolution):
    return ((coords[:, 1].long() * resolution + coords[:, 2]) * resolution + coords[:, 3]).long()


def dense_projection(grid, maps, camera):
    matrices, _ = compute_multiview_projection_matrices(camera['transform_matrix'],
        camera['distance'], grid.front_view_transform_matrix)
    parts = [grid(maps[:, source].reshape(2, 32, 32, 2048),
        camera['camera_angle_x'][:, source+1], camera['distance'][:, source+1],
        camera['mesh_scale'], matrices[:, source+1]).float() for source in range(3)]
    return (parts[0] + parts[1] + parts[2]) / 4


def test_packed_membership_matches_native_rotated_grid_for_every_camera():
    grid, coords, camera = fixture()
    dense = native_ray_membership(grid, **camera, views=(1, 2, 3))
    actual = sparse_source_ray_membership(grid, coords, camera)
    expected = dense[coords[:, 0].long(), :, flat_indices(coords, 4)]
    assert torch.equal(actual, expected)
    order = torch.tensor([3, 0, 4, 1, 2])
    assert torch.equal(sparse_source_ray_membership(grid, coords[order], camera), actual[order])


def test_packed_pooling_matches_real_dense_members_without_padding_observations_and_gradients():
    features = torch.randn(5, 7, requires_grad=True)
    batch = torch.tensor([1, 0, 1, 0, 0])
    rays = torch.tensor([[2, 1, -1], [2, 1, 4], [2, 3, 4], [2, -1, 4], [9, 1, 4]])
    packed = pool_sparse_source_ray_queries(features, rays, batch, batch_size=2)
    padded = features.new_zeros(2, 5, 7)
    padded_rays = torch.full((2, 3, 5), -1, dtype=torch.long)
    for row in range(5):
        padded[batch[row], row] = features[row]
        padded_rays[batch[row], :, row] = rays[row]
    reference = pool_ray_queries(padded, padded_rays, num_rays=1024)
    assert torch.equal(packed.values, reference.values)
    assert torch.equal(packed.voxel_count, reference.voxel_count)
    assert torch.equal(packed.valid, reference.valid)
    weight = torch.randn_like(packed.values)
    actual_grad = torch.autograd.grad((packed.values * weight).sum(), features, retain_graph=True)[0]
    expected_grad = torch.autograd.grad((reference.values * weight).sum(), features)[0]
    assert torch.equal(actual_grad, expected_grad)


@pytest.mark.parametrize('chunk', [127, 2048])
def test_sparse_dino_and_NAF_delivery_matches_native_dense_projection_and_gradient(chunk):
    grid, coords, camera = fixture()
    maps = torch.randn(2, 3, 1024, 2048, requires_grad=True)
    mass = torch.rand(2, 3, 1024, 1, requires_grad=True)
    original = torch.randn_like(maps)
    actual = project_sparse_pbr_source_delta(grid, maps, mass, original, coords, camera,
        channel_chunk=chunk)
    dense = dense_projection(grid, maps - mass * original, camera)
    expected = dense[coords[:, 0].long(), flat_indices(coords, 4)]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-6)
    weight = torch.randn_like(actual)
    a = torch.autograd.grad((actual * weight).sum(), (maps, mass), retain_graph=True)
    b = torch.autograd.grad((expected * weight).sum(), (maps, mass))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=3e-5, rtol=2e-5)
    assert actual[:, :1024].abs().sum() > 0 and actual[:, 1024:].abs().sum() > 0


def test_zero_source_delta_preserves_native_exact_and_R_bypasses_zero_semantic_gate():
    grid, coords, camera = fixture()
    original = torch.randn(2, 3, 1024, 2048)
    mass = torch.rand(2, 3, 1024, 1) - .5
    zero = project_sparse_pbr_source_delta(grid, mass * original, mass, original, coords, camera)
    assert not zero.any()
    residual = torch.full_like(original, .2)
    actual = project_sparse_pbr_source_delta(grid, residual, torch.zeros_like(mass), original, coords, camera)
    torch.testing.assert_close(actual, torch.full_like(actual, .15), atol=1e-7, rtol=0)


def test_border_delivery_keeps_native_prior_rule_without_visibility_write_gate():
    grid, coords, camera = fixture()
    camera['camera_angle_x'].fill_(.01)
    assert (sparse_source_ray_membership(grid, coords, camera) == -1).all()
    maps = torch.full((2, 3, 1024, 2048), .2)
    actual = project_sparse_pbr_source_delta(grid, maps, maps[..., :1] * 0,
        torch.ones_like(maps), coords, camera)
    # Native projection samples the border, including for out-of-frustum rows.
    torch.testing.assert_close(actual, torch.full_like(actual, .15), atol=1e-7, rtol=0)


def test_invalid_coords_and_feature_layout_are_rejected():
    grid, coords, camera = fixture()
    for column, value in [(0, 2), (1, -1), (3, 4)]:
        bad = coords.clone(); bad[0, column] = value
        with pytest.raises(ValueError):
            sparse_source_ray_membership(grid, bad, camera)
    maps = torch.zeros(2, 3, 1024, 1024)
    with pytest.raises(ValueError):
        project_sparse_pbr_source_delta(grid, maps, maps[..., :1], maps, coords, camera)
    with pytest.raises(ValueError):
        pool_sparse_source_ray_queries(torch.zeros(5, 2), torch.zeros(5, 3, dtype=torch.long),
            torch.tensor([0, 0, 2, 0, 0]), batch_size=2)
