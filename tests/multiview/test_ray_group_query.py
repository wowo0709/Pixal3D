"""Ray grouping preserves camera identity and shares one query along the ray."""
import pytest
import torch

from pixal3d.modules.ray_group_query import pool_ray_queries


def test_each_camera_has_its_own_ray_groups_and_empty_rays_are_invalid():
    x = torch.tensor([[[1., 2.], [3., 4.], [9., 10.], [20., 30.]]])
    ids = torch.tensor([[[0, 0, 1, -1], [1, 0, 1, -1]]])
    result = pool_ray_queries(x, ids, num_rays=3)
    torch.testing.assert_close(result.values[0, 0, 0], torch.tensor([2., 3.]))
    torch.testing.assert_close(result.values[0, 1, 0], torch.tensor([3., 4.]))
    torch.testing.assert_close(result.values[0, 1, 1], torch.tensor([5., 6.]))
    assert torch.equal(result.voxel_count, torch.tensor([[[2, 1, 0], [1, 2, 0]]]))
    assert not result.valid[..., 2].any()
    assert not result.values[..., 2, :].any()


def test_within_ray_voxel_permutation_preserves_the_single_ray_query():
    torch.manual_seed(21)
    x = torch.randn(1, 7, 13)
    ids = torch.tensor([[[0, 1, 0, 2, 1, 0, -1]]])
    permutation = torch.tensor([5, 4, 0, 3, 1, 2, 6])
    assert torch.equal(ids, ids[..., permutation])
    normal = pool_ray_queries(x, ids, num_rays=4)
    altered = pool_ray_queries(x[:, permutation], ids, num_rays=4)
    torch.testing.assert_close(normal.values, altered.values, rtol=1e-6, atol=1e-7)
    assert torch.equal(normal.valid, altered.valid)


def test_pooling_gradient_goes_to_all_and_only_members_of_the_selected_ray():
    x = torch.randn(1, 5, 3, requires_grad=True)
    ids = torch.tensor([[[0, 0, 1, -1, 2]]])
    pool_ray_queries(x, ids, num_rays=4).values[0, 0, 0].sum().backward()
    torch.testing.assert_close(x.grad[0, :2], torch.full((2, 3), .5))
    assert not x.grad[0, 2:].any()


def test_batches_do_not_share_ray_queries():
    x = torch.tensor([[[1.], [3.]], [[10.], [30.]]])
    ids = torch.zeros(2, 1, 2, dtype=torch.long)
    result = pool_ray_queries(x, ids, num_rays=1)
    torch.testing.assert_close(result.values.flatten(), torch.tensor([2., 20.]))


@pytest.mark.parametrize('ids', [torch.tensor([[[0, 2]]]), torch.tensor([[[-2, 0]]]),
                                  torch.tensor([[[0., 1.]]])])
def test_invalid_ray_ids_cannot_silently_enter_a_group(ids):
    with pytest.raises(ValueError):
        pool_ray_queries(torch.ones(1, 2, 3), ids, num_rays=2)
