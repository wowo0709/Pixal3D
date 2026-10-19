"""Ray features must pass through each original camera's native projector."""
from types import SimpleNamespace

import pytest
import torch

from pixal3d.modules.ray_group_matching import RayGroupConditioner
from pixal3d.modules.ray_group_query import native_ray_membership
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    ProjGrid, compute_multiview_projection_matrices,
)
from pixal3d.trainers.flow_matching.ray_group_matching import (
    RayGroupImageConditionedProjFlowMatchingCFGTrainer, project_ray_delta,
)


def camera():
    transforms = torch.eye(4).repeat(1, 4, 1, 1)
    transforms[0, :, 0, 3] = torch.tensor([0., .2, -.15, .1])
    transforms[0, :, 1, 3] = 1.8
    return dict(transform_matrix=transforms, distance=transforms[:, :, :3, 3].norm(dim=-1),
                camera_angle_x=torch.tensor([[1.1, 1.05, 1.2, 1.]]), mesh_scale=torch.tensor([1.1]))


def project_views(grid, maps, cam):
    matrices, _ = compute_multiview_projection_matrices(cam['transform_matrix'], cam['distance'], grid.front_view_transform_matrix)
    views = [grid(maps[:, view], cam['camera_angle_x'][:, view], cam['distance'][:, view],
                  cam['mesh_scale'], matrices[:, view]) for view in range(4)]
    return (views[0].float() + views[1].float() + views[2].float() + views[3].float()) / 4


def test_ray_delta_equals_reprojecting_the_modified_native_maps():
    torch.manual_seed(101)
    grid, cam = ProjGrid(4, 518), camera()
    maps = torch.randn(1, 4, 32, 32, 4)
    native = project_views(grid, maps, cam)
    delta = torch.randn(1, 3, 1024, 4) * .05
    modified = maps.clone()
    modified[:, 1:] += delta.reshape(1, 3, 32, 32, 4)
    expected = project_views(grid, modified, cam)
    actual = native + project_ray_delta(grid, delta, cam)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert torch.equal(native + project_ray_delta(grid, torch.zeros_like(delta), cam), native)
    changed_camera = dict(cam, transform_matrix=cam['transform_matrix'].roll(1, dims=1))
    assert not torch.allclose(project_ray_delta(grid, delta, changed_camera), actual - native)


def test_actual_ray_trainer_zero_start_keeps_native_projection_and_global_condition():
    torch.manual_seed(12)
    grid, cam = ProjGrid(2, 518), camera()
    parent = object.__new__(RayGroupImageConditionedProjFlowMatchingCFGTrainer)
    parent._image_cond_module = SimpleNamespace(proj_grid=grid)
    parent.deform_conditioner = RayGroupConditioner(arm='c1', implementation='r9-native-v5',
        key_dim=1024, value_dim=1024, query_dim=1024, topk=64, hidden_dim=256,
        projection_mode='native-bilinear', score_temperature=.5, ray_mode='semantic-attention',
        task_attention=dict(decoder=True, score_prior=True, train_decoder=False, train_score=False,
                            content_mode='residual-zero', chunk_size=64, initial_content_scale=16.))
    target = torch.randn(1, 1024, 1024)
    sources = torch.randn(1, 3, 1024, 1024)
    maps = torch.cat((target[:, None], sources), dim=1).reshape(1, 4, 32, 32, 1024)
    native = project_views(grid, maps, cam)
    ids = native_ray_membership(grid, **cam, views=(0,))[:, 0]
    assert (ids >= 0).all()
    transport = dict(target_keys=torch.randn_like(target), target_query=target,
        source_keys=torch.randn_like(sources), source_values=sources,
        source_valid=torch.ones(1, 3, 1024, dtype=torch.bool),
        target_valid=torch.ones(1, 1024, dtype=torch.bool), patch_grid=ids.reshape(1, 2, 2, 2))
    global_cond = torch.randn(1, 1, 1024)
    result = parent._inject_deform_transport({'proj': native.clone(), 'global': global_cond.clone()},
        transport, None, anchor_condition={'proj': native}, camera_info=cam)
    assert torch.equal(result['proj'], native)
    assert torch.equal(result['global'], global_cond)
    (result['proj'] * torch.randn_like(result['proj'])).sum().backward()
    assert parent.deform_conditioner.fusion_gate.grad.abs() > 0
    assert parent.deform_conditioner.retargeter.net[-1].weight.grad.abs().sum() > 0
    assert parent._last_deform_condition.query_valid.shape == (1, 3 * 1024)
    with pytest.raises(ValueError, match='old anchor-patch'):
        parent._inject_deform_transport({'proj': native.clone()},
            dict(transport, semantic_indices=torch.zeros(1, 1024, 64, dtype=torch.long)), None,
            anchor_condition={'proj': native}, camera_info=cam)


def test_new_trainer_registry_is_separate_from_existing_voxel_trainer():
    from pixal3d import trainers
    assert trainers.RayGroupImageConditionedProjFlowMatchingCFGTrainer is RayGroupImageConditionedProjFlowMatchingCFGTrainer
    assert trainers.VoxelQueryImageConditionedProjFlowMatchingCFGTrainer.__module__.endswith('.voxel_query_matching')
