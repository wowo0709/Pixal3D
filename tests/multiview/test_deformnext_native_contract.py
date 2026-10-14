from types import SimpleNamespace
import numpy as np
import pytest
import torch
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    ImageConditionedProjMixin, ProjGrid, compute_multiview_projection_matrices,
    project_points_to_image_batch,
)
from experiments.deformnext_retargeting.projection_maps import anchor_patch_indices


def cameras():
    angle = torch.tensor(0.7)
    rotation = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0],
                             [torch.sin(angle), torch.cos(angle), 0], [0, 0, 1.]])
    transform = torch.eye(4).repeat(1, 4, 1, 1)
    transform[:, :, :3, :3] = rotation
    transform[:, :, 1, 3] = -1.8
    return {'camera_angle_x': torch.full((1, 4), 0.8),
            'distance': torch.full((1, 4), 1.8),
            'mesh_scale': torch.tensor([0.96]), 'transform_matrix': transform}


def test_vanilla_bridge_preserves_native_four_view_inputs_exactly():
    bridge = object.__new__(ImageConditionedProjMixin)
    image = torch.randn(1, 4, 3, 8, 8)
    info = cameras()
    expected = {'global': torch.randn(1, 5, 6), 'proj': torch.randn(1, 8, 6)}
    negative = {key: torch.zeros_like(value) for key, value in expected.items()}
    calls = []
    def encode(images, **kwargs):
        calls.append((images, kwargs))
        return expected, negative
    bridge.encode_image_proj = encode
    actual, neg = bridge._encode_deform_views(image, info, None)
    assert actual is expected and neg is negative
    assert len(calls) == 1 and calls[0][0] is image
    assert calls[0][1]['transform_matrix'] is info['transform_matrix']


def test_semantic_bridge_retains_native_global_and_canonical_anchor():
    bridge = object.__new__(ImageConditionedProjMixin)
    info = cameras()
    grid = ProjGrid(grid_resolution=2, image_resolution=512)
    bridge.image_cond_model = SimpleNamespace(fixed_projection_transform=grid.front_view_transform_matrix)
    calls = []
    global_native = torch.full((1, 5, 6), 7.)
    def encode(images, **kwargs):
        calls.append((images, kwargs))
        value = 4. if images.ndim == 5 else 2.
        out = {'global': global_native if images.ndim == 5 else global_native * 0,
               'proj': torch.full((1, 8, 6), value)}
        return out, {key: torch.zeros_like(v) for key, v in out.items()}
    bridge.encode_image_proj = encode
    from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
    bridge.deform_conditioner = DeformationAwareConditioner(
        arm='b1', key_dim=2, value_dim=6, query_dim=1, topk=1)
    bridge.training_models = {}
    data = {'source_values': torch.full((1, 3, 1, 6), 10.),
            'source_valid': torch.ones(1, 3, 1, dtype=torch.bool),
            'semantic_indices': torch.zeros(1, 1, 1, dtype=torch.long),
            'patch_grid': torch.zeros(1, 2, 2, 2, dtype=torch.long)}
    out, _ = bridge._encode_deform_views(torch.ones(1, 4, 3, 8, 8), info, data)
    torch.testing.assert_close(out['global'], global_native)
    torch.testing.assert_close(out['proj'], torch.full((1, 8, 6), 8.))
    canonical, _ = compute_multiview_projection_matrices(
        info['transform_matrix'], info['distance'], grid.front_view_transform_matrix)
    torch.testing.assert_close(calls[1][1]['transform_matrix'], canonical[:, 0])
    assert not torch.equal(canonical[:, 0], info['transform_matrix'][:, 0])


@pytest.mark.parametrize('resolution', [16, 64])
def test_cache_projection_matches_native_anchor_frame(resolution):
    info = cameras()
    row = {'mesh_scale': 0.96, 'camera_angle_x': info['camera_angle_x'][0].tolist(),
           'transform_matrix': info['transform_matrix'][0].tolist()}
    grid = ProjGrid(grid_resolution=resolution, image_resolution=512)
    coords = np.array([[0, 0, 0], [resolution-1]*3, [resolution//2, 3, 7]])
    expected_camera, _ = compute_multiview_projection_matrices(
        info['transform_matrix'], info['distance'], grid.front_view_transform_matrix)
    points = grid.grid_points.reshape(resolution, resolution, resolution, 3)[tuple(coords.T)] / 0.96 / 2
    pixels, _, _ = project_points_to_image_batch(
        points[None], expected_camera[:, 0], info['camera_angle_x'][:, 0], 512)
    xy = torch.floor(pixels[0] / 16).long().clamp(0, 31)
    expected = (xy[:, 1] * 32 + xy[:, 0]).numpy()
    np.testing.assert_array_equal(anchor_patch_indices(row, coords, resolution), expected)


def test_generated_v1_has_no_transport_and_pbr_uses_retained_count(tmp_path):
    from experiments.deformnext_retargeting.configure import HERE, build
    args = SimpleNamespace(
        base_config_root=HERE / 'base_configs', output_root=tmp_path / 'runs',
        data_root=tmp_path / 'data', file4_mirror_root=None,
        training_assets=9149, pbr_training_assets=9137, epochs=5,
        ss_checkpoint='/weights/ss.pt', pbr_checkpoint='/weights/pbr.pt',
        q2c_checkpoint='/weights/q2c.pt', omega_checkpoint='/weights/omega.pt',
        num_workers=0, dino='/weights/dino', wandb_project=None,
    )
    ss = build('ss64', 'v1', args)
    pbr = build('pbr1024', 'v1', args)
    for cfg in (ss, pbr):
        assert cfg['trainer']['args']['deform_conditioner'] is None
        assert cfg['dataset']['args']['transport_arm'] == 'v1'
        assert 'match_root' not in cfg['dataset']['args']
        assert cfg['experiment']['validate_at_start'] is True
    assert ss['trainer']['args']['max_steps'] == 480
    assert pbr['trainer']['args']['max_steps'] == 1905
    assert pbr['validation']['every_steps'] == 381
