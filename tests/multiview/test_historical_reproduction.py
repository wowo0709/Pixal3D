"""Historical contracts: per-view writes, exact controls and endpoint gradients."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.deformnext_retargeting import historical_reproduction as h
from experiments.deformnext_retargeting.historical_geometry import camera_rays_for_pixels, project_points
from experiments.deformnext_retargeting.reproduction_transport_context import native_slot_context
from experiments.deformnext_retargeting.cache_reproduction_routes import frame_contract


def test_reference_algorithms_match_pinned_historical_ast():
    directory = Path(h.__file__).parent
    authority = json.loads((directory / 'historical_reproduction_authority.json').read_text())
    for filename, expected in authority['function_AST_sha256'].items():
        actual = {n.name: hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
                  for n in ast.parse((directory / filename).read_text()).body
                  if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        assert actual == expected


def test_identity_reads_and_wrong_reads_preserve_each_views_evidence_multiset():
    source = torch.arange(3 * 32 * 32 * 2).reshape(1, 3, 32, 32, 2).float()
    grid = h.identity_patch_grid(1, 3, 32, 32, device=torch.device('cpu'))
    assert torch.equal(h._sample_neighborhood(source, grid, radius=0)[..., 0, :], source)
    valid = torch.zeros(1, 3, 32, 32, dtype=torch.bool)
    valid[:, :, 2:7, 3:9] = True
    before = torch.get_rng_state().clone()
    permutation = h.wrong_permutation(valid, pair_id='same-asset')
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(permutation, h.wrong_permutation(valid, pair_id='same-asset'))
    donor = grid.reshape(1, 3, 1024, 2).gather(2, permutation.reshape(1, 3, 1024, 1).expand(-1, -1, -1, 2))
    sampled = h._sample_neighborhood(source, donor.reshape_as(grid), radius=0)[..., 0, :]
    for view in range(3):
        mask = valid[0, view]
        assert not torch.any((sampled[0, view][mask] == source[0, view][mask]).all(-1))
        assert torch.equal(sampled[0, view][mask].sort(dim=0).values, source[0, view][mask].sort(dim=0).values)
        assert torch.equal(sampled[0, view][~mask], source[0, view][~mask])


def test_native_fusion_identity_fallback_and_soft_mass_applied_once():
    target = torch.full((1, 7, 2), 4.)
    source = torch.full((1, 3, 7, 2), 8.)
    full = torch.ones(1, 3, 7, 1)
    assert torch.equal(h.target_first_static_projection(target, source, full), torch.full_like(target, 7.))
    assert torch.equal(h.target_first_static_projection(target, source * 0, full * 0), target)
    # source already includes the bilinear support numerator: do not square its mass.
    assert torch.equal(h.target_first_static_projection(target, source * .25, full * .25), torch.full_like(target, 4.75))


def test_patched_camera_pixels_round_trip_in_exact_historical_convention():
    pixels = np.array([[7.5, 7.5], [255.5, 255.5], [503.5, 503.5]], np.float32)
    camera = np.eye(4, dtype=np.float32)
    camera[:3, 3] = [1, 2, 3]
    rays = camera_rays_for_pixels(pixels, camera, np.pi / 3, resolution=512)
    points = rays[:, :3] + rays[:, 3:] * 2
    actual, depth, valid = project_points(points, camera, np.pi / 3, resolution=512)
    np.testing.assert_allclose(actual, pixels, atol=2e-5)
    assert valid.all()
    np.testing.assert_allclose(depth, 2.)


def test_geometry_channels_use_predicted_volume_and_native_xyz_order():
    original = np.full((64, 64, 64), -2., np.float32)
    original[8:28, 12:36, 16:40] = 3
    desired = np.roll(original, 8, axis=0)
    geometry = h.geometry_from_logits(original, desired)
    assert geometry.shape == (1, 4096, 9)
    assert torch.isfinite(geometry).all()
    torch.testing.assert_close(geometry[..., 6:], geometry[..., :3] - geometry[..., 3:6])
    expected = torch.nn.functional.avg_pool3d(torch.from_numpy(desired)[None, None] / 8, 4, 4)
    torch.testing.assert_close(geometry[..., 0].reshape(1, 1, 16, 16, 16), expected)


def test_zero_adapter_preserves_native_and_full_rollout_propagates_gradient():
    adapter = h.GeometryFactorialVoxelAdapter(channels=2, hidden=8, geometry_hidden=4)
    target = torch.randn(1, 4096, 2)
    mismatch = torch.randn_like(target)
    geometry = torch.randn(1, 4096, 9)
    corrected, _ = adapter(target, mismatch, geometry, mode='r4_content')
    assert torch.equal(corrected, mismatch)
    frozen = torch.nn.Linear(2, 2).requires_grad_(False)
    class Sampler:
        def _get_model_prediction(self, model, sample, time, **condition):
            velocity = model(sample) + condition['cond']['proj']
            return sample, sample, velocity
    cond = {'cond': {'global': torch.zeros(1, 1, 2), 'proj': corrected},
            'neg_cond': {'global': torch.zeros(1, 1, 2), 'proj': torch.zeros_like(corrected)}}
    output = h.differentiable_flow_euler_samples(Sampler(), frozen, torch.zeros_like(target), cond,
                                               steps=3, rescale_t=1., checkpoint_model=True)
    output.square().mean().backward()
    assert adapter.residual[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in frozen.parameters())


def test_endpoint_loss_keeps_source_stale_visible_and_two_resolutions():
    logits = torch.zeros(1, 1, 64, 64, 64, requires_grad=True)
    regions = {}
    for resolution in [64, 32]:
        masks = {}
        for i, name in enumerate(['source_positive', 'stale', 'visible_positive', 'visible_negative']):
            mask = torch.zeros(1, 1, resolution, resolution, resolution, dtype=torch.bool)
            mask[..., 2 + i, 3 + i, 4 + i] = True
            masks[name] = mask
        regions[resolution] = masks
    loss, terms = h.locked_endpoint_loss(logits, regions)
    assert set(terms) == {64, 32}
    assert float(loss.detach()) == pytest.approx(3 * np.log(2))
    loss.backward()
    assert logits.grad.abs().sum() > 0


def test_frame_fingerprint_includes_ss_scale_even_when_rgb_and_geometry_are_same():
    row = dict(mesh_scale=1., latent_geometry_schema='deformnext-m1-render-world-v4',
               target_world_center=[0, 0, 0], target_world_extent=1.)
    assert frame_contract(row) != frame_contract({**row, 'mesh_scale': 2.})


def test_scoped_per_view_transport_keeps_global_negative_and_restores_hook():
    from types import SimpleNamespace
    class Grid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.grid_points = torch.zeros(1, 4096, 3)
            self.image_resolution = 512
        def forward(self, maps, fov, distance, scale, transform):
            return maps.mean((1, 2))[:, None].expand(-1, 4096, -1).clone()
    class Trainer:
        def __init__(self):
            self.image_cond_model = object()
            self._image_cond_module = SimpleNamespace(proj_grid=Grid())
        def _encode_deform_views(self, images, camera, transport, **kwargs):
            outputs = []
            for view in range(4):
                matrix = torch.eye(4)[None].clone()
                matrix[:, 2, 3] = 2 + view
                outputs.append(self._image_cond_module.proj_grid(
                    torch.full((1, 32, 32, 2), float(view + 1)),
                    torch.ones(1), torch.ones(1), torch.ones(1), matrix))
            return {'global': torch.full((1, 1, 2), 9.), 'proj': sum(outputs) / 4}, {
                'global': torch.zeros(1, 1, 2), 'proj': torch.zeros_like(outputs[0])}
    trainer = Trainer()
    images = torch.zeros(1, 4, 3, 8, 8)
    grid = h.identity_patch_grid(1, 3, 32, 32, device=torch.device('cpu'))
    valid = torch.ones(1, 3, 32, 32, dtype=torch.bool)
    for mode, expected in [('native', 2.5), ('source-off', 1.), ('gt-corr', 2.5), ('wrong-corr', 2.5)]:
        events = []
        with native_slot_context(trainer, mode, grid, valid, ['asset'], events):
            cond, negative = trainer._encode_deform_views(images, {}, None)
            assert torch.equal(cond['global'], torch.full((1, 1, 2), 9.))
            assert not negative['proj'].any()
            assert torch.all(cond['proj'] == expected)
        assert '_encode_deform_views' not in trainer.__dict__
        assert not trainer._image_cond_module.proj_grid._forward_hooks
        assert events[0]['native_joint_proj_max_abs'] == 0.
    # The new bridge is explicitly separate from the historical fallback: no
    # supported write must preserve the native source projection, not drop it.
    for support, weight in [(torch.zeros_like(valid), 1.), (valid, 0.)]:
        events = []
        with native_slot_context(trainer, 'gt-corr', grid, support, ['asset'], events,
                                 fallback='native-spatial', source_weight=weight):
            cond, negative = trainer._encode_deform_views(images, {}, None)
            assert torch.all(cond['proj'] == 2.5)
            assert torch.all(cond['global'] == 9.)
            assert not negative['proj'].any()
        assert events[0]['fallback'] == 'native-spatial'
        if weight == 0:
            assert events[0]['zero_weight_native_projection_exact']
    with pytest.raises(RuntimeError, match='scoped failure'):
        with native_slot_context(trainer, 'gt-corr', grid, valid, ['asset'], []):
            raise RuntimeError('scoped failure')
    assert '_encode_deform_views' not in trainer.__dict__


def test_real_route_loader_requires_frame_identity_and_diagnostic_schema(tmp_path):
    from experiments.deformnext_retargeting.reproduce_native_transport import load_routes
    from experiments.deformnext_retargeting.cache_reproduction_routes import SCHEMA
    (tmp_path / 'data').mkdir()
    row = dict(sample_id='sample', conditioning_input_sha256='rgb', mesh_scale=1.,
               latent_geometry_schema='deformnext-m1-render-world-v4',
               target_world_center=[0, 0, 0], target_world_extent=1.)
    np.savez(tmp_path / 'data/sample.npz', schema=SCHEMA, sample_id='sample',
             conditioning_input_sha256='rgb', frame_contract=frame_contract(row), diagnostic_only=True,
             source_pixel=np.zeros((3, 1024, 2), np.float32), valid=np.ones((3, 1024), bool),
             route_hidden_coords64=np.array([[2, 3, 4]]), route_hidden_coords32=np.array([[1, 1, 2]]))
    grid, valid, regions = load_routes(tmp_path, [row], torch.device('cpu'))
    assert grid.shape == (1, 3, 32, 32, 2) and valid.all()
    assert len(regions) == 1
    with pytest.raises(ValueError, match='identity mismatch'):
        load_routes(tmp_path, [{**row, 'mesh_scale': 2.}], torch.device('cpu'))


def test_sampled_hidden_surface_metric_has_distinct_scope_and_native_or_pooling():
    from experiments.deformnext_retargeting.reproduce_native_transport import region_metrics
    gt = torch.zeros(64, 64, 64, dtype=torch.bool)
    pred = torch.zeros_like(gt)
    gt[2, 2, 2] = gt[3, 3, 3] = True
    pred[2, 2, 2] = True
    result = region_metrics(pred, gt, {64: np.array([[2, 2, 2], [3, 3, 3]]), 32: np.array([[1, 1, 1]])})
    assert result['route_sampled_hidden_gt_recall64'] == .5
    assert result['route_sampled_hidden_gt_recall32'] == 1.
