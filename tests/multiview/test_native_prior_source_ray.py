from types import SimpleNamespace
from unittest.mock import patch

import torch
import pytest

from pixal3d.modules.native_prior_source_ray import source_ray_delta, project_native_prior_source_delta
from pixal3d.modules.target_first_ray_v2 import TargetFirstRayEvidence
from pixal3d.modules.target_first_ray import add_source_ray_correction
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin, ProjGrid
from pixal3d.trainers.flow_matching.native_prior_source_ray import NativePriorSourceRayImageConditionedProjFlowMatchingCFGTrainer as Trainer


def camera_fixture():
    grid = ProjGrid(16, 512)
    matrices = torch.eye(4).repeat(1, 4, 1, 1)
    matrices[:, :, 2, 3] = 2
    camera = dict(transform_matrix=matrices, distance=torch.ones(1, 4) * 2,
                  camera_angle_x=torch.ones(1, 4), mesh_scale=torch.ones(1))
    return grid, camera


def test_zero_source_and_equal_source_replacement_preserve_native_carrier_exact():
    grid, camera = camera_fixture()
    original = torch.randn(1, 3, 1024, 1024)
    mass = torch.rand(1, 3, 1024, 1) - .5
    for numerator, weight in [(torch.zeros_like(original), torch.zeros_like(mass)), (mass * original, mass)]:
        delta = project_native_prior_source_delta(grid, numerator, weight, original, camera)
        assert not delta.any()
        native = {'proj': torch.randn(1, 4096, 1024), 'global': torch.randn(1, 1, 1024)}
        corrected = add_source_ray_correction(native, delta)
        assert all(torch.equal(corrected[key], value) for key, value in native.items())
        assert corrected['global'] is native['global']


def test_retarget_residual_stays_outside_zero_matching_gate_and_uses_native_rays():
    grid, camera = camera_fixture()
    original = torch.ones(1, 3, 1024, 1024)
    mass = torch.zeros(1, 3, 1024, 1)
    residual = torch.full_like(original, .2)
    delta = project_native_prior_source_delta(grid, residual, mass, original, camera)
    torch.testing.assert_close(delta, torch.full_like(delta, .15), atol=1e-7, rtol=0)
    assert (delta.abs().sum(-1) > 0).all()


def test_source_replacement_subtracts_original_source_instead_of_target():
    grid, camera = camera_fixture()
    source = torch.arange(1, 4).float().view(1, 3, 1, 1).expand(1, 3, 1024, 1024)
    mass = torch.full((1, 3, 1024, 1), .125)
    numerator = mass * (source + 1) + .02
    delta = project_native_prior_source_delta(grid, numerator, mass, source, camera)
    torch.testing.assert_close(delta, torch.full_like(delta, (.125 + .02) * .75), atol=1e-7, rtol=0)


def test_semantic_gate_and_retarget_output_have_independent_initial_gradients():
    original = torch.randn(1, 3, 1024, 1024)
    evidence = original + .3
    gate = torch.tensor(0., requires_grad=True)
    residual = torch.tensor(0., requires_grad=True)
    use = torch.full((1, 3, 1024, 1), .5)
    mass = use * torch.tanh(gate)
    numerator = mass * evidence + use * residual
    source_ray_delta(numerator, mass, original).sum().backward()
    assert gate.grad > 0 and residual.grad > 0


@pytest.mark.parametrize('mode', ['semantic', 'native-spatial'])
def test_actual_trainer_wiring_preserves_full_native_prior_and_queries_only_target(mode):
    grid, camera = camera_fixture()
    native = {'proj': torch.randn(1, 4096, 1024), 'global': torch.randn(1, 1, 1024)}
    negative = {key: torch.randn_like(value) for key, value in native.items()}
    target = {'proj': torch.randn_like(native['proj']), 'global': torch.randn_like(native['global'])}
    captured = []

    class Conditioner:
        correspondence_mode = mode
        active = False
        def __call__(self, **kwargs):
            captured.append(kwargs)
            mass = torch.full((1, 3, 1024, 1), .125 if self.active else 0.)
            numerator = (mass * (kwargs['source_values'] + 1) + .02 if self.active
                         else torch.zeros(1, 3, 1024, 1024))
            return TargetFirstRayEvidence(numerator, mass, SimpleNamespace(), torch.zeros(1, 3072))

    trainer = object.__new__(Trainer)
    trainer.image_attn_mode = 'proj'
    trainer.image_cond_model = SimpleNamespace(proj_grid=grid, fixed_projection_transform=grid.front_view_transform_matrix)
    trainer.deform_conditioner = Conditioner()
    trainer.encode_image_proj = lambda images, **kwargs: (target, negative)
    images = torch.rand(1, 4, 3, 8, 8)
    transport = dict(target_keys=torch.randn(1, 1024, 1024), source_keys=torch.randn(1, 3, 1024, 1024),
        source_values=torch.randn(1, 3, 1024, 1024), source_valid=torch.ones(1, 3, 1024, dtype=torch.bool),
        target_query=torch.randn(1, 1024, 1024), target_valid=torch.ones(1, 1024, dtype=torch.bool),
        patch_grid=(torch.arange(4096) % 1024).reshape(1, 16, 16, 16))
    with patch.object(ImageConditionedProjMixin, '_encode_deform_views', return_value=(native, negative)) as base:
        untouched, neg = trainer._encode_deform_views(images, camera, None)
        assert untouched is native and neg is negative and not captured
        result, result_negative = trainer._encode_deform_views(images, camera, transport)
        assert base.call_count == 2
        assert all(call.args[1] is images and call.args[3] is None for call in base.call_args_list)
        trainer.deform_conditioner.active = True
        activated, activated_negative = trainer._encode_deform_views(images, camera, transport)
    assert all(torch.equal(result[key], value) for key, value in native.items())
    assert result_negative is negative
    assert captured[0]['native_voxels'] is target['proj']
    assert captured[0]['source_values'] is transport['source_values']
    assert not {'deformation', 'semantic_indices', 'matching_logits'} & captured[0].keys()
    torch.testing.assert_close(activated['proj'], native['proj'] + (.125 + .02) * .75, atol=1e-6, rtol=0)
    assert activated['global'] is native['global'] and activated_negative is negative
    if mode == 'native-spatial':
        assert captured[0]['spatial_values'].shape == (1, 3072, 3, 1024)


def test_actual_nonzero_conditioner_missing_query_preserves_native_prior_and_DDP_edges():
    from pixal3d.modules.target_first_observed_query import ObservedQueryTargetFirstRayConditioner
    model = ObservedQueryTargetFirstRayConditioner(arm='c1', implementation='r9-native-v5',
        key_dim=1024, value_dim=1024, query_dim=1024, hidden_dim=256,
        topk=64, score_temperature=.05, ray_mode='semantic-top1', candidate_aggregation='top1', projection_mode='native-bilinear')
    with torch.no_grad():
        model.fusion_gate.fill_(.2)
        model.retargeter.net[-1].weight.fill_(.01)
    source = torch.randn(1, 3, 1024, 1024)
    output = model(native_voxels=torch.randn(1, 5, 1024),target_omega_voxels=torch.randn(1, 5, 1024),
        target_native_voxels=torch.randn(1, 5, 1024),observed_query=torch.zeros(1, 5,dtype=torch.bool),
        ray_ids=torch.tensor([[[0, 0, 1, 1, -1],[2, 2, 1, 1, -1],[3, 3, 4, 4, -1]]]),
        source_keys=torch.randn_like(source),source_values=source,source_valid=torch.ones(1, 3, 1024,dtype=torch.bool))
    delta = source_ray_delta(output.source_numerator, output.semantic_weight, source)
    assert not delta.any() and not output.use_probability.any()
    delta.sum().backward()
    for name,value in model.named_parameters():
        if value.requires_grad:
            assert value.grad is not None and torch.isfinite(value.grad).all() and not value.grad.any(), name
