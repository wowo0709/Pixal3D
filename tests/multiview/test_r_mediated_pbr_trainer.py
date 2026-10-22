from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from pixal3d.modules.target_first_ray_v2 import TargetFirstRayEvidence
from pixal3d.trainers.flow_matching.r_mediated_pbr_source import (
    RMediatedPBRNativeTrainer, pbr_source_kwargs,
)
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin
from test_pbr_source_ray_geometry import fixture


def source_spec(mode):
    spec = dict(arm='c1', implementation='r9-native-v5', key_dim=1024, query_dim=1024,
        value_dim=2048, hidden_dim=256, topk=64, g_max=.25, q2c_checkpoint='/declared/matcher.pt',
        freeze_q2c=True, candidate_aggregation='top1', projection_mode='native-bilinear',
        score_temperature=.05, ray_chunk_size=64, appearance_mode=mode)
    if mode != 'top1-only':
        spec['residual_logit_bound'] = 1.
    return spec


@pytest.mark.parametrize('mode', ['top1-only', 'top1-plus-feature', 'raw-topk-plus-feature'])
def test_PBR_recipe_physically_separates_head_and_NoDQ_dimensions(mode):
    spec = source_spec(mode)
    kwargs = pbr_source_kwargs(spec)
    assert kwargs['appearance_mode'] == mode
    for update in [dict(value_dim=1024), dict(projection_mode='nearest'),
            dict(deformation_dim=2048), dict(freeze_q2c=False), dict(q2c_checkpoint=None)]:
        with pytest.raises(ValueError):
            pbr_source_kwargs(dict(spec, **update))
    if mode == 'top1-only':
        with pytest.raises(ValueError):
            pbr_source_kwargs(dict(spec, residual_logit_bound=1.))


def test_legacy_cached_correspondences_and_SS_grid_do_not_enter_new_ray_model():
    trainer = object.__new__(RMediatedPBRNativeTrainer)
    trainer.deform_conditioner_config = source_spec('top1-only')
    values = {k: torch.zeros(1) for k in ['source_values', 'source_valid', 'target_query',
        'patch_ids', 'patch_grid', 'semantic_indices', 'matching_logits', 'candidate_valid']}
    pack = {'transport_' + key: value for key, value in values.items()}
    actual = trainer._pop_deform_transport(pack)
    assert set(actual) == {'source_values', 'source_valid', 'target_query', 'patch_ids'}
    assert not pack
    for key in ['deformation', 'spatial_indices', 'target_context', 'source_context']:
        pack = {'transport_' + k: v for k, v in values.items()}
        pack['transport_' + key] = torch.zeros(1)
        with pytest.raises(ValueError):
            trainer._pop_deform_transport(pack)


@pytest.mark.parametrize('mode', ['top1-only', 'top1-plus-feature', 'raw-topk-plus-feature'])
def test_actual_sparse_PBR_consumer_keeps_native_global_negative_and_target_RGB_carrier(mode):
    grid, coords, camera = fixture()
    camera['coords'] = coords
    native = {'global': torch.randn(2, 5, 1024), 'proj': torch.randn(len(coords), 2048)}
    negative = {k: torch.randn_like(v) for k, v in native.items()}
    target = {'global': torch.randn_like(native['global']), 'proj': torch.randn_like(native['proj'])}
    captured = []
    class Conditioner:
        appearance_mode = mode
        active = False
        def __call__(self, **kwargs):
            captured.append(kwargs)
            mass = torch.full((2, 3, 1024, 1), .125 if self.active else 0.)
            numerator = mass * (kwargs['source_values']+1)
            if self.active:
                numerator = numerator+.02
            return TargetFirstRayEvidence(numerator, mass, SimpleNamespace(), torch.zeros(2, 3072))
    trainer = object.__new__(RMediatedPBRNativeTrainer)
    trainer.image_attn_mode = 'proj'
    trainer.image_cond_model = SimpleNamespace(proj_grid=grid,
        fixed_projection_transform=grid.front_view_transform_matrix)
    trainer.deform_conditioner = Conditioner()
    target_calls = []
    def encode(images, **kwargs):
        target_calls.append((images, kwargs))
        return target, negative
    trainer.encode_image_proj = encode
    transport = dict(target_keys=torch.randn(2, 1024, 1024),
        source_keys=torch.randn(2, 3, 1024, 1024), source_values=torch.randn(2, 3, 1024, 2048),
        source_valid=torch.ones(2, 3, 1024, dtype=torch.bool), target_query=torch.randn(2, 1024, 1024),
        target_valid=torch.ones(2, 1024, dtype=torch.bool), patch_ids=torch.tensor([7, 9, 11, 12, 13]))
    images = torch.rand(2, 4, 3, 8, 8)
    with patch.object(ImageConditionedProjMixin, '_encode_deform_views', return_value=(native, negative)) as base:
        unchanged, neg = trainer._encode_deform_views(images, camera, None)
        assert unchanged is native and neg is negative and not captured
        cold, neg = trainer._encode_deform_views(images, camera, transport)
        trainer.deform_conditioner.active = True
        learned, learned_negative = trainer._encode_deform_views(images, camera, transport)
    assert base.call_count == 3 and all(call.args[3] is None for call in base.call_args_list)
    assert all(torch.equal(cold[key], value) for key, value in native.items())
    assert learned['global'] is native['global'] and learned_negative is negative and neg is negative
    torch.testing.assert_close(learned['proj'], native['proj']+(.125+.02)*.75, atol=1e-6, rtol=0)
    if mode == 'top1-only':
        assert not target_calls and captured[0]['native_query_rows'] is None
    else:
        assert len(target_calls) == 2
        assert torch.equal(target_calls[0][0], images[:, 0])
        assert torch.equal(captured[0]['native_query_rows'], target['proj'][..., :1024])
        assert target_calls[0][1]['coords'] is coords
    assert captured[0]['source_values'] is transport['source_values']
    assert torch.equal(captured[0]['batch_ids'], coords[:, 0])
    assert not {'spatial_indices', 'deformation', 'semantic_indices', 'matching_logits'} & set(captured[0])
