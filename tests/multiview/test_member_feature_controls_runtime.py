import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from pixal3d.modules.target_first_observed_spatial import native_spatial_ray_candidates
from pixal3d.modules.target_first_ray_v2 import TargetFirstRayEvidence
from pixal3d.trainers.flow_matching.member_feature_controls import MemberFeatureControlImageConditionedProjFlowMatchingCFGTrainer as Trainer
from pixal3d.trainers.flow_matching.member_posterior_source_ray import MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer as Parent
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin, ProjGrid
from experiments.deformnext_retargeting.member_feature_control_recipe import CONTROLS
from experiments.deformnext_retargeting import member_feature_control_runtime as runtime
from test_member_feature_controls import pair


@pytest.mark.parametrize('kind', list(CONTROLS))
def test_physical_module_is_registered_before_native_optimizer_initialization(kind, tmp_path, monkeypatch):
    full, _, _ = pair(kind)
    path = tmp_path / 'utility.pt'; torch.save(full.ray_readout.score_selector.state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    completed = tmp_path / 'completed.json'; completed.write_text(json.dumps(dict(
        status='PASS_FIXED_SEVEN_EPOCH_TARGET_PATCH_UTILITY_TRAINING', epochs=7, optimizer_updates=112, pure_utility_export_SHA256=digest)))
    trainer = object.__new__(Trainer); trainer.image_cond_model = object(); trainer.deform_conditioner = None; trainer.models = {}
    trainer.deform_conditioner_config = dict(arm='c1', implementation='r9-native-v5', key_dim=1024, value_dim=1024,
        query_dim=1024, hidden_dim=256, topk=64, score_temperature=.5, projection_mode='native-bilinear',
        ray_mode='semantic-top1', ray_chunk_size=4, candidate_aggregation='top1', correspondence_mode=CONTROLS[kind]['correspondence'])
    seen = []
    def native(self, **kwargs):
        assert self.models['deform_conditioner'] is self.deform_conditioner
        seen.append(sum(p.requires_grad for p in self.deform_conditioner.parameters()))
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    monkeypatch.setattr(Parent, 'init_models_and_more', native)
    trainer.init_models_and_more(feature_control=kind, ray_feature_utility=dict(checkpoint=str(path), SHA256=digest, completed=str(completed)))
    assert seen == [CONTROLS[kind]['source_tensors']]
    assert trainer._ray_feature_utility_initialization['utility_initialization_applied'] == (kind != 'top1-feature')
    if kind == 'top1-feature':
        assert not hasattr(trainer.deform_conditioner.ray_readout, 'score_selector')
    else:
        assert all(torch.equal(v, trainer.deform_conditioner.ray_readout.score_selector.state_dict()[k]) for k, v in full.ray_readout.score_selector.state_dict().items())


def test_spatial_trainer_uses_actual_native_camera_projection_and_keeps_global_negative():
    grid = ProjGrid(16, 512); transforms = torch.eye(4).repeat(1, 4, 1, 1); transforms[:, :, 2, 3] = 2
    camera = dict(transform_matrix=transforms, distance=torch.full((1, 4), 2.), camera_angle_x=torch.ones(1, 4), mesh_scale=torch.ones(1))
    native = dict(proj=torch.randn(1, 4096, 1024), **{'global': torch.randn(1, 1, 1024)})
    negative = {key: torch.randn_like(value) for key, value in native.items()}; target = {key: torch.randn_like(value) for key, value in native.items()}
    seen = []
    class Conditioner:
        active = False
        def __call__(self, **kwargs):
            seen.append(kwargs); mass = torch.full((1, 3, 1024, 1), .125 if self.active else 0.)
            numerator = mass * (kwargs['source_values'] + 1) + (.02 if self.active else 0.)
            return TargetFirstRayEvidence(numerator, mass, SimpleNamespace(), torch.zeros(1, 3072))
    trainer = object.__new__(Trainer); trainer.feature_control = 'spatial-feature-attention'; trainer.image_attn_mode = 'proj'; trainer.deform_conditioner = Conditioner()
    trainer.image_cond_model = SimpleNamespace(proj_grid=grid, fixed_projection_transform=grid.front_view_transform_matrix)
    trainer.encode_image_proj = lambda images, **kw: (target, negative)
    images = torch.rand(1, 4, 3, 8, 8)
    transport = dict(target_keys=torch.randn(1, 1024, 1024), source_keys=torch.randn(1, 3, 1024, 1024),
        source_values=torch.randn(1, 3, 1024, 1024), source_valid=torch.ones(1, 3, 1024, dtype=torch.bool),
        target_query=torch.randn(1, 1024, 1024), target_valid=torch.ones(1, 1024, dtype=torch.bool), patch_grid=(torch.arange(4096) % 1024).reshape(1, 16, 16, 16))
    with patch.object(ImageConditionedProjMixin, '_encode_deform_views', return_value=(native, negative)) as base:
        untouched, neg = trainer._encode_deform_views(images, camera, None)
        cold, _ = trainer._encode_deform_views(images, camera, transport)
        trainer.deform_conditioner.active = True; active, _ = trainer._encode_deform_views(images, camera, transport)
    assert untouched is native and neg is negative and base.call_count == 3
    assert all(torch.equal(cold[key], value) for key, value in native.items())
    args = seen[0]
    expected = native_spatial_ray_candidates(grid, transport['source_keys'], transport['source_values'], args['ray_ids'], camera)
    assert torch.equal(args['spatial_keys'], expected[0]) and torch.equal(args['spatial_values'], expected[1])
    assert args['frame_target_keys'] is transport['target_keys']
    assert not {'deformation', 'semantic_indices', 'matching_logits'} & args.keys()
    torch.testing.assert_close(active['proj'], native['proj'] + (.125 + .02) * .75, atol=1e-6, rtol=0)
    assert active['global'] is native['global']
    with pytest.raises(ValueError, match='no GT'):
        trainer._encode_deform_views(images, camera, dict(transport, semantic_indices=torch.zeros(1)))


def test_failed_phase_restores_arguments_and_creates_no_success_marker(tmp_path, monkeypatch):
    from experiments.deformnext_retargeting import probe_member_feature_controls as proof
    path = tmp_path / 'config.json'; path.write_text(json.dumps(dict(trainer=dict(args=dict(feature_control='top1-feature')))))
    def fail():
        assert '--phase' not in sys.argv
        raise RuntimeError('expected control phase failure')
    monkeypatch.setattr(proof, 'main', fail)
    monkeypatch.setattr(sys, 'argv', ['runtime', '--phase', 'native', '--config', str(path), '--output', str(tmp_path)])
    previous = sys.argv
    with pytest.raises(RuntimeError, match='expected control phase failure'): runtime.main()
    assert sys.argv is previous and not (tmp_path / 'member-feature-control-runtime.json').exists()
