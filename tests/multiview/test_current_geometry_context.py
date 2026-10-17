from types import SimpleNamespace

import pytest
import torch

from experiments.deformnext_retargeting.current_geometry_context import geometry_content_context
from experiments.deformnext_retargeting.geometry_content_adapter import GeometryContentRetargetAdapter


class NativeTrainer:
    def __init__(self):
        self.image_cond_model = object()
        self._image_cond_module = SimpleNamespace(proj_grid=torch.nn.Identity())
        self.global_features = torch.randn(1, 2, 4)
        self.negative = {'global': -self.global_features}

    def _encode_deform_views(self, images, camera_info, transport, **kwargs):
        projected = [self._image_cond_module.proj_grid(torch.full((1, 4096, 4), float(i+1))) for i in range(4)]
        return {'proj': sum(projected) / 4, 'global': self.global_features}, self.negative


def test_actual_projection_context_is_initially_native_with_global_negative_identity_and_gradient():
    trainer = NativeTrainer()
    adapter = GeometryContentRetargetAdapter(with_query=True, channels=4)
    images = torch.zeros(1, 4, 3, 1, 1)
    geometry = torch.randn(1, 4096, 9)
    events = []
    with geometry_content_context(trainer, adapter, geometry, events):
        condition, negative = trainer._encode_deform_views(images, {}, None)
        assert events[0]['native_projection_exact']
        assert condition['global'] is trainer.global_features
        assert negative is trainer.negative
        condition['proj'].square().mean().backward()
        assert adapter.residual[-1].weight.grad.abs().sum() > 0
    assert '_encode_deform_views' not in trainer.__dict__
    assert not trainer._image_cond_module.proj_grid._forward_hooks


def test_no_query_rejects_zero_channels_and_scoped_hooks_restore_on_failure():
    trainer = NativeTrainer()
    adapter = GeometryContentRetargetAdapter(with_query=False, channels=4)
    with pytest.raises(ValueError, match='physically absent'):
        with geometry_content_context(trainer, adapter, torch.zeros(1, 4096, 9), []):
            pass
    assert adapter.content_encoder is None and adapter.residual[0].normalized_shape == (15,)
    with pytest.raises(ValueError, match='four RGB views'):
        with geometry_content_context(trainer, adapter, None, []):
            trainer._encode_deform_views(torch.zeros(1, 3, 3, 1, 1), {}, None)
    assert '_encode_deform_views' not in trainer.__dict__
    assert not trainer._image_cond_module.proj_grid._forward_hooks
