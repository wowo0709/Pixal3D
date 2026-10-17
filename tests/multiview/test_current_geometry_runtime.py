from types import SimpleNamespace

import pytest
import torch

from experiments.deformnext_retargeting.current_geometry_runtime import capture_native


def trainer(count=4, wrong_mean=False):
    projection = torch.nn.Identity()
    value = SimpleNamespace(image_cond_model=object(), _image_cond_module=SimpleNamespace(proj_grid=projection))
    def condition(**pack):
        per_view = [projection(torch.ones(1, 4096, 4) * i) for i in range(count)]
        combined = sum(per_view) / count + float(wrong_mean)
        return {'cond': {'proj': combined, 'global': torch.ones(1, 2, 4)},
                'neg_cond': {'proj': torch.zeros_like(combined), 'global': torch.zeros(1, 2, 4)}}
    value.get_inference_cond = condition
    return value


def test_cache_captures_actual_target_projection_and_preserves_native_condition():
    instance = trainer()
    condition, target = capture_native(instance, {})
    assert target.eq(0).all()
    assert condition['cond']['proj'].eq(1.5).all()
    assert condition['cond']['global'].eq(1).all()
    assert condition['neg_cond']['global'].eq(0).all()
    assert not instance._image_cond_module.proj_grid._forward_hooks


@pytest.mark.parametrize('count,wrong_mean', [(3, False), (4, True)])
def test_cache_rejects_different_views_or_projection_mean_and_restores_hook(count, wrong_mean):
    instance = trainer(count, wrong_mean)
    with pytest.raises(ValueError):
        capture_native(instance, {})
    assert not instance._image_cond_module.proj_grid._forward_hooks
