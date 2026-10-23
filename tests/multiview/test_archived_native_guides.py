from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from experiments.deformnext_retargeting.archived_native_guides import save_guides, load_guides


def native_inputs():
    shape = SimpleNamespace(coords=torch.tensor([[0, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.int32),
                            feats=torch.tensor([[.25, -1.], [2., 0.]], dtype=torch.bfloat16))
    # Keep negative/zero logits and a parent without children; neither may be pruned.
    guides = [SimpleNamespace(coords=shape.coords.clone(), feats=torch.tensor(
        [[-2., 0., .25, 1., -.5, 3., 0., 4.], [-1.]*8], dtype=torch.bfloat16),
        _scale=(Fraction(1, 2**i),)*3) for i in range(4)]
    return shape, guides


def test_roundtrip_preserves_native_logits_dtype_order_rational_scale_and_dead_branches(tmp_path):
    shape, guides = native_inputs()
    path = tmp_path/'guides.pt'
    save_guides(path, shape, guides)
    recovered = load_guides(path, shape, lambda **kw: SimpleNamespace(**kw))
    for expected, actual in zip(guides, recovered):
        assert torch.equal(expected.coords, actual.coords)
        assert expected.feats.dtype == actual.feats.dtype
        assert torch.equal(expected.feats.view(torch.uint8), actual.feats.view(torch.uint8))
        assert expected._scale == actual.scale


@pytest.mark.parametrize('field', ['coords', 'feats'])
def test_rejects_different_generated_shape_or_latent_row_order(tmp_path, field):
    shape, guides = native_inputs()
    path = tmp_path/'guides.pt'
    save_guides(path, shape, guides)
    setattr(shape, field, getattr(shape, field).flip(0))
    with pytest.raises(ValueError, match='exact original'):
        load_guides(path, shape, SimpleNamespace)


def test_rejects_changed_saved_guide_values(tmp_path):
    shape, guides = native_inputs()
    path = tmp_path/'guides.pt'
    save_guides(path, shape, guides)
    payload = torch.load(path, weights_only=True)
    payload['guides'][0]['feats'][0, 0] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match='unchanged actual'):
        load_guides(path, shape, SimpleNamespace)


def test_refuses_overwrite_and_incomplete_hierarchy(tmp_path):
    shape, guides = native_inputs()
    path = tmp_path/'guides.pt'
    with pytest.raises(ValueError, match='four actual'):
        save_guides(path, shape, guides[:3])
    save_guides(path, shape, guides)
    with pytest.raises(FileExistsError):
        save_guides(path, shape, guides)
