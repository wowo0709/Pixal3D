from types import SimpleNamespace

import pytest
import torch

from experiments.deformnext_retargeting.probe_r_mediated_pbr_native_v2 import preload_frozen_naf,warm_after_constructor


def trainer():
    extractor=SimpleNamespace(use_naf_upsample=True,naf_model=None)
    def load():
        torch.randn(9)
        extractor.naf_model=torch.nn.Linear(3,3).eval().requires_grad_(False)
    extractor._load_naf=load
    return SimpleNamespace(image_cond_model=extractor,multiview_stage='pbr1024',
        optimizer=SimpleNamespace(state={}),step=0)


def test_lazy_frozen_weights_initialized_without_consuming_paired_noise():
    instance=trainer();before=torch.get_rng_state().clone()
    actual=preload_frozen_naf(instance)
    assert torch.equal(before,torch.get_rng_state())
    assert actual['NAF_frozen_eval'] and actual['optimizer_updates']==0
    assert instance.image_cond_model.naf_model is not None
    with pytest.raises(ValueError,match='fresh native'):
        preload_frozen_naf(instance)


def test_error_restores_rng_and_constructor_hook(tmp_path):
    instance=trainer()
    def broken():
        torch.randn(4)
        raise RuntimeError('synthetic load failure')
    instance.image_cond_model._load_naf=broken
    before=torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):preload_frozen_naf(instance)
    assert torch.equal(before,torch.get_rng_state())
    class Base:
        def __init__(self):raise RuntimeError('synthetic native constructor failure')
    old=Base.__init__
    with pytest.raises(RuntimeError):
        with warm_after_constructor(Base):Base()
    assert Base.__init__ is old
