from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from experiments.deformnext_retargeting.native_guide_capture_context import capture_native_guides


def fixture():
    shape = SimpleNamespace(coords=torch.tensor([[0,0,0,0]], dtype=torch.int32), feats=torch.ones(1,2))
    meshes = [object()]
    guides = [SimpleNamespace(coords=shape.coords, feats=torch.ones(1,8), _scale=(Fraction(1,2**i),)*3) for i in range(4)]
    class Pipeline:
        def decode_shape_slat(self, actual, resolution):
            assert actual is shape
            return meshes,guides
    return Pipeline,shape,meshes,guides


def test_original_objects_and_CPU_RNG_are_preserved_and_method_restored(tmp_path):
    cls,shape,meshes,guides=fixture(); original=cls.decode_shape_slat; state=torch.get_rng_state().clone()
    with capture_native_guides(cls,tmp_path/'guides.pt') as events:
        actual_meshes,actual_guides=cls().decode_shape_slat(shape,1024)
        assert actual_meshes is meshes and actual_guides is guides
        assert len(events)==1 and events[0]['actual_guide_count']==4
        assert torch.equal(state,torch.get_rng_state())
    assert cls.decode_shape_slat is original


def test_restores_original_method_after_a_consumer_exception(tmp_path):
    cls,shape,_,_=fixture();original=cls.decode_shape_slat
    with pytest.raises(RuntimeError,match='consumer'):
        with capture_native_guides(cls,tmp_path/'guides.pt'):
            cls().decode_shape_slat(shape,1024)
            raise RuntimeError('consumer')
    assert cls.decode_shape_slat is original


def test_rejects_an_extra_decode_instead_of_overwriting_actual_guides(tmp_path):
    cls,shape,_,_=fixture()
    with capture_native_guides(cls,tmp_path/'guides.pt'):
        cls().decode_shape_slat(shape,1024)
        with pytest.raises(ValueError,match='one original'):
            cls().decode_shape_slat(shape,1024)


def test_rejects_the_wrong_resolution_before_the_original_decoder_runs(tmp_path):
    cls,shape,_,_=fixture();original=cls.decode_shape_slat
    with pytest.raises(ValueError,match='native1024'):
        with capture_native_guides(cls,tmp_path/'guides.pt'):
            cls().decode_shape_slat(shape,512)
    assert cls.decode_shape_slat is original
    assert not(tmp_path/'guides.pt').exists()
