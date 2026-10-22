from types import SimpleNamespace

import pytest
import torch

from experiments.deformnext_retargeting.r_mediated_native_runtime import native_condition_only,tensor_sha


def test_native_trace_hash_keeps_bfloat_dtype_and_shape_identity():
    value=torch.tensor([1.,2.],dtype=torch.bfloat16,requires_grad=True)
    assert tensor_sha(value)==tensor_sha(value.detach().clone())
    assert tensor_sha(value)!=tensor_sha(value.reshape(1,2))
    assert tensor_sha(torch.zeros(1,dtype=torch.float32))!=tensor_sha(torch.zeros(1,dtype=torch.int32))


@pytest.mark.parametrize('fail',[False,True])
def test_native_reference_forward_restores_actual_source_method(monkeypatch,fail):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin
    received=[]
    def reference(self,images,camera,transport):
        received.append(transport)
        if fail:raise RuntimeError('owned reference failure')
        return 'native'
    monkeypatch.setattr(ImageConditionedProjMixin,'_encode_deform_views',reference)
    original=lambda *args:'source'
    trainer=SimpleNamespace(_encode_deform_views=original)
    try:
        if fail:
            with pytest.raises(RuntimeError,match='owned reference'):
                with native_condition_only(trainer):trainer._encode_deform_views('images','camera','input-only-source')
        else:
            with native_condition_only(trainer):assert trainer._encode_deform_views('images','camera','source')=='native'
    finally:
        assert trainer._encode_deform_views is original
    assert received==[None]
