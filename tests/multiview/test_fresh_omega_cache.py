import torch
import pytest

from experiments.deformnext_retargeting.verify_omega_feature_cache import compare_tokens


def features():
    return torch.cat([torch.ones(4,1024,1024),torch.ones(4,1024,1024)*2],-1)


def test_expected_fp16_storage_rounding_is_distinguished_from_exactness():
    fresh=features()+.0003;cached=fresh.half()
    result=compare_tokens(cached,fresh)
    assert result['pass'] and result['fresh_FP16_exact_to_cache']
    assert len(result['comparisons'])==8


def test_global_agreement_cannot_hide_frame_features_from_the_wrong_encoder():
    fresh=features();cached=fresh.clone();cached[...,:1024]*=1.02
    result=compare_tokens(cached,fresh)
    assert not result['pass']
    assert all(r['pass']==(r['component']=='GLOBAL') for r in result['comparisons'])


def test_wrong_frame_global_order_is_rejected_by_independent_component_metrics():
    fresh=features();result=compare_tokens(fresh.roll(1024,-1),fresh)
    assert not result['pass']


def test_one_wrong_source_view_cannot_hide_in_a_four_view_mean():
    fresh=features();cached=fresh.clone();cached[2,:,:1024]*=1.02
    result=compare_tokens(cached,fresh)
    assert not result['pass']
    assert [(r['view'],r['component']) for r in result['comparisons'] if not r['pass']]==[(2,'FRAME')]


@pytest.mark.parametrize('shape',[(4,1024,1024),(3,1024,2048),(4,256,2048)])
def test_different_descriptor_layout_cannot_be_compared_as_omega_l12(shape):
    with pytest.raises(ValueError,match='layout'):compare_tokens(torch.ones(shape),torch.ones(shape))


def test_nonfinite_or_zero_corrupt_keys_do_not_count_as_encoder_parity():
    a=features();a[0,0,0]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):compare_tokens(a,features())
    with pytest.raises(ValueError,match='zero'):compare_tokens(torch.zeros(4,1024,2048),torch.zeros(4,1024,2048))
