import pytest
import torch

from experiments.deformnext_retargeting.source_condition_field_oracle import condition_field_oracle


class Encoder:
    deform_conditioner = None

    def _encode_deform_views(self, images, camera_info, transport, *, control='normal'):
        value = images.mean((1, 2, 3, 4))
        condition = dict(global_=value[:, None] + 10, proj=value[:, None, None] + 20)
        condition['global'] = condition.pop('global_')
        return condition, {k: torch.zeros_like(v) for k, v in condition.items()}


def original_inputs(images):
    original = images.clone()
    original[:, 1:3] -= .25
    return original, [str(i) for i in range(len(images))]


@pytest.mark.parametrize('field', ['proj', 'global'])
def test_exactly_one_condition_field_changes_and_rng_is_preserved(field):
    images = torch.ones(2, 4, 3, 4, 4)
    model = Encoder()
    base, negative = model._encode_deform_views(original_inputs(images)[0], {}, None)
    teacher, _ = model._encode_deform_views(images, {}, None)
    method, rng, events = Encoder._encode_deform_views, torch.random.get_rng_state(), []
    with condition_field_oracle(Encoder, original_inputs, field, events):
        actual, actual_negative = model._encode_deform_views(images, {}, None)
    other = 'global' if field == 'proj' else 'proj'
    assert torch.equal(actual[field], teacher[field])
    assert torch.equal(actual[other], base[other])
    assert all(torch.equal(actual_negative[k], v) for k, v in negative.items())
    assert torch.equal(rng, torch.random.get_rng_state())
    assert Encoder._encode_deform_views is method
    assert events[0]['unselected_field_bit_exact'] and events[0]['RNG_unchanged']


def test_restores_encoder_on_failure_and_rejects_target_changes():
    images, method = torch.ones(1, 4, 3, 4, 4), Encoder._encode_deform_views
    with pytest.raises(ValueError):
        with condition_field_oracle(Encoder, lambda x: (x * 0, ['one']), 'proj', []):
            Encoder()._encode_deform_views(images, {}, None)
    assert Encoder._encode_deform_views is method


def test_rejects_encoding_that_consumes_sampling_rng():
    class RandomEncoder(Encoder):
        def _encode_deform_views(self, *args, **kwargs):
            torch.rand(1)
            return super()._encode_deform_views(*args, **kwargs)
    method = RandomEncoder._encode_deform_views
    with pytest.raises(ValueError, match='RNG'):
        with condition_field_oracle(RandomEncoder, original_inputs, 'global', []):
            RandomEncoder()._encode_deform_views(torch.ones(1, 4, 3, 4, 4), {}, None)
    assert RandomEncoder._encode_deform_views is method
