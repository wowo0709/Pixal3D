import torch
import pytest
from types import SimpleNamespace

from experiments.deformnext_retargeting.r_native_signal_probe import consumer_signal, distribution, matching_signal, native_signal_context


def test_observer_distinguishes_precast_delta_from_actual_bfloat_delivery_without_mutation():
    native = dict(global_=torch.ones(1), proj=torch.ones(1, 2, 2, dtype=torch.bfloat16))
    native['global'] = native.pop('global_')
    delta = torch.full((1, 2, 2), .0001)
    delivered = dict(global_=native['global'], proj=native['proj'] + delta.to(torch.bfloat16))
    delivered['global'] = delivered.pop('global_')
    rng = torch.random.get_rng_state()
    versions = native['proj']._version, delta._version, delivered['proj']._version
    result = consumer_signal(native, delta, delivered)
    assert result['pre_cast_projected_delta_RMS'] > 0
    assert result['actual_consumer_delta_RMS'] == 0
    assert result['changed_consumer_channel_fraction'] == 0
    assert versions == (native['proj']._version, delta._version, delivered['proj']._version)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_no_candidate_empty_statistics_are_not_fabricated_as_perfect_matching():
    assert distribution(torch.tensor([])) == dict(count=0, mean=None, min=None, max=None)
    with pytest.raises(ValueError, match='nonfinite'): distribution(torch.tensor([float('nan')]))


def test_observer_reports_whether_learned_heads_escape_a_concentrated_matching_prior():
    module = SimpleNamespace(last_feature_probabilities=(torch.tensor([0]), torch.full((1, 4, 2), .5)),
        last_R_input_appearance=torch.ones(1, 3), last_R_only_output=torch.zeros(1, 3),
        _last_member_retained_mass=torch.ones(1), _last_distinct_member_count=torch.ones(1))
    attention = SimpleNamespace(matching_logits=torch.tensor([[[.99, .01]]]).log(),
        query_valid=torch.ones(1, 1, dtype=torch.bool), fusion_scale=torch.tensor(0.))
    output = SimpleNamespace(attention=attention, use_probability=torch.tensor([.8]))
    record = matching_signal(module, output)
    assert record['matching_prior_top1_mass']['mean'] == pytest.approx(.99)
    assert record['learned_head_mass_outside_matching_top1']['mean'] == pytest.approx(.5)
    assert record['learned_head_effective_candidates']['mean'] == pytest.approx(2.)
    assert record['R_output_RMS'] == 0


def test_probe_context_restores_bindings_even_after_inference_failure():
    from pixal3d.modules.r_mediated_source_contrast import RMediatedSourceContrastConditioner
    from pixal3d.trainers.flow_matching import member_posterior_source_ray as consumer
    forward, add = RMediatedSourceContrastConditioner.forward, consumer.add_source_ray_correction
    with pytest.raises(RuntimeError, match='owned'):
        with native_signal_context(dict(matching=[], consumer=[])):
            assert consumer.add_source_ray_correction is not add
            raise RuntimeError('owned observer failure')
    assert RMediatedSourceContrastConditioner.forward is forward
    assert consumer.add_source_ray_correction is add
