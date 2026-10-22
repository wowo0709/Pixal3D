"""Read-only instrumentation of matching selection and actual native projection delivery."""
from contextlib import contextmanager
import math

import torch


def distribution(values):
    values = values.detach().float().reshape(-1)
    if not len(values):
        return dict(count=0, mean=None, min=None, max=None)
    if not torch.isfinite(values).all():
        raise ValueError('nonfinite native signal probe input')
    return dict(count=values.numel(), mean=float(values.mean()), min=float(values.min()), max=float(values.max()))


def consumer_signal(native, delta, delivered):
    baseline = native['proj'].detach().float()
    change = delta.detach().float()
    actual = delivered['proj'].detach().float() - baseline
    base_rms = float(baseline.square().mean().sqrt())
    raw_rms = float(change.square().mean().sqrt())
    actual_rms = float(actual.square().mean().sqrt())
    if (not math.isfinite(base_rms + raw_rms + actual_rms)
            or delivered['global'] is not native['global']):
        raise ValueError('native global object or finite delivery changed')
    return dict(native_proj_RMS=base_rms, pre_cast_projected_delta_RMS=raw_rms,
        actual_consumer_delta_RMS=actual_rms,
        actual_delta_over_native_RMS=actual_rms / max(base_rms, 1e-30),
        nonzero_delta_fraction=float((change != 0).float().mean()),
        changed_consumer_channel_fraction=float((actual != 0).float().mean()),
        target_native_global_object_preserved=True)


def matching_signal(module, output):
    selected, probability = module.last_feature_probabilities
    prior_logits = output.attention.matching_logits.flatten(0, 1)[selected]
    prior = prior_logits.float().softmax(-1)
    p = probability.float()
    if p.shape != (len(selected), 4, prior.shape[-1]):
        raise ValueError('actual selected four-head posterior and matching prior required')
    entropy = lambda q: -(q * q.clamp_min(1e-30).log()).sum(-1)
    prior_top1 = prior.argmax(-1)
    top1_mass = p.gather(-1, prior_top1[:, None, None].expand(-1, 4, 1)).squeeze(-1)
    return dict(active_source_rays=int(len(selected)),
        total_source_rays=output.attention.query_valid.numel(),
        query_valid_fraction=float(output.attention.query_valid.float().mean()),
        source_use_probability=distribution(output.use_probability),
        matching_prior_top1_mass=distribution(prior.max(-1).values),
        matching_prior_effective_candidates=distribution(entropy(prior).exp()),
        learned_head_effective_candidates=distribution(entropy(p).exp()),
        learned_head_mass_outside_matching_top1=distribution(1 - top1_mass),
        learned_head_KL_from_matching_prior=distribution((p * (
            p.clamp_min(1e-30).log() - prior[:, None].clamp_min(1e-30).log())).sum(-1)),
        R_input_appearance_RMS=float(module.last_R_input_appearance.float().square().mean().sqrt()) if len(selected) else 0.,
        R_output_RMS=float(module.last_R_only_output.float().square().mean().sqrt()) if len(selected) else 0.,
        semantic_fusion_scale=float(output.attention.fusion_scale),
        member_retained_mass=distribution(module._last_member_retained_mass),
        member_count=distribution(module._last_distinct_member_count))


@contextmanager
def native_signal_context(events):
    from pixal3d.modules.r_mediated_source_contrast import RMediatedSourceContrastConditioner
    from pixal3d.trainers.flow_matching import member_posterior_source_ray as consumer
    original_forward = RMediatedSourceContrastConditioner.forward
    original_add = consumer.add_source_ray_correction

    def forward(module, *args, **kwargs):
        if torch.is_grad_enabled() or module.training:
            raise ValueError('signal probe is an inference-only observer')
        output = original_forward(module, *args, **kwargs)
        events['matching'].append(matching_signal(module, output))
        return output

    def add(native, delta):
        output = original_add(native, delta)
        events['consumer'].append(consumer_signal(native, delta, output))
        return output

    RMediatedSourceContrastConditioner.forward = forward
    consumer.add_source_ray_correction = add
    try:
        yield
    finally:
        RMediatedSourceContrastConditioner.forward = original_forward
        consumer.add_source_ray_correction = original_add
