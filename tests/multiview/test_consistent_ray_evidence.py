"""Helper math and constructor contracts; no CPU neural forward/training."""
import copy
import math

import pytest
import torch
from torch import nn

from experiments.deformnext_retargeting.consistent_evidence_native_recipe import (
    NAME, SCHEMA, STUDY_SCHEMA, SOURCE_COUNTS, VARIANTS, construct_conditioner, validate_config,
)
from pixal3d.modules.bounded_member_feature_source_ray import bounded_probabilities, masked_probabilities
from pixal3d.modules.consistent_ray_evidence import AnchoredRawRayReadout, anchored_probabilities, raw_evidence
from test_r_mediated_native_recipe import recipe


def config(variant='E_FIXED_RAW', phase='science'):
    reference = recipe()
    cfg = copy.deepcopy(reference)
    cfg['trainer']['name'] = NAME
    cfg['trainer']['args'].pop('ray_feature_utility')
    cfg['trainer']['args']['deform_conditioner'].update(
        ray_mode='semantic-weighted', candidate_aggregation='weighted')
    cfg['use_wandb'] = phase == 'science'
    cfg['experiment'].update(native_variant=variant, conditioning_schema=SCHEMA,
        followup_study_schema=STUDY_SCHEMA, evidence_mode=VARIANTS[variant],
        actual_source_tensors=SOURCE_COUNTS[variant],
        learned_attention_head_tensors=7 if variant == 'E_LEARNED_RAW' else 0,
        learned_projection_tensors=0, shared_carrier_use_retarget_evidence=True,
        physical_deformation_query_channels=0, appearance_mode='consistent-raw-topk',
        residual_logit_bound=None, engineering_only=phase != 'science')
    return cfg, reference


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('phase', ['proof', 'capacity', 'science'])
def test_only_shared_evidence_and_probabilities_change(variant, phase):
    cfg, reference = config(variant, phase)
    assert validate_config(cfg, phase, reference) == variant


@pytest.mark.parametrize('damage', ['top1', 'uniform', 'head-count', 'V', 'old-family',
    'cap', 'teacher', 'batch', 'split', 'lr', 'candidate', 'DQ', 'matching', 'native', 'phase'])
def test_reject_confound_or_false_architecture(damage):
    cfg, reference = config('E_LEARNED_RAW')
    e, t = cfg['experiment'], cfg['trainer']['args']
    if damage == 'top1': t['deform_conditioner']['candidate_aggregation'] = 'top1'
    elif damage == 'uniform': e['evidence_mode'] = 'uniform'
    elif damage == 'head-count': e['actual_source_tensors'] = 19
    elif damage == 'V': e['learned_projection_tensors'] = 2
    elif damage == 'old-family': e['module_pipeline_schema'] = 'unified-module-pipeline-v2'
    elif damage == 'cap': e['residual_logit_bound'] = 1.
    elif damage == 'teacher': t['ray_feature_utility'] = reference['trainer']['args']['ray_feature_utility']
    elif damage == 'batch': t['batch_size_per_gpu'] = 48
    elif damage == 'split': cfg['data_dir'] = 'different'
    elif damage == 'lr': t['optimizer']['args']['lr'] = 1e-4
    elif damage == 'candidate': t['deform_conditioner']['topk'] = 32
    elif damage == 'DQ': t['deform_conditioner']['deformation_dim'] = 2048
    elif damage == 'matching': t['deform_conditioner']['score_temperature'] = .5
    elif damage == 'native': e['shared_carrier_use_retarget_evidence'] = False
    else: cfg['use_wandb'] = False
    with pytest.raises(ValueError): validate_config(cfg, 'science', reference)


def test_initial_posterior_is_exact_and_preserves_zero_support():
    prior = torch.tensor([[.8, .2, 0.], [0., 0., 0.]]).log()
    valid = torch.tensor([[True, True, True], [False, False, False]])
    initial = torch.tensor([[[1., 2., float('-inf')]]]).expand(2, 4, -1).clone()
    probability = anchored_probabilities(initial, initial, prior, valid)
    expected = masked_probabilities(prior[:, None].expand(-1, 4, -1), valid)
    assert torch.equal(probability, expected)
    assert torch.isfinite(probability).all() and probability[1].count_nonzero() == 0
    assert probability[:, :, 2].count_nonzero() == 0


def test_learned_scores_can_overcome_a_wrong_high_confidence_prior():
    prior = torch.tensor([[.99, .01]]).log()
    initial = torch.zeros(1, 4, 2)
    current = initial.clone(); current[:, :, 1] = 8.
    valid = torch.ones(1, 2, dtype=torch.bool)
    p = anchored_probabilities(current, initial, prior, valid)
    old = bounded_probabilities(current, initial, prior, valid, 1.)
    assert p[0, :, 1].min() > .96
    assert old[0, :, 1].max() < .07
    assert torch.allclose(p, anchored_probabilities(current+7., initial, prior, valid))


def test_raw_values_keep_scale_and_mask_invalid_nan_features():
    values = torch.tensor([[[10., -2.], [2., 6.], [float('nan'), float('nan')]]])
    p = torch.tensor([[[.75, .25, 0.]]]).expand(-1, 4, -1)
    valid = torch.tensor([[True, True, False]])
    evidence, weights = raw_evidence(values, p, valid)
    assert torch.equal(evidence, torch.tensor([[8., 0.]]))
    assert torch.equal(weights, p[:, 0])


def test_all_four_score_heads_receive_gradient_through_raw_evidence():
    prior = torch.tensor([[.9, .1]]).log()
    scores = torch.zeros(1, 4, 2, requires_grad=True)
    valid = torch.ones(1, 2, dtype=torch.bool)
    p = anchored_probabilities(scores, torch.zeros_like(scores), prior, valid)
    evidence, _ = raw_evidence(torch.tensor([[[1.], [5.]]]), p, valid)
    evidence.sum().backward()
    assert torch.isfinite(scores.grad).all() and (scores.grad.abs().sum(-1) > 0).all()


def test_nonfinite_supported_learned_score_is_an_error():
    scores = torch.zeros(1, 4, 2); scores[0, 0, 1] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        anchored_probabilities(scores, torch.zeros_like(scores), torch.zeros(1, 2),
                               torch.ones(1, 2, dtype=torch.bool))


@pytest.fixture
def mocked_matcher(monkeypatch):
    from pixal3d.modules import r9_matching
    class Matcher(nn.Module):
        def __init__(self):
            super().__init__(); self.frozen = nn.Parameter(torch.ones(1))
        def forward(self, *args, **kwargs):
            raise AssertionError('CPU neural matching forward is forbidden in these checks')
    monkeypatch.setattr(r9_matching, 'PromotedR9Matcher', Matcher)


def test_physical_core10_plus_head7_and_frozen_matcher_without_rng_shift(mocked_matcher):
    modules = []
    for variant in VARIANTS:
        kwargs = config(variant)[0]['trainer']['args']['deform_conditioner']
        kwargs['freeze_q2c'] = False
        before = torch.random.get_rng_state()
        module = construct_conditioner(variant, kwargs)
        assert torch.equal(before, torch.random.get_rng_state())
        assert sum(p.requires_grad for p in module.parameters()) == SOURCE_COUNTS[variant]
        assert module._actual_initial_core10_equal_Top1
        assert module.retargeter.input_dim == 2048 and module.fusion_gate.item() == 0.
        module.train(); assert not module.q2c.training
        assert not any(p.requires_grad for p in module.q2c.parameters())
        modules.append(module)
    first = {n: p for n, p in modules[0].named_parameters() if p.requires_grad}
    second = dict(modules[1].named_parameters())
    assert all(torch.equal(p, second[n]) for n, p in first.items())
    head = modules[1].ray_readout
    assert len(list(head.parameters())) == 7
    assert not any(hasattr(head, name) for name in ['value', 'output', 'residual_logit_bound'])
    assert modules[0].ray_readout is None
    with pytest.raises(RuntimeError): modules[0].load_state_dict(modules[1].state_dict(), strict=True)


@pytest.mark.parametrize('corrupt', ['value', 'dtype', 'shape'])
def test_readout_schema_is_checked_not_silently_overwritten(corrupt):
    head = AnchoredRawRayReadout()
    state = head.state_dict(); key = 'anchored_raw_readout_schema'
    if corrupt == 'value': state[key] = state[key]+1
    elif corrupt == 'dtype': state[key] = state[key].float()
    else: state[key] = state[key].reshape(())
    with pytest.raises(RuntimeError, match='schema differ'): head.load_state_dict(state, strict=True)


def test_conditioner_rejects_other_mode_or_legacy_top1_schema(mocked_matcher):
    kwargs = config()[0]['trainer']['args']['deform_conditioner']; kwargs['freeze_q2c'] = False
    module = construct_conditioner('E_FIXED_RAW', kwargs)
    state = module.state_dict()
    state['consistent_ray_evidence_schema'] = torch.tensor([2], dtype=torch.uint8)
    with pytest.raises(RuntimeError, match='schema differ'): module.load_state_dict(state, strict=True)
    state.pop('consistent_ray_evidence_schema')
    state['r_mediated_top1_source_schema'] = torch.tensor([1], dtype=torch.uint8)
    with pytest.raises(RuntimeError): module.load_state_dict(state, strict=True)


@pytest.mark.parametrize('variant', VARIANTS)
def test_real_forward_uses_one_evidence_at_all_three_boundaries_without_CPU_NN(mocked_matcher, monkeypatch, variant):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from pixal3d.modules import consistent_ray_evidence as module_source
    from pixal3d.modules.native_prior_source_ray import source_ray_delta
    kwargs = config(variant)[0]['trainer']['args']['deform_conditioner']; kwargs['freeze_q2c'] = False
    module = construct_conditioner(variant, kwargs)
    class RetargetProbe(nn.Module):
        @contextmanager
        def native_source_context(self, source):
            self.own_source = source; yield
        def forward(self, inputs, g_max):
            self.appearance = inputs[:, :1024].clone()
            return torch.full_like(self.appearance, 2.)
    class UseProbe(nn.Module):
        def forward(self, query, evidence, valid):
            self.evidence = evidence.clone()
            return valid.float()*.5
    class HeadProbe(nn.Module):
        def forward(self, native, context, keys, values, valid, prior):
            p = torch.zeros(len(values), 4, 64)
            p[:, :, 0] = .75; p[:, :, 1] = .25
            return raw_evidence(values, p, valid)
    module.retargeter = RetargetProbe(); module.evidence_use = UseProbe()
    if variant == 'E_LEARNED_RAW': module.ray_readout = HeadProbe()
    ids = torch.zeros(1, 3072, 64, dtype=torch.long); ids[:, :, 1] = 1
    valid = torch.zeros_like(ids, dtype=torch.bool); valid[:, 0, :2] = True
    p = torch.zeros_like(ids, dtype=torch.float); p[:, 0, 0] = .25; p[:, 0, 1] = .75
    posterior = SimpleNamespace(candidate_indices=ids, candidate_valid=valid, probabilities=p,
        contextual_query=torch.zeros(1, 3072, 1024), retained_mass=torch.zeros(1, 3072),
        distinct_member_count=torch.zeros(1, 3072, dtype=torch.long))
    monkeypatch.setattr(module_source, 'member_ray_posteriors', lambda *a, **kw: posterior)
    monkeypatch.setattr(module_source, 'pool_ray_queries',
        lambda value, *a, **kw: SimpleNamespace(values=value[:, :3072].reshape(1, 3, 1024, 1024)))
    native = torch.zeros(1, 4096, 1024)
    source = torch.zeros(1, 3, 1024, 1024); source[:, 0, 0] = 10.; source[:, 0, 1] = 20.
    with torch.no_grad():
        module.fusion_gate.fill_(math.atanh(.2))
        output = module(native_voxels=native, target_omega_voxels=native, target_native_voxels=native,
            frame_target_keys=None, target_patch_ids=None, ray_ids=None, observed_query=None,
            source_keys=source, source_values=source, source_valid=None)
    expected = 17.5 if variant == 'E_FIXED_RAW' else 12.5
    evidence = output.attention.base_evidence
    assert evidence[0, 0].eq(expected).all()
    assert module.retargeter.appearance.eq(expected).all()
    assert torch.equal(module.evidence_use.evidence, evidence)
    assert torch.equal(module.last_selected_evidence, evidence)
    assert module.retargeter.own_source.eq(10.).all()
    delta = source_ray_delta(output.source_numerator, output.semantic_weight, source)
    assert torch.allclose(delta[0, 0, 0], torch.full((1024,), .1*(expected-10.)+1.))
    assert delta[:, :, 1:].count_nonzero() == 0
    assert delta[:, 1:].count_nonzero() == 0


def test_checkpoint_pair_cannot_alias_original_full_or_another_source_run(tmp_path):
    import hashlib
    import json
    from experiments.deformnext_retargeting.evaluate_consistent_evidence_checkpoint import BUNDLE_SCHEMA, validate_bundle
    cfg, reference = config('E_LEARNED_RAW')
    path = tmp_path/'reference.json'; path.write_text(json.dumps(reference))
    cfg['experiment'].update(reference_science_config=str(path),
        reference_science_config_SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
    cfg['output_dir'] = '/actual/E_LEARNED_RAW/native'
    bundle = dict(schema=BUNDLE_SCHEMA, variant='E_LEARNED_RAW', step=480,
        checkpoint='/actual/E_LEARNED_RAW/native/ckpts/denoiser_step0000480.pt',
        conditioner_checkpoint='/actual/E_LEARNED_RAW/native/ckpts/deform_conditioner_step0000480.pt',
        native_config='/actual/E_LEARNED_RAW/native/config.json', source_teacher_updates=0,
        native_task_objective='SS_flow_mse', native_validation={'step': 480})
    assert validate_bundle(bundle, cfg) == 'E_LEARNED_RAW'
    for damage in ['schema', 'source', 'variant']:
        damaged = dict(bundle)
        if damage == 'schema': damaged['schema'] = 'unified-module-projected-SS-checkpoint-pair-v1'
        elif damage == 'source': damaged['conditioner_checkpoint'] = '/actual/Full/native/ckpts/deform_conditioner_step0000480.pt'
        else: damaged['variant'] = 'E_FIXED_RAW'
        with pytest.raises(ValueError): validate_bundle(damaged, cfg)
