"""Matched control math/structure and arithmetic probes; no CPU NN forwards."""
import ast
from contextlib import contextmanager
import inspect
import math
import textwrap
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.deformnext_retargeting.shared_evidence_ablation_recipe import VARIANTS, construct_conditioner
from pixal3d.modules.consistent_ray_evidence import AnchoredRawRayReadout, ConsistentRayEvidenceConditioner
from pixal3d.modules.shared_evidence_ablation import (
    SharedEvidenceAblationConditioner, UnprioritizedRawRayReadout,
    uniform_probabilities, unprioritized_probabilities,
)
from test_consistent_ray_evidence import config


@pytest.fixture
def matcher(monkeypatch):
    from pixal3d.modules import r9_matching
    class FrozenProbe(nn.Module):
        def __init__(self):
            super().__init__(); self.weight = nn.Parameter(torch.ones(1))
        def forward(self, *a, **kw):
            raise AssertionError('CPU matcher NN forward is forbidden')
    monkeypatch.setattr(r9_matching, 'PromotedR9Matcher', FrozenProbe)


def make(variant):
    kwargs = config(VARIANTS[variant][0])[0]['trainer']['args']['deform_conditioner']
    kwargs['freeze_q2c'] = False
    return construct_conditioner(variant, kwargs)


def test_uniform_keeps_every_valid_candidate_and_empty_support_zero():
    valid = torch.tensor([[True, True, False], [False, False, False]])
    p = uniform_probabilities(valid)
    assert p.shape == (2, 4, 3)
    assert torch.equal(p[0], torch.tensor([[.5, .5, 0.]]).expand(4, -1))
    assert p[1].count_nonzero() == 0 and torch.isfinite(p).all()


def test_no_prior_keeps_same_head_and_eligible_support_but_still_learns():
    prior = torch.tensor([[.99, .01, 0.], [0., 0., 0.]]).log()
    valid = torch.tensor([[True, True, True], [False, False, False]])
    scores = torch.zeros(2, 4, 3, requires_grad=True)
    p = unprioritized_probabilities(scores, torch.zeros_like(scores), prior, valid)
    assert torch.equal(p[0], torch.tensor([[.5, .5, 0.]]).expand(4, -1))
    assert p[1].count_nonzero() == 0
    p[0, :, 1].sum().backward()
    assert (scores.grad[0].abs().sum(-1) > 0).all()
    assert scores.grad[1].count_nonzero() == 0
    changed = scores.detach().clone(); changed[0, :, 1] = 4.
    assert unprioritized_probabilities(changed, torch.zeros_like(changed), prior, valid)[0, :, 1].min() > .98


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_no_prior_does_not_hide_corrupt_prior(bad):
    with pytest.raises(ValueError):
        unprioritized_probabilities(torch.zeros(1, 4, 2), torch.zeros(1, 4, 2),
                                    torch.tensor([[bad, 0.]]), torch.ones(1, 2, dtype=torch.bool))


@pytest.mark.parametrize('variant', VARIANTS)
def test_only_declared_parameters_change_and_constructor_preserves_rng(matcher, variant):
    before = torch.random.get_rng_state(); module = make(variant)
    assert torch.equal(before, torch.random.get_rng_state())
    assert module._actual_initial_retained_parameters_equal_parent
    assert sum(p.requires_grad for p in module.parameters()) == VARIANTS[variant][3]
    assert not any(p.requires_grad for p in module.q2c.parameters())
    module.train(); assert not module.q2c.training
    if 'NO_R' in variant:
        assert module.retargeter is None
        assert not any(k.startswith('retargeter.') for k in module.state_dict())
    elif variant == 'A_FIXED_UNIFORM':
        assert module.ray_readout is None
    else:
        assert isinstance(module.ray_readout, UnprioritizedRawRayReadout)
        assert len(list(module.ray_readout.parameters())) == 7
        assert not hasattr(module.ray_readout, 'value') and not hasattr(module.ray_readout, 'output')


def test_uniform_is_not_a_pure_score_control_of_a_learned_model(matcher):
    kwargs = config()[0]['trainer']['args']['deform_conditioner']; kwargs['freeze_q2c'] = False
    with pytest.raises(ValueError):
        SharedEvidenceAblationConditioner(ablation='uniform-k', evidence_mode='learned-anchored', **kwargs)


@pytest.mark.parametrize('damage', ['value', 'dtype', 'shape'])
def test_wrong_control_checkpoint_schema_is_rejected(matcher, damage):
    m = make('A_FIXED_UNIFORM'); state = m.state_dict(); key = 'shared_evidence_ablation_schema'
    if damage == 'value': state[key] = state[key]+1
    elif damage == 'dtype': state[key] = state[key].float()
    else: state[key] = state[key].reshape(())
    with pytest.raises(RuntimeError, match='schema differ'): m.load_state_dict(state, strict=True)


def test_no_prior_forward_changes_only_the_probability_operator():
    reference = ast.parse(textwrap.dedent(inspect.getsource(AnchoredRawRayReadout.forward)))
    actual = ast.parse(textwrap.dedent(inspect.getsource(UnprioritizedRawRayReadout.forward)))
    found = []
    for node in ast.walk(actual):
        if isinstance(node, ast.Name) and node.id == 'unprioritized_probabilities':
            found.append(node); node.id = 'anchored_probabilities'
    assert len(found) == 1
    assert ast.dump(actual) == ast.dump(reference)


def test_shared_forward_has_only_explicit_weighting_and_R_presence_controls():
    reference = ast.parse(textwrap.dedent(inspect.getsource(ConsistentRayEvidenceConditioner.forward)))
    actual = ast.parse(textwrap.dedent(inspect.getsource(SharedEvidenceAblationConditioner.forward)))
    class RestoreReference(ast.NodeTransformer):
        def visit_IfExp(self, node):
            if ast.unparse(node.test) == "self.ablation == 'uniform-k'":return node.orelse
            return self.generic_visit(node)
        def visit_If(self, node):
            if ast.unparse(node.test) == 'self.retargeter is not None':return node.body
            return self.generic_visit(node)
    restored = RestoreReference().visit(actual)
    assert ast.dump(restored) == ast.dump(reference)


def test_native_trainer_initialization_differs_only_in_the_source_constructor():
    from experiments.deformnext_retargeting.consistent_evidence_native_recipe import build_trainer as parent_builder
    from experiments.deformnext_retargeting.shared_evidence_ablation_recipe import build_trainer
    parent = ast.parse(textwrap.dedent(inspect.getsource(parent_builder)))
    actual = ast.parse(textwrap.dedent(inspect.getsource(build_trainer)))
    def init(tree):
        return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'init_models_and_more')
    assert ast.dump(init(actual)) == ast.dump(init(parent))


def test_unknown_variant_cannot_build_an_unbound_trainer():
    from experiments.deformnext_retargeting.shared_evidence_ablation_recipe import build_trainer
    with pytest.raises(ValueError):build_trainer('E_FIXED_RAW')


@pytest.mark.parametrize('variant', ['A_FIXED_UNIFORM', 'A_FIXED_NO_R', 'A_LEARNED_NO_R'])
def test_same_evidence_reaches_use_R_and_native_source_subtraction(matcher, monkeypatch, variant):
    from pixal3d.modules import shared_evidence_ablation as source
    from pixal3d.modules.consistent_ray_evidence import raw_evidence
    from pixal3d.modules.native_prior_source_ray import source_ray_delta
    m = make(variant)
    class RetargetProbe(nn.Module):
        @contextmanager
        def native_source_context(self, own):self.own = own; yield
        def forward(self, inputs, g_max):
            self.appearance = inputs[:, :1024].clone(); return torch.full_like(self.appearance, 2.)
    class UseProbe(nn.Module):
        def forward(self, query, evidence, valid):self.evidence = evidence.clone(); return valid.float()*.5
    class HeadProbe(nn.Module):
        def forward(self, native, context, keys, values, valid, prior):
            p = torch.zeros(len(values), 4, 64); p[:, :, 0] = .75; p[:, :, 1] = .25
            return raw_evidence(values, p, valid)
    if m.retargeter is not None:m.retargeter = RetargetProbe()
    m.evidence_use = UseProbe()
    if m.ray_readout is not None:m.ray_readout = HeadProbe()
    ids = torch.zeros(1, 3072, 64, dtype=torch.long); ids[:, :, 1] = 1
    valid = torch.zeros_like(ids, dtype=torch.bool); valid[:, 0, :2] = True
    p = torch.zeros_like(ids, dtype=torch.float); p[:, 0, 0] = .25; p[:, 0, 1] = .75
    posterior = SimpleNamespace(candidate_indices=ids, candidate_valid=valid, probabilities=p,
        contextual_query=torch.zeros(1, 3072, 1024), retained_mass=torch.zeros(1, 3072),
        distinct_member_count=torch.zeros(1, 3072, dtype=torch.long))
    monkeypatch.setattr(source, 'member_ray_posteriors', lambda *a, **kw:posterior)
    monkeypatch.setattr(source, 'pool_ray_queries', lambda value,*a,**kw:
        SimpleNamespace(values=value[:, :3072].reshape(1, 3, 1024, 1024)))
    native = torch.zeros(1, 4096, 1024); values = torch.zeros(1, 3, 1024, 1024)
    values[:, 0, 0] = 10.; values[:, 0, 1] = 20.
    with torch.no_grad():
        m.fusion_gate.fill_(math.atanh(.2))
        result = m(native_voxels=native, target_omega_voxels=native, target_native_voxels=native,
            frame_target_keys=None, target_patch_ids=None, ray_ids=None, observed_query=None,
            source_keys=values, source_values=values, source_valid=None)
    expected = 15. if variant == 'A_FIXED_UNIFORM' else 12.5 if variant == 'A_LEARNED_NO_R' else 17.5
    assert result.attention.base_evidence[0, 0].eq(expected).all()
    assert torch.equal(m.last_selected_evidence, m.evidence_use.evidence)
    delta = source_ray_delta(result.source_numerator, result.semantic_weight, values)
    extra = 1. if m.retargeter is not None else 0.
    assert torch.allclose(delta[0, 0, 0], torch.full((1024,), .1*(expected-10.)+extra))
    assert delta[:, :, 1:].count_nonzero() == 0 and delta[:, 1:].count_nonzero() == 0
    if m.retargeter is not None:
        assert m.retargeter.appearance.eq(expected).all() and m.retargeter.own.eq(10.).all()
    else:assert result.attention.retarget_residual.count_nonzero() == 0
