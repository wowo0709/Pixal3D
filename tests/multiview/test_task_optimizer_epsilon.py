"""The opt-in epsilon must affect only utility updates and preserve other groups."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pixal3d.modules.task_matching_attention import TaskCandidateAttention
from pixal3d.utils.optimizer_groups import split_task_attention_group


class Conditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion_gate = nn.Parameter(torch.ones(()))
        self.retargeter = nn.Linear(4, 4)
        self.task_attention = TaskCandidateAttention(width=4, hidden=3, content_mode='residual-zero')
        self.frozen_head = nn.Linear(4, 4).requires_grad_(False)


def groups(base, conditioner):
    return [{'model_name': 'default', 'params': list(base.parameters()), 'lr': 1e-5},
            {'model_name': 'deform_conditioner',
             'params': [p for p in conditioner.parameters() if p.requires_grad], 'lr': 1e-4}]


def test_none_retains_exact_legacy_groups_and_parameter_order():
    original = groups(nn.Linear(4, 4), Conditioner())
    assert split_task_attention_group(original, None, None) is original


def test_split_is_nonmutating_complete_and_preserves_learning_rates():
    base, conditioner = nn.Linear(4, 4), Conditioner()
    original = groups(base, conditioner)
    before = [[id(p) for p in group['params']] for group in original]
    actual = split_task_attention_group(original, conditioner, 1e-10)
    assert [[id(p) for p in group['params']] for group in original] == before
    assert [g['model_name'] for g in actual] == ['default', 'deform_conditioner', 'deform_conditioner.task_attention']
    assert [g['lr'] for g in actual] == [1e-5, 1e-4, 1e-4]
    assert all('eps' not in g for g in actual[:2]) and actual[2]['eps'] == 1e-10
    assert [id(p) for g in actual for p in g['params']] == [p for group in before for p in group]
    assert {id(p) for p in actual[2]['params']} == {id(p) for p in conditioner.task_attention.parameters()}
    assert not {id(p) for p in conditioner.frozen_head.parameters()} & {id(p) for g in actual for p in g['params']}


def test_real_adam_step_changes_only_utility_with_tiny_gradients():
    base = nn.Linear(4, 4)
    conditioner = Conditioner()
    with torch.no_grad():
        for p in [*base.parameters(), *conditioner.parameters()]:
            p.fill_(1.)
    other_base, other_conditioner = copy.deepcopy(base), copy.deepcopy(conditioner)
    original = torch.optim.AdamW(groups(base, conditioner), betas=(.9, .95), eps=1e-8, weight_decay=0)
    modified = torch.optim.AdamW(split_task_attention_group(groups(other_base, other_conditioner),
                                                           other_conditioner, 1e-10),
                                  betas=(.9, .95), eps=1e-8, weight_decay=0)
    for model in [base, other_base]:
        for p in model.parameters():
            p.grad = torch.full_like(p, 1e-6)
    for model in [conditioner, other_conditioner]:
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.grad = torch.full_like(p, 1e-9 if name.startswith('task_attention.') else 1e-6)
    original.step()
    modified.step()
    assert all(torch.equal(p, q) for p, q in zip(base.parameters(), other_base.parameters()))
    for (name, p), (_, q) in zip(conditioner.named_parameters(), other_conditioner.named_parameters()):
        if name.startswith('task_attention.'):
            assert float((1 - q.detach()).abs().mean()) > 5 * float((1 - p.detach()).abs().mean())
        else:
            assert torch.equal(p, q)
    assert [g['eps'] for g in modified.param_groups] == [1e-8, 1e-8, 1e-10]


@pytest.mark.parametrize('epsilon', [True, 0, -1, float('nan'), float('inf'), '1e-10'])
def test_invalid_epsilon_rejected(epsilon):
    model = Conditioner()
    with pytest.raises(ValueError, match='finite and positive'):
        split_task_attention_group(groups(nn.Linear(4, 4), model), model, epsilon)


def test_missing_module_or_missing_optimizer_membership_rejected():
    model = Conditioner()
    original = groups(nn.Linear(4, 4), model)
    with pytest.raises(ValueError, match='requires the task module'):
        split_task_attention_group(original, SimpleNamespace(task_attention=None), 1e-10)
    removed = next(model.task_attention.parameters())
    original[1]['params'] = [p for p in original[1]['params'] if p is not removed]
    with pytest.raises(ValueError, match='all trainable task parameters'):
        split_task_attention_group(original, model, 1e-10)


def test_duplicate_parameter_membership_rejected():
    model = Conditioner()
    original = groups(nn.Linear(4, 4), model)
    original[0]['params'].append(original[1]['params'][0])
    with pytest.raises(ValueError, match='multiple groups'):
        split_task_attention_group(original, model, 1e-10)


def test_reference_profile_requires_explicit_new_epsilon_study(tmp_path):
    from experiments.deformnext_retargeting.frozen_matcher_utility import validate_profile
    cfg = {'ckpt': 'none', 'load_dir': '',
           'trainer': {'args': {'task_attention_optimizer_eps': 1e-10}}}
    with pytest.raises(ValueError, match='explicit epsilon-study profile'):
        validate_profile(cfg, tmp_path)


def test_three_group_optimizer_and_warmup_resume_preserves_updates():
    from pixal3d.trainers.utils import LinearWarmupLRScheduler
    base, conditioner = nn.Linear(4, 4), Conditioner()
    optimizer = torch.optim.AdamW(
        split_task_attention_group(groups(base, conditioner), conditioner, 1e-10),
        betas=(.9, .95), eps=1e-8, weight_decay=.01)
    scheduler = LinearWarmupLRScheduler(optimizer, warmup_steps=25)
    for p in [*base.parameters(), *conditioner.parameters()]:
        if p.requires_grad:
            p.grad = torch.full_like(p, 1e-9)
    optimizer.step()
    scheduler.step()
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_scheduler = copy.deepcopy(scheduler.state_dict())
    resumed_base, resumed_conditioner = copy.deepcopy(base), copy.deepcopy(conditioner)
    resumed_optimizer = torch.optim.AdamW(
        split_task_attention_group(groups(resumed_base, resumed_conditioner), resumed_conditioner, 1e-10),
        betas=(.9, .95), eps=1e-8, weight_decay=.01)
    resumed_scheduler = LinearWarmupLRScheduler(resumed_optimizer, warmup_steps=25)
    resumed_optimizer.load_state_dict(saved_optimizer)
    resumed_scheduler.load_state_dict(saved_scheduler)
    for p in [*base.parameters(), *conditioner.parameters(),
              *resumed_base.parameters(), *resumed_conditioner.parameters()]:
        if p.requires_grad:
            p.grad = torch.full_like(p, 1e-9)
    optimizer.step()
    resumed_optimizer.step()
    scheduler.step()
    resumed_scheduler.step()
    assert all(torch.equal(p, q) for p, q in zip(base.parameters(), resumed_base.parameters()))
    assert all(torch.equal(p, q) for p, q in zip(conditioner.parameters(), resumed_conditioner.parameters()))
    assert [g['eps'] for g in resumed_optimizer.param_groups] == [1e-8, 1e-8, 1e-10]
    assert [g['lr'] for g in resumed_optimizer.param_groups] == [g['lr'] for g in optimizer.param_groups]
    assert all(float(state['step']) == 2 for state in resumed_optimizer.state.values())


@pytest.mark.parametrize('mode', [None, 'amp'])
@pytest.mark.parametrize('epsilon', [None, 1e-10])
def test_basic_factory_wires_groups_and_warmup_without_cuda(mode, epsilon):
    from pixal3d.trainers.basic import BasicTrainer
    base, conditioner = nn.Linear(4, 4), Conditioner()
    holder = SimpleNamespace(
        world_size=1, models={'denoiser': base, 'deform_conditioner': conditioner},
        model_learning_rates={'deform_conditioner': 1e-4},
        task_attention_optimizer_eps=epsilon, mix_precision_mode=mode,
        mix_precision_dtype=torch.bfloat16, is_master=False,
        optimizer_config={'name': 'AdamW', 'args': {'lr': 1e-5, 'eps': 1e-8}},
        lr_scheduler_config={'name': 'LinearWarmupLRScheduler', 'args': {'warmup_steps': 25}},
        elastic_controller_config=None, grad_clip=None,
    )
    BasicTrainer.init_models_and_more(holder)
    expected_names = ['default', 'deform_conditioner']
    expected_lr = [4e-7, 4e-6]
    expected_eps = [1e-8, 1e-8]
    if epsilon is not None:
        expected_names.append('deform_conditioner.task_attention')
        expected_lr.append(4e-6)
        expected_eps.append(epsilon)
    assert [g['model_name'] for g in holder.optimizer.param_groups] == expected_names
    assert [g['lr'] for g in holder.optimizer.param_groups] == pytest.approx(expected_lr)
    assert [g['eps'] for g in holder.optimizer.param_groups] == expected_eps
    actual = [id(p) for g in holder.optimizer.param_groups for p in g['params']]
    assert actual == [id(p) for p in holder.model_params]
