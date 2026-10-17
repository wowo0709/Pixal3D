"""Check pinned runtime math before allocating a real frozen SS rollout."""
import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.deformnext_retargeting import historical_geometry_runtime as runtime
from experiments.deformnext_retargeting.historical_reproduction import GeometryFactorialVoxelAdapter
from experiments.deformnext_retargeting.reproduce_geometry_content import ARMS, reference_rows


def test_native_sampler_scoring_condition_and_noise_match_historical_ast():
    directory = Path(runtime.__file__).parent
    authority = json.loads((directory / 'historical_geometry_runtime_authority.json').read_text())
    nodes = {n.name: n for n in ast.parse(Path(runtime.__file__).read_text()).body if hasattr(n, 'name')}
    for name, origin in authority['function_origins'].items():
        assert hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest() == origin['AST_sha256']
    assert ARMS == ['r2_feat', 'r4_content']


@pytest.mark.parametrize('arm', ARMS)
def test_zero_adapter_preserves_native_global_negative_and_projected_source_control(arm):
    torch.manual_seed(9)
    adapter = GeometryFactorialVoxelAdapter(channels=4, hidden=8, geometry_hidden=3)
    target = torch.randn(1, 4096, 4)
    mismatch = torch.randn_like(target)
    pair = {
        'clean_target_condition': {'cond': {'global': torch.ones(1, 5, 4), 'proj': target},
                                   'neg_cond': {'global': torch.zeros(1, 5, 4), 'proj': target * 0}},
        'native_mismatch_condition': {'cond': {'global': torch.full((1, 5, 4), 2.), 'proj': mismatch},
                                      'neg_cond': {'global': torch.full((1, 5, 4), -1.), 'proj': mismatch * 0}},
    }
    geometry = torch.randn(1, 4096, 9)
    normal, _, native, _ = runtime.adapted_condition(pair, adapter, geometry, arm, torch.device('cpu'))
    off, _, _, _ = runtime.adapted_condition(pair, adapter, geometry, arm, torch.device('cpu'), source_off=True)
    assert torch.equal(normal['cond']['proj'], mismatch)
    assert torch.equal(off['cond']['proj'], target)
    for condition in [normal, off]:
        assert torch.equal(condition['cond']['global'], native['cond']['global'])
        for key in ['global', 'proj']:
            assert torch.equal(condition['neg_cond'][key], native['neg_cond'][key])


def test_reproduction_noise_is_stable_and_preserves_global_rng():
    before = torch.get_rng_state().clone()
    values = [runtime._seeded_noise('pair', seed, shape=(1, 8, 2, 2, 2), device='cpu') for seed in range(3)]
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(value, runtime._seeded_noise('pair', seed, shape=value.shape, device='cpu'))
               for seed, value in enumerate(values))
    assert not torch.equal(values[0], values[1])
    assert not torch.equal(values[1], values[2])


def test_geometry_shuffle_preserves_nine_channel_tuples_without_fixed_locations():
    geometry = torch.arange(4096 * 9).reshape(1, 4096, 9)
    order = runtime.geometry_derangement('pair')
    assert not torch.any(order == torch.arange(4096))
    assert torch.equal(order.sort().values, torch.arange(4096))
    assert torch.equal(geometry[:, order].sort(dim=1).values, geometry)


def test_reference_export_excludes_confirm_pairs_and_requires_locked_train_dev_counts():
    rows = [{'pair_id': f'train{i}', 'split': 'train'} for i in range(32)]
    rows += [{'pair_id': f'dev{i}', 'split': 'dev'} for i in range(8)]
    rows += [{'pair_id': f'confirm{i}', 'split': 'confirm'} for i in range(6)]
    selected = reference_rows({'records': rows})
    assert len(selected) == 40
    assert all(r['split'] in ['train', 'dev'] for r in selected)
    with pytest.raises(ValueError, match='TRAIN32/DEV8'):
        reference_rows({'records': rows[1:]})
    with pytest.raises(ValueError, match='TRAIN32/DEV8'):
        reference_rows({'records': [rows[1], *rows[1:]]})
