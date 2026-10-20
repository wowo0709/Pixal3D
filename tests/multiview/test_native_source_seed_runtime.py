"""Only the two explicit seed sites may differ in a confirmation diagnostic."""
import ast
from pathlib import Path

import pytest

from experiments.deformnext_retargeting.native_source_seed_runtime import PROBES, seed202_ast


@pytest.mark.parametrize('native', [False, True])
def test_rebinding_preserves_unrelated_constants_and_rng_calls(native):
    source = """
def main():
    if cfg['training_seed'] != 101: raise ValueError('seed')
    other = 101
    math_value = value * 101
    setup_rng(0, 80217)
""" + ('    setup_rng(0, 101)\n' if native else '')
    tree, counts = seed202_ast(source, native_probe=native)
    text = ast.unparse(tree)
    assert "cfg['training_seed'] != 202" in text
    assert 'other = 101' in text and 'value * 101' in text
    assert 'setup_rng(0, 80217)' in text
    assert counts == dict(seed_guard=1, initial_RNG=int(native))


def test_all_frozen_family_probes_have_exactly_the_declared_seed_sites():
    root = Path(__file__).resolve().parents[2] / 'experiments/deformnext_retargeting'
    for native in [True, False]:
        names = {pair[0 if native else 1] for pair in PROBES.values()}
        for name in names:
            path = root / (name+'.py');before = path.read_bytes()
            tree, counts = seed202_ast(before.decode(), native_probe=native)
            compile(tree, '<seed202-check>', 'exec')
            assert counts == dict(seed_guard=1, initial_RNG=int(native))
            assert path.read_bytes() == before


@pytest.mark.parametrize('source,native', [
    ('if cfg["training_seed"] != 101: pass', True),
    ('if cfg["training_seed"] != 202: pass', False),
    ('if cfg["training_seed"] != 101: pass\nsetup_rng(0, 101)', False),
    ('if cfg["training_seed"] != 101: pass\nif cfg.get("training_seed") != 101: pass', False),
])
def test_changed_or_missing_sites_fail_instead_of_weakening_guards(source, native):
    with pytest.raises(ValueError):
        seed202_ast(source, native_probe=native)
