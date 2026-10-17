import ast
import hashlib
import json
from pathlib import Path

import pytest

from experiments.deformnext_retargeting import historical_geometry_statistics as s
from experiments.deformnext_retargeting.summarize_geometry_reproduction import leaf_errors


def test_original_pair_bootstrap_is_pinned_and_does_not_treat_noise_as_new_pairs():
    path = Path(s.__file__)
    authority = json.loads(path.with_name('historical_geometry_statistics_authority.json').read_text())
    function = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef))
    assert hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest() == authority['AST_sha256']
    result = s._ci([.04] * 8, [str(i) for i in range(8)])
    assert result['mean'] == pytest.approx(.04)
    assert result['ci95'] == pytest.approx([.04, .04])
    assert len(result['pair_values']) == 8
    with pytest.raises(RuntimeError, match='pair-level'):
        s._ci([.04] * 72, [str(i) for i in range(8)])


def test_endpoint_audit_compares_nested_regional_bce_terms_as_well_as_totals():
    actual = {'regional_bce': 1., 'regional_bce_terms': {'64': {'source_positive': 2.}}}
    expected = {'regional_bce': 1., 'regional_bce_terms': {'64': {'source_positive': 2.001}}}
    assert max(leaf_errors(actual, expected)) == pytest.approx(.001)
    assert not any(leaf_errors(actual, actual))
    with pytest.raises(ValueError, match='keys'):
        leaf_errors(actual, {'regional_bce': 1.})
