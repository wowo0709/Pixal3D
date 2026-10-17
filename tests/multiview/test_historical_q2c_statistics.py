"""Keep original OOF cue/seed/pair reduction and paired uncertainty intact."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

from experiments.deformnext_retargeting import historical_q2c_statistics as s


def test_aggregation_and_pair_bootstrap_match_original_ast():
    directory = Path(s.__file__).parent
    authority = json.loads((directory / 'historical_q2c_statistics_authority.json').read_text())
    nodes = {n.name: n for n in ast.parse(Path(s.__file__).read_text()).body if hasattr(n, 'name')}
    for name, origin in authority['function_origins'].items():
        assert hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest() == origin['AST_sha256']


def rows():
    return [dict(backbone='OMEGA_L12_FRAME_RAW', split='DEV', fold=fold, pair_id=pair,
                 seed=seed, arm=arm, original_cue_id=cue, query_slot=0, query_id=-1,
                 task_rgb_error=(.04 + .02 * fold) - (.01 if arm == 'D1-Q2C' else 0),
                 wrong_rgb_error=.1, gt_rgb_error=0.)
            for fold, pair in enumerate(['a', 'b']) for seed in [42017, 42018, 42019]
            for arm in ['D0-PAIRWISE', 'D1-Q2C'] for cue in [0, 1]]


def test_rgb_gain_aggregates_cues_and_model_seeds_before_pair_bootstrap():
    aggregate = s.aggregate_rows(rows())
    assert aggregate['pair_count'] == 2
    assert aggregate['arms']['D0-PAIRWISE']['task_rgb_error'] == pytest.approx(.05)
    assert aggregate['arms']['D1-Q2C']['task_rgb_error'] == pytest.approx(.04)
    contrast = s.contrast(aggregate, 'D0-PAIRWISE', 'D1-Q2C')
    assert contrast['mean'] == pytest.approx(.01)
    assert contrast['ci95'] == pytest.approx([.01, .01])
    assert contrast['seed_gains'] == pytest.approx([.01] * 3)
    assert contrast['pair_count'] == 2


def test_oof_reproduction_refuses_missing_seed_duplicate_and_cue_level_ratios():
    with pytest.raises(ValueError, match='coverage'):
        s.aggregate_rows(rows()[:-1])
    with pytest.raises(ValueError, match='duplicate'):
        s.aggregate_rows([*rows(), rows()[0]])
    with pytest.raises(ValueError, match='cue-level ratio'):
        s.aggregate_rows([{**r, 'g_oracle': 0.} for r in rows()])
