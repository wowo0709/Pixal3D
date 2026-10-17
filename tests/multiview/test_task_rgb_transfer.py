"""Current-data task labels preserve barycentrics, split isolation and cue firewall."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.deformnext_retargeting import historical_surface_cue as original
from experiments.deformnext_retargeting.cache_task_rgb_transfer import (
    patch_operator, stable_seed, choose, freeze,
)


def test_value_generator_matches_original_ast_and_has_disjoint_role_seeds():
    path = Path(original.__file__)
    authority = json.loads(path.with_name('historical_surface_cue_authority.json').read_text())
    function = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef))
    assert hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest() == authority['AST_sha256']
    train = {stable_seed('asset', 'label', cue) for cue in range(16)}
    evaluation = {stable_seed('asset', 'evaluation', cue) for cue in range(16)}
    assert len(train) == len(evaluation) == 16 and not train & evaluation
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    before = np.random.get_state()
    rgb = original.procedural_surface_cue(vertices, active_vertices=np.array([True, False, True]), seed=7)
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1])
    assert np.all(rgb[1] == 0) and np.all(rgb[[0, 2]] >= .08)


def test_patch_operator_interpolates_barycentric_foreground_means_without_area_shortcut():
    faces = np.array([[0, 1, 2]])
    face = np.zeros(16, dtype=np.int64)
    bary = np.tile([.25, .25, .5], (16, 1))
    valid = np.ones(16, dtype=bool)
    alpha = np.ones((4, 4), np.float32)
    alpha[:2, :2] = 0
    alpha[0, 0] = .5
    operator, mass = patch_operator(face, bary, valid, alpha, faces, 3, image_size=4, patch_size=2)
    assert mass.tolist() == [.125, 1., 1., 1.]
    colors = np.eye(3)
    expected = np.tile([.25, .25, .5], (4, 1))
    np.testing.assert_allclose(operator @ colors, expected)
    assert np.allclose(np.asarray(operator.sum(1)).reshape(-1), 1.)


def test_missing_rays_do_not_acquire_rgb_or_vertex_correspondences():
    face = np.full(16, 2**32-1, dtype=np.int64)
    valid = np.zeros(16, bool)
    operator, mass = patch_operator(face, np.zeros((16, 3)), valid, np.ones((4, 4)),
        np.array([[0, 1, 2]]), 3, image_size=4, patch_size=2)
    assert operator.nnz == 0 and not mass.any()
    with pytest.raises(ValueError, match='barycentric'):
        patch_operator(np.zeros(16, np.int64), np.zeros((16, 3)), np.ones(16, bool), np.ones((4, 4)),
            np.array([[0, 1, 2]]), 3, image_size=4, patch_size=2)


def test_pilot_selection_is_deterministic_balanced_and_cluster_disjoint(tmp_path):
    def row(i, role):
        return {'sample_id': f'{role}-{i}', 'category': 'a' if i % 2 else 'b',
                'source_cluster_id': f'{role}-cluster-{i // 2}'}
    rows = [row(i, 'train') for i in range(12)]
    selected = choose(rows, 4, 'train')
    assert selected == choose(list(reversed(rows)), 4, 'train')
    assert len({r['source_cluster_id'] for r in selected}) == 4
    assert {r['category'] for r in selected} == {'a', 'b'}
    panels = tmp_path / 'panels'; panels.mkdir()
    for name, role in [('head_train', 'train'), ('head_calibration', 'cal'), ('head_validation', 'val')]:
        (panels / (name + '.jsonl')).write_text(''.join(json.dumps(row(i, role)) + '\n' for i in range(12)))
    out = tmp_path / 'frozen'
    freeze(panels, out, train_count=4, calibration_count=2)
    report = json.loads((out / 'population.json').read_text())
    assert report['source_clusters_disjoint'] and not report['validation_508_used_for_training_or_task_evaluation']
    overlap = row(0, 'val'); overlap['source_cluster_id'] = 'train-cluster-0'
    with (panels / 'head_validation.jsonl').open('a') as stream:
        stream.write(json.dumps(overlap) + '\n')
    with pytest.raises(ValueError, match='overlap'):
        freeze(panels, tmp_path / 'invalid', train_count=4, calibration_count=2)
