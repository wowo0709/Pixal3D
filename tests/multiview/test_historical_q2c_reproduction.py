"""Guard the actual historical task, zero start, candidate budget and split I/O."""
import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.deformnext_retargeting import historical_q2c_training as h
from experiments.deformnext_retargeting.reproduce_q2c import checked, record_from_npz


def test_copied_task_sampler_and_optimizer_methods_match_historical_ast():
    directory = Path(h.__file__).parent
    authority = json.loads((directory / 'historical_q2c_authority.json').read_text())
    nodes = {n.name: n for n in ast.parse(Path(h.__file__).read_text()).body if hasattr(n, 'name')}
    for name, origin in authority['function_origins'].items():
        assert hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest() == origin['AST_sha256']
    methods = {n.name: n for n in nodes['R9TrainingRunner'].body if isinstance(n, ast.FunctionDef)}
    assert set(methods) == set(authority['runner_method_origins'])
    for name, origin in authority['runner_method_origins'].items():
        assert hashlib.sha256(ast.dump(methods[name], include_attributes=False).encode()).hexdigest() == origin['AST_sha256']


def record():
    generator = np.random.default_rng(2)
    source = generator.normal(size=(64, 1024)).astype(np.float32)
    query = generator.normal(size=(3, 1024)).astype(np.float32)
    labels = generator.uniform(size=(16, 64, 3)).astype(np.float32)
    ids = np.array([2, 4, 6])
    return {'query': {12: query}, 'source': {12: source},
            'candidates': np.tile(np.arange(64), (3, 1)),
            'raw_scores': generator.normal(0, .01, size=(3, 64)).astype(np.float32),
            'label_rgb': labels, 'query_vertex_indices': ids, 'target_rgb': labels[:, ids]}


def runner(arm, data):
    info = h.TrainingTuple(wave='exp4', arm=arm, decoder_arm=arm)
    return h.R9TrainingRunner(h._normalize_records({'asset': data}), temperature=.005,
                             device='cpu', tuple_info=info, upstream_state={})


def test_d0_d1_share_identical_head_candidates_sampling_and_initial_scores():
    data = record()
    d0 = runner('D0-PAIRWISE', data)
    d1 = runner('D1-Q2C', data)
    for key, expected in d0.model.fusion_head.state_dict().items():
        assert torch.equal(expected, d1.model.fusion_head.state_dict()[key])
    a, b = d0.next_batch(), d1.next_batch()
    assert a.sampled_query_sha256 == b.sampled_query_sha256
    assert torch.equal(a.candidates, b.candidates)
    assert torch.equal(a.label_rgb, b.label_rgb)
    sa, sb = d0._forward(a), d1._forward(b)
    assert torch.equal(sa['scores'], sb['scores'])
    assert torch.equal(sa['scores'], a.raw_scores)
    d1._check_zero(b, sb)
    assert d1.zero_start_proof['decoder_equals_d0']
    assert len(d0.model.decoder.blocks) == 0
    assert len(d1.model.decoder.blocks) == 1


def test_rgb_labels_affect_loss_but_never_descriptor_scorer_inputs():
    data = record()
    model = runner('D1-Q2C', data)
    batch = model.next_batch()
    scores = model._forward(batch)['scores']
    changed = replace(batch, label_rgb=batch.label_rgb + .5, target_rgb=batch.target_rgb * 0)
    assert torch.equal(scores, model._forward(changed)['scores'])
    prediction = h.fuse_label_values(scores, batch.label_rgb, temperature=.005)
    actual = torch.einsum('qk,cqkd->cqd', torch.softmax(scores / .005, -1), batch.label_rgb)
    assert torch.equal(prediction, actual)
    loss = (prediction - batch.target_rgb).abs().mean()
    loss.backward()
    assert model.model.decoder.blocks[0].attention_gate.grad is not None
    assert torch.isfinite(model.model.decoder.blocks[0].attention_gate.grad)


def test_replay_loader_rejects_changed_files_and_inconsistent_rgb_targets(tmp_path):
    path = tmp_path / 'record.npz'
    data = record()
    np.savez(path, query_L12=data['query'][12], source_L12=data['source'][12],
             **{k: v for k, v in data.items() if k not in ['query', 'source']})
    ref = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    loaded = record_from_npz(checked(ref))
    h._normalize_records({'asset': loaded})
    loaded['target_rgb'] = loaded['target_rgb'] + .1
    with pytest.raises(ValueError, match='target RGB'):
        h._normalize_records({'asset': loaded})
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='checksum'):
        checked(ref)


def test_visibility_matched_wrong_map_has_no_active_group_fixed_points():
    visible = np.array([True] * 5 + [False] * 7)
    wrong = h.wrong_indices(visible, cue_seed=7, lock_seed=23)
    assert np.all(wrong != np.arange(len(visible)))
    np.testing.assert_array_equal(visible[wrong], visible)
    np.testing.assert_array_equal(np.sort(wrong), np.arange(len(visible)))
