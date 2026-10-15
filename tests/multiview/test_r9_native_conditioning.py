"""Regression contracts for the promoted matcher and native-preserving fusion."""
import ast
import hashlib
import pytest
import torch
from pixal3d.modules import r9_matching as r9
from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin


def test_selected_scorer_definitions_remain_verbatim_authority():
    from pathlib import Path
    tree = ast.parse(Path(r9.__file__).read_text())
    hashes = {}
    for node in tree.body:
        name = getattr(node, 'name', None)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in r9.VERBATIM_AST_SHA256:
            hashes[name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    assert hashes == r9.VERBATIM_AST_SHA256


def test_promoted_count_init_and_nonzero_learning_signal():
    torch.manual_seed(7)
    scorer = r9.PromotedR9Matcher()
    assert sum(p.numel() for p in scorer.parameters()) == 928453
    q = torch.randn(2, 1024)
    k = torch.randn(2, 64, 1024)
    raw = torch.randn(2, 64)
    out = scorer.score_candidates(q, k, raw)
    torch.testing.assert_close(out, raw, rtol=0, atol=0)
    (out[:, 0] - out[:, 1]).sum().backward()
    for p in (scorer.decoder.blocks[0].attention_gate,
              scorer.decoder.blocks[0].ffn_gate, scorer.fusion_head.output.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_query_only_warmup_pruning_preserves_loss_and_parameter_gradients():
    from experiments.deformnext_retargeting.train_q2c import topology_listwise_loss
    import copy
    torch.manual_seed(31)
    full = r9.PromotedR9Matcher()
    # Exercise the trained contextual path as well as its zero initialization.
    with torch.no_grad():
        full.decoder.blocks[0].attention_gate.fill_(.2)
        full.decoder.blocks[0].ffn_gate.fill_(.1)
        full.fusion_head.output.weight.normal_(0, .02)
    pruned = copy.deepcopy(full)
    q, k, raw = torch.randn(6, 1024), torch.randn(6, 64, 1024), torch.randn(6, 64)
    indices = torch.arange(64).expand(6, -1)
    valid = torch.ones(6, 64, dtype=torch.bool)
    positives = torch.tensor([[3], [999], [5], [999], [7], [999]])
    visible = torch.tensor([[True], [True], [True], [False], [True], [False]])
    eligible = ((indices[..., None] == positives[:, None]) & visible[:, None]).any(-1).any(-1)
    loss, counts = topology_listwise_loss(full.score_candidates(q, k, raw), indices, valid, positives, visible)
    selected_loss, _ = topology_listwise_loss(pruned.score_candidates(q[eligible], k[eligible], raw[eligible]),
        indices[eligible], valid[eligible], positives[eligible], visible[eligible])
    torch.testing.assert_close(loss, selected_loss)
    assert counts['visible_queries'] == 4 and counts['proposal_misses'] == 1 and counts['null_queries'] == 2
    loss.backward(); selected_loss.backward()
    for (name, p), (other_name, other) in zip(full.named_parameters(), pruned.named_parameters()):
        assert name == other_name
        if p.grad is None:
            assert other.grad is None
        else:
            torch.testing.assert_close(p.grad, other.grad, atol=2e-6, rtol=2e-4)


def make(arm, width=6):
    return DeformationAwareConditioner(arm=arm, key_dim=1024, value_dim=width,
        query_dim=4, topk=64, **({'deformation_dim': 2048} if arm == 'c2' else {}))


def data(width=6):
    torch.manual_seed(91)
    indices = torch.arange(64).reshape(1, 1, 64).expand(1, 3, -1)
    return dict(source_values=torch.randn(1, 3, 24, width),
                source_valid=torch.ones(1, 3, 24, dtype=torch.bool),
                semantic_indices=indices, matching_logits=torch.randn(1, 3, 64),
                target_valid=torch.tensor([[True, True, False]]),
                patch_grid=torch.tensor([[[[0, 1], [2, 0]], [[1, 2], [0, 1]]]]))


@pytest.mark.parametrize('arm', ['b1', 'b2', 'b3', 'c1', 'c2'])
@pytest.mark.parametrize('sparse', [False, True])
def test_all_arms_exact_native_start_and_first_step_gradient(arm, sparse):
    bridge = object.__new__(ImageConditionedProjMixin)
    bridge.deform_conditioner = model = make(arm)
    bridge.training_models = {}
    pack = data()
    if arm in ('c1', 'c2'): pack['target_query'] = torch.randn(1, 3, 4)
    if arm == 'c2': pack['deformation'] = torch.randn(1, 3, 2048)
    coords = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 0]]) if sparse else None
    native = torch.randn(2, 6) if sparse else torch.randn(1, 8, 6)
    anchor = torch.randn_like(native)
    global_features = torch.randn(1, 1, 4)
    def forward():
        return bridge._inject_deform_transport(
            {'proj': native.clone(), 'global': global_features}, pack, coords,
            anchor_condition={'proj': anchor})
    result = forward()
    torch.testing.assert_close(result['proj'], native, rtol=0, atol=0)
    assert result['global'] is global_features
    result['proj'].square().sum().backward()
    assert model.fusion_gate.grad.abs() > 0
    if arm in ('c1', 'c2'):
        assert model.retargeter.net[-1].weight.grad.abs().sum() > 0
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.grad is not None], lr=0.01)
    optimizer.step()
    learned = forward()['proj']
    assert (learned - native).abs().sum() > 0
    # Query 2 has no observable input geometry and must retain native values,
    # including after gates and adapter have become nonzero.
    if not sparse:
        invalid = pack['patch_grid'].flatten(1) == 2
        torch.testing.assert_close(learned[invalid], native[invalid], rtol=0, atol=0)


def test_pbr_retains_native_hr_when_semantic_gate_opens():
    bridge = object.__new__(ImageConditionedProjMixin)
    bridge.deform_conditioner = make('b3', 2048)
    bridge.training_models = {}
    bridge.deform_conditioner.fusion_gate.data.fill_(1)
    native = torch.randn(1, 8, 2048)
    out = bridge._inject_deform_transport({'proj': native.clone()}, data(2048), None,
                                         anchor_condition={'proj': torch.randn_like(native)})['proj']
    torch.testing.assert_close(out[..., 1024:], native[..., 1024:], rtol=0, atol=0)
    assert (out[..., :1024] - native[..., :1024]).abs().sum() > 0


def test_score_aggregation_shift_invariance_and_empty_fallback():
    model = make('b3')
    torch.manual_seed(3)
    logits = torch.randn(2, 5, 64)
    values = torch.randn(2, 5, 64, 6)
    views = torch.randint(0, 3, (2, 5, 64))
    valid = torch.rand(2, 5, 64) > .2
    valid[0, 0] = False
    expected = model._attention(logits, values, valid, views, 3)
    for shift in (-100., 10., 100.):
        out = model._attention(logits + shift, values, valid, views, 3)
        for a, b in zip(out, expected):
            assert torch.isfinite(a).all()
            # Float32 addition at +/-100 loses ~1e-5 in the input; tau=.005
            # amplifies that unavoidable quantization. This still rejects
            # clamping logits before softmax (which destroys shift invariance).
            torch.testing.assert_close(a, b, rtol=1e-3, atol=2e-4)
    assert expected[0][0, 0].count_nonzero() == 0


def test_legacy_checkpoint_cannot_load_as_promoted(tmp_path):
    path = tmp_path / 'old.pt'
    torch.save({'q2c': {}}, path)
    with pytest.raises(ValueError, match='legacy pairwise'):
        DeformationAwareConditioner(arm='b2', key_dim=1024, value_dim=6,
            query_dim=4, q2c_checkpoint=str(path), freeze_q2c=True)


def test_frame_extraction_preserves_raw_first_branch():
    x = torch.randn(4, 20, 2048)
    torch.testing.assert_close(r9.frame_raw(x), x[..., :1024], rtol=0, atol=0)
    with pytest.raises(ValueError, match='verified 2048-D'):
        r9.frame_raw(x[..., :1024])


def test_topology_loss_counts_visible_misses_and_nulls_separately():
    from experiments.deformnext_retargeting.train_q2c import topology_listwise_loss
    logits = torch.tensor([[1., 0.], [0., 1.], [3., 1.]], requires_grad=True)
    indices = torch.tensor([[2, 3], [4, 5], [6, 7]])
    positives = torch.tensor([[2], [9], [6]])
    visible = torch.tensor([[True], [True], [False]])
    loss, counts = topology_listwise_loss(logits, indices, torch.ones_like(indices).bool(), positives, visible)
    assert counts == dict(eligible_queries=1, top1_correct=1, raw_top1_correct=1,
                         visible_queries=2, proposal_misses=1, null_queries=1)
    shifted, _ = topology_listwise_loss(logits + 100, indices, torch.ones_like(indices).bool(), positives, visible)
    torch.testing.assert_close(loss, shifted, rtol=1e-4, atol=1e-5)
    loss.backward()
    assert logits.grad[0].abs().sum() > 0
    assert logits.grad[1:].count_nonzero() == 0
    empty, counts = topology_listwise_loss(logits, indices, torch.zeros_like(indices).bool(), positives, visible)
    assert torch.isfinite(empty) and empty == 0
    assert counts['eligible_queries'] == 0


def test_alpha_cutout_teacher_does_not_confuse_transparency_with_camera_shift():
    from experiments.deformnext_retargeting.topology_labels import mask_agreement
    geometry = torch.zeros(20, 20, dtype=torch.bool); geometry[4:16, 4:16] = True
    observed = geometry.clone(); observed[5:15, 5:15] = False
    iou, coverage = mask_agreement(geometry, observed, alpha_cutouts=True)
    assert iou < .5 and coverage == 1
    with pytest.raises(ValueError, match='disagrees'):
        mask_agreement(geometry, torch.roll(observed, 8, 0), alpha_cutouts=True)


def test_c2_raw_context_remains_2048_while_matching_is_frame_1024():
    model = make('c2')
    q, source = torch.randn(1, 3, 1024), torch.randn(1, 3, 24, 1024)
    out = model(target_keys=q, source_keys=source,
                source_values=torch.randn(1, 3, 24, 6), source_valid=torch.ones(1, 3, 24).bool(),
                target_valid=torch.ones(1, 3).bool(), target_query=torch.randn(1, 3, 4),
                target_context=torch.randn(1, 3, 2048), source_context=torch.randn(1, 3, 24, 2048))
    assert model.retargeter.input_dim == 6 + 4 + 2048
    assert out.retarget_residual.count_nonzero() == 0
    assert make('c1').retargeter.input_dim == 6 + 4


def test_frame_proposals_are_identical_inside_amp_and_across_batch_sizes():
    torch.manual_seed(315)
    query, source = torch.randn(2, 5, 1024), torch.randn(2, 3, 24, 1024)
    valid = torch.ones(2, 3, 24, dtype=torch.bool)
    valid[:, :, 0] = False
    reference = r9.frame_topk(query, source, valid)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        amp = r9.frame_topk(query, source, valid)
    for a, b in zip(reference, amp): torch.testing.assert_close(a, b, rtol=0, atol=0)
    for i in range(2):
        single = r9.frame_topk(query[i:i+1], source[i:i+1], valid[i:i+1])
        for a, b in zip(reference, single): torch.testing.assert_close(a[i:i+1], b, rtol=0, atol=0)


def test_learned_d1_scoring_keeps_fp32_contract_inside_amp():
    torch.manual_seed(11)
    module = make('b3')
    with torch.no_grad():
        module.q2c.decoder.blocks[0].attention_gate.fill_(.3)
        module.q2c.decoder.blocks[0].ffn_gate.fill_(.1)
        module.q2c.fusion_head.output.weight.fill_(.003)
    q, k, raw = torch.randn(1, 3, 1024).half(), torch.randn(1, 3, 64, 1024).half(), torch.randn(1, 3, 64)
    expected = module._q2c_logits(q, k, raw)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = module._q2c_logits(q, k, raw)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('arm', ['b1', 'b2', 'b3', 'c1', 'c2'])
def test_insufficient_source_support_retains_native_after_gate_opens(arm):
    bridge = object.__new__(ImageConditionedProjMixin)
    bridge.deform_conditioner = model = make(arm); bridge.training_models = {}
    model.fusion_gate.data.fill_(1)
    pack = data(); pack['source_valid'].flatten()[35:] = False
    if arm in ('c1', 'c2'):
        pack['target_query'] = torch.randn(1, 3, 4)
        model.retargeter.net[-1].bias.data.fill_(1)
    if arm == 'c2': pack['deformation'] = torch.randn(1, 3, 2048)
    native = torch.randn(1, 8, 6)
    out = bridge._inject_deform_transport({'proj': native.clone()}, pack, None,
        anchor_condition={'proj': torch.randn_like(native)})['proj']
    torch.testing.assert_close(out, native, rtol=0, atol=0)
    q = torch.randn(1, 3, 1024); source = torch.randn(1, 3, 24, 1024)
    scores, _ = r9.frame_topk(q, source, pack['source_valid'], topk=64)
    assert not scores.isfinite().any()


def test_score_attention_inherits_r8_train_locked_temperature():
    valid=torch.ones(1,2,dtype=torch.bool);logits=torch.tensor([[.5,.505]])
    assert r9.MATCHING_TEMPERATURE==.005
    weights=r9.categorical_weights(logits,valid)
    torch.testing.assert_close(weights,torch.softmax((logits-logits.max())/.005,-1))
    assert weights[0,1]>.7
    assert r9.categorical_weights(logits,~valid).count_nonzero()==0
