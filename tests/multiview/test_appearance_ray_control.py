import torch

from experiments.deformnext_retargeting.appearance_ray_control import appearance_path_off
from pixal3d.modules.direct_ray_retargeting import DirectRayRetargetingConditioner


def test_only_appearance_is_muted_and_same_probability_raw_query_projection_survive():
    torch.manual_seed(621)
    model = DirectRayRetargetingConditioner(arm='c2', key_dim=1024, value_dim=1024,
        query_dim=1024, topk=64, implementation='r9-native-v5', candidate_aggregation='top1',
        projection_mode='nearest', score_temperature=.05, deformation_dim=2048, feature_chunk_size=1)
    torch.nn.init.normal_(model.ray_readout.output.weight, std=.01)
    torch.nn.init.normal_(model.retargeter.deformation_projection.weight, std=.01)
    source, target = torch.randn(1, 3, 1024, 2048), torch.randn(1, 1, 2048)
    pack = dict(source_values=torch.randn(1, 3, 1024, 1024), source_valid=torch.ones(1, 3, 1024, dtype=torch.bool),
        target_query=torch.randn(1, 1, 1024), semantic_indices=torch.arange(64).reshape(1, 1, 64),
        matching_logits=torch.randn(1, 1, 64), candidate_valid=torch.ones(1, 1, 64, dtype=torch.bool),
        target_valid=torch.ones(1, 1, dtype=torch.bool), target_keys=target[..., :1024],
        source_keys=source[..., :1024], target_context=target, source_context=source)
    captures = []
    handle = model.retargeter.deformation_projection.register_forward_hook(
        lambda module, args, output: captures.append((args[0].clone(), output.clone())))
    probabilities = []
    p_handle = model.ray_readout.register_forward_hook(
        lambda module, args, output: probabilities.append(output[1].clone()))
    state = {key: value.clone() for key, value in model.state_dict().items()}
    original = model.ray_readout.forward
    with torch.no_grad():
        model(**pack)
        before = model.last_feature_attention_stats['direct_retargeted_feature_RMS']
        events = []
        with appearance_path_off(model, events):
            model(**pack)
            assert model.last_feature_attention_stats['direct_retargeted_feature_RMS'] == 0
        assert before > 0
        assert torch.equal(captures[0][0], captures[1][0])
        assert torch.equal(captures[0][1], captures[1][1]) and captures[1][1].abs().sum() > 0
        assert torch.equal(probabilities[0], probabilities[1])
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    assert model.ray_readout.forward == original
    assert any(row.get('DQ_projection_retained') for row in events)
    handle.remove()
    p_handle.remove()
