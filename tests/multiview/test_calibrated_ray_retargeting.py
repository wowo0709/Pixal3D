import pytest
import torch

from pixal3d.modules.calibrated_ray_retargeting import (
    CalibratedDirectRayNoQueryConditioner, CalibratedDirectRayRetargetingConditioner,
    file_sha, install_calibrated_matching, load_calibrated_matching)
from pixal3d.modules.direct_ray_retargeting_no_query import DirectRayRetargetingNoQueryConditioner
from pixal3d.modules.ray_feature_attention import RayFeatureReadout


def spec(tmp_path):
    torch.manual_seed(531)
    head = RayFeatureReadout()
    torch.nn.init.normal_(head.score_selector.content_output.weight, std=.01)
    reference = {k: v.clone() for k, v in head.state_dict().items()}
    calibrated = {k: v.clone() for k, v in reference.items()}
    calibrated['query.weight'].fill_(.002)
    calibrated['key.weight'].add_(.001)
    cold, warm = tmp_path/'cold.pt', tmp_path/'warm.pt'
    torch.save({'ray_readout.'+k: v for k, v in reference.items()}, cold)
    torch.save(calibrated, warm)
    return dict(checkpoint=str(warm), checkpoint_SHA256=file_sha(warm),
                reference_conditioner=str(cold), reference_SHA256=file_sha(cold))


def args():
    return dict(arm='c1', key_dim=1024, value_dim=1024, query_dim=1024, topk=64,
        implementation='r9-native-v5', candidate_aggregation='top1', projection_mode='nearest',
        score_temperature=.05, feature_chunk_size=1)


def test_seven_matching_tensors_transfer_to_both_native_value_widths(tmp_path):
    matching = load_calibrated_matching(spec(tmp_path))
    for width in [1024, 2048]:
        head = RayFeatureReadout(width)
        before = {key: value.clone() for key, value in head.state_dict().items()}
        install_calibrated_matching(head, matching)
        assert torch.equal(head.value.weight, before['value.weight'])
        assert torch.equal(head.output.weight, before['output.weight'])
        assert all(torch.equal(head.state_dict()[key], value) for key, value in matching.items())


def test_initializer_rejects_wrong_sha_or_non_matching_tensor_changes(tmp_path):
    declared = spec(tmp_path)
    with pytest.raises(ValueError, match='SHA256'):
        load_calibrated_matching(dict(declared, checkpoint_SHA256='0'*64))
    state = torch.load(declared['checkpoint'], weights_only=True)
    state['value.weight'].add_(.01)
    torch.save(state, declared['checkpoint'])
    declared['checkpoint_SHA256'] = file_sha(declared['checkpoint'])
    with pytest.raises(ValueError, match='only Q/K'):
        load_calibrated_matching(declared)


def test_calibrated_forward_keeps_original_R_function_with_physical_query_choices(tmp_path):
    torch.manual_seed(532)
    declared = spec(tmp_path)
    native = DirectRayRetargetingNoQueryConditioner(**args())
    torch.nn.init.normal_(native.retargeter.net[-1].weight, std=.01)
    with torch.no_grad():
        native.retargeter.gate_logit.fill_(.2)
        native.fusion_gate.fill_(.1)
    no_query = CalibratedDirectRayNoQueryConditioner(**args(), calibrated_matching=declared)
    full = CalibratedDirectRayRetargetingConditioner(**dict(args(), arm='c2', deformation_dim=2048),
                                                  calibrated_matching=declared)
    for module in [no_query, full]:
        module.load_reference(native.reference_state())
        assert module.assert_calibrated_initializer()['matching_tensors_loaded'] == 7
    assert not hasattr(no_query.retargeter, 'deformation_projection')
    assert full.retargeter.input_dim == 4096 and no_query.retargeter.input_dim == 2048
    source = torch.randn(1, 3, 1024, 2048)
    target = torch.randn(1, 1, 2048)
    pack = dict(source_values=torch.randn(1, 3, 1024, 1024),
        source_valid=torch.ones(1, 3, 1024, dtype=torch.bool), target_query=torch.randn(1, 1, 1024),
        semantic_indices=torch.arange(64).reshape(1, 1, 64), matching_logits=torch.randn(1, 1, 64),
        candidate_valid=torch.ones(1, 1, 64, dtype=torch.bool), target_valid=torch.ones(1, 1, dtype=torch.bool),
        target_keys=target[..., :1024], source_keys=source[..., :1024])
    with torch.no_grad():
        baseline, absent = native(**pack), no_query(**pack)
        present = full(**dict(pack, target_context=target, source_context=source))
    assert torch.equal(baseline.evidence, absent.evidence)
    assert torch.equal(baseline.evidence, present.evidence)
    assert torch.equal(baseline.retarget_residual, present.retarget_residual)
    # Catch accidental utility reloads in a continuation constructor.
    with torch.no_grad():
        no_query.ray_readout.score_selector.log_content_scale.add_(.01)
    with pytest.raises(ValueError, match='overwritten'):
        no_query.assert_calibrated_initializer()
