import pytest
import torch

from experiments.deformnext_retargeting.r9_diagnostic_context import r9_diagnostic_context
from pixal3d.modules.deform_conditioner import DeformationAwareConditioner


def example():
    torch.manual_seed(5)
    model = DeformationAwareConditioner(arm='c1', key_dim=1024, value_dim=4, query_dim=4, topk=64, hidden_dim=8)
    with torch.no_grad():
        model.fusion_gate.fill_(.2)
        model.retargeter.net[-1].bias.fill_(.1)
    valid = torch.ones(2, 3, 32, dtype=torch.bool)
    valid[0, :, -2:] = False
    valid[1, :, -1:] = False
    values = torch.randn(2, 3, 32, 4)
    values[~valid] = 0
    data = dict(source_values=values, source_valid=valid,
        semantic_indices=torch.arange(64).repeat(2, 5, 1),
        matching_logits=torch.randn(2, 5, 64), target_valid=torch.ones(2, 5, dtype=torch.bool),
        target_query=torch.randn(2, 5, 4))
    return model, data


def test_r9_wrong_object_changes_values_only_keeps_scores_support_and_restores_after_error():
    model, data = example()
    originals = {key: value.clone() for key, value in data.items()}
    parameters = {key: value.clone() for key, value in model.state_dict().items()}
    normal, events = model(**data), []
    captured = []
    with pytest.raises(RuntimeError, match='sampling failed'):
        with r9_diagnostic_context(model, 'wrong-object', events):
            hook = model.register_forward_pre_hook(lambda m, args, kw: captured.append(kw), with_kwargs=True)
            try:
                changed = model(**data)
            finally:
                hook.remove()
            for key in data:
                if key != 'source_values':
                    assert captured[0][key] is data[key]
            assert captured[0]['source_values'][~data['source_valid']].eq(0).all()
            assert not torch.equal(changed.evidence, normal.evidence)
            torch.testing.assert_close(changed.matching_logits, normal.matching_logits)
            torch.testing.assert_close(changed.query_valid, normal.query_valid)
            torch.testing.assert_close(changed.fusion_scale, normal.fusion_scale)
            raise RuntimeError('sampling failed')
    assert events[0]['zero_donor_vectors'] == 0
    torch.testing.assert_close(model(**data).evidence, normal.evidence)
    for key, value in originals.items():
        torch.testing.assert_close(data[key], value)
    for key, value in parameters.items():
        torch.testing.assert_close(model.state_dict()[key], value)


def test_r9_adapter_off_and_full_path_off_have_distinct_contracts():
    model, data = example()
    normal = model(**data)
    with r9_diagnostic_context(model, 'adapter-off', []):
        adapter_off = model(**data)
    assert adapter_off.retarget_residual.eq(0).all()
    torch.testing.assert_close(adapter_off.query_valid, normal.query_valid)
    torch.testing.assert_close(adapter_off.matching_logits, normal.matching_logits)
    assert not torch.equal(adapter_off.evidence, normal.evidence)
    with r9_diagnostic_context(model, 'source-off', []):
        source_off = model(**data)
    assert not source_off.query_valid.any() and source_off.evidence.eq(0).all()
    torch.testing.assert_close(source_off.fusion_scale, normal.fusion_scale)


def test_r9_wrong_object_rejects_missing_camera_donors_without_leaving_hook():
    model, data = example()
    data['source_valid'][1, 0] = False
    with pytest.raises(ValueError, match='donor camera'):
        with r9_diagnostic_context(model, 'wrong-object', []):
            model(**data)
    assert not model._forward_pre_hooks
    with pytest.raises(ValueError, match='original r9'):
        with r9_diagnostic_context(torch.nn.Identity(), 'source-off', []):
            pass


def test_r9_wrong_address_preserves_each_joint_camera_distribution_and_scores():
    model,data=example();original={key:value.clone() for key,value in data.items()}
    normal=model(**data);events=[];captured=[]
    with r9_diagnostic_context(model,'wrong-address',events):
        hook=model.register_forward_pre_hook(lambda m,args,kw:captured.append(kw),with_kwargs=True)
        try:changed=model(**data)
        finally:hook.remove()
    swapped=captured[0]['source_values'];valid=data['source_valid']
    torch.testing.assert_close(swapped[~valid],data['source_values'][~valid])
    for sample in range(2):
        for view in range(3):
            before=data['source_values'][sample,view,valid[sample,view]]
            after=swapped[sample,view,valid[sample,view]]
            unique_before,count_before=torch.unique(before,dim=0,return_counts=True)
            unique_after,count_after=torch.unique(after,dim=0,return_counts=True)
            torch.testing.assert_close(unique_before,unique_after)
            torch.testing.assert_close(count_before,count_after)
            assert (before!=after).any(-1).all()
    torch.testing.assert_close(changed.matching_logits,normal.matching_logits)
    torch.testing.assert_close(changed.query_valid,normal.query_valid)
    assert not torch.equal(changed.evidence,normal.evidence)
    assert events[0]['changed_source_address_sites']==int(valid.sum())
    assert not model._forward_pre_hooks
    for key,value in original.items():torch.testing.assert_close(data[key],value)
    torch.testing.assert_close(model(**data).evidence,normal.evidence)
