from types import SimpleNamespace

import pytest
import torch

from experiments.deformnext_retargeting.probe_source_teacher_native_initialization import temporary_source_module
from experiments.deformnext_retargeting.source_teacher_initialization import CLASSES, source_input_binding
from pixal3d.modules.member_feature_source_ray import MemberFeatureSourceRayConditioner
from test_source_teacher_initialization import saved, KWARGS
from test_source_contrast_member_feature import fixture


@pytest.mark.parametrize('aliases',[False,True])
def test_readonly_reference_swap_restores_both_native_mappings_on_exception(aliases):
    original,reference=torch.nn.Linear(2,2),torch.nn.Linear(2,2)
    models={'deform_conditioner':original}
    trainer=SimpleNamespace(world_size=1,deform_conditioner=original,models=models,
        training_models=models if aliases else dict(models))
    optimizer=torch.optim.AdamW(original.parameters(),lr=1e-4)
    parameter_ids={id(p)for p in optimizer.param_groups[0]['params']}
    with pytest.raises(RuntimeError,match='reference failure'):
        with temporary_source_module(trainer,reference):
            assert trainer.deform_conditioner is reference
            assert trainer.models['deform_conditioner']is reference
            assert trainer.training_models['deform_conditioner']is reference
            assert parameter_ids=={id(p)for p in original.parameters()}
            raise RuntimeError('reference failure')
    assert trainer.deform_conditioner is original
    assert trainer.models['deform_conditioner']is original
    assert trainer.training_models['deform_conditioner']is original
    assert not optimizer.state


def test_refuses_swapping_wrapped_or_distributed_source_models():
    original,reference=object(),object()
    trainer=SimpleNamespace(world_size=2,deform_conditioner=original,
        models={'deform_conditioner':original},training_models={'deform_conditioner':original})
    with pytest.raises(ValueError,match='world1'):
        with temporary_source_module(trainer,reference):pass
    assert trainer.deform_conditioner is original


@pytest.mark.parametrize('variant',['original-NoDQ','source-contrast-NoDQ'])
def test_loaded_warm_factory_forward_matches_standalone_source_without_weight_changes(tmp_path,variant):
    old,contrast,data=fixture();reference=old if variant=='original-NoDQ'else contrast
    with torch.no_grad():
        for p in reference.parameters():
            if p.requires_grad:p.add_(.02)
    spec=saved(tmp_path,reference,variant);trainer_module=SimpleNamespace(MemberFeatureSourceRayConditioner=MemberFeatureSourceRayConditioner)
    with source_input_binding(trainer_module,variant=variant,teacher=spec):
        module=trainer_module.MemberFeatureSourceRayConditioner(**KWARGS)
        module.q2c.load_state_dict(reference.q2c.state_dict())
        module.load_utility({k:torch.zeros_like(v)for k,v in reference.ray_readout.score_selector.state_dict().items()})
    reference.eval();module.eval()
    with torch.no_grad():a,b=reference(**data),module(**data)
    for key in ['source_numerator','semantic_weight','use_probability']:
        assert torch.equal(getattr(a,key),getattr(b,key))
    assert torch.equal(a.attention.candidate_indices,b.attention.candidate_indices)
    assert all(torch.equal(v,reference.state_dict()[k])for k,v in module.state_dict().items())
