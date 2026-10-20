import sys

import pytest

from experiments.deformnext_retargeting import member_posterior_runtime as runtime


def test_process_local_constructor_registration_is_restored_after_failed_phase(monkeypatch):
    from pixal3d.trainers.flow_matching import native_prior_source_ray as original
    from pixal3d.trainers.flow_matching.member_posterior_source_ray import MemberPosteriorSourceRayImageConditionedProjFlowMatchingCFGTrainer as Member
    from experiments.deformnext_retargeting import train_native_prior_source_ray as training
    from experiments.deformnext_retargeting import probe_native_prior_source_ray as proof
    previous=(original.NativePriorSourceRayImageConditionedProjFlowMatchingCFGTrainer,training.NAME,training.validate_native_prior_recipe)
    class ExpectedFailure(Exception):pass
    def fail():
        assert original.NativePriorSourceRayImageConditionedProjFlowMatchingCFGTrainer is Member
        assert training.NAME==runtime.NAME and training.validate_native_prior_recipe is runtime.validate_member_recipe
        assert '--phase' not in sys.argv
        raise ExpectedFailure()
    monkeypatch.setattr(proof,'main',fail)
    monkeypatch.setattr(sys,'argv',['member_posterior_runtime','--phase','native','--config','unused.json'])
    argv=sys.argv
    with pytest.raises(ExpectedFailure):runtime.main()
    assert sys.argv is argv
    assert (original.NativePriorSourceRayImageConditionedProjFlowMatchingCFGTrainer,training.NAME,training.validate_native_prior_recipe)==previous
