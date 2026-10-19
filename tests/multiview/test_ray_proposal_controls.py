from types import SimpleNamespace

import pytest
import torch

from experiments.deformnext_retargeting.ray_proposal_controls import (
    constituent_patch_union, ray_proposal_control, runtime,
    RayGroupImageConditionedProjFlowMatchingCFGTrainer as Trainer,
)


def test_source_disabled_native_support_never_manufactures_candidates():
    ids, valid = constituent_patch_union(target_keys=torch.zeros(1, 1024, 1024),
        source_keys=torch.zeros(1, 3, 1024, 1024), source_valid=torch.zeros(1, 3, 1024, dtype=torch.bool),
        ray_ids=torch.zeros(1, 3, 2, dtype=torch.long), patch_ids=torch.zeros(1, 2, dtype=torch.long),
        query_valid=torch.zeros(1, 3072, dtype=torch.bool))
    assert ids.shape == (1, 3072, 64) and not valid.any() and not ids.any()


def test_teacher_rejected_and_all_runtime_functions_restore_after_exception():
    original = (runtime.RayGroupConditioner.forward, runtime.semantic_ray_proposals, Trainer._inject_deform_transport)
    with pytest.raises(ValueError, match='teacher'):
        with ray_proposal_control('patch-union', []):
            Trainer._inject_deform_transport(SimpleNamespace(), {}, {'positive_indices': torch.ones(1)}, None)
    assert original == (runtime.RayGroupConditioner.forward, runtime.semantic_ray_proposals, Trainer._inject_deform_transport)


def test_unknown_policy_is_not_silently_accepted():
    with pytest.raises(ValueError, match='explicit'):
        with ray_proposal_control('GT-best', []):
            pass
