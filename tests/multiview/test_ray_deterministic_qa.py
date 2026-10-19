import pytest

from experiments.deformnext_retargeting.verify_ray_deterministic_conditioning_contract import validate_environment


def config():
    return {'trainer': {'name': 'RayGroupImageConditionedProjFlowMatchingCFGTrainer'}}


def environment():
    return {'CUDA_VISIBLE_DEVICES': '41', 'MATCHING_V2_ALLOWED_GPUS': '41,42',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}


def test_explicit_assigned_single_gpu_and_workspace_are_required():
    validate_environment(config(), environment())
    values = environment()
    values['CUDA_VISIBLE_DEVICES'] = '43'
    with pytest.raises(ValueError, match='assigned'):
        validate_environment(config(), values)


def test_a_visible_gpu_pair_cannot_silently_use_only_the_first_device():
    values = environment()
    values['CUDA_VISIBLE_DEVICES'] = '41,42'
    with pytest.raises(ValueError, match='one explicitly'):
        validate_environment(config(), values)


def test_missing_workspace_is_rejected_before_cuda_initialization():
    values = environment()
    values.pop('CUBLAS_WORKSPACE_CONFIG')
    with pytest.raises(ValueError, match='before CUDA'):
        validate_environment(config(), values)


def test_other_trainers_cannot_use_the_ray_qa_adapter():
    with pytest.raises(ValueError, match='ray-group'):
        validate_environment({'trainer': {'name': 'OtherTrainer'}}, environment())
