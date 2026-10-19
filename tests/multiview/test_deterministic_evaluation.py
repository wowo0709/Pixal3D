import hashlib

import pytest

from experiments.deformnext_retargeting.evaluate_deterministic_checkpoint import (
    validate_entry_path, validate_environment,
)


def test_evaluation_requires_one_assigned_gpu_and_the_verified_workspace():
    env = {'CUDA_VISIBLE_DEVICES':'41','MATCHING_V2_ALLOWED_GPUS':'41,42',
        'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    validate_environment(env)
    env['CUDA_VISIBLE_DEVICES']='43'
    with pytest.raises(ValueError,match='assigned'):
        validate_environment(env)


def test_visible_gpu_pair_cannot_be_silently_truncated():
    with pytest.raises(ValueError,match='one explicitly'):
        validate_environment({'CUDA_VISIBLE_DEVICES':'41,42','MATCHING_V2_ALLOWED_GPUS':'41,42',
            'CUBLAS_WORKSPACE_CONFIG':':4096:8'})


def test_missing_workspace_cannot_silently_fall_back():
    with pytest.raises(ValueError,match='before CUDA'):
        validate_environment({'CUDA_VISIBLE_DEVICES':'41','MATCHING_V2_ALLOWED_GPUS':'41'})


def test_evaluator_is_bound_to_both_checkout_path_and_source_bytes(tmp_path):
    path=tmp_path/'experiments/deformnext_retargeting/evaluate_native_checkpoint.py'
    path.parent.mkdir(parents=True)
    path.write_text('original evaluator\n')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    validate_entry_path(path,tmp_path,digest)
    with pytest.raises(ValueError,match='checkout'):
        validate_entry_path(path,tmp_path/'another',digest)
    path.write_text('changed evaluator\n')
    with pytest.raises(ValueError,match='source changed'):
        validate_entry_path(path,tmp_path,digest)
