import numpy as np
import pytest
import torch

from experiments.deformnext_retargeting.evaluate_source_feature_teacher_replay import (
    moments, summarize, validate_validation_teacher,
)


def test_reporting_moments_distinguish_correct_opposite_and_absent_directions():
    truth = torch.tensor([[[1., 2.], [3., 4.]]]); mask = torch.ones(1, 2, dtype=torch.bool)
    right = summarize(moments(truth, truth, mask))
    wrong = summarize(moments(-truth, truth, mask))
    empty = summarize(moments(torch.zeros_like(truth), truth, mask))
    assert right['MSE_ratio'] == 0 and right['cosine'] == 1
    assert wrong['MSE_ratio'] == 4 and wrong['cosine'] == -1
    assert empty['MSE_ratio'] == 1 and empty['cosine'] is None
    masked = summarize(moments(truth, truth, torch.zeros_like(mask)))
    assert masked['elements'] == 0 and masked['corrected_MSE'] is None


def test_validation_reporting_rejects_train_labels_and_foreign_asset():
    row = dict(sample_id='validation_a', split='validation')
    label = dict(schema='diagnostic-validation32-source-projection-replay-v1', split='validation',
        not_for_training=True, sample_id='validation_a',
        teacher_source9_10_FP16=np.empty((2,1024,1024),dtype=np.float16),
        teacher_alpha9_10=np.empty((2,1024),dtype=bool))
    validate_validation_teacher(label, row)
    for patch in [dict(split='train'),dict(not_for_training=False),dict(sample_id='train_a')]:
        with pytest.raises(ValueError, match='read-only validation32'):
            validate_validation_teacher(dict(label,**patch), row)
