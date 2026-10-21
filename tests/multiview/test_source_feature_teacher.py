from types import SimpleNamespace

import pytest
import torch

from experiments.deformnext_retargeting.source_feature_teacher import (
    source_feature_teacher_loss, validate_teacher_records,
)


def tensors():
    source = torch.zeros(1, 3, 1024, 1024)
    correction = torch.zeros_like(source, requires_grad=True)
    return source, correction, SimpleNamespace(source_numerator=correction,
        semantic_weight=torch.zeros(1, 3, 1024, 1))


def test_loss_has_the_correct_direction_full_patch_weight_and_no_camera11_label():
    source, correction, out = tensors()
    teacher = torch.full((1, 2, 1024, 1024), .1)
    loss = source_feature_teacher_loss(out, source, teacher, split='train', camera_ids=[9, 10])
    assert loss.item() == pytest.approx(.01)
    loss.backward()
    assert (correction.grad[:, :2] < 0).all()
    assert not correction.grad[:, 2].any()
    with torch.no_grad():
        correction[:, :2].copy_(teacher)
        correction[:, 2].fill_(100)
    assert source_feature_teacher_loss(out, source, teacher, split='train', camera_ids=[9, 10]) == 0


@pytest.mark.parametrize('split,cameras', [('validation', [9, 10]), ('test', [9, 10]), ('train', [8, 9])])
def test_rejects_validation_or_wrong_camera_supervision(split, cameras):
    source, _, out = tensors()
    with pytest.raises(ValueError, match='TRAIN M1 RGB'):
        source_feature_teacher_loss(out, source, source[:, :2], split=split, camera_ids=cameras)


def test_rejects_train_val_source_overlap_and_validation_replay_labels():
    row = dict(sample_id='a', source_cluster_id='source_a', split='train',
        conditioning_camera_ids=[8, 9, 10, 11], evaluation_target_images=['nine', 'ten'])
    assert validate_teacher_records([row], {'heldout'}) == ({'a'}, {'source_a'})
    for rows in [[row, row], [dict(row, source_cluster_id='heldout')],
                 [dict(row, not_for_training=True)], [dict(row, split='validation')]]:
        with pytest.raises(ValueError, match='unique TRAIN'):
            validate_teacher_records(rows, {'heldout'})
