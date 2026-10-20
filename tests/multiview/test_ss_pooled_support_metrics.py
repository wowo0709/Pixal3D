import hashlib
import numpy as np

from experiments.deformnext_retargeting.ss_pooled_support_metrics import pooled_support_metrics,artifact_support_metrics
from experiments.deformnext_retargeting.cascade_artifact_inputs import native_shape_support


def test_within_cell_SS64_change_can_leave_actual_shape32_input_exact():
    first=np.zeros((64,64,64),dtype=bool);second=np.zeros_like(first);truth=np.zeros_like(first)
    first[2,4,6]=True;second[3,5,7]=True;truth[3,5,7]=True
    a=pooled_support_metrics(first,truth);b=pooled_support_metrics(second,truth)
    assert not np.array_equal(first,second) and a['voxel_iou']==b['voxel_iou']==1.
    assert a['actual_native_SS32_coordinate_SHA256']==b['actual_native_SS32_coordinate_SHA256']
    assert a['actual_native_SS32_coordinate_SHA256']==hashlib.sha256(native_shape_support(first).numpy().tobytes()).hexdigest()


def test_archive_diagnostic_uses_actual_prediction_and_decoded_GT_bytes(tmp_path):
    pred=np.zeros((64,64,64),dtype=bool);gt=np.zeros_like(pred)
    pred[63,63,63]=True;pred[2,4,6]=True;gt[2,4,6]=True
    path=tmp_path/'actual-occupancy.npz'
    np.savez(path,prediction_occupancy_packed=np.packbits(pred),target_occupancy_packed=np.packbits(gt),occupancy_shape=np.array([64,64,64]))
    result=artifact_support_metrics(path)
    assert (result['tp'],result['fp'],result['fn'])==(1,1,0) and result['voxel_iou']==.5
    assert result['native_shape512_SS32_tokens']==2 and result['GT_support_is_for_scoring_only']


def test_empty_prediction_is_recorded_without_a_GT_or_prior_replacement():
    empty=np.zeros((64,64,64),dtype=bool);gt=empty.copy();gt[4,8,2]=True
    result=pooled_support_metrics(empty,gt)
    assert result['native_shape512_SS32_tokens']==0 and not result['nonempty_generation_input']
    assert result['voxel_iou']==0 and (result['tp'],result['fp'],result['fn'])==(0,0,1)
