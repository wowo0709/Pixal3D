import pytest

from experiments.deformnext_retargeting.summarize_semantic_epoch import aggregate,validate_epoch


def row():
    rows=[{'sample_id':str(i),'tp':2,'fp':1,'fn':2,'target_voxels_sha256':'a'*64,
        'target_voxel_shape':[64,64,64]} for i in range(32)]
    return {'step':192,'weights':'raw','quick_panel_ids':[str(i) for i in range(32)],'quick_per_asset':rows}


def test_same_case_aggregation_remains_micro_counts_not_mean_of_logged_case_metrics():
    r=row();r['quick_per_asset'][0].update(tp=100,fp=0,fn=0)
    result=aggregate(validate_epoch(r,192));assert result['micro_voxel_iou']==pytest.approx(162/255)


@pytest.mark.parametrize('field,value',[('step',96),('weights','ema')])
def test_previous_epoch_or_ema_cannot_be_mislabeled_current_raw(field,value):
    r=row();r[field]=value
    with pytest.raises(ValueError,match='raw epoch'):validate_epoch(r,192)


def test_same_set_in_different_order_is_rejected():
    r=row();r['quick_per_asset'].reverse()
    with pytest.raises(ValueError,match='ordered'):validate_epoch(r,192)


def test_same32_must_have_actual_64cube_target_geometry_hashes():
    r=row();r['quick_per_asset'][0]['target_voxel_shape']=[32,32,32]
    with pytest.raises(ValueError,match='hashes'):validate_epoch(r,192)
