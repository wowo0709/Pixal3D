import pytest

from experiments.deformnext_retargeting.summarize_semantic_topk_search import aggregate,validate_evaluation


def evaluation():
    return {'status':'PASS','stage':'ss64','step':480,'weights':'raw','sampling_steps':12,'guidance':3.,
        'generation_batch_size':8,'optimizer_updates_during_evaluation':0,'target_modality':'rgb',
        'conditioning_mode':'edited-rgb','diagnostic_only':False,'input_camera_ids':[8,9,10,11],
        'input_states':['M1','M0','M0','M0'],
        'per_asset':[{'sample_id':str(i),'target_voxels_sha256':'a'*64,'tp':2,'fp':1,'fn':3} for i in range(128)]}


@pytest.mark.parametrize('key,bad',[('step',384),('target_modality','depth'),('optimizer_updates_during_evaluation',1),('diagnostic_only',True)])
def test_summary_rejects_wrong_checkpoint_depth_or_training_during_eval(key,bad):
    e=evaluation();validate_evaluation(e);e[key]=bad
    with pytest.raises(ValueError,match='protocol'):validate_evaluation(e)


def test_summary_requires_camera08_and_actual_target_hashes():
    e=evaluation();e['input_camera_ids']=[9,9,10,11]
    with pytest.raises(ValueError,match='four-view'):validate_evaluation(e)
    e=evaluation();e['per_asset'][0].pop('target_voxels_sha256')
    with pytest.raises(ValueError,match='hashes'):validate_evaluation(e)


def test_micro_metric_uses_counts_instead_of_averaging_case_IoU():
    rows=[{'per_asset':[{'tp':2,'fp':0,'fn':0},{'tp':0,'fp':4,'fn':4}]}]
    m=aggregate(rows)
    assert m['micro_voxel_iou']==.2 and m['micro_voxel_f1']==1/3
