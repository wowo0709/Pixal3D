import copy

import numpy as np
import pytest

from experiments.deformnext_retargeting.firstwave_wandb import validate_visual_sources,near


def fixture():
    rows=[{'sample_id':str(i),'source_cluster_id':str(i),'seed':i,'tp':2,'fp':1,'fn':2,
        'target_voxels_sha256':'a'*64,'target_voxel_shape':[64,64,64]} for i in range(128)]
    image=np.zeros((128*64,3*64,3),dtype=np.uint8)
    return {'one':rows,'two':copy.deepcopy(rows)},{'one':image,'two':image.copy()}


def test_existing_same_order_actual_gt_can_be_presented_without_new_inference():
    records,images=fixture();assert validate_visual_sources(records,images)==[str(i) for i in range(128)]


@pytest.mark.parametrize('field,value',[('seed',999),('source_cluster_id','different'),('target_voxels_sha256','b'*64),('target_voxel_shape',[32,32,32])])
def test_projection_similarity_cannot_hide_different_noise_source_or_3d_target(field,value):
    records,images=fixture();records['two'][0][field]=value
    with pytest.raises(ValueError,match='3D GT'):validate_visual_sources(records,images)


def test_matching_3d_hash_metadata_cannot_hide_different_actual_gt_pixels():
    records,images=fixture();images['two'][0,0,0]=1
    with pytest.raises(ValueError,match='pixels'):validate_visual_sources(records,images)


def test_shared_wrong_resolution_cannot_be_labelled_as_ss64():
    records,images=fixture()
    for rows in records.values():
        for row in rows:row['target_voxel_shape']=[32,32,32]
    with pytest.raises(ValueError,match='3D GT'):validate_visual_sources(records,images)


def test_changed_case_order_cannot_be_presented_as_same_cases():
    records,images=fixture();records['two'].reverse()
    with pytest.raises(ValueError,match='order'):validate_visual_sources(records,images)


def test_duplicate_cases_are_not_a_complete128_panel():
    records,images=fixture();records['one'][1]=records['one'][0]
    with pytest.raises(ValueError,match='distinct'):validate_visual_sources(records,images)


def test_remote_numeric_check_rejects_nonfinite_or_different_results():
    near(.4,.4)
    for value in [float('nan'),float('inf'),.40001]:
        with pytest.raises(ValueError):near(value,.4)
