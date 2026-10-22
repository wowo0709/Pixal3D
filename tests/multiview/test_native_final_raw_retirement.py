from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.native_final_raw_retirement import validate_raw


def inputs(tmp_path,alias='R_top1_only'):
    root=tmp_path/'owned'/alias/'asset';path=root/'geometry/prediction.npz';path.parent.mkdir(parents=True);path.write_bytes(b'raw')
    zero=alias.startswith('zero-')
    case=dict(status='PASS_ACTUAL_FROZEN_NATIVE_SIX_MODEL_TEST508_CASE',actual_completed_native_case=True,alias=alias,
        sample_id='one',actual_SS_prediction_SHA256='SS',surface_reference_join=dict(status='PASS_ACTUAL_LOCAL_NATIVE_GEOMETRY_SHARED_GT_SURFACE_JOIN'),
        actual_shared_GT_material_pixel_identity_in_observations=True,scored_image_observations=[dict(camera_id=c)for c in [8,9,10]],
        terminals=[dict(returncode=0)]*(3 if zero else 4),raw_archive_roots=[str(path.parent)])
    generation=dict(status='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION',sample_id='one',
        SS_prediction_SHA256='SS',GT_geometry_or_coords_used_in_generation=False,GT_loaded_only_after_generation=True,fixed_shape_models=True)
    return path,dict(case_root=root,owned_root=tmp_path/'owned',case=case,generation=generation,
        canonical_entry=dict(sha256='actual',bytes=3),actual_SHA256='actual',readers=[])


@pytest.mark.parametrize('alias',['zero-source4','zero-edited4','vanilla','R_top1_only','R_feature','R_raw_topk'])
def test_only_case_owned_completed_canonical_raw_duplicates_can_retire(tmp_path,alias):
    path,kw=inputs(tmp_path,alias)
    token=validate_raw(path,**kw)
    assert token['local_duplicate_only']and token['bytes']==3 and path.is_file()


@pytest.mark.parametrize('damage',['reader','failed-phase','not-complete','wrong-GT','different-file4','source-checkpoint','outside-owned','different-SS'])
def test_readers_failed_producers_or_foreign_files_prevent_retirement(tmp_path,damage):
    path,kw=inputs(tmp_path)
    if damage=='reader':kw['readers']=['actual consumer']
    elif damage=='failed-phase':kw['case']['terminals'][0]={'returncode':1}
    elif damage=='not-complete':kw['case']['actual_completed_native_case']=False
    elif damage=='wrong-GT':kw['generation']['GT_geometry_or_coords_used_in_generation']=True
    elif damage=='different-file4':kw['canonical_entry']['sha256']='other'
    elif damage=='source-checkpoint':path=path.with_name('denoiser_step0001900.pt');path.write_bytes(b'raw')
    elif damage=='outside-owned':kw['owned_root']=tmp_path/'different'
    elif damage=='different-SS':kw['generation']['SS_prediction_SHA256']='other'
    with pytest.raises(ValueError):validate_raw(path,**kw)
