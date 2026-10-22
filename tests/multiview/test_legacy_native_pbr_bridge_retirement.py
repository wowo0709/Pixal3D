from pathlib import Path

import pytest

from experiments.deformnext_retargeting.legacy_native_pbr_bridge import CASE_STATUS
from experiments.deformnext_retargeting.legacy_native_pbr_bridge_retirement import validate_case_raw
from test_legacy_native_pbr_bridge import pair_fixture


def inputs(tmp_path):
    owned=tmp_path/'cases';root=owned/'spatial'/'asset';folder=root/'matched';folder.mkdir(parents=True)
    path=folder/'prediction.npz';path.write_bytes(b'closed raw')
    case=dict(status=CASE_STATUS,SS_variant='feature-spatial',sample_id='s',PBR_mode='vanilla',
        existing_geometry_reused_read_only=True,new_SS_or_Shape_inferences=0,
        original_Shape5124210_Shape102416850=True,paired_original_and_trained_texture_initial_noise_exact=True,
        paired_raw_geometry_and_native_predicted_LOD_identical=True,
        actual_prior_and_matched_three_camera_original_material_M1_shared_scores=True,
        original_generated_geometry_not_aligned_or_replaced=True,optimizer_updates=0,
        terminals=[dict(returncode=0,actual_session_drained=True)for _ in range(6)],
        original_geometry_root=str(tmp_path/'original'),matched_generation=str(folder),SS_prediction_SHA256='SS')
    generation=pair_fixture()[1]
    return path,dict(case_root=root,owned_root=owned,case=case,generation=generation,
        canonical_entry=dict(sha256='digest',bytes=path.stat().st_size),actual_SHA256='digest',readers=[])


def test_selects_only_case_owned_new_texture_duplicate(tmp_path):
    p,k=inputs(tmp_path);token=validate_case_raw(p,**k)
    assert token['local_duplicate_only'] and token['original_legacy_geometry_not_selected']
    assert p.is_file()  # Admission itself never deletes data.


@pytest.mark.parametrize('error',['original','reader','wrong-role','outside','hash','bytes','live-session','wrong-generation'])
def test_refuses_original_inputs_or_unverified_and_consumed_outputs(tmp_path,error):
    p,k=inputs(tmp_path)
    if error=='original':k['case']['original_geometry_root']=str(k['case_root'])
    elif error=='reader':k['readers']=['pending inference']
    elif error=='wrong-role':
        p=p.parent.parent/'geometry'/'prediction.npz';p.parent.mkdir();p.write_bytes(b'closed raw')
    elif error=='outside':k['owned_root']=tmp_path/'unrelated'
    elif error=='hash':k['actual_SHA256']='different'
    elif error=='bytes':k['canonical_entry']['bytes']+=1
    elif error=='live-session':k['case']['terminals'][0]['actual_session_drained']=False
    else:k['generation']['PBR_arm']='top1-only'
    with pytest.raises(ValueError):validate_case_raw(p,**k)
