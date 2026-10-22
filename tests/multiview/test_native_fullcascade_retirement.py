import pytest

from experiments.deformnext_retargeting.native_fullcascade_retirement import validate_case_raw


@pytest.mark.parametrize('role',['geometry','prior','matched'])
def test_only_exact_owned_completed_canonical_case_raw_can_retire(tmp_path,role):
    owned=tmp_path/'our-cascade';root=owned/'variant/asset';folder=root/role;folder.mkdir(parents=True)
    path=folder/'prediction.npz';path.write_bytes(b'synthetic duplicate')
    case=dict(status='PASS_ACTUAL_NATIVE_SS480_AND_MATCHED_PBR1900_FULLCASCADE_CASE',SS_variant='R_raw_topk',
        PBR_mode='raw-topk-plus-feature',sample_id='one',SS_prediction_SHA256='SS',
        paired_original_and_trained_texture_initial_noise_exact=True,
        paired_raw_geometry_and_native_predicted_LOD_identical=True,
        actual_prior_and_matched_three_camera_original_material_M1_shared_scores=True,terminals=[dict(returncode=0)]*7,
        geometry_raw=str(root/'geometry'),prior_generation=str(root/'prior'),matched_generation=str(root/'matched'))
    g=dict(sample_id='one',SS_prediction_SHA256='SS',GT_geometry_or_coords_used_in_generation=False)
    if role=='geometry':g.update(status='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION',
        GT_loaded_only_after_generation=True,fixed_shape_models=True)
    else:g.update(status='PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE',
        checkpoint_kind='original16850-engineering'if role=='prior'else'trained-native1900',
        engineering_only=role=='prior',PBR_arm='vanilla'if role=='prior'else'raw-topk-plus-feature',
        source_weights_unchanged=True,geometry_array_byte_identity={'vertices':True})
    kw=dict(case_root=root,owned_root=owned,case=case,generation=g,
        canonical_entry=dict(sha256='actual',bytes=path.stat().st_size),actual_SHA256='actual',readers=[])
    assert validate_case_raw(path,**kw)['local_duplicate_only']and path.exists()
    for changed in[dict(readers=['renderer']),dict(owned_root=tmp_path/'other'),dict(actual_SHA256='wrong'),
        dict(case=dict(case,paired_raw_geometry_and_native_predicted_LOD_identical=False)),
        dict(generation=dict(g,GT_geometry_or_coords_used_in_generation=True))]:
        with pytest.raises(ValueError):validate_case_raw(path,**dict(kw,**changed))
    original=tmp_path/'original.npz';original.write_bytes(path.read_bytes());path.unlink();path.symlink_to(original)
    with pytest.raises(ValueError):validate_case_raw(path,**kw)
    assert original.exists()
