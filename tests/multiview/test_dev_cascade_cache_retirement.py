from copy import deepcopy
from hashlib import sha256

import pytest

from experiments.deformnext_retargeting.dev_cascade_cache_retirement import (
    ARMS,candidates,own_contract,validate_candidate,
)


def fixture(tmp_path,arm='A_FIXED_UNIFORM'):
    sample='sample-DEV';owned=tmp_path/'own';root=owned/arm/'cases'/sha256(sample.encode()).hexdigest()[:16]
    for p in candidates(root):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'canonical raw')
    case=dict(SS_arm=arm,sample_id=sample,status=own_contract(arm).CASE_STATUS,
        new_training=False,TEST_forward=False,same_original_material_M1_reference=True,
        all_original_numerical_source_files_unmodified=True,reused_completed_native_generation=None,
        new_Shape_cascade_calls=1,new_PBR_replay_calls=1,geometry_root=str(root/'geometry'),
        final_generation_root=str(root/'PBR'),shared_root=str(root/'shared'),SS_prediction_SHA256='a'*64,
        actual_owned_terminals=[dict(phase=p,returncode=0,actual_session_drained=True)
            for p in ['geometry','PBR','render','shared']])
    geometry=dict(status='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION',sample_id=sample,
        SS_prediction_SHA256='a'*64,GT_geometry_or_coords_used_in_generation=False,
        GT_loaded_only_after_generation=True,per_output_alignment=False)
    replay=dict(status='PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE',sample_id=sample,
        PBR_arm='top1-only',checkpoint_kind='trained-native1900',engineering_only=False,
        source_weights_unchanged=True,geometry_array_byte_identity=dict(vertices=True,faces=True))
    return root,dict(case_root=root,owned_root=owned,case=case,geometry=geometry,replay=replay,
        canonical_entry=dict(bytes=len(b'canonical raw'),sha256=sha256(b'canonical raw').hexdigest()),
        actual_local_SHA256=sha256(b'canonical raw').hexdigest(),readers=[])


@pytest.mark.parametrize('arm',ARMS)
def test_accepts_only_three_explicit_completed_control_duplicates(tmp_path,arm):
    root,args=fixture(tmp_path,arm)
    for path in candidates(root):assert validate_candidate(path,**args)['canonical_file4_retained']
    assert all(p.exists() for p in candidates(root))  # Validator never unlinks.


@pytest.mark.parametrize('relative',['geometry/prediction.npz','geometry/native-shape-guides.pt',
    'shared/camera08.npz','render/camera09.npz','ckpts/denoiser_step0000480.pt'])
def test_protects_later_metric_inputs_guides_images_and_checkpoints(tmp_path,relative):
    root,args=fixture(tmp_path);path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'canonical raw')
    with pytest.raises(ValueError):validate_candidate(path,**args)
    assert path.exists()


@pytest.mark.parametrize('problem',['active-reader','failed-render','undrained','wrong-status',
    'TEST','reused','wrong-sample','GT-generation','realigned','different-replay-geometry',
    'remote-sha','remote-size','wrong-owner','symlink'])
def test_rejects_unverified_or_still_consumed_raw(tmp_path,problem):
    root,args=fixture(tmp_path);args=deepcopy(args);path=candidates(root)[0]
    if problem=='active-reader':args['readers']=['future raw consumer']
    elif problem=='failed-render':args['case']['actual_owned_terminals'][2]['returncode']=1
    elif problem=='undrained':args['case']['actual_owned_terminals'][3]['actual_session_drained']=False
    elif problem=='wrong-status':args['case']['status']='pending'
    elif problem=='TEST':args['case']['TEST_forward']=True
    elif problem=='reused':args['case']['reused_completed_native_generation']={'old':'case'}
    elif problem=='wrong-sample':args['case']['sample_id']='other'
    elif problem=='GT-generation':args['geometry']['GT_geometry_or_coords_used_in_generation']=True
    elif problem=='realigned':args['geometry']['per_output_alignment']=True
    elif problem=='different-replay-geometry':args['replay']['geometry_array_byte_identity']['vertices']=False
    elif problem=='remote-sha':args['canonical_entry']['sha256']='b'*64
    elif problem=='remote-size':args['canonical_entry']['bytes']+=1
    elif problem=='wrong-owner':args['owned_root']=tmp_path/'someone-else'
    else:
        target=tmp_path/'original.npz';target.write_bytes(path.read_bytes());path.unlink();path.symlink_to(target)
    with pytest.raises(ValueError):validate_candidate(path,**args)
    assert path.exists()
