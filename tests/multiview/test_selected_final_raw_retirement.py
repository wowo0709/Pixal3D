"""Synthetic completed consumers and tiny files; no deletion or neural work."""
from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.cascade_view_roles import file_sha256 as sha
from experiments.deformnext_retargeting.common_geometry_pbr_contract import MODES as PBR_MODES
from experiments.deformnext_retargeting.pbr_replay_artifacts import GEOMETRY_FIELDS
from experiments.deformnext_retargeting.pipeline_population_contract_v2 import ARMS
from experiments.deformnext_retargeting.selected_final_population import compose_models
from experiments.deformnext_retargeting.selected_final_raw_retirement import required_consumers,validate_raw
from test_selected_final_population import parents


def candidate(tmp_path,arm='V1',mode='top1-only',alias='V1',role='geometry'):
    parent,controls=parents(tmp_path,arm,mode);models=compose_models(parent,controls)
    owned=tmp_path/'owned';root=owned/alias;folder=root/role;folder.mkdir(parents=True)
    path=folder/'prediction.npz';path.write_bytes(b'synthetic canonical raw duplicate')
    frozen=dict(path='/closed/actual-parent.json',SHA256='a'*64);control=dict(path='/closed/actual-controls.json',SHA256='b'*64)
    names=required_consumers(alias,role,parent=parent,controls=controls)
    group=models[alias]['geometry_group'];canonical=next(n for n,m in models.items() if m['geometry_group']==group)
    consumers={}
    for name in names:
        consumers[name]=dict(status='PASS_ACTUAL_SELECTED_PHYSICAL_TEST508_CASE',alias=name,sample_id='test0',source_cluster_id='testcluster0',
            actual_completed_native_case=True,frozen_model_binding=deepcopy(models[name]),parent_freeze=frozen,control_freeze=control,
            actual_SS_prediction_SHA256='c'*64,geometry_source_root=str(owned/canonical/'geometry'),
            actual_native_guide_archive_SHA256='d'*64,actual_shared_GT_material_pixel_identity_in_observations=True,
            surface_reference_join={'status':'PASS_ACTUAL_LOCAL_NATIVE_GEOMETRY_SHARED_GT_SURFACE_JOIN'},
            scored_image_observations=[dict(camera_id=c) for c in [8,9,10]],
            terminals=[dict(returncode=0,actual_session_drained=True)],case_output=str(owned/name),
            raw_archive_roots=[str(owned/canonical/'geometry'),str(owned/name/'PBR')])
    generation=dict(status='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION' if role=='geometry' else 'PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE',
        sample_id='test0',SS_prediction_SHA256='c'*64,GT_geometry_or_coords_used_in_generation=False,
        GT_loaded_only_after_generation=True,fixed_shape_models=True,checkpoint_kind='trained-native1900',engineering_only=False,
        PBR_arm=models[alias].get('PBR_mode'),source_weights_unchanged=True,geometry_array_byte_identity=dict.fromkeys(GEOMETRY_FIELDS,True))
    args=dict(case_root=root,owned_root=owned,case=consumers[alias],generation=generation,consumers=consumers,
        parent=parent,controls=controls,parent_freeze=frozen,control_freeze=control,
        canonical_entry=dict(sha256=sha(path),bytes=path.stat().st_size),actual_SHA256=sha(path),readers=[])
    return path,args


@pytest.mark.parametrize('arm',ARMS)
@pytest.mark.parametrize('mode',PBR_MODES)
def test_every_physical_candidate_has_its_actual_frozen_raw_consumers(tmp_path,arm,mode):
    p,c=parents(tmp_path,arm,mode);models=compose_models(p,c);alias=p['selected_alias']
    group=models[alias]['geometry_group'];owner=next(n for n,m in models.items() if m['geometry_group']==group)
    names=required_consumers(owner,'geometry',parent=p,controls=c)
    assert set(names)=={n for n,m in models.items() if m['geometry_group']==group}
    assert required_consumers(alias,'PBR',parent=p,controls=c)==[alias]


def test_shared_raw_geometry_waits_for_both_actual_PBR_compositions(tmp_path):
    path,args=candidate(tmp_path);before=deepcopy(args);token=validate_raw(path,**args)
    assert token['actual_frozen_raw_consumers']==['V1','SELECTED'] and args==before and path.exists()
    args['consumers'].pop('SELECTED')
    with pytest.raises(ValueError):validate_raw(path,**args)
    assert path.exists()


@pytest.mark.parametrize('damage',['failed-consumer','other-source','other-weights','other-SS','other-guides',
    'other-parent','other-control','no-score-pixels','live-session','missing-camera','unrelated-consumer',
    'bad-SHA','bad-size','reader','different-GT-protocol','not-own-root'])
def test_no_raw_admission_with_incomplete_foreign_or_active_consumers(tmp_path,damage):
    path,args=candidate(tmp_path);other=args['consumers']['SELECTED']
    if damage=='failed-consumer':other['actual_completed_native_case']=False
    elif damage=='other-source':other['source_cluster_id']='other'
    elif damage=='other-weights':other['frozen_model_binding']['ss64']['checkpoint']['SHA256']='f'*64
    elif damage=='other-SS':other['actual_SS_prediction_SHA256']='f'*64
    elif damage=='other-guides':other['actual_native_guide_archive_SHA256']='f'*64
    elif damage=='other-parent':other['parent_freeze']=dict(path='/other-parent',SHA256='f'*64)
    elif damage=='other-control':other['control_freeze']=dict(path='/other-control',SHA256='f'*64)
    elif damage=='no-score-pixels':other['actual_shared_GT_material_pixel_identity_in_observations']=False
    elif damage=='live-session':other['terminals'][0]['actual_session_drained']=False
    elif damage=='missing-camera':other['scored_image_observations'].pop()
    elif damage=='unrelated-consumer':args['consumers']['RETARGET']=deepcopy(other)
    elif damage=='bad-SHA':args['actual_SHA256']='f'*64
    elif damage=='bad-size':args['canonical_entry']['bytes']+=1
    elif damage=='reader':args['readers']=['actual open reader']
    elif damage=='different-GT-protocol':args['generation']['GT_loaded_only_after_generation']=False
    else:args['case_root']=args['case_root'].parent/'different-owner'
    with pytest.raises(ValueError):validate_raw(path,**args)
    assert path.exists()


def test_reused_case_can_retire_own_PBR_but_cannot_own_its_producers_geometry(tmp_path):
    path,args=candidate(tmp_path,alias='SELECTED',role='PBR')
    assert validate_raw(path,**args)['actual_frozen_raw_consumers']==['SELECTED']
    with pytest.raises(ValueError):required_consumers('SELECTED','geometry',parent=args['parent'],controls=args['controls'])
    with pytest.raises(ValueError):required_consumers('zero-source4','PBR',parent=args['parent'],controls=args['controls'])
    args['generation']['geometry_array_byte_identity'].pop('shape_feats')
    with pytest.raises(ValueError):validate_raw(path,**args)
