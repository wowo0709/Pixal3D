from copy import deepcopy
from hashlib import sha256
import json
import os

import numpy as np
import psutil
import pytest

from experiments.deformnext_retargeting.cascade_view_roles import ClosedInputs
from experiments.deformnext_retargeting.operational_reused_guide_retirement import (
    pending_guide_reads,reader_snapshot,retained_case_files,validate_candidate,
)
from experiments.deformnext_retargeting.selected_final_case_integrity import CASE_STATUS
from experiments.deformnext_retargeting.selected_final_metric_inputs import read_pixels


def fixture(tmp_path):
    sid='frozen-complete-reused';alias='SELECTED';root=tmp_path/'old/cases'/sha256(sid.encode()).hexdigest()[:24]/alias
    guide=root/'geometry/native-shape-guides.pt';guide.parent.mkdir(parents=True);guide.write_bytes(b'canonical guide duplicate')
    models={a:dict(actual_model=a) for a in ['zero-source4','zero-edited4','V1','RETARGET','SELECTED']}
    cases={a:dict(audit=dict(path=str(root.parent/a/'audit.json'))) for a in models}
    m=dict(groups={sid:dict(cases=cases)},models=models)
    g=dict(status='PASS_ACTUAL_COMPLETE_ASSET_GROUP_NATIVE_CASES_AND_RAW_RETIREMENT',sample_id=sid,
        cases=deepcopy(cases),retirements={f'actual-{i}':{} for i in range(8)})
    case=dict(status=CASE_STATUS,actual_completed_native_case=True,GT_not_used_in_generation=True,optimizer_updates=0,
        sample_id=sid,alias=alias,case_output=str(root),geometry_source_root=str(guide.parent),
        frozen_model_binding=deepcopy(models[alias]),actual_native_guide_archive_SHA256=sha256(guide.read_bytes()).hexdigest())
    kwargs=dict(row=dict(sample_id=sid,alias=alias),manifest=m,group=g,case=case,
        terminal=dict(returncode=0,actual_session_drained=True,case_output=str(root)),
        canonical_entry=dict(sha256=case['actual_native_guide_archive_SHA256'],bytes=guide.stat().st_size),
        digest=case['actual_native_guide_archive_SHA256'],readers=[])
    return guide,kwargs


@pytest.mark.parametrize('problem',[None,'partial','unretired','failed','GT','training','wrong-model',
    'wrong-root','relabeled','failed-worker','undrained','reader','changed-guide','wrong-NAS-hash','wrong-NAS-size','symlink'])
def test_only_unread_exact_original_completed_canonical_guides_are_eligible(tmp_path,problem):
    path,k=fixture(tmp_path)
    if problem=='partial':k['group']['cases'].pop('V1')
    elif problem=='unretired':k['group']['retirements'].pop('actual-0')
    elif problem=='failed':k['case']['actual_completed_native_case']=False
    elif problem=='GT':k['case']['GT_not_used_in_generation']=False
    elif problem=='training':k['case']['optimizer_updates']=1
    elif problem=='wrong-model':k['case']['frozen_model_binding']={}
    elif problem=='wrong-root':k['case']['geometry_source_root']=str(tmp_path/'live/geometry')
    elif problem=='relabeled':k['case']['alias']='V1'
    elif problem=='failed-worker':k['terminal']['returncode']=1
    elif problem=='undrained':k['terminal']['actual_session_drained']=False
    elif problem=='reader':k['readers']=['actual pending guide reader']
    elif problem=='changed-guide':k['digest']='b'*64
    elif problem=='wrong-NAS-hash':k['canonical_entry']['sha256']='b'*64
    elif problem=='wrong-NAS-size':k['canonical_entry']['bytes']+=1
    elif problem=='symlink':
        actual=tmp_path/'actual-guide';path.rename(actual);path.symlink_to(actual)
    if problem is None:assert validate_candidate(path,**k)['canonical_file4_retained']
    else:
        with pytest.raises(ValueError):validate_candidate(path,**k)
    assert path.exists()


@pytest.mark.parametrize('name',['prediction.npz','camera08.npz','checkpoint.pt','GT.pt'])
def test_pixels_raws_and_checkpoints_cannot_be_guide_candidates(tmp_path,name):
    path,k=fixture(tmp_path);other=path.parent/name;other.write_bytes(path.read_bytes())
    with pytest.raises(ValueError):validate_candidate(other,**k)
    assert other.exists() and path.exists()


@pytest.mark.parametrize('status',['pending','running','HELD','future-unknown'])
def test_pending_and_unknown_consumers_protect_guide_files_and_parent_folders(tmp_path,status):
    path,k=fixture(tmp_path);plan=tmp_path/'reader.json'
    plan.write_text(json.dumps(dict(generation=str(path.parent),closed_inputs={str(path):'digest'})))
    jobs=[dict(name='actual-reader',command=['python','--plan',str(plan)])]
    assert pending_guide_reads(jobs,{'actual-reader':dict(status=status)},[path])==['actual-reader']
    assert pending_guide_reads(jobs,{'actual-reader':dict(status='completed')},[path])==[]
    plan.unlink()
    with pytest.raises(ValueError):pending_guide_reads(jobs,{},[path])


def test_second_snapshot_sees_an_actual_open_guide(tmp_path,monkeypatch):
    path,k=fixture(tmp_path);jp,sp=tmp_path/'jobs.json',tmp_path/'states.json';jp.write_text('[]');sp.write_text('{}')
    policy=dict(reader_queues=[dict(jobs=str(jp),states=str(sp))])
    monkeypatch.setattr(psutil,'process_iter',lambda:[psutil.Process(os.getpid())])
    assert not reader_snapshot(policy,[path])[0]
    with path.open('rb'):assert reader_snapshot(policy,[path])[0]==['actual open process '+str(os.getpid())]
    assert path.exists()


def test_original_scored_pixel_consumer_is_exact_after_fixture_guide_offload(tmp_path):
    guide,k=fixture(tmp_path);folder=guide.parent.parent/'shared';folder.mkdir()
    path=folder/'camera08.npz'
    np.savez_compressed(path,prediction=np.zeros((3,512,512),np.float32),
        prediction_alpha=np.ones((512,512),np.float32),GT_shared_renderer=np.ones((3,512,512),np.float32),
        GT_alpha=np.ones((512,512),np.float32))
    receipt=folder/'file4-closure.json';receipt.write_text(json.dumps(dict(status='PASS',local_equals_actual_file4_SHA256=True,
        files=[dict(relative_path=str(path.relative_to(tmp_path)),sha256=sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)])))
    kwargs=dict(alias='SELECTED',sid=k['row']['sample_id'],cluster='source-cluster',check=ClosedInputs(tmp_path))
    before=read_pixels(path,**kwargs)
    guide.unlink()  # Disposable fixture only; production requires the closed fresh NAS intent.
    after=read_pixels(path,**kwargs)
    assert after==before and path.exists()


@pytest.mark.parametrize('problem',[None,'missing-camera','missing-generation','missing-case','missing-score','symlink-score'])
def test_every_scored_archive_and_its_metadata_must_stay_present(tmp_path,problem):
    guide,k=fixture(tmp_path);root=guide.parent.parent
    paths=[guide.parent/'generation.json',root/'audit.json',root/'shared/render-evaluation.json',
        *[root/'shared'/f'camera{cid:02d}.npz' for cid in [8,9,10]]]
    for p in paths:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'actual fixture retained input')
    if problem=='missing-camera':paths[3].unlink()
    elif problem=='missing-generation':paths[0].unlink()
    elif problem=='missing-case':paths[1].unlink()
    elif problem=='missing-score':paths[2].unlink()
    elif problem=='symlink-score':
        outside=tmp_path/'outside-score';paths[2].rename(outside);paths[2].symlink_to(outside)
    if problem is None:assert retained_case_files(guide)==paths
    else:
        with pytest.raises(ValueError):retained_case_files(guide)
    assert guide.exists()
