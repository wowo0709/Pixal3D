from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.deformnext_retargeting import fixed_pbr_render_geometry as fixed
from experiments.deformnext_retargeting import run_selected_campaign_case as adapter
from experiments.deformnext_retargeting.cascade_view_roles import ClosedInputs, file_sha256 as sha
from experiments.deformnext_retargeting.selected_shared_render_geometry import (
    FIELD, FrozenGeometryNumpy, bind_owner_render_geometry, required_sources,
    verify_case_render_geometry,
)
from experiments.deformnext_retargeting.selected_final_phase import SCHEMA as PHASE_SCHEMA, validate_binding


def close_fixture(folder, base):
    files = [p for p in folder.iterdir() if p.is_file() and p.name != 'file4-closure.json']
    (folder/'file4-closure.json').write_text(json.dumps(dict(status='PASS',
        local_equals_actual_file4_SHA256=True, files=[dict(relative_path=str(p.relative_to(base)),
        sha256=sha(p), bytes=p.stat().st_size) for p in files])))


def inputs(tmp_path, monkeypatch):
    # CPU-only archive fixtures. Production and the real GPU check use1m faces.
    monkeypatch.setattr(fixed, 'FACE_BUDGET', 1)
    common = tmp_path/'owner/geometry';render = tmp_path/'owner/render';model = tmp_path/'model/PBR'
    for p in [common, render, model]:p.mkdir(parents=True)
    mesh = dict(vertices=np.array([[0,0,0],[1,0,0],[0,1,0]], np.float32),
                faces=np.array([[0,1,2],[0,1,2]], np.int32))
    raw = dict(**mesh,coords=np.array([[1,2,3]],np.int32),origin=np.zeros(3,np.float32),
        voxel_size=np.array(.01,np.float32),voxel_shape=np.array([1024]*3,np.int32),
        shape_coords=np.array([[1,2,3]],np.int32),shape_feats=np.ones((1,32),np.float32),
        attrs=np.ones((1,6),np.float32),texture_feats=np.ones((1,32),np.float32))
    np.savez_compressed(common/'prediction.npz',**raw)
    own = deepcopy(raw);own['attrs'] *= .7;own['texture_feats'] *= .3
    np.savez_compressed(model/'prediction.npz',**own)
    (model/'generation.json').write_text(json.dumps(dict(sample_id='s',GT_geometry_or_coords_used_in_generation=False)))
    frozen = deepcopy(mesh);frozen['vertices'][0,0] = .01;frozen['faces'] = frozen['faces'][:1]
    np.savez_compressed(render/'render-geometry.npz',**frozen)
    for p in [common,render,model]:close_fixture(p,tmp_path)
    native = dict(generation_root=str(model),output=str(tmp_path/'model/render'),
                  canonical_local_root=str(tmp_path),row=dict(sample_id='s'))
    case = dict(sample_id='s',geometry_group='SS1',geometry_source_root=str(common),
        geometry_reference=dict(alias='V1'),phases=[dict(name='render',plan=native)],
        raw_archive_roots=[str(common),str(model)])
    owner = dict(alias='V1',sample_id='s',geometry_group='SS1',geometry_source_root=str(common),
                 case_output=str(tmp_path/'owner'),actual_completed_native_case=True)
    sources = {p:sha(p) for p in required_sources()}
    bound = bind_owner_render_geometry(case,owner,ClosedInputs(tmp_path),sources)
    return case,owner,bound,frozen,own


def test_binding_retains_raw_roots_model_inputs_and_original_phase_count(tmp_path,monkeypatch):
    case,owner,bound,_,_ = inputs(tmp_path,monkeypatch)
    assert bound['raw_archive_roots'] == case['raw_archive_roots']
    assert len(bound['phases']) == len(case['phases'])
    assert bound['phases'][0]['plan']['generation_root'] == case['phases'][0]['plan']['generation_root']
    assert FIELD not in case['phases'][0]['plan']
    assert bound['render_geometry_reference']['owner_alias'] == owner['alias']


def test_original_geometry_producer_plan_is_not_changed():
    case = dict(geometry_reference=None)
    assert bind_owner_render_geometry(case,None,None,{}) is case


def test_render_input_override_is_rejected_in_a_model_sampling_phase():
    with pytest.raises(ValueError,match='cannot alter a model'):
        validate_binding(dict(schema=PHASE_SCHEMA,kind='native-guide-replay'),{FIELD:{}})


@pytest.mark.parametrize('damage',['sample','group','source','incomplete'])
def test_other_or_unbound_owner_cannot_supply_render_triangles(tmp_path,monkeypatch,damage):
    case,owner,_,_,_ = inputs(tmp_path,monkeypatch);sources={p:sha(p) for p in required_sources()}
    if damage=='sample':owner['sample_id']='other'
    elif damage=='group':owner['geometry_group']='other'
    elif damage=='incomplete':owner['actual_completed_native_case']=False
    else:sources={}
    with pytest.raises(ValueError):bind_owner_render_geometry(case,owner,ClosedInputs(tmp_path),sources)


def test_proxy_changes_only_prediction_geometry_and_never_global_numpy(tmp_path,monkeypatch):
    _,_,bound,frozen,own=inputs(tmp_path,monkeypatch);native=bound['phases'][0]['plan'];original_load=np.load
    path=Path(native['generation_root'])/'prediction.npz';before=sha(path)
    proxy=FrozenGeometryNumpy(np,native)
    with proxy.load(path) as arrays:
        assert np.array_equal(arrays['vertices'],frozen['vertices'])
        assert np.array_equal(arrays['attrs'],own['attrs'])
        assert np.array_equal(arrays['texture_feats'],own['texture_feats'])
    np.savez(tmp_path/'GT-only.npz',GT=np.array([123],np.float32))
    with proxy.load(tmp_path/'GT-only.npz') as z:assert z['GT'][0]==123
    output=Path(native['output']);output.mkdir();np.savez_compressed(output/'render-geometry.npz',**frozen)
    proof=proxy.verify()
    assert proof['rendered_vertices_faces_exact'] and proof['new_raw_archive_files']==0
    assert np.load is original_load and sha(path)==before
    assert len(list(Path(native['generation_root']).glob('prediction*.npz')))==1


def test_post_render_same_face_count_with_different_vertices_is_rejected(tmp_path,monkeypatch):
    _,_,bound,frozen,_=inputs(tmp_path,monkeypatch);native=bound['phases'][0]['plan'];proxy=FrozenGeometryNumpy(np,native)
    with proxy.load(Path(native['generation_root'])/'prediction.npz'):pass
    output=Path(native['output']);output.mkdir();frozen['vertices'][1,0]+=.01
    np.savez_compressed(output/'render-geometry.npz',**frozen)
    with pytest.raises(ValueError,match='exact frozen'):proxy.verify()


def test_a_second_prediction_read_cannot_silently_reuse_unchecked_arrays(tmp_path,monkeypatch):
    _,_,bound,_,_=inputs(tmp_path,monkeypatch);native=bound['phases'][0]['plan'];proxy=FrozenGeometryNumpy(np,native)
    with proxy.load(Path(native['generation_root'])/'prediction.npz'):pass
    with pytest.raises(ValueError,match='one explicit'):proxy.load(Path(native['generation_root'])/'prediction.npz')


def test_case_validation_requires_actual_phase_proof_not_only_geometry_metadata(tmp_path,monkeypatch):
    _,_,bound,frozen,_=inputs(tmp_path,monkeypatch);native=bound['phases'][0]['plan'];proxy=FrozenGeometryNumpy(np,native)
    with proxy.load(Path(native['generation_root'])/'prediction.npz'):pass
    output=Path(native['output']);output.mkdir();np.savez_compressed(output/'render-geometry.npz',**frozen)
    proof=proxy.verify();phase=dict(status='PASS_ACTUAL_ORIGINAL_NUMERICAL_SELECTED_CASE_PHASE',
                                  kind='assigned-native-render',proof={FIELD:proof})
    (output/'selected-phase-entry.json').write_text(json.dumps(phase));close_fixture(output,tmp_path)
    assert verify_case_render_geometry(bound,ClosedInputs(tmp_path))['rendered_vertices_faces_exact']
    phase['proof'][FIELD]['reference']['owner_alias']='wrong'
    (output/'selected-phase-entry.json').write_text(json.dumps(phase));close_fixture(output,tmp_path)
    with pytest.raises(ValueError,match='phase proof'):verify_case_render_geometry(bound,ClosedInputs(tmp_path))


def test_campaign_adapter_restores_every_hook_after_a_real_execution_error(tmp_path,monkeypatch):
    path=tmp_path/'context.json';path.write_text(json.dumps(dict(campaign_credit={'fixture':True},
        campaign_token='token',source_SHA256={p:sha(p) for p in adapter.required_sources()})))
    monkeypatch.setenv('SELECTED_FINAL_CAMPAIGN_TOKEN','token');monkeypatch.setattr(adapter,'source_guard',lambda c:None)
    def fail(path):raise RuntimeError('fixture-native-failure')
    monkeypatch.setattr(adapter.native,'execute',fail)
    before=(adapter.native.DiskBudget,adapter.native.load_case,adapter.native.validate_case_reports)
    with pytest.raises(RuntimeError,match='fixture-native-failure'):adapter.execute(path)
    assert (adapter.native.DiskBudget,adapter.native.load_case,adapter.native.validate_case_reports)==before
