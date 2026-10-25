"""Synthetic metadata/pixels only; no actual TEST quality or neural forward."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.deformnext_retargeting.cascade_view_roles import ClosedInputs, file_sha256 as sha
from experiments.deformnext_retargeting.export_final_qualitative import (
    case_images, frozen_admission, read_scored_image, render_case, report_admission, required_sources)
from experiments.deformnext_retargeting.final_qualitative_selection import (
    PLAN_STATUS, outcome_blind, policy, select_cases, validate_declaration)
from experiments.deformnext_retargeting.metric_image_identity import scored_image_observation


def metadata(population=None):
    ids = ['a', 'z', *[f'fixture-{i:03d}' for i in range(2, 508)]]
    if population is None:
        population = dict(sample_ids=ids, source_clusters_by_sample={sid: f'cluster-{i%134:03d}' for i,sid in enumerate(ids)},
            manifest=dict(path='/fixture/test.jsonl', SHA256='a'*64))
    ids = population['sample_ids']
    rows = [dict(sample_id=sid, source_uid='fixture-uid-'+population['source_clusters_by_sample'][sid],
        source_cluster_id=population['source_clusters_by_sample'][sid], split='test') for sid in ids]
    declared = dict(status=PLAN_STATUS, policy=policy(), all_metadata=rows, assets=508, source_clusters=134,
        manifest=deepcopy(population['manifest']), selected_outcome_blind_examples=outcome_blind(rows),
        no_candidate_predictions_or_TEST_quality_values_read=True, no_candidate_NN_forward_or_training_or_new_render=True,
        actual_final_freeze_created=False, actual_final_qualitative_images_created=False, final_model_choice_not_made=True)
    return population, declared


def inputs(selected='SELECTED'):
    population, declared = metadata()
    names = ['zero-source4', 'zero-edited4', 'V1', 'RETARGET'] + (['SELECTED'] if selected == 'SELECTED' else [])
    models = {name: {} for name in names}; per = {}
    for role in ['source9_10', 'all8_9_10']:
        per[role] = {name: [dict(sample_id=sid, source_cluster_id=population['source_clusters_by_sample'][sid],
            CD=.1, FScore=.5, LPIPS=.2, DreamSim=.3) for sid in population['sample_ids']] for name in names}
    return dict(declared=declared, population=population, selected_alias=selected, models=models, statistics=dict(per_asset=per))


def set_delta(args, index, **values):
    alias = args['selected_alias']
    for metric, delta in values.items():
        role = 'all8_9_10' if metric in ['CD', 'FScore'] else 'source9_10'
        baseline = dict(CD=.1, FScore=.5, LPIPS=.2, DreamSim=.3)[metric]
        args['statistics']['per_asset'][role][alias][index][metric] = baseline + delta


def test_known_metadata_hash_order_retains_two_sources_and_is_input_order_independent():
    rows = [dict(sample_id=sid, source_uid=cluster, source_cluster_id=cluster, split='test')
        for sid,cluster in [('a','A'),('z','A'),('b','B'),('c','C')]]
    assert [r['sample_id'] for r in outcome_blind(rows)] == ['z', 'b']
    assert outcome_blind(rows) == outcome_blind(list(reversed(rows)))
    with pytest.raises(ValueError, match='four prescribed'):
        outcome_blind([dict(r, quality=.9) for r in rows])


def test_even_category_selects_lower_median_instead_of_best_case():
    args = inputs()
    for i in range(6):
        set_delta(args, i, CD=-.01, FScore=.01*(i+1), LPIPS=-.01, DreamSim=-.01)
    before = deepcopy(args); result = select_cases(**args)
    assert args == before
    category = result['categories']['joint-improvement']
    assert category['eligible_cases'] == 6 and category['selected_sample_id'] == 'fixture-002'
    assert category['selected_sample_id'] != 'fixture-005'


def test_numeric_ties_use_declared_hash_not_lexical_ID_and_merge_categories():
    args = inputs()
    for i in [0, 1]:
        set_delta(args, i, CD=-.01, FScore=.02, LPIPS=-.01, DreamSim=-.01)
    assert select_cases(**args)['categories']['joint-improvement']['selected_sample_id'] == 'z'
    args = inputs(); set_delta(args, 0, CD=.01, FScore=-.02, LPIPS=.01, DreamSim=.01)
    result = select_cases(**args); rows = [r for r in result['cases'] if r['sample_id'] == 'a']
    assert len(rows) == 1 and {'geometry-loss','appearance-loss'} <= set(rows[0]['categories'])


def test_empty_categories_and_exact_zero_changes_never_get_replacement_cases():
    result = select_cases(**inputs())
    assert len(result['cases']) == 2
    assert all(row['eligible_cases'] == 0 and row['selected_sample_id'] is None
        and row['empty_category_not_replaced'] for row in result['categories'].values())


@pytest.mark.parametrize('selected,comparator',[('SELECTED','RETARGET'),('RETARGET','V1'),('V1','RETARGET')])
def test_comparator_follows_frozen_identity_and_source_metrics_keep_their_role(selected,comparator):
    args = inputs(selected); set_delta(args, 0, LPIPS=.01, DreamSim=.01)
    # All-camera appearance says the opposite; it must not replace source9/10.
    args['statistics']['per_asset']['all8_9_10'][selected][0].update(LPIPS=.01, DreamSim=.01)
    result = select_cases(**args)
    assert result['comparator'] == comparator
    assert result['categories']['appearance-loss']['selected_sample_id'] == 'a'


@pytest.mark.parametrize('damage',['policy','blind-ID','manifest','quality-read','missing-row','wrong-source','row-order','NaN'])
def test_declared_policy_full_population_and_finite_roles_cannot_be_replaced(damage):
    args = inputs()
    if damage == 'policy':args['declared']['policy']['diagnostic_choice'] = 'best case'
    elif damage == 'blind-ID':args['declared']['selected_outcome_blind_examples'][0]['sample_id'] = 'z'
    elif damage == 'manifest':args['declared']['manifest']['SHA256'] = 'b'*64
    elif damage == 'quality-read':args['declared']['no_candidate_predictions_or_TEST_quality_values_read'] = False
    elif damage == 'missing-row':args['statistics']['per_asset']['source9_10']['V1'].pop()
    elif damage == 'wrong-source':args['statistics']['per_asset']['source9_10']['V1'][0]['source_cluster_id'] = 'other'
    elif damage == 'row-order':args['statistics']['per_asset']['all8_9_10']['V1'].reverse()
    else:args['statistics']['per_asset']['source9_10']['SELECTED'][0]['LPIPS'] = float('nan')
    with pytest.raises(ValueError):select_cases(**args)


def test_missing_preTEST_qualitative_pin_fails_before_opening_quality_reports(tmp_path,monkeypatch):
    from test_selected_final_population import parents
    from experiments.deformnext_retargeting.selected_final_metric_summary import metric_contract
    import experiments.deformnext_retargeting.export_final_qualitative as entry
    parent, controls = parents(tmp_path)
    parent['evaluation_protocol']['selected_final_metric_contract'] = metric_contract(parent,draws=16)
    values = dict(parent=parent, controls=controls); reads = []
    def read(spec,check):
        reads.append(spec['path']); return values[spec['path']]
    monkeypatch.setattr(entry, 'read_file', read)
    context = dict(schema=entry.SCHEMA, source_SHA256={p:sha(p) for p in required_sources()},
        source_inputs=dict(parent_freeze={'path':'parent'}, control_freeze={'path':'controls'},
            selection_plan={'path':'declaration'}, eight_metrics={'path':'DO_NOT_READ'}, ten_metrics={'path':'DO_NOT_READ'}))
    with pytest.raises(ValueError, match='before any TEST prediction'):
        frozen_admission(context, None)
    assert reads == ['parent', 'controls']


def test_valid_preTEST_pins_accept_metadata_without_opening_quality_reports(tmp_path,monkeypatch):
    from test_selected_final_population import parents
    from experiments.deformnext_retargeting.selected_final_metric_summary import metric_contract
    import experiments.deformnext_retargeting.export_final_qualitative as entry
    parent, controls = parents(tmp_path); population, declared = metadata(parent['population'])
    pins = {p:sha(p) for p in required_sources()}
    specs = {name:dict(path='/fixture/'+name+'.json',SHA256='a'*64,receipt='/fixture/file4-closure.json')
        for name in ['parent_freeze','control_freeze','selection_plan']}
    specs['test_manifest'] = dict(population['manifest'],receipt='/fixture/file4-closure.json')
    parent['evaluation_protocol'].update(selected_final_metric_contract=metric_contract(parent,draws=16),
        qualitative_selection_plan=specs['selection_plan'])
    parent['evaluation_protocol']['source_SHA256'].update(pins)
    values = {specs['parent_freeze']['path']:parent, specs['control_freeze']['path']:controls,
        specs['selection_plan']['path']:declared, specs['test_manifest']['path']:declared['all_metadata']}
    reads = []
    def read(spec,check,**kwargs):
        reads.append(spec['path']); return values[spec['path']]
    monkeypatch.setattr(entry,'read_file',read)
    result = frozen_admission(dict(schema=entry.SCHEMA,source_SHA256=pins,source_inputs=specs),None)
    assert result[0] is parent and result[4] is declared and len(result[2])==5
    assert reads == [specs[k]['path'] for k in ['parent_freeze','control_freeze','selection_plan','test_manifest']]


def complete_reports(tmp_path,arm='E_FIXED_RAW',mode='top1-only'):
    from test_selected_final_metric_summary import inputs as fixture, distribution
    from experiments.deformnext_retargeting.selected_final_metric_summary import eight_metrics,combine_ten
    from experiments.deformnext_retargeting.selected_final_population import compose_models
    args = fixture(tmp_path,arm,mode);eight = eight_metrics(**args)
    ten = combine_ten(eight,distribution(eight),parent=args['parent'],controls=args['controls'])
    specs = dict(parent_freeze={'path':'/fixture/freeze.json'},control_freeze={'path':'/fixture/controls.json'})
    ten.update(source_inputs=specs,all_ten_metrics_collected=True)
    kwargs = dict(parent=args['parent'],models=compose_models(args['parent'],args['controls']),
        contract=eight['metric_contract'],inputs=specs)
    return eight,ten,kwargs


@pytest.mark.parametrize('arm,mode,count',[('E_FIXED_RAW','top1-only',5),('C_DECODER','top1-only',4)])
def test_complete_metric_report_accepts_distinct_and_merged_selected_models(tmp_path,arm,mode,count):
    eight,ten,kwargs = complete_reports(tmp_path,arm,mode)
    index = report_admission(eight,ten,**kwargs)
    assert len(index)==count and all(len(rows)==1524 for rows in index.values())


@pytest.mark.parametrize('damage',['eight-only','changed-role-score','duplicate-archive','different-freeze','missing-model'])
def test_partial_or_different_model_metrics_cannot_supply_final_figures(tmp_path,damage):
    eight,ten,kwargs = complete_reports(tmp_path)
    if damage=='eight-only':ten['all_ten_metrics_collected']=False
    elif damage=='changed-role-score':
        ten['role_statistics']=deepcopy(ten['role_statistics'])
        ten['role_statistics']['per_asset']['source9_10']['V1'][0]['LPIPS']+=.01
    elif damage=='duplicate-archive':
        rows=eight['scored_image_observations']['V1'];rows[1]['archive']=rows[0]['archive'];rows[1]['archive_SHA256']=rows[0]['archive_SHA256']
    elif damage=='different-freeze':ten['source_inputs']=dict(ten['source_inputs'],parent_freeze={'path':'/fixture/other.json'})
    else:ten['models']=dict(ten['models']);ten['models'].pop('zero-edited4')
    with pytest.raises(ValueError):report_admission(eight,ten,**kwargs)


def archive(tmp_path,alias,camera,*,GT=.5,prediction=.25):
    root = tmp_path / alias; root.mkdir(exist_ok=True); path = root / f'camera{camera:02d}.npz'
    pixels = dict(prediction=np.full((3,512,512),prediction,np.float32),
        GT_shared_renderer=np.full((3,512,512),GT,np.float32), GT_alpha=np.ones((512,512),np.float32))
    np.savez_compressed(path,**pixels)
    row = scored_image_observation(pixels,sample_id='fixture',camera_id=camera,source_cluster_id='source',
        archive=str(path),archive_SHA256=sha(path))
    receipt = root/'file4-closure.json'
    entries = json.loads(receipt.read_text())['files'] if receipt.exists() else []
    entries = [r for r in entries if r['relative_path'] != str(path.relative_to(tmp_path))]
    entries.append(dict(relative_path=str(path.relative_to(tmp_path)),sha256=sha(path),bytes=path.stat().st_size))
    # A local engineering fixture only, never a published file4 scientific receipt.
    receipt.write_text(json.dumps(dict(status='PASS',local_equals_actual_file4_SHA256=True,files=entries)))
    return row


@pytest.mark.parametrize('damage',['changed-file','float-identity','wrong-camera','crop','dtype'])
def test_actual_scored_pixels_camera_conversion_and_archives_are_not_guessed(tmp_path,damage):
    row = archive(tmp_path,'V1',8); path = tmp_path/'V1/camera08.npz'
    if damage == 'changed-file':path.write_bytes(path.read_bytes()+b'changed')
    elif damage == 'float-identity':row['prediction_float_RGB_SHA256'] = '0'*64
    elif damage == 'wrong-camera':row['camera_id'] = 9
    else:
        with np.load(path) as z:pixels = {k:z[k].copy() for k in z.files}
        pixels['prediction'] = pixels['prediction'][:,:256,:256] if damage=='crop' else pixels['prediction'].astype(np.float64)
        np.savez_compressed(path,**pixels);row['archive_SHA256'] = sha(path)
        receipt = path.parent/'file4-closure.json';a = json.loads(receipt.read_text());a['files'][0].update(sha256=sha(path),bytes=path.stat().st_size);receipt.write_text(json.dumps(a))
    with pytest.raises(ValueError):read_scored_image(row,ClosedInputs(tmp_path))


def test_scored_case_join_and_scientific_plot_preserve_all_cameras_and_shared_GT(tmp_path):
    index = {}; models = {name:{} for name in ['zero-source4','zero-edited4','V1','RETARGET','SELECTED']}
    for alias in models:
        index[alias] = {('fixture',camera):archive(tmp_path,alias,camera) for camera in [8,9,10]}
    images, observed = case_images('fixture',models=models,index=index,check=ClosedInputs(tmp_path))
    assert len(images) == 18 and len(observed) == 15
    assert all(image.shape==(512,512,3) and image.dtype==np.uint8 for image in images.values())
    assert images['V1',8][0,0].tolist() == [64,64,64]
    case = dict(sample_id='fixture',categories=['engineering-only'],per_model_metrics={a:dict(LPIPS=.2,DreamSim=.3,CD=.1,FScore=.5) for a in models})
    outputs = render_case(case,images=images,models=models,output=tmp_path,number=1,selected_alias='RETARGET',comparator='V1')
    assert all((tmp_path/name).stat().st_size>0 for name in outputs)
    index['RETARGET']['fixture',10] = archive(tmp_path,'RETARGET',10,GT=.51)
    with pytest.raises(ValueError,match='identical GT'):
        case_images('fixture',models=models,index=index,check=ClosedInputs(tmp_path))
