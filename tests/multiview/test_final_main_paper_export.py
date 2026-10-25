"""Synthetic complete-report metadata only; no actual TEST or neural metric."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.deformnext_retargeting.export_final_main_table import FIELDS, STATUS, build
from test_final_qualitative_export import complete_reports


def reports(tmp_path,arm='E_FIXED_RAW',mode='top1-only'):
    eight,ten,kwargs=complete_reports(tmp_path,arm,mode)
    from test_selected_final_population import parents
    # Use the exact producer's already validated physical control object.
    _,controls=parents(tmp_path,arm,mode)
    return eight,ten,dict(parent=kwargs['parent'],controls=controls,inputs=kwargs['inputs'])


@pytest.mark.parametrize('arm,mode,count',[('E_FIXED_RAW','top1-only',5),('C_DECODER','top1-only',4),('V1','vanilla',4)])
def test_actual_deduplicated_row_order_ten_fields_and_raw_signed_KID(tmp_path,arm,mode,count):
    eight,ten,kwargs=reports(tmp_path,arm,mode)
    for alias in ten['models']:
        ten['primary_ten_metric_table'][alias]['KID']=-1.25e-10
        ten['distribution_metrics'][alias]['kernel_inception_distance_mean']=-1.25e-10
    before=deepcopy((eight,ten,kwargs));result=build(eight,ten,**kwargs)
    assert result['status']==STATUS and len(result['models'])==count
    assert result['models'][:4]==['zero-source4','zero-edited4','V1','RETARGET']
    assert all(tuple(row)==FIELDS and row['KID']=='-1.25e-10' for row in result['formatted_table'].values())
    assert all(row['KID']==-1.25e-10 for row in result['unrounded_primary_table'].values())
    assert '\\pending' not in result['tex'] and '\\textbf' not in result['tex']
    assert result['pairwise_primary_family_size']==(16 if count==5 else 8)
    assert not result['full_goal_complete']
    assert all(r['FID']['interpretation']=='descriptive_only' and r['KID']['family_CI95'] is None for r in result['comparisons'].values())
    # The exporter must not modify even the unrounded input values it displays.
    assert (eight,ten,kwargs)==before


def test_family_evidence_uses_metric_direction_and_does_not_call_uncertainty_equivalence(tmp_path):
    eight,ten,kwargs=reports(tmp_path)
    key='SELECTED_minus_RETARGET'
    for record in [eight,ten]:
        record['primary_pairwise_family_intervals'][key]['deformation_weighted_family_CI95']['LPIPS']=[-.1,.1]
    result=build(eight,ten,**kwargs);values=result['comparisons'][key]
    assert values['LPIPS']['interpretation']=='uncertain_not_equivalent'
    assert values['PSNR']['interpretation']=='benefit_supported'
    assert values['FScore']['interpretation']=='harm_supported'
    assert values['CD']['interpretation']=='harm_supported'
    assert 'do not establish equivalence or non-inferiority' in result['writing_scope_notes']


@pytest.mark.parametrize('damage',['source-role','source-weighting','KID-clamp','KID-scale','distribution-count',
    'foreign-interval','missing-endpoint','family-size','wrong-delta','NaN'])
def test_other_means_distribution_units_or_unadjusted_intervals_cannot_be_exported(tmp_path,damage):
    eight,ten,kwargs=reports(tmp_path);ten=deepcopy(ten);key='SELECTED_minus_RETARGET'
    if damage=='source-role':ten['primary_ten_metric_table']['V1']['LPIPS']=.99
    elif damage=='source-weighting':
        eight['statistics']['roles']['source9_10']['means']['V1']['deformation_weighted']['LPIPS']+=.01
        ten['role_statistics']=deepcopy(eight['statistics']);ten['primary_ten_metric_table']['V1']['LPIPS']+=.01
    elif damage=='KID-clamp':ten['distribution_metrics']['V1']['unbiased_KID_clamped']=True
    elif damage=='KID-scale':ten['primary_ten_metric_table']['V1']['KID']*=1000
    elif damage=='distribution-count':ten['distribution_metrics']['V1']['images_per_pool']=32
    elif damage=='foreign-interval':ten['primary_pairwise_family_intervals'][key]['deformation_weighted_family_CI95']['LPIPS']=[-.1,.1]
    else:
        for record in [eight,ten]:
            row=record['primary_pairwise_family_intervals'][key]
            if damage=='missing-endpoint':row['deformation_weighted_family_CI95'].pop('CD')
            elif damage=='family-size':row['endpoint_family_size']=1
            elif damage=='wrong-delta':row['deformation_weighted_delta']['CD']+=1
            else:row['deformation_weighted_family_CI95']['CD']=[float('nan'),1]
    with pytest.raises(ValueError):build(eight,ten,**kwargs)


def test_unpinned_paper_source_stops_before_any_TEST_quality_read(tmp_path,monkeypatch):
    import experiments.deformnext_retargeting.export_final_main_table as entry
    from experiments.deformnext_retargeting.cascade_view_roles import file_sha256 as sha
    from experiments.deformnext_retargeting.selected_final_metric_summary import metric_contract
    from test_selected_final_population import parents
    parent,controls=parents(tmp_path)
    parent['evaluation_protocol']['selected_final_metric_contract']=metric_contract(parent,draws=16)
    reads=[];values=dict(parent=parent,controls=controls)
    def read(spec,check):
        reads.append(spec['path']);return values[spec['path']]
    class MetadataFixtureCheck:
        def __init__(self,base):pass
        def check(self,*args):pass
    monkeypatch.setattr(entry,'read_file',read);monkeypatch.setattr(entry,'ClosedInputs',MetadataFixtureCheck)
    context=dict(schema=entry.SCHEMA,repo=str(Path(entry.__file__).resolve().parents[2]),
        CPU_affinity=[92,93],canonical_local_root=str(tmp_path),source_SHA256={p:sha(p) for p in entry.required_sources()},
        source_inputs=dict(parent_freeze={'path':'parent'},control_freeze={'path':'controls'},selection_plan={'path':'policy'},
            eight_metrics={'path':'DO_NOT_READ'},ten_metrics={'path':'DO_NOT_READ'}))
    path=tmp_path/'context.json';path.write_text(json.dumps(context))
    with pytest.raises(ValueError,match='before TEST'):entry.export(path)
    assert reads==['parent','controls']
