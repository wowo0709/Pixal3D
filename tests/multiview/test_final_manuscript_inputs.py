"""Synthetic complete-report metadata only; no real TEST results or NN."""
from copy import deepcopy

import pytest

from experiments.deformnext_retargeting import export_final_main_table as main_export
from experiments.deformnext_retargeting import export_final_qualitative as qual_export
from experiments.deformnext_retargeting.final_qualitative_selection import select_cases
from experiments.deformnext_retargeting.prepare_final_manuscript_inputs import (
    qualitative_inputs,supplementary_tables,validate_exports,widen_main_table,
)
from test_final_main_paper_export import reports
from test_final_qualitative_export import metadata,inputs,set_delta


def fixture(tmp_path):
    eight,ten,kwargs=reports(tmp_path)
    parent,controls=kwargs['parent'],kwargs['controls']
    declared=metadata(parent['population'])[1]
    specs=dict(kwargs['inputs'],selection_plan={'path':'/fixture/declaration.json'},
        eight_metrics={'path':'/fixture/eight.json'},ten_metrics={'path':'/fixture/ten.json'})
    main=main_export.build(eight,ten,parent=parent,controls=controls,inputs=specs)
    main['source_inputs']=deepcopy(specs)
    selection=select_cases(declared=declared,population=parent['population'],selected_alias=parent['selected_alias'],
        models=eight['models'],statistics=eight['statistics'])
    figures=[dict(sample_id=r['sample_id'],categories=r['categories'],outputs=[f'example-{i:02d}.pdf',f'example-{i:02d}.png'])
        for i,r in enumerate(selection['cases'],1)]
    lookup={a:{(r['sample_id'],r['camera_id']):r for r in rows} for a,rows in eight['scored_image_observations'].items()}
    observed=[dict(alias=a,**lookup[a][r['sample_id'],cid]) for r in selection['cases'] for a in eight['models'] for cid in [8,9,10]]
    qual=dict(status=qual_export.STATUS,figures=figures,source_inputs=deepcopy(specs),exact_original_float_and_official_uint8_images=True,
        shared_GT_RGB_and_alpha_exact=True,model_specific_crop_alignment_or_new_render=False,CPU_model_forward_or_training_updates=0,
        actual_scored_image_observations=observed)
    return eight,ten,main,qual,selection,dict(parent=parent,controls=controls,declared=declared)


def test_full_width_main_is_reversible_and_keeps_every_cell_caption_and_signed_kid(tmp_path):
    eight,ten,main,qual,selection,kwargs=fixture(tmp_path)
    before=deepcopy((eight,ten,main,qual,selection))
    original=validate_exports(eight,ten,main,qual,selection,**kwargs)['tex']
    text=widen_main_table(original)
    assert r'\begin{table*}' in text and r'\resizebox{\textwidth}' in text
    assert text.replace(r'\begin{table*}',r'\begin{table}').replace(r'\end{table*}',r'\end{table}').replace(
        r'\resizebox{\textwidth}',r'\resizebox{\linewidth}')==original
    assert '-0.001' in text
    assert (eight,ten,main,qual,selection)==before
    with pytest.raises(ValueError):widen_main_table(text)


@pytest.mark.parametrize('problem',['unfinished-ten','changed-main','changed-interval','different-report','different-pixel',
    'rerender','CPU-forward','selection-changed'])
def test_incomplete_or_relabelled_exports_cannot_supply_final_inputs(tmp_path,problem):
    eight,ten,main,qual,selection,kwargs=deepcopy(fixture(tmp_path))
    if problem=='unfinished-ten':ten['all_ten_metrics_collected']=False
    elif problem=='changed-main':main['unrounded_primary_table']['V1']['LPIPS']+=.1
    elif problem=='changed-interval':main['comparisons']['SELECTED_minus_V1']['CD']['interpretation']='benefit_supported'
    elif problem=='different-report':qual['source_inputs']['ten_metrics']['path']='/fixture/other.json'
    elif problem=='different-pixel':qual['actual_scored_image_observations'][0]['prediction_float_RGB_SHA256']='wrong'
    elif problem=='rerender':qual['model_specific_crop_alignment_or_new_render']=True
    elif problem=='CPU-forward':qual['CPU_model_forward_or_training_updates']=1
    else:selection['cases'][0]['categories']=['picked-after-viewing']
    with pytest.raises(ValueError):validate_exports(eight,ten,main,qual,selection,**kwargs)


def test_existing_supplementary_means_and_all16_intervals_are_read_only(tmp_path):
    eight,ten,main,qual,selection,kwargs=fixture(tmp_path);before=deepcopy((eight,main))
    files=supplementary_tables(eight,main)
    assert set(files)=={'foreground-sensitivity.tex','adjusted-contrasts.tex'}
    assert files['adjusted-contrasts.tex'].count('Selected minus ')==16
    assert 'do not isolate a PBR effect' in files['foreground-sensitivity.tex']
    assert 'does not prove equivalence' in files['adjusted-contrasts.tex']
    assert all('FinalReportReady' not in text for text in files.values())
    assert (eight,main)==before
    eight['statistics']['roles']['source9_10']['means']['V1']['deformation_weighted']['DreamSim_GTcrop']+=.01
    with pytest.raises(ValueError):supplementary_tables(eight,main)


@pytest.mark.parametrize('problem',[None,'reordered','missing','renamed'])
def test_first_original_blind_stays_main_and_every_other_panel_stays_supplement(tmp_path,problem):
    eight,ten,main,qual,selection,kwargs=fixture(tmp_path);figures=deepcopy(qual['figures'])
    if problem=='reordered':figures[:2]=reversed(figures[:2])
    elif problem=='missing':figures.pop()
    elif problem=='renamed':figures[0]['outputs'][0]='better-example.pdf'
    if problem is not None:
        with pytest.raises(ValueError):qualitative_inputs(selection,figures)
    else:
        documents,assets=qualitative_inputs(selection,figures)
        assert 'example-01.pdf' in documents['qualitative-main.tex']
        assert 'example-02.pdf' not in documents['qualitative-main.tex']
        assert 'example-02.pdf' in documents['qualitative-supplement.tex']
        assert all(assets.count(name)==1 for name in assets) and len(assets)==2*len(figures)
        assert not any('FinalReportReady' in text for text in documents.values())


def test_empty_categories_stay_empty_and_a_main_overlap_is_referenced_without_duplicating():
    args=inputs();empty=select_cases(**args)
    def figures(selection):
        return [dict(sample_id=r['sample_id'],categories=r['categories'],outputs=[f'example-{i:02d}.pdf',f'example-{i:02d}.png'])
            for i,r in enumerate(selection['cases'],1)]
    documents,assets=qualitative_inputs(empty,figures(empty))
    assert documents['qualitative-supplement.tex'].count('No eligible example')==3 and len(assets)==4
    sid=empty['outcome_blind_examples'][0]['sample_id'];index=args['population']['sample_ids'].index(sid)
    set_delta(args,index,CD=.01,FScore=-.01,LPIPS=.01,DreamSim=.01)
    overlap=select_cases(**args);documents,assets=qualitative_inputs(overlap,figures(overlap))
    assert len(assets)==4 and 'example-01.pdf' not in documents['qualitative-supplement.tex']
    assert r'\ref{fig:final-qualitative-01}' in documents['qualitative-supplement.tex']
