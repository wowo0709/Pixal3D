"""Bind closed actual TEST exports into portable manuscript input files.

This is publication metadata/layout only. It never runs a metric/model,
changes model selection, writes result-dependent prose or declares a final
paper ready. The original exporters and their pre-TEST policy stay frozen.
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil

from .cascade_view_roles import ClosedInputs, file_sha256 as sha
from .collect_module_seed_confirmation import read_file
from . import export_final_main_table as main_export
from . import export_final_qualitative as qual_export
from .final_qualitative_selection import select_cases
from .run_selected_final_campaign import close, write
from .selected_campaign_resources import source_guard
from .selected_final_metric_summary import PAIRWISE_ROLES
from .selected_pipeline_freeze_v2 import exact
from .selected_pipeline_model_contract import require


SCHEMA='actual-final-manuscript-input-binding-v1'
STATUS='PASS_ACTUAL_FINAL_MANUSCRIPT_INPUTS_BOUND_WRITING_AND_REVIEW_PENDING'


def widen_main_table(text):
    require(text.count(r'\begin{table}')==text.count(r'\end{table}')==1
        and text.count(r'\resizebox{\linewidth}')==1,'one exact original Main table float and dimension required')
    new=text.replace(r'\begin{table}',r'\begin{table*}').replace(r'\end{table}',r'\end{table*}')
    new=new.replace(r'\resizebox{\linewidth}',r'\resizebox{\textwidth}')
    require(new.replace(r'\begin{table*}',r'\begin{table}').replace(r'\end{table*}',r'\end{table}')
        .replace(r'\resizebox{\textwidth}',r'\resizebox{\linewidth}')==text,
        'every Main cell, caption, label and numerical unit must remain identical')
    return new


def supplementary_tables(eight,main):
    """Display existing means and adjusted intervals; no new metric/bootstrap."""
    means={};ids=[r['sample_id'] for r in eight['statistics']['per_asset']['source9_10'][main['models'][0]]]
    require(len(ids)==508,'full508 existing role rows required')
    fields=['DreamSim','DreamSim_GTcrop','CLIP','CLIP_GTcrop','PSNR','foreground_PSNR']
    for alias in main['models']:
        means[alias]={}
        for role in ['source9_10','target8']:
            rows=eight['statistics']['per_asset'][role][alias]
            reported=eight['statistics']['roles'][role]['means'][alias]['deformation_weighted']
            require([r['sample_id'] for r in rows]==ids,'same ordered508 supplementary camera-role rows required')
            for metric in fields:
                value=reported[metric]
                require(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
                    and math.isclose(value,math.fsum(r[metric] for r in rows)/508,rel_tol=0,abs_tol=1e-12),
                    'supplementary means must equal the original producer values')
            means[alias][role]=deepcopy(reported)
    lines=[r'\begin{table*}[t]',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{lrrrr}',r'\toprule',
        r'\multicolumn{5}{l}{A. Source appearance: full frame and common GT-defined crop}\\',
        r'Model&DreamSim full$\downarrow$&DreamSim crop$\downarrow$&CLIP full$\uparrow$&CLIP crop$\uparrow$\\',r'\midrule']
    for alias in main['models']:
        src=means[alias]['source9_10']
        lines.append(main_export.LABELS[alias]+'&'+'&'.join(format(src[k],'.6f') for k in fields[:4])+r'\\')
    lines.extend([r'\midrule',r'\multicolumn{5}{l}{B. Pixel alignment: full frame and alpha-union foreground}\\',
        r'Model&\shortstack{Source PSNR\\full$\uparrow$}&\shortstack{Source PSNR\\foreground$\uparrow$}&\shortstack{Target PSNR\\full$\uparrow$}&\shortstack{Target PSNR\\foreground$\uparrow$}\\',r'\midrule'])
    for alias in main['models']:
        src,tgt=means[alias]['source9_10'],means[alias]['target8']
        values=[src['PSNR'],src['foreground_PSNR'],tgt['PSNR'],tgt['foreground_PSNR']]
        lines.append(main_export.LABELS[alias]+'&'+'&'.join(format(v,'.5f') for v in values)+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}',
        r'\caption{Existing supplementary scores for the complete frozen TEST508 pipelines. Source means use cameras9/10; target means use camera8. Crops use the common GT-alpha bounding box with10\% padding. Foreground PSNR uses the union of predicted/GT alpha$>0.5$ and can have output-specific support. These descriptive composed-pipeline sensitivities do not isolate a PBR effect, change the primary family, or reselect a checkpoint.}',
        r'\label{tab:final-foreground}',r'\end{table*}'])
    contrast=[r'\begin{table*}[t]',r'\centering\small',r'\begin{tabular}{llrrl}',r'\toprule',
        r'Contrast&Endpoint&Difference&Family-adjusted95\% interval&Interpretation\\',r'\midrule']
    count=0
    for name,values in main['comparisons'].items():
        baseline=name.split('_minus_',1)[1]
        label='Selected minus '+main_export.LABELS[baseline]
        for metric in PAIRWISE_ROLES:
            row=values[metric];lo,hi=row['family_CI95'];fmt=main_export.FORMATS[metric]
            status={'benefit_supported':'benefit supported','harm_supported':'harm supported',
                'uncertain_not_equivalent':'uncertain'}[row['interpretation']]
            endpoint={'M_IoU':'M-IoU','FScore':'F-Score'}.get(metric,metric)
            contrast.append(label+'&'+endpoint+'&'+format(row['delta'],fmt)+'&$['+
                format(lo,fmt)+','+format(hi,fmt)+']$&'+status+r'\\');count+=1
    require(count==main['pairwise_primary_family_size'],'all original adjusted primary endpoints required')
    contrast.extend([r'\bottomrule',r'\end{tabular}',
        r'\caption{All predeclared selected-versus-control primary comparisons, using the existing source-cluster-adjusted intervals. Source LPIPS/DreamSim/CLIP use9/10; target PSNR/SSIM/M-IoU use8; CD/F-Score use generated surfaces. Values are rounded for display; interpretation uses the original unrounded intervals and metric directions. An interval containing zero does not prove equivalence. FID/KID remain descriptive and are not included in this inferential family.}',
        r'\label{tab:final-adjusted-contrasts}',r'\end{table*}'])
    return {'foreground-sensitivity.tex':'\n'.join(lines)+'\n','adjusted-contrasts.tex':'\n'.join(contrast)+'\n'}


def qualitative_inputs(selection,figures):
    expected=[(r['sample_id'],r['categories']) for r in selection['cases']]
    require([(r['sample_id'],r['categories']) for r in figures]==expected
        and len(expected)>=2 and [r['sample_id'] for r in figures[:2]]==
            [r['sample_id'] for r in selection['outcome_blind_examples']],
        'both original outcome-blind panels and every unique conditional panel must retain order')
    blocks=[];assets=[];numbers={}
    for number,figure in enumerate(figures,1):
        stem=f'example-{number:02d}'
        require(figure['outputs']==[stem+'.pdf',stem+'.png'],'exact original uncropped figure filenames required')
        assets.extend(figure['outputs']);numbers[figure['sample_id']]=number
        category=', '.join(figure['categories'])
        blocks.append('\n'.join([r'\begin{figure*}[p]',r'\centering',
            r'\includegraphics[width=\textwidth]{final-results/figures/'+stem+'.pdf}',
            r'\caption{Actual scored frozen TEST example. Rows are target8/source9/source10; columns are the common original-material M1 reference and every frozen model. Selection: '+category+
            r'. Full-frame panels use the scored archive and official uint8 conversion, without model-specific crop, alignment or rerender. Source annotations use cameras9/10. Conditional categories illustrate their stated conditions and do not estimate population success rates.}',
            r'\label{fig:final-qualitative-'+f'{number:02d}'+'}',r'\end{figure*}'])+'\n')
    notes=[r'\paragraph{Predeclared conditional eligibility.}']
    for name,row in selection['categories'].items():
        sid=row['selected_sample_id']
        selected=('No eligible example; no replacement was introduced.' if sid is None else
            r'The selected lower-median example appears in Fig.~\ref{fig:final-qualitative-'+f'{numbers[sid]:02d}'+'}.')
        notes.append(name+': '+str(row['eligible_cases'])+' eligible. '+selected)
    notes.append('Overlapping conditions share one panel. The first outcome-blind panel stays in the Main text even when it also satisfies a conditional category; it is referenced here rather than duplicated.')
    return {'qualitative-main.tex':blocks[0],
        'qualitative-supplement.tex':'\n'.join(notes)+'\n'+''.join(blocks[1:])},assets


def validate_exports(eight,ten,main,qual,selection,*,parent,controls,declared):
    inputs=main['source_inputs'];expected=main_export.build(eight,ten,parent=parent,controls=controls,inputs=inputs)
    require(main.get('status')==main_export.STATUS and qual.get('status')==qual_export.STATUS,
        'actual completed original Main and qualitative exports required')
    for key in ['models','selected_alias','unrounded_primary_table','formatted_table','comparisons','pairwise_primary_family_size']:
        require(exact(main.get(key),expected[key]),'actual Main export changed: '+key)
    require(all(exact(qual['source_inputs'].get(k),inputs[k]) for k in
        ['parent_freeze','control_freeze','selection_plan','eight_metrics','ten_metrics'])
        and qual.get('exact_original_float_and_official_uint8_images') is True
        and qual.get('shared_GT_RGB_and_alpha_exact') is True
        and qual.get('model_specific_crop_alignment_or_new_render') is False
        and qual.get('CPU_model_forward_or_training_updates')==0,
        'same complete frozen report and exact original scored figures required')
    selected=select_cases(declared=declared,population=parent['population'],selected_alias=parent['selected_alias'],
        models=eight['models'],statistics=eight['statistics'])
    require(exact(selection,selected),'original predeclared selection cannot be changed for publication')
    index=qual_export.report_admission(eight,ten,parent=parent,models=eight['models'],contract=eight['metric_contract'],inputs=inputs)
    observed=qual['actual_scored_image_observations'];expected_observed=[]
    for case in selection['cases']:
        for alias in eight['models']:
            for cid in [8,9,10]:expected_observed.append(dict(alias=alias,**index[alias][case['sample_id'],cid]))
    require(exact(observed,expected_observed),'every displayed figure must retain its actual scored image observation')
    return expected


def prepare(path):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','explicit CPU-only publication metadata preparation required')
    path=Path(path);context=json.loads(path.read_text())
    require(context.get('schema')==SCHEMA and context['repo']==str(Path(__file__).resolve().parents[2]),
        'actual final publication binding context and checkout required')
    os.sched_setaffinity(0,set(context['CPU_affinity']));os.nice(10);source_guard(context)
    require(len(context['source_SHA256'])==505 and str(Path(__file__).resolve()) in context['operational_source_SHA256']
        and all(sha(p)==d for p,d in context['operational_source_SHA256'].items()),'original505 and separate actual publication source pins required')
    base=Path(context['canonical_local_root']);check=ClosedInputs(base);check.check(path,path.parent/'file4-closure.json')
    main=read_file(context['source_inputs']['main_export'],check);qual=read_file(context['source_inputs']['qualitative_export'],check)
    for report,module in [(main,main_export),(qual,qual_export)]:
        require(exact(report.get('source_SHA256'),{p:context['source_SHA256'][p] for p in module.required_sources()}),
            'actual original completed exporter source binding required')
    parent_inputs=qual['source_inputs'];required=qual_export.required_sources()
    parent,controls,models,contract,declared=qual_export.frozen_admission(dict(schema=qual_export.SCHEMA,
        source_SHA256={p:context['source_SHA256'][p] for p in required},source_inputs=parent_inputs),check)
    require(all(parent['evaluation_protocol']['source_SHA256'].get(p)==d for p,d in context['source_SHA256'].items()
        if p!=context['closure_command'][1]),'all numerical/exporter sources must retain their pre-TEST bytes')
    eight=read_file(main['source_inputs']['eight_metrics'],check);ten=read_file(main['source_inputs']['ten_metrics'],check)
    qualroot=Path(context['source_inputs']['qualitative_export']['path']).parent
    selection=read_file(dict(path=str(qualroot/'selection.json'),SHA256=sha(qualroot/'selection.json'),receipt=str(qualroot/'file4-closure.json')),check)
    expected=validate_exports(eight,ten,main,qual,selection,parent=parent,controls=controls,declared=declared)
    mainpath=Path(context['source_inputs']['main_export']['path']).parent/'main-table.tex'
    check.check(mainpath,mainpath.parent/'file4-closure.json');require(mainpath.read_text()==expected['tex'],'exact original Main TeX required')
    documents={'main-table.tex':widen_main_table(mainpath.read_text()),**supplementary_tables(eight,main)}
    qualitative,assets=qualitative_inputs(selection,qual['figures']);documents.update(qualitative)
    for name in assets:check.check(qualroot/name,qualroot/'file4-closure.json')
    output=Path(context['output']);require(output.is_absolute() and output.resolve()==output and output!=base
        and output.is_relative_to(base) and not output.exists(),'new external publication input bundle required')
    figures=output/'final-results/figures';figures.mkdir(parents=True,exist_ok=False)
    for name,text in documents.items():(figures.parent/name).write_text(text)
    for name in assets:
        shutil.copyfile(qualroot/name,figures/name);require(sha(figures/name)==sha(qualroot/name),'figure copying must preserve actual exported bytes')
    evidence=dict(unrounded_primary_table=main['unrounded_primary_table'],comparisons=main['comparisons'],
        original_selection=selection,manuscript_writing_complete=False,visual_review_complete=False,full_goal_complete=False)
    write(output/'writing-evidence.json',evidence)
    source_guard(context)
    for p,d in context['operational_source_SHA256'].items():require(sha(p)==d,'publication metadata source changed')
    for p,d in check.inputs.items():require(sha(p)==d,'actual closed publication input changed during binding')
    audit=dict(status=STATUS,source_inputs=context['source_inputs'],actual_models=list(models),assets=508,source_clusters=134,
        actual_primary_cells=len(models)*10,actual_adjusted_intervals=main['pairwise_primary_family_size'],
        actual_qualitative_panels=len(qual['figures']),actual_copied_figure_SHA256={n:sha(figures/n) for n in assets},
        Main_layout_change='table->table* and linewidth->textwidth only; invertible, every cell/caption unchanged',
        actual_documents=list(documents),missing_result_dependent_sections=['abstract-findings.tex','results.tex','discussion.tex','conclusion.tex','provenance.tex'],
        final_flag_not_written=True,original505_unchanged=True,no_metric_or_model_forward_or_new_bootstrap=True,
        no_TEST_model_reselection_or_qualitative_reselection=True,manuscript_writing_complete=False,visual_review_complete=False,
        verified_inputs=check.inputs,full_goal_complete=False)
    write(output/'audit.json',audit);close(output,context,'manuscript-inputs');return audit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True,type=Path)
    print(json.dumps(prepare(p.parse_args().plan)),flush=True)


if __name__=='__main__':main()
