"""Source-bound final Main table and bounded comparison evidence.

All quantities come from the actual frozen TEST508 ten-metric collector.
No model execution, metric substitution, model choice or bootstrap is run.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

from .cascade_view_roles import ClosedInputs, file_sha256 as sha
from .collect_module_seed_confirmation import read_file
from .export_final_qualitative import report_admission, required_sources as qualitative_sources
from .final_qualitative_selection import validate_declaration
from .selected_final_metric_summary import PAIRWISE_ROLES, validate_contract
from .selected_final_population import compose_models
from .selected_pipeline_freeze_v2 import METRIC_DIRECTIONS, exact
from .selected_pipeline_model_contract import require


SCHEMA = 'final-main-paper-table-closed-inputs-v1'
STATUS = 'PASS_ACTUAL_FROZEN_TEST508_MAIN_TEX_AND_COMPARISON_EVIDENCE'
FIELDS = ('LPIPS','DreamSim','CLIP','FID','KID','PSNR','SSIM','M_IoU','CD','FScore')
FORMATS = dict(LPIPS='.6f',DreamSim='.6f',CLIP='.6f',FID='.6f',KID='.8g',
    PSNR='.4f',SSIM='.6f',M_IoU='.6f',CD='.7f',FScore='.6f')
LABELS = {'zero-source4':'Zero-shot, source-only RGB4',
    'zero-edited4':'Zero-shot, edited target + source RGB3',
    'V1':'Vanilla V1','RETARGET':'Top-1 R','SELECTED':'Selected model'}


def required_sources():
    return sorted(set(qualitative_sources()) | {str(Path(__file__).resolve())})


def build(eight, ten, *, parent, controls, inputs):
    models = compose_models(parent, controls); contract = validate_contract(parent)
    report_admission(eight,ten,parent=parent,models=models,contract=contract,inputs=inputs)
    table = ten['primary_ten_metric_table']; ids = parent['population']['sample_ids']
    require(set(table)==set(models) and all(set(row)==set(FIELDS) for row in table.values()),
        'all ten original final table fields and every deduplicated model required')
    for alias,row in table.items():
        require(all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in row.values()),
            'finite actual reported values required; no placeholder cells')
        for metric,role in PAIRWISE_ROLES.items():
            records = eight['statistics']['per_asset'][role][alias]
            require([r['sample_id'] for r in records]==ids, 'the complete original508 rows required for camera-role means')
            mean = math.fsum(r[metric] for r in records)/508
            reported = eight['statistics']['roles'][role]['means'][alias]['deformation_weighted'][metric]
            require(math.isclose(row[metric],mean,rel_tol=0,abs_tol=1e-12)
                and math.isclose(row[metric],reported,rel_tol=0,abs_tol=1e-12),
                'use original deformation-weighted source9/10, target8 and surface means')
        distribution = ten['distribution_metrics'][alias]
        require(distribution['images_per_pool']==1524 and distribution['feature_dimension']==2048
            and distribution['statistic_dtype']=='float64' and distribution['unbiased_KID_clamped'] is False
            and distribution['KID_subset_std_is_not_CI'] is True
            and row['FID']==distribution['frechet_inception_distance']
            and row['KID']==distribution['kernel_inception_distance_mean'],
            'retain raw signed KID/FID and the original official complete image population')
    comparisons = ten['primary_pairwise_family_intervals']
    require(exact(comparisons,eight['primary_pairwise_family_intervals']),
        'the final table must retain the actual eight-metric adjusted interval record')
    pairs = contract['primary_contrasts']; expected = [a+'_minus_'+b for a,b in pairs]
    require(set(comparisons)==set(expected) and contract['pairwise_primary_family_size']==8*len(pairs),
        'only the actual predeclared selected-versus-baseline family is used')
    evidence = {}
    for a,b in pairs:
        key = a+'_minus_'+b; report = comparisons[key]
        require(report['endpoint_family_size']==contract['pairwise_primary_family_size']
            and report['source_clusters']==134
            and set(report['deformation_weighted_family_CI95'])==set(PAIRWISE_ROLES)
            and set(report['deformation_weighted_delta'])==set(PAIRWISE_ROLES),
            'all eight adjusted source-cluster intervals and their declared family required')
        evidence[key] = {}
        for metric in PAIRWISE_ROLES:
            delta = report['deformation_weighted_delta'][metric]
            lo,hi = report['deformation_weighted_family_CI95'][metric]
            require(all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in [delta,lo,hi])
                and lo<=hi and math.isclose(delta,table[a][metric]-table[b][metric],rel_tol=0,abs_tol=1e-12),
                'actual paired differences and finite ordered family intervals required')
            lower = METRIC_DIRECTIONS[metric]=='lower'
            benefit = hi<0 if lower else lo>0
            harm = lo>0 if lower else hi<0
            evidence[key][metric] = dict(delta=delta,family_CI95=[lo,hi],
                direction=METRIC_DIRECTIONS[metric],scope=PAIRWISE_ROLES[metric],
                interpretation='benefit_supported' if benefit else 'harm_supported' if harm else 'uncertain_not_equivalent')
        for metric in ['FID','KID']:
            evidence[key][metric] = dict(delta=table[a][metric]-table[b][metric],
                family_CI95=None,direction='lower',scope='global_all_three_cameras',interpretation='descriptive_only')
    formatted = {alias:{m:format(table[alias][m],FORMATS[m]) for m in FIELDS} for alias in models}
    lines = [r'\begin{table}[t]',r'\centering\scriptsize',r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{lrrrrrrrrrr}',r'\toprule',
        r'&\multicolumn{3}{c}{Source preservation}&\multicolumn{2}{c}{Global naturalness}&\multicolumn{5}{c}{Target alignment}\\',
        r'Model&LPIPS$\downarrow$&DreamSim$\downarrow$&CLIP$\uparrow$&FID$\downarrow$&KID$\downarrow$&PSNR$\uparrow$&SSIM$\uparrow$&M-IoU$\uparrow$&CD$\downarrow$&F-Score$\uparrow$\\',r'\midrule']
    for alias in models:
        lines.append(LABELS[alias]+'&'+'&'.join(formatted[alias][m] for m in FIELDS)+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}}',
        r'\caption{Actual frozen TEST508 results from134 source clusters. Source LPIPS/DreamSim/CLIP average cameras9/10; target PSNR/SSIM/2D mask IoU use camera8. Surface CD is symmetric unsquared Euclidean distance; F-Score uses threshold0.02 in the common normalized frame. Official Inception2048 FID/KID use the same1524-image prediction/reference pools. KID is signed and shown in raw units, without clipping or rescaling. References use original materials on edited M1 geometry. These are input-camera evaluations. The selected physical model is '+parent['selected_alias']+r'; a merged baseline is not repeated as another model.}',
        r'\label{tab:final-main}',r'\end{table}'])
    notes = ('Main table values use the original camera-role means and raw distribution estimates. '
        'The paired family conditions on the frozen checkpoints and generation protocol; it does not quantify training-seed variability. '
        'Intervals containing zero do not establish equivalence or non-inferiority. '
        'FID/KID differences are descriptive; KID subset variability is not a source-cluster interval. '
        'Geometry and source appearance require separate interpretation. '
        'This export does not establish individual module necessity, universal superiority, or completion of manuscript writing.\n')
    return dict(status=STATUS,models=list(models),selected_alias=parent['selected_alias'],
        unrounded_primary_table=table,formatted_table=formatted,comparisons=evidence,
        pairwise_primary_family_size=contract['pairwise_primary_family_size'],
        tex='\n'.join(lines)+'\n',writing_scope_notes=notes,full_goal_complete=False)


def export(path):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','explicit CPU-only final paper export required')
    path=Path(path);context=json.loads(path.read_text())
    require(context.get('schema')==SCHEMA and context.get('repo')==str(Path(__file__).resolve().parents[2])
        and set(required_sources())<=set(context['source_SHA256']), 'complete actual paper exporter context/source binding required')
    affinity=context['CPU_affinity'];require(bool(affinity) and all(type(c) is int for c in affinity)
        and set(affinity)<=os.sched_getaffinity(0),'assigned available metadata cores required')
    os.sched_setaffinity(0,set(affinity));os.nice(10)
    base=Path(context['canonical_local_root']);check=ClosedInputs(base);check.check(path,path.parent/'file4-closure.json')
    for p,d in context['source_SHA256'].items():require(sha(p)==d,'paper exporter source changed')
    inputs=context['source_inputs'];parent=read_file(inputs['parent_freeze'],check);controls=read_file(inputs['control_freeze'],check)
    compose_models(parent,controls);validate_contract(parent)
    require(all(parent['evaluation_protocol']['source_SHA256'].get(p)==d for p,d in context['source_SHA256'].items())
        and exact(parent['evaluation_protocol'].get('qualitative_selection_plan'),inputs['selection_plan']),
        'freeze the original declaration and actual paper/figure source bytes before TEST')
    declared=read_file(inputs['selection_plan'],check);validate_declaration(declared,parent['population'])
    eight=read_file(inputs['eight_metrics'],check);ten=read_file(inputs['ten_metrics'],check)
    result=build(eight,ten,parent=parent,controls=controls,inputs=inputs)
    output=Path(context['output'])
    require(output.is_absolute() and output!=base and output.resolve()==output and output.is_relative_to(base)
        and not output.exists(),'new owned canonical paper export directory required')
    output.mkdir(parents=True)
    (output/'main-table.tex').write_text(result['tex']);(output/'writing-scope-notes.txt').write_text(result['writing_scope_notes'])
    (output/'comparison-evidence.json').write_text(json.dumps(result['comparisons'],indent=2,allow_nan=False)+'\n')
    for p,d in context['source_SHA256'].items():require(sha(p)==d,'source changed during paper export')
    for p,d in check.inputs.items():require(sha(p)==d,'closed input changed during paper export')
    audit={k:v for k,v in result.items() if k not in ['tex','writing_scope_notes']}
    audit.update(source_inputs=inputs,source_SHA256=context['source_SHA256'],verified_inputs=check.inputs,
        original_per_image_metrics_not_recomputed=True,camera_role_means_checked_without_replacing_values=True,
        neural_forward_or_training_updates=0,automatic_model_adoption=False,manuscript_writing_complete=False)
    (output/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    manifest=Path(context['manifest_directory'])/(context['name']+'-files.json');require(not manifest.exists(),'new owned closure manifest required')
    manifest.write_text(json.dumps([str(p.relative_to(base)) for p in sorted(output.iterdir()) if p.is_file()],indent=2)+'\n')
    subprocess.run([*context['closure_command'],'--manifest',str(manifest),'--receipt',
        str((output/'file4-closure.json').relative_to(base)),
        '--scope','Actual frozen TEST508 Main TeX and exact existing adjusted-interval evidence; no new metric, model selection, causal module or final writing claim'],check=True)
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    result=export(parser.parse_args().plan);print(json.dumps(dict(status=result['status'],models=result['models'])),flush=True)


if __name__=='__main__':
    main()
