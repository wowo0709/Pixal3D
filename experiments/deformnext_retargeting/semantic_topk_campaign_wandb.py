"""Publish completed SS model-search evidence, including conditional replication.

Logging this report never marks the persistent model-search goal complete.
Incomplete development arms or selected-model confirmation are errors, not
silently omitted methods. Model binaries remain on canonical file4.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from .firstwave_wandb import ORDER as FIRST_ORDER, PROJECT, read, sha, near
from .semantic_topk_search import write_new
from .semantic_topk_candidates import candidate_source
from .semantic_topk_zero_replay import MODES, zero_paths, validate_zero
from .summarize_semantic_topk_search import MODELS, aggregate, validate_evaluation


RUN_ID = 'ss-semantic-topk-campaign-v1'
NEW_MODELS = [*MODELS, 'static-top1-c1']
ORDER = [*FIRST_ORDER, *[name for name in NEW_MODELS if name not in FIRST_ORDER]]
LABELS = dict(zip(FIRST_ORDER, ['Zero source4', 'Zero edited+3', 'V1', 'Top1',
                              'Soft T0.02', 'B3 T0.005', 'C1 T0.005']))
LABELS.update({'static-top1-c1':'Top1+C1', 'static-c1':'Static C1 T0.05',
               'task-full':'Task full', 'task-no-decoder':'Task no decoder',
               'task-no-prior':'Task no score prior', 'task-native-spatial':'Task native spatial'})
COMPONENTS = {
    'TopK weighting / no retargeter':'soft-topk-vs-top1-reference',
    'C1 retargeter / matching-score Top1 held fixed':'static-top1-c1-vs-top1-reference',
    'TopK weighting / same C1 retargeter':'static-c1-vs-static-top1-c1',
    'Learned matching path / same C1 retargeter':'task-full-vs-static-c1',
    'Matching decoder / physical D0 control':'task-full-vs-task-no-decoder',
    'Scalar score prior / decoder retained':'task-full-vs-task-no-prior',
    'Semantic64 versus native spatial3 / different candidate counts':'task-full-vs-task-native-spatial',
    'Previous C1 T0.005 versus matched Top1+C1':'previous-c1-vs-static-top1-c1',
}


def require(path, status, files):
    value = read(path)
    if value.get('status') != status:
        raise ValueError('completed evidence required: '+str(path))
    files.append(path)
    return value


def validate_noise_identity(records, assets, clusters):
    """Bind every model/noise to the same ordered cases and actual 64-cube GT."""
    if not records or any(len(es) != 3 for es in records.values()):
        raise ValueError('three actual sampling noises required for every method')
    reference = next(iter(records.values()))
    ids = [r['sample_id'] for r in reference[0]['per_asset']]
    if len(ids) != assets or len(set(ids)) != assets:
        raise ValueError('complete distinct panel required')
    sources = {r['source_cluster_id'] for r in reference[0]['per_asset']}
    if len(sources) != clusters:
        raise ValueError('source cluster population differs')
    if any([r['sample_id'] for r in e['per_asset']] != ids for es in records.values() for e in es):
        raise ValueError('original panel order differs')
    for ix in range(assets):
        baseline = reference[0]['per_asset'][ix]
        if len({e['per_asset'][ix]['seed'] for e in reference}) != 3:
            raise ValueError('three distinct sampling seeds required per case')
        for es in records.values():
            for noise, e in enumerate(es):
                row = e['per_asset'][ix]
                if (not row.get('target_voxels_sha256') or row.get('target_voxel_shape') != [64,64,64]
                    or any(row[k] != baseline[k] for k in ['source_cluster_id','target_voxels_sha256'])
                    or row['tp']+row['fn'] != baseline['tp']+baseline['fn']
                    or row['seed'] != reference[noise]['per_asset'][ix]['seed']):
                    raise ValueError('actual GT/source/noise identity differs')
    return ids


def conditional_scope(candidate, confirmation=None, additional=None, correspondence=None):
    """A PASS computation may still give negative scientific evidence."""
    if candidate is None:
        if any(value is not None for value in [confirmation, additional, correspondence]):
            raise ValueError('unselected model cannot have selected-model confirmation')
        return {'selected_candidate':None, 'seed_confirmation_completed':False,
                'additional121_completed':False, 'positive_in_both_training_seeds':False,
                'positive_on_additional121_both_seeds':False,
                'same_retargeter_Top1_replication_across_training_seeds':False,
                'full_campaign_complete':False, 'model_adopted':False}
    statuses = ['PASS_FIXED_FINAL_SEED202_CONFIRMATION',
                'PASS_ADDITIONAL121_FIXED_SELECTED_MODEL_VALIDATION',
                'PASS_SELECTED_MODEL_SAME_POOL_VALUE_ADDRESS_DIAGNOSTIC']
    for value, status in zip([confirmation, additional, correspondence], statuses):
        if value is None or value.get('status') != status or value.get('selected_candidate') != candidate:
            raise ValueError('selected-model confirmation/additional/correspondence is incomplete or differs')
    return {'selected_candidate':candidate, 'seed_confirmation_completed':True,
            'additional121_completed':True,
            'positive_in_both_training_seeds':confirmation['positive_CI_and_all_noise_direction_at_both_seeds'],
            'positive_on_additional121_both_seeds':additional['positive_CI_and_all_noises_on_both_additional_training_seeds'],
            'same_retargeter_Top1_replication_across_training_seeds':False,
            'full_campaign_complete':False, 'model_adopted':False}


def validate_saved_qa(qa,spec):
    expected=('PASS_CURRENT_C1_CACHE_AMP_CFG_CONTRACT' if not spec.get('task_attention') and spec['arm']=='c1'
              else 'PASS_TRAINED_SS_CACHE_AMP_CFG_CONTRACT')
    if qa.get('status')!=expected:
        raise ValueError('saved-model QA schema differs from actual conditioner recipe')
    if (qa['optimizer_updates'] != 0 or not qa['actual_four_RGB_cameras_exact']
        or not qa['p1_all_branches_exact_zero'] or len(qa['sample_ids']) != 2):
        raise ValueError('actual trained RGB/cache/CFG evidence differs')


def proof_files(root, job, name, seed, files):
    recipe = root/'final-weight-audits-v1'/job/'final-recipe-snapshot.json'
    cfg=read(recipe)
    qapath=root/'functional-QA-v1'/f's{seed}-{name}'/'summary.json'
    validate_saved_qa(read(qapath),cfg['trainer']['args']['deform_conditioner']);files.append(qapath)
    weights = require(root/'final-weight-audits-v1'/job/'audit.json',
                      'PASS_ACTUAL_FINAL_TENSORS_AND_OPTIMIZER', files)
    if cfg['training_seed'] != seed:
        raise ValueError('actual optimizer training seed differs')
    files.append(recipe)
    if (not weights['all1061_denoiser_tensors_finite'] or weights['frozen_base_tensors_exact'] != 701
        or weights['LoRA_tensors_updated'] != 360 or weights['actual_optimizer_step'] != 480
        or weights['physical_deformation_query_channels'] != 0):
        raise ValueError('actual final saved weights differ')
    require(root/'final-weight-audits-v1'/job/'file4-closure.json', 'PASS', files)
    for epoch in range(1,6):
        closure = require(root/'epoch-audits-v1'/job/f'epoch{epoch:02d}'/'file4-closure.json', 'PASS', files)
        if closure.get('local_equals_actual_file4_SHA256') is not True:
            raise ValueError('actual canonical file4 epoch closure missing')


def previous_c1_proof_files(root, files):
    """Reuse the original C1 proofs with their actual paths and checkpoint hashes."""
    base,arm=candidate_source(root,'previous-c1');parent=root.parent
    recipe=base/'configs'/(arm+'.json');cfg=read(recipe);files.append(recipe)
    qa_path=parent/'conditioning-contract-v2/summary.json'
    qa=read(qa_path);validate_saved_qa(qa,cfg['trainer']['args']['deform_conditioner']);files.append(qa_path)
    if (cfg['training_seed']!=101 or qa['source_recipe']['path']!=str(recipe)
        or qa['source_recipe']['sha256']!=sha(recipe) or not qa['saved_frozen_head_exact_to_selected_checkpoint']):
        raise ValueError('existing C1 recipe/frozen-head proof differs')
    weights_path=base/'all-three-final-weight-audit-v1.json'
    weights=require(weights_path,'PASS_ALL_THREE_FINAL_MODEL_TENSORS_AND_FROZEN_BASE',files)
    actual=[r for r in weights['actual_final_weights'] if r['arm']=='c1']
    if len(actual)!=1:raise ValueError('existing C1 final weight proof is not unique')
    actual=actual[0]
    if (actual['actual_final_denoiser']!=qa['trained_denoiser'] or actual['actual_conditioner']!=qa['trained_conditioner']
        or not actual['all_denoiser_tensors_finite'] or not actual['all_conditioner_tensors_finite']
        or not actual['all_frozen_base_tensors_dtype_shape_and_values_exact_to_SS4210']
        or actual['frozen_base_tensor_keys']!=701 or actual['lora_tensor_keys_changed']!=360
        or actual['frozen_Q2C_head_keys']!=24 or not weights['B3_C1_all24_frozen_Q2C_tensors_exact']
        or not actual['actual_retargeter_present'] or actual['RAW_deformation_query_channels']!=0):
        raise ValueError('existing C1 actual tensors/checkpoint identity differs')
    epoch_path=base/'epoch-audits-v2/c1/epoch05/audit.json'
    epoch=require(epoch_path,'PASS_EPOCH_CHECKPOINT_AUDIT',files)
    expected={'training_seed':101,'step':480,'all_optimizer_steps':480,'all_optimizer_moments_finite':True,
              'global_batch':96,'world_size':2,'microbatch_per_rank':48,'accumulation':1}
    if any(epoch.get(k)!=v for k,v in expected.items()):raise ValueError('existing C1 actual optimizer differs')
    closures=[(parent/'provenance/current-C1-QA-v2-completed-file4.json',[qa_path]),
        (parent/'provenance/all-three-final-weight-audit-v1-file4.json',[weights_path]),
        (parent/'provenance/matched-all-three-final-quality-and-remote-v1-file4.json',
         [recipe,*[base/'evaluations/c1'/f'noise-{n}/evaluation.json' for n in range(3)]]),
        (base/'epoch-audits-v2/c1/epoch05/file4-closure.json',[epoch_path])]
    for closure_path,paths in closures:
        closure=require(closure_path,'PASS',files)
        if not closure.get('local_equals_actual_file4_SHA256'):raise ValueError('existing C1 canonical file4 proof missing')
        rows={r['relative_path']:r for r in closure['files']}
        for path in paths:
            row=rows.get(str(path.relative_to(parent)))
            if row is None or row['sha256']!=sha(path) or row['bytes']!=path.stat().st_size:
                raise ValueError('existing C1 file4 proof input changed')
            files.append(path)
        if closure_path==closures[-1][0]:
            for ref in [qa['trained_denoiser'],qa['trained_conditioner']]:
                row=rows.get(str(Path(ref['path']).relative_to(parent)))
                if row is None or any(row[k]!=ref[k] for k in ['sha256','bytes']):
                    raise ValueError('existing C1 file4 checkpoint identity differs')


def collect(root):
    """Read all required immutable results before producing any output."""
    from .semantic_topk_selection import selection
    files=[]
    first = require(root/'firstwave-fixed-final-v1/report.json', 'PASS_FIXED_FINAL_FIRST_WAVE_ONLY', files)
    seven = require(root/'seven-arm-final-report.json', 'PASS_SEVEN_ARM_FIXED_FINAL_DEVELOPMENT_SEARCH', files)
    matched = require(root/'matched-retargeter-top1-report.json', 'PASS_MATCHED_RETARGETER_TOP1_FIXED_FINAL', files)
    decision_path = root/'candidate-selection-v1.json';decision=read(decision_path);files.append(decision_path)
    recomputed = selection(seven, matched)
    if any(decision.get(k) != recomputed[k] for k in ['status','selected_candidate','candidate_evidence']):
        raise ValueError('actual fixed-final selection differs')
    for name in ['seven-arm-final-report.json','matched-retargeter-top1-report.json']:
        if decision['actual_fixed_final_report_SHA256'][name] != sha(root/name):
            raise ValueError('selection input changed')
    dose = require(root/'firstwave-fusion-dose-v1/summary.json',
                   'PASS_COMPLETED_SAME_WEIGHT_AGGREGATION_FUSION_DIAGNOSTIC', files)
    value = require(root/'task-value-address-v1/summary.json',
                    'PASS_COMPLETED_TASK_VALUE_ADDRESS_DIAGNOSTIC', files)
    omega=require(root/'omega-backbone-QA-v1/summary.json','PASS_FRESH_OMEGA7_FEATURE_CACHE_PARITY',files)
    if (omega['completed_updates']!=23597 or not omega['strict_base_and_adapter_key_load']
        or not omega['all_parameters_frozen_eval'] or len(omega['cases'])!=2
        or omega['Omega_forward_passes']!=2 or omega['SS_generations']!=0 or omega['optimizer_updates']!=0
        or omega['input_camera_ids']!=[8,9,10,11] or omega['input_states']!=['M1','M0','M0','M0']):
        raise ValueError('actual fresh Omega7 backbone proof differs')
    backbone_metrics={}
    for ix,case in enumerate(omega['cases']):
        if len(case['feature_parity']['comparisons'])!=8 or not case['feature_parity']['pass'] or not case['cached_proposals_exact_to_match_cache']:
            raise ValueError('all per-view FRAME/GLOBAL feature checks required')
        for c in case['feature_parity']['comparisons']:
            if not math.isfinite(c['relative_RMS']) or not math.isfinite(c['mean_cosine']) or c['relative_RMS']>.005 or c['mean_cosine']<.9999:
                raise ValueError('actual fresh Omega7 numerical differences exceed declared limits')
            for key in ['relative_RMS','mean_cosine','max_abs_error']:
                backbone_metrics[f'backbone_QA/case{ix}/view{c["view"]}/{c["component"]}/{key}']=c[key]
        backbone_metrics[f'backbone_QA/case{ix}/proposal_set_overlap']=case['fresh_proposal_set_overlap']
        files.append(root/'omega-backbone-QA-v1'/f'case{ix}.json')
    omega_closure=require(root/'provenance/omega-backbone-QA-v1-file4.json','PASS',files)
    if not omega_closure['local_equals_actual_file4_SHA256']:raise ValueError('actual Omega7 proof file4 closure missing')
    files.append(root/'provenance/omega-selected-weights-v1.json')
    records={};images={};old=root.parent/'ss-seed-confirmation-v1'
    for name in ORDER:
        if name in MODES:
            paths=zero_paths(root,root.parent,name)
        else:
            base,arm=(old,name.removeprefix('previous-')) if name.startswith('previous-') else (root,name)
            paths=[base/'evaluations'/arm/f'noise-{n}'/'evaluation.json' for n in range(3)]
        records[name]=[read(path) for path in paths];files.extend(paths)
        for e in records[name]:
            if name in MODES:validate_zero(e,MODES[name])
            else:validate_evaluation(e)
        panel=paths[0].parent/'panel.png';files.append(panel);images[name]=np.asarray(Image.open(panel).convert('RGB'))
    ids=validate_noise_identity(records,128,93)
    models={name:aggregate(es) for name,es in records.items()}
    for report in [first,seven,matched]:
        for name,model in report['models'].items():
            if model != models[name]:raise ValueError('reported model differs from actual counts')
    for name in NEW_MODELS:proof_files(root,'train-'+name,name,101,files)
    seedroot=root/'seed-confirmation-v1'
    proof_files(root,'confirm-seed202-train-top1-reference','top1-reference',202,files)
    candidate=decision['selected_candidate'];confirmation=additional=correspondence=None
    if candidate is not None:
        confirmation=require(seedroot/'final-confirmation-report.json','PASS_FIXED_FINAL_SEED202_CONFIRMATION',files)
        additional=require(root/'additional-validation-v1/final-report.json',
                           'PASS_ADDITIONAL121_FIXED_SELECTED_MODEL_VALIDATION',files)
        correspondence=require(root/'selected-correspondence-v1/summary.json',
                               'PASS_SELECTED_MODEL_SAME_POOL_VALUE_ADDRESS_DIAGNOSTIC',files)
        if additional['candidate_selection_SHA256']!=sha(decision_path):raise ValueError('additional model selection changed')
        proof_files(root,'confirm-seed202-train-'+candidate,candidate,202,files)
        if candidate=='previous-c1':previous_c1_proof_files(root,files)
    markdown=[root/'firstwave-fixed-final-v1/report.md',root/'seven-arm-final-report.md']
    if candidate is not None:markdown += [seedroot/'final-confirmation-report.md',root/'additional-validation-v1/final-report.md']
    for path in markdown:
        if not path.is_file():raise FileNotFoundError(path)
        files.append(path)
    scope=conditional_scope(candidate,confirmation,additional,correspondence)
    scope.update(fresh_Omega7_verified_cases=2,fresh_Omega_proposals_exact_all_cases=omega['fresh_proposals_exact_all_cases'])
    sets={'development128':{'models':models,'contrasts':{**seven['contrasts'],**matched['contrasts']}}}
    galleries={'development128':{'records':records,'images':images,'ids':ids,
        'groups':[FIRST_ORDER,['previous-c1',*[name for name in NEW_MODELS if name not in FIRST_ORDER]]],
        'scope':'Development128 / noise0. Numbers are actual 3D voxel IoU;pictures are occupancy max-projections.'}}
    seedrecords={}
    for name in ['top1-reference', *([candidate] if candidate is not None else [])]:
        paths=[seedroot/'evaluations'/name/f'noise-{n}'/'evaluation.json' for n in range(3)]
        seedrecords[name]=[read(p) for p in paths];files.extend(paths)
        for e in seedrecords[name]:validate_evaluation(e)
    validate_noise_identity(seedrecords,128,93)
    # Bind the second training seed to the original development sampling seeds/GT.
    validate_noise_identity({**records,**{'seed202-'+k:v for k,v in seedrecords.items()}},128,93)
    sets['seed202']={'models':{k:aggregate(v) for k,v in seedrecords.items()},'contrasts':{}}
    if confirmation is not None:
        if sets['seed202']['models']!=confirmation['models']:raise ValueError('seed202 counts differ')
        sets['seed202']['contrasts']={'candidate-vs-Top1':confirmation['per_training_seed_contrast']['202']}
        folder=root/'additional-validation-v1';contract=read(folder/'evaluation-contract.json');files.append(folder/'evaluation-contract.json')
        extra={};extraimages={}
        for name in contract['models']:
            paths=[folder/'evaluations'/name/f'noise-{n}'/'evaluation.json' for n in range(3)]
            extra[name]=[read(p) for p in paths];files.extend(paths)
            panel=paths[0].parent/'panel.png';files.append(panel);extraimages[name]=np.asarray(Image.open(panel).convert('RGB'))
            actual=aggregate(extra[name]);reported=additional['models'][name]
            if any(reported[k]!=v for k,v in actual.items()):raise ValueError('additional121 counts differ')
        extraids=validate_noise_identity(extra,121,39)
        if len(extra)!=14:raise ValueError('all fourteen additional-panel methods required')
        if {r['source_cluster_id'] for r in next(iter(extra.values()))[0]['per_asset']} & {r['source_cluster_id'] for r in records['top1-reference'][0]['per_asset']}:
            raise ValueError('additional source clusters overlap development')
        sets['additional121']={'models':additional['models'],'contrasts':additional['contrasts']}
        names=list(extra)
        galleries['additional121']={'records':extra,'images':extraimages,'ids':extraids,
            'groups':[names[:7],names[7:]],'scope':'Source-disjoint additional121 / noise0. Fixed selected model;validation,not an independent test.'}
    sets['same-weight-fusion-dose']={'models':{name:{'micro_voxel_iou':m['micro_IoU'],
        'micro_voxel_f1':m['micro_F1']} for name,m in dose['models'].items()},'contrasts':dose['contrasts']}
    sets['task-full-value-address']={'models':{},'contrasts':{'normal-vs-wrong-address':value['normal_minus_wrong_address']}}
    if correspondence is not None:
        sets['selected-value-address']={'models':{},'contrasts':{'normal-vs-wrong-address':correspondence['normal_minus_wrong_address']}}
    attention_metrics={}
    for name in ['static-c1','task-full','task-no-decoder','task-no-prior','task-native-spatial']:
        folder=(root/'attention-statistics-v1'/name if name in ['static-c1','task-full']
                else root/'functional-QA-v1'/('s101-'+name))
        path=folder/'summary.json';qa=read(path);files.append(path)
        cfg=read(root/'configs'/(name+'.json'))
        validate_saved_qa(qa,cfg['trainer']['args']['deform_conditioner'])
        original=read(root/'functional-QA-v1'/('s101-'+name)/'summary.json')
        if any(qa[k]!=original[k] for k in ['source_recipe','trained_denoiser','trained_conditioner','sample_ids']):
            raise ValueError('attention statistics saved model/cases differ')
        stats=qa['actual_attention_weight_statistics']
        if not stats or stats['active_queries']<=0 or len(stats['per_case'])!=2:
            raise ValueError('actual active attention statistics missing')
        for key in ['candidate_slots','active_queries','mean_max_weight','mean_entropy_nats','mean_effective_candidates','max_row_sum_error']:
            value=stats[key]
            if not math.isfinite(value):raise ValueError('nonfinite actual attention statistic')
            attention_metrics[f'attention_QA/{name}/{key}']=value
        scale=qa['actual_content_attention_scale']
        if scale is not None:
            if not math.isfinite(scale):raise ValueError('nonfinite actual attention scale')
            attention_metrics[f'attention_QA/{name}/content_attention_scale']=scale
    scope['attention_statistics_scope']='Two identical saved-model cases;native has3 spatial candidates versus64 semantic patches. Descriptive statistics,not dataset coverage or quality evidence.'
    return {'sets':sets,'galleries':galleries,'scope':scope,'files':list(dict.fromkeys(files)),
            'backbone_metrics':backbone_metrics,'attention_metrics':attention_metrics}


def publication_jobs(root, candidate, selection_job):
    """Attach reporting after actual model work; filesystem gates watch auditors."""
    specs=[('train-'+name,name,101) for name in NEW_MODELS]
    specs.append(('confirm-seed202-train-top1-reference','top1-reference',202))
    if candidate is not None:specs.append(('confirm-seed202-train-'+candidate,candidate,202))
    dependencies=[selection_job,'summarize-seven-arm-semantic-topk-search',
        'summarize-matched-retargeter-top1','summarize-zero-shot-replay-for-topk',
        'summarize-task-value-address-diagnostic','summarize-firstwave-fusion-dose',
        'fresh-omega7-feature-cache-parity']
    dependencies += [f'functional-QA-s{seed}-{name}' for _,name,seed in specs]
    dependencies += ['attention-statistics-s101-'+name for name in ['static-c1','task-full']]
    dependencies += [f'confirm-seed202-final-top1-reference-noise{n}' for n in range(3)]
    if candidate is not None:
        dependencies += ['summarize-semantic-topk-seed202','summarize-additional121-semantic-topk']
    required=[str(root/'provenance/omega-backbone-QA-v1-file4.json')]
    required += [str(root/'attention-statistics-v1'/name/'summary.json') for name in ['static-c1','task-full']]
    for job,_,_ in specs:
        required.append(str(root/'final-weight-audits-v1'/job/'file4-closure.json'))
        required += [str(root/'epoch-audits-v1'/job/f'epoch{e:02d}'/'file4-closure.json') for e in range(1,6)]
    folder=root/'campaign-wandb-v1';jobs=[]
    for mode,output in [('render',folder/'presentation/summary.json'),
                        ('publish',folder/'publication.json'),
                        ('readback',folder/'remote-readback/audit.json')]:
        jobs.append({'name':'campaign-report-'+mode,'resource':'cpu','priority':5,
            'depends_on':dependencies if not jobs else [jobs[-1]['name']],
            'requires':required if mode=='render' else [],'outputs':[str(output)],
            'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.semantic_topk_campaign_wandb',
                       '--root',str(root),'--mode',mode]})
    return jobs


def render(root):
    data=collect(root);folder=root/'campaign-wandb-v1/presentation'
    if folder.exists():raise FileExistsError('preserve previous presentation')
    # Check all pixels before writing any presentation, including the additional panel.
    for gallery in data['galleries'].values():
        images=gallery['images'];reference=next(iter(images.values()));count=len(gallery['ids'])
        if any(a.shape!=(count*64,192,3) or not np.array_equal(a[:,:64],reference[:,:64]) for a in images.values()):
            raise ValueError('actual GT projection pixels/resolution differ')
    folder.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    summaries={}
    for name,gallery in data['galleries'].items():
        images=gallery['images'];ids=gallery['ids'];pages=0
        with PdfPages(folder/(name+'-all-cases.pdf')) as pdf:
            for start in range(0,len(ids),8):
                for group_index,group in enumerate(gallery['groups']):
                    figure,axes=plt.subplots(8,8,figsize=(13.5,12),layout='constrained')
                    for row in range(8):
                        case=start+row
                        for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([])
                        if case>=len(ids):
                            for ax in axes[row]:ax.set_axis_off()
                            continue
                        axes[row,0].imshow(images[group[0]][case*64:(case+1)*64,:64],interpolation='nearest')
                        axes[row,0].set_ylabel(f'{case+1}: '+ids[case].split('_')[1][:8],fontsize=7)
                        if row==0:axes[row,0].set_title('GT',fontsize=8)
                        for column,method in enumerate(group,1):
                            axes[row,column].imshow(images[method][case*64:(case+1)*64,64:128],interpolation='nearest')
                            r=gallery['records'][method][0]['per_asset'][case]
                            axes[row,column].set_xlabel(f"IoU64 {r['tp']/(r['tp']+r['fp']+r['fn']):.3f}",fontsize=7)
                            if row==0:axes[row,column].set_title(LABELS.get(method,method),fontsize=7)
                        for column in range(len(group)+1,8):axes[row,column].set_axis_off()
                    figure.suptitle(gallery['scope']+'\nOriginal case order;no best-case selection or new inference.',fontsize=9)
                    pdf.savefig(figure);pages+=1
                    if start==0:figure.savefig(folder/f'{name}-first8-group{group_index}.png',dpi=160)
                    plt.close(figure)
        summaries[name]={'cases':len(ids),'pages':pages,'sample_ids':ids,'all_actual_GT_pixels_and_3D_hashes_exact':True}
    write_new(folder/'summary.json',{'status':'PASS_COMPLETED_CAMPAIGN_PRESENTATION','panels':summaries,
        'scope':data['scope'],'new_inference_or_training':False,'input_SHA256':{str(p):sha(p) for p in data['files']}})
    print('PASS_COMPLETED_CAMPAIGN_PRESENTATION',flush=True)


def payload(root):
    data=collect(root);folder=root/'campaign-wandb-v1/presentation';summary=read(folder/'summary.json')
    if summary['status']!='PASS_COMPLETED_CAMPAIGN_PRESENTATION' or summary['scope']!=data['scope']:
        raise ValueError('completed matching presentation required')
    if summary['input_SHA256']!={str(p):sha(p) for p in data['files']}:raise ValueError('actual presentation inputs changed')
    metrics={**data['backbone_metrics'],**data.get('attention_metrics',{})};tables={}
    for name,values in data['sets'].items():
        rows=[];contrasts=[]
        for method,model in values['models'].items():
            rows.append([method,model['micro_voxel_iou'],model['micro_voxel_f1']])
            for key in ['micro_voxel_iou','micro_voxel_f1']:metrics[f'{name}/{method}/{key}']=model[key]
        for label,c in values['contrasts'].items():
            contrasts.append([label,c['gain'],*c['paired_cluster_bootstrap_ci95']])
            for key,value in zip(['gain','ci95_lower','ci95_upper'],[c['gain'],*c['paired_cluster_bootstrap_ci95']]):
                metrics[f'{name}/contrasts/{label}/{key}']=value
        if rows:tables[name+'/models']={'columns':['method','micro_IoU','micro_F1'],'data':rows}
        if contrasts:tables[name+'/contrasts']={'columns':['contrast','gain_IoU','ci95_lower','ci95_upper'],'data':contrasts}
    components=[]
    for label,key in COMPONENTS.items():
        c=data['sets']['development128']['contrasts'][key]
        components.append([label,c['gain'],*c['paired_cluster_bootstrap_ci95']])
    tables['development128/component_effects']={'columns':['comparison_scope','gain_IoU','ci95_lower','ci95_upper'],'data':components}
    files=data['files']+[p for p in folder.iterdir() if p.is_file()]
    if any(p.suffix=='.pt' for p in files):raise ValueError('model binaries belong on canonical file4')
    return {'metrics':metrics,'tables':tables,'images':{p.stem:str(p) for p in folder.glob('*.png')},
        'artifact_files':{str(p.relative_to(root.parent)):str(p) for p in files},'scope':data['scope'],
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}


def publish(root):
    os.environ.setdefault('WANDB_DISABLE_CODE','true')
    import wandb
    folder=root/'campaign-wandb-v1'
    if (folder/'publication.json').exists():raise FileExistsError('already published;read back the same run')
    spec=payload(root);write_new(folder/'publication-spec.json',spec)
    run=wandb.init(entity=PROJECT.split('/')[0],project=PROJECT.split('/')[1],id=RUN_ID,resume='never',
        name='SS semantic TopK completed experiment comparison',dir=str(folder),
        settings=wandb.Settings(x_disable_stats=True,x_disable_meta=True),
        config={'source_commit':spec['source_commit'],'scope':spec['scope'],
            'primary_metric':'SS64 micro voxel IoU/F1;aligned decoded target latent',
            'training_in_this_run':False,'same_retargeter_Top1_replication_across_training_seeds':False})
    run.log(spec['metrics']);run.log({**{k:wandb.Table(**v) for k,v in spec['tables'].items()},
                                    **{k:wandb.Image(p) for k,p in spec['images'].items()}})
    artifact=wandb.Artifact(RUN_ID,type='evaluation-report')
    for name,path in spec['artifact_files'].items():artifact.add_file(path,name=name)
    logged=run.log_artifact(artifact);logged.wait();version=logged.version
    run.summary.update({'status':'PASS_COMPLETED_SS_CAMPAIGN_PUBLICATION','model_weights_uploaded':False,**spec['scope']})
    url=run.url;run.finish()
    write_new(folder/'publication.json',{'status':'PASS_WANDB_FINISHED_SS_CAMPAIGN_REPORT','run_id':RUN_ID,
        'run_url':url,'artifact_version':version,'spec_SHA256':sha(folder/'publication-spec.json'),
        'full_goal_complete':False,'independent_readback_required':True})
    print(json.dumps({'status':'PASS_WANDB_FINISHED_SS_CAMPAIGN_REPORT','run_url':url}),flush=True)


def readback(root):
    folder=root/'campaign-wandb-v1';receipt=read(folder/'publication.json');spec=read(folder/'publication-spec.json')
    if receipt['spec_SHA256']!=sha(folder/'publication-spec.json'):raise ValueError('publication intent changed')
    if (folder/'remote-readback/audit.json').exists():raise FileExistsError('preserve successful readback')
    index=0
    while (folder/f'remote-readback/attempt-{index}').exists():index+=1
    output=folder/f'remote-readback/attempt-{index}';output.mkdir(parents=True)
    # Start an empty download cache, independent of the uploading process.
    os.environ['WANDB_CACHE_DIR']=str(output/'fresh-download-cache')
    import wandb
    api=wandb.Api(timeout=30)
    for attempt in range(7):
        api.flush();actual=api.run(PROJECT+'/'+receipt['run_id'])
        if actual.state=='finished' and actual.summary.get('status')=='PASS_COMPLETED_SS_CAMPAIGN_PUBLICATION':break
        if attempt==6:raise RuntimeError('upload pending;poll the same run without republishing')
        time.sleep(10)
    for key,value in spec['scope'].items():
        if actual.summary.get(key)!=value:raise ValueError('actual published scope differs')
    if actual.summary.get('model_weights_uploaded') is not False:raise ValueError('model binary logging scope differs')
    for key,value in spec['metrics'].items():near(actual.summary[key],value)
    for key,expected in spec['tables'].items():
        meta=actual.summary[key];path=Path(actual.file(meta['path']).download(root=str(output/key.replace('/','_')),replace=False).name)
        table=read(path)
        if table['columns']!=expected['columns'] or len(table['data'])!=len(expected['data']):raise ValueError('remote table shape differs')
        for left,right in zip(table['data'],expected['data']):
            if left[0]!=right[0] or len(left)!=len(right):raise ValueError('remote table identities differ')
            for a,b in zip(left[1:],right[1:]):near(a,b)
    for key,local in spec['images'].items():
        meta=actual.summary[key];path=Path(actual.file(meta['path']).download(root=str(output/key),replace=False).name)
        if sha(path)!=sha(Path(local)) or sha(path)!=meta['sha256']:raise ValueError('actual PNG bytes differ')
    artifact=api.artifact(PROJECT+'/'+RUN_ID+':'+receipt['artifact_version'])
    downloaded=Path(artifact.download(root=str(output/'artifact')))
    for name,local in spec['artifact_files'].items():
        if sha(downloaded/name)!=sha(Path(local)):raise ValueError('actual artifact differs: '+name)
    write_new(folder/'remote-readback/audit.json',{'status':'PASS_ACTUAL_SS_CAMPAIGN_WANDB_READBACK',
        'run_url':actual.url,'actual_remote_state':actual.state,'numeric_fields':len(spec['metrics']),
        'tables':{k:len(v['data']) for k,v in spec['tables'].items()},
        'downloaded_PNGs_byte_exact':len(spec['images']),'downloaded_artifact_files_byte_exact':len(spec['artifact_files']),
        'fresh_download_cache':True,'source_commit':spec['source_commit'],'scope':spec['scope'],'full_goal_complete':False})
    print('PASS_ACTUAL_SS_CAMPAIGN_WANDB_READBACK',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=['render','publish','readback'],required=True);a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,available[-2:])
    {'render':render,'publish':publish,'readback':readback}[a.mode](a.root)


if __name__=='__main__':main()
