"""Test value-address utility on the actually selected saved SS model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from .semantic_topk_search import write_new
from .summarize_semantic_topk_search import validate_evaluation
from .summarize_task_value_diagnostic import summarize as value_summary


SUMMARY_JOB='summarize-selected-model-correspondence'


def jobs_and_contract(root,work,candidate,dependency):
    if candidate not in {'soft-topk','static-c1','task-full','task-no-decoder','task-no-prior'}:
        raise ValueError('correspondence diagnostic requires a selected semantic Top-K model')
    folder=root/'selected-correspondence-v1';jobs=[]
    if candidate=='task-full':
        # Reuse actual already-queued intervention, rather than duplicate GPU
        # evaluation just because Full happened to win model selection.
        evaluations=root/'task-value-address-v1'
        dependencies=[dependency,'summarize-task-value-address-diagnostic']
    else:
        evaluations=folder
        config=root/'configs'/(candidate+'.json');run=root/'runs_ss64'/candidate/'ckpts'
        dependencies=[]
        for control in ['normal','wrong-address']:
            name='selected-correspondence-'+control;output=evaluations/control
            dependencies.append(name)
            jobs.append({'name':name,'resource':'evaluation','priority':3,'depends_on':[dependency],
                'requires':[str(config),str(run/'denoiser_step0000480.pt'),str(run/'deform_conditioner_step0000480.pt')],
                'outputs':[str(output/'evaluation.json'),str(output/'panel.png')],
                'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.evaluate_native_checkpoint',
                    '--stage','ss64','--config-root',str(config.parent),'--arm-config',str(config),
                    '--checkpoint',str(run/'denoiser_step0000480.pt'),
                    '--conditioner-checkpoint',str(run/'deform_conditioner_step0000480.pt'),
                    '--step','480','--output',str(output),'--panel-file',str(work/'panels/panel-32.json'),
                    '--seed-map-file',str(work/'panels/seed-map.json'),'--generation-batch','8',
                    '--task-value-control',control]})
    jobs.append({'name':SUMMARY_JOB,'resource':'cpu','priority':3,'depends_on':dependencies,
        'outputs':[str(folder/'summary.json')],
        'command':[sys.executable,'-u','-m','experiments.deformnext_retargeting.semantic_topk_selected_correspondence',
            '--root',str(root)]})
    return jobs,{'status':'QUEUED_NOT_EVALUATED','candidate':candidate,
        'normal':str(evaluations/'normal/evaluation.json'),'altered':str(evaluations/'wrong-address/evaluation.json'),
        'reference':str(root/'evaluations'/candidate/'noise-0/evaluation.json'),
        'panel':str(work/'panels/panel-32.json'),'seed_map':str(work/'panels/seed-map.json'),
        'scope':'Same selected seed101 raw480 model/common32/noise0,zero SGD. Keep keys,logits,weights,valid support and every query candidate-value multiset;permute value-to-weight pairing only. Not a new training ablation or native-spatial substitute.'}


def summarize(normal,altered,reference):
    validate_evaluation(reference)
    for field in ['checkpoint','conditioner_checkpoint','arm_config','manifest_sha256','generation_batch_size','stage','arm']:
        if normal.get(field) is None or any(e.get(field)!=reference.get(field) for e in [normal,altered]):
            raise ValueError('selected diagnostic must bind the same actual saved model/protocol: '+field)
    if normal.get('diagnostic_only') is not False or altered.get('diagnostic_only') is not True:
        raise ValueError('selected intervention must be explicitly marked diagnostic')
    if any(e.get('transport_control')!='normal' or e.get('r9_path_control')!='normal' for e in [normal,altered]):
        raise ValueError('do not combine selected value-only intervention with source/input controls')
    result=value_summary(normal,altered,reference)
    if result['normal_minus_wrong_address']['paired_target_hashes_verified']!=32:
        raise ValueError('all actual paired GT hashes must be verified')
    result.update(status='PASS_SELECTED_MODEL_SAME_POOL_VALUE_ADDRESS_DIAGNOSTIC',
        quality_effect_supported_at_this_panel=result['normal_minus_wrong_address']['paired_cluster_bootstrap_ci95'][0]>0,
        scope='Selected saved-model/common32/one-noise address sensitivity with exactly preserved per-query candidate-value multisets. A CI including zero does not reject the utility of semantic candidate retrieval or count as a fresh-training ablation.')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    folder=a.root/'selected-correspondence-v1';contract_path=folder/'contract.json'
    contract=json.loads(contract_path.read_text());decision=json.loads((a.root/'candidate-selection-v1.json').read_text())
    if decision['selected_candidate']!=contract['candidate']:
        raise ValueError('selected correspondence model changed')
    paths=[Path(contract[k]) for k in ['normal','altered','reference']]
    records=[json.loads(p.read_text()) for p in paths]
    result=summarize(*records)
    ids=json.loads(Path(contract['panel']).read_text())['sample_ids']
    seeds=json.loads(Path(contract['seed_map']).read_text())
    if len(ids)!=32 or any([r['sample_id'] for r in e['per_asset']]!=ids for e in records[:2]):
        raise ValueError('actual selected correspondence panel differs from common32')
    if any(r['seed']!=seeds[r['sample_id']] for e in records[:2] for r in e['per_asset']):
        raise ValueError('actual selected correspondence noise differs')
    result.update(selected_candidate=contract['candidate'],contract_SHA256=hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        evaluation_SHA256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},full_goal_complete=False)
    write_new(folder/'summary.json',result)
    print(json.dumps({'status':result['status'],'candidate':contract['candidate'],
        'gain':result['normal_minus_wrong_address']['gain']}),flush=True)


if __name__=='__main__':main()
