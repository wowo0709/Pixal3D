"""Compile fresh deterministic SS panels without relabeling legacy validations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .compare_ss_noise_panels import compare
from .deterministic_study_contract import legacy_epoch_comparison, validate_evaluation, verify_replay
from .ray_group_study import ARMS, STATIC, source_record, validate_profile
from .semantic_topk_search import write_new
from .summarize_semantic_topk_search import aggregate, task_gradient_audit
from .summarize_ss_module_study import audit_training
from .summarize_task_value_diagnostic import summarize as value_summary


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sources(plan):
    contract = read(plan['source_contract'])
    receipt = read(plan['source_receipt'])
    if receipt.get('status') != 'PASS' or receipt.get('local_equals_actual_file4_SHA256') is not True:
        raise ValueError('actual file4 source closure missing')
    entries = {row['relative_path']: row for row in receipt['files']}
    for row in contract['runtime_components']:
        if sha(row['runtime_path']) != row['sha256'] or sha(row['archived_path']) != row['sha256']:
            raise ValueError('frozen source changed: ' + row['runtime_path'])
        relative = str(Path(row['archived_path']).relative_to(plan['receipt_root']))
        if entries.get(relative, {}).get('sha256') != row['sha256']:
            raise ValueError('source archive differs from actual file4 receipt')


def verify_model(model):
    records = [read(path) for path in model['final_paths']]
    if len(records) != 3:
        raise ValueError('three generation noises required')
    for record in records:
        validate_evaluation(record, assets=128, step=model['step'], mode=model['mode'])
    ids = read(model['panel128'])['sample_ids']
    for record,seed_file in zip(records,model['seed_maps']):
        seeds = read(seed_file)
        if ([row['sample_id'] for row in record['per_asset']]!=ids
                or any(row['seed']!=seeds[row['sample_id']] for row in record['per_asset'])):
            raise ValueError('actual fixed panel or prescribed noise differs')
        for key in ['checkpoint','conditioner_checkpoint','arm_config','manifest_sha256','arm']:
            if record.get(key)!=records[0].get(key):
                raise ValueError('model/config identity differs between noises')
        for key in ['entry_source_SHA256','adapter_SHA256']:
            if record['deterministic_evaluation'][key]!=records[0]['deterministic_evaluation'][key]:
                raise ValueError('evaluator source differs between noises')
    identity = lambda row: (row['sample_id'], row['source_cluster_id'], row['target_voxels_sha256'])
    reference = [identity(row) for row in records[0]['per_asset']]
    if len({row[1] for row in reference}) != 93:
        raise ValueError('fixed128 must contain93 source clusters')
    for record in records[1:]:
        if [identity(row) for row in record['per_asset']] != reference:
            raise ValueError('final panel/GT order differs between noises')
    for rows in zip(*(record['per_asset'] for record in records)):
        if len({row['seed'] for row in rows}) != 3:
            raise ValueError('three distinct per-asset generation noises required')
    small = read(model['replay_path'])
    if [row['sample_id'] for row in small['per_asset']]!=read(model['panel32'])['sample_ids']:
        raise ValueError('independent replay is not the prescribed fixed32 panel')
    proof = verify_replay(small, records[0], step=model['step'], mode=model['mode'])
    if model.get('legacy_validation'):
        old = [json.loads(line) for line in Path(model['legacy_validation']).read_text().splitlines()]
        proof['legacy_epoch_comparison'] = legacy_epoch_comparison(next(row for row in old if row['step']==model['step']), small)
    return records, proof


def replay(plan, name):
    check_sources(plan)
    _, proof = verify_model(plan['models'][name])
    model = plan['models'][name]
    proof['inputs'] = {path: sha(path) for path in [*model['final_paths'], model['replay_path']]}
    check_sources(plan)
    write_new(Path(model['replay_proof']), proof)
    print(json.dumps({'model':name, 'status':proof['status']}), flush=True)


def values(plan, name):
    check_sources(plan)
    model = plan['models'][name]
    normal, altered, reference = [read(path) for path in [model['replay_path'], model['wrong_address_path'], model['final_paths'][0]]]
    validate_evaluation(altered, assets=32, step=480, value_control='wrong-address')
    for key in ['entry_source_SHA256', 'adapter_SHA256']:
        if altered['deterministic_evaluation'][key] != normal['deterministic_evaluation'][key]:
            raise ValueError('value diagnostic evaluator source differs')
    verify_replay(normal, reference)
    proof = value_summary(normal, altered, reference)
    proof['inputs'] = {path:sha(path) for path in [model['replay_path'], model['wrong_address_path'], model['final_paths'][0]]}
    check_sources(plan)
    write_new(Path(model['value_proof']), proof)
    print(json.dumps({'model':name, 'status':proof['status']}), flush=True)


def audit_ray(root, name):
    cfg = read(root/'configs'/(name+'.json'))
    validate_profile(cfg, root)
    initial = read(root.parent/'seven-arm-final-report.json')['training_audits']['static-c1']['initial_iou']
    training = audit_training(root, name, initial)
    final_folder = root/'final-weight-audits-v1'/('train-'+name)
    final = read(final_folder/'audit.json')
    qa = read(root/'functional-QA-v1'/('s101-'+name)/'summary.json')
    expected = dict(status='PASS_ACTUAL_FINAL_TENSORS_AND_OPTIMIZER', frozen_base_tensors_exact=701,
        LoRA_tensors_updated=360, optimizer_states=366 if name in STATIC else 371,
        actual_optimizer_step=480, physical_deformation_query_channels=0, retargeter_input_channels=2048,
        matching_tensors_present={'decoder':0 if name=='ray-no-decoder' else 18,'score':6},
        matching_tensors_changed={'decoder':[],'score':[]})
    if (any(final.get(key)!=value for key,value in expected.items())
            or qa.get('status')!='PASS_SAVED_RAY_GROUP_RGB_BF16_CFG_CONTRACT' or qa.get('optimizer_updates')!=0):
        raise ValueError('saved ray tensor/conditioning audit differs: '+name)
    for folder in [final_folder, *[root/'epoch-audits-v1'/('train-'+name)/f'epoch{n:02d}' for n in range(1,6)]]:
        closure = read(folder/'file4-closure.json')
        audit = folder/'audit.json'
        if (closure.get('status')!='PASS' or closure.get('local_equals_actual_file4_SHA256') is not True
                or not any(row['relative_path']==str(audit.relative_to(root.parent.parent)) and row['sha256']==sha(audit) for row in closure['files'])):
            raise ValueError('actual file4 epoch/final audit bytes differ')
    training['task_branch_gradients'] = task_gradient_audit(root, name)
    lineage = read(root/'score-prior-control-v1/source-isolation.json')
    current = source_record()['component_SHA256']
    parent = lineage['parent_components_SHA256']
    registry = 'experiments/deformnext_retargeting/ray_group_study.py'
    if set(parent)!=set(current) or any(parent[key]!=current[key] for key in parent if key!=registry):
        raise ValueError('original tensor model/data/trainer source changed')
    binding = read(root/'verification'/name/'ray-source-binding-train.json')
    allowed = [current] if name=='ray-no-score-prior' else [parent,current]
    if not binding['source_stable_through_execution'] or binding['component_SHA256'] not in allowed:
        raise ValueError('actual training source lineage differs')
    return training


def summarize(plan):
    check_sources(plan)
    expected_names = set(ARMS)|{'v1','old-top1-R','zero-source-only','zero-edited-rgb'}
    if set(plan['models']) != expected_names:
        raise ValueError('all six ray arms and four contextual baselines required')
    records, replays, training, value_proofs = {}, {}, {}, {}
    root = Path(plan['ray_root'])
    for name, model in plan['models'].items():
        records[name], replays[name] = verify_model(model)
        saved = read(model['replay_proof'])
        if saved['inputs'] != {path:sha(path) for path in [*model['final_paths'], model['replay_path']]}:
            raise ValueError('replay proof no longer binds actual evaluations')
        if name in ARMS:
            training[name] = audit_ray(root, name)
            value_proofs[name] = read(model['value_proof'])
            wanted = [model['replay_path'],model['wrong_address_path'],model['final_paths'][0]]
            if value_proofs[name]['inputs'] != {path:sha(path) for path in wanted}:
                raise ValueError('value proof no longer binds actual evaluations')
    pairs = [('ray-full',name) for name in ARMS if name!='ray-full'] + [('ray-weighted','ray-top1')]
    pairs += [(name,base) for name in ARMS for base in ['v1','old-top1-R','zero-source-only','zero-edited-rgb']]
    contrasts = {candidate+'-vs-'+base:compare([(str(n),records[base][n],records[candidate][n]) for n in range(3)]) for candidate,base in pairs}
    report = dict(status='PASS_SIX_RAY_ARMS_DETERMINISTIC_FINAL_REPORT',
        models={name:aggregate(value) for name,value in records.items()}, contrasts=contrasts,
        saved_checkpoint_replays=replays, training_audits=training, value_address_diagnostics=value_proofs,
        training_seed=101, generation_noise_repeats=3, model_adopted=False,
        legacy_epoch_results_unchanged=True,
        scope='Fixed raw480 (zero-shot step0),development128/93source clusters,three generation noises;one training seed. Exploratory contrasts without multiplicity correction. Exact independent deterministic32-to128 replay;not a claim of exact legacy training-epoch replay.',
        interpretation='NoScore remains Top1. NoCorr uses native spatial3 versus semantic64. No-decoder retains D1-selected scalar head initialization. Old Top1-R changes query/carrier;context only. Both zero baselines score desired M1 GT. No training-seed replication or held-out-test claim.',
        inputs={path:sha(path) for model in plan['models'].values() for path in [*model['final_paths'],model['replay_path'],model['replay_proof']]})
    lines=['# Deterministic SS ray study','',report['scope'],'','| Model | IoU | F1 |','| --- | ---: | ---: |']
    lines += [f"| {name} | {r['micro_voxel_iou']:.6f} | {r['micro_voxel_f1']:.6f} |" for name,r in report['models'].items()]
    lines += ['','| Contrast | IoU gain, pp | 95% source-cluster CI, pp |','| --- | ---: | --- |']
    for name,r in contrasts.items():
        lo,hi=r['paired_cluster_bootstrap_ci95']
        lines.append(f"| {name} | {100*r['gain']:+.3f} | [{100*lo:+.3f},{100*hi:+.3f}] |")
    lines += ['',report['interpretation']]
    check_sources(plan)
    write_new(Path(plan['report_json']),report)
    with Path(plan['report_md']).open('x') as stream:
        stream.write('\n'.join(lines)+'\n')
    print(report['status'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--mode',choices=['replay','value','summarize'],required=True)
    p.add_argument('--model')
    args=p.parse_args(); plan=read(args.plan)
    if args.mode=='summarize': summarize(plan)
    elif args.mode=='replay': replay(plan,args.model)
    else: values(plan,args.model)


if __name__=='__main__':
    main()
