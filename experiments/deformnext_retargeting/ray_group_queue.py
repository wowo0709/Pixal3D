"""Queue the matched ray-group study and bind final contrasts to saved evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from .compare_ss_noise_panels import compare
from .orthogonal_utility_followups import close_job, cpu, require_source_contract
from .ray_group_study import ARMS, STATIC, source_record, validate_profile
from .semantic_topk_confirmation import make_jobs
from .semantic_topk_functional_qa import job as qa_job
from .semantic_topk_search import write_new
from .semantic_topk_selected_correspondence import summarize as value_summary
from .summarize_frozen_utility_followup import zero_records
from .summarize_semantic_topk_search import aggregate, task_gradient_audit, validate_evaluation
from .summarize_ss_module_study import audit_final_replay, audit_training


MODULE = 'experiments.deformnext_retargeting.ray_group_queue'
ORDER = ['ray-full', 'ray-weighted', 'ray-top1', 'ray-native-spatial', 'ray-no-decoder']
PAIRS = {name: (['0', '1'] if name in {'ray-weighted', 'ray-native-spatial'} else ['2', '3']) for name in ARMS}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commands_from_stage(jobs, stage):
    # The live scheduler intentionally runs from the frozen reference checkout.
    # Both cwd and PYTHONPATH must select this new trainer registry in children.
    for job in jobs:
        job['command'] = ['env', '-C', str(stage), 'PYTHONPATH=' + str(stage)] + job['command']
    return jobs


def arm_jobs(args, name):
    root, campaign, work, previous = args.root, args.root.parent, args.work_root, args.previous_work
    cfg = root / 'configs' / (name + '.json')
    run = root / 'runs_ss64' / name
    jobs = make_jobs(root, previous, name, PAIRS[name], [], 101, prefix='')
    jobs[0]['command'][3] = 'experiments.deformnext_retargeting.ray_group_study'
    training = jobs[0]['name']
    jobs[0]['requires'] += [str(args.source_receipt), str(root / 'verification' / name /
        'cohost-final-native-v4/ray-source-binding-proof.json')]
    for job in jobs:
        job['requires'].append(str(args.source_receipt))
    qa = qa_job(root, root, name, 101, previous / 'panels/panel-128.json')
    jobs.append(qa)
    epochs = []
    for n in range(1, 6):
        label = f'audit-epoch-{name}-{n:02d}'
        epochs.append(label)
        folder = root / 'epoch-audits-v1' / training / f'epoch{n:02d}'
        jobs.append({'name': label, 'resource': 'cpu', 'priority': 0,
            'requires': [str(args.source_receipt), str(run / 'ckpts' / f'misc_step{96*n:07d}.pt'), str(run / 'validation.jsonl')],
            'outputs': [str(folder / 'file4-closure.json'), str(folder / 'once-driver-result-v1.json')],
            'command': cpu() + ['-m', 'experiments.deformnext_retargeting.run_saved_audit_once',
                '--kind', 'epoch', '--implementation', str(args.epoch_template), '--expected-sha', args.epoch_sha,
                '--root', str(root), '--work-root', str(work), '--job-name', training, '--config', str(cfg), '--epoch', str(n)]})
    folder = root / 'final-weight-audits-v1' / training
    final = 'audit-final-' + name
    jobs.append({'name': final, 'resource': 'cpu', 'priority': 0, 'depends_on': [training, epochs[-1]],
        'requires': [str(args.source_receipt)],
        'outputs': [str(folder / 'file4-closure.json'), str(folder / 'once-driver-result-v1.json')],
        'command': cpu() + ['-m', 'experiments.deformnext_retargeting.run_saved_audit_once', '--kind', 'final',
            '--implementation', str(args.final_template), '--expected-sha', args.final_sha, '--root', str(root),
            '--work-root', str(work), '--job-name', training, '--config', str(cfg)]})
    values = root / 'value-address-v1' / name
    value_jobs = []
    for control in ['normal', 'wrong-address']:
        label = name + '-ray-value-' + control
        value_jobs.append(label)
        output = values / control
        jobs.append({'name': label, 'resource': 'evaluation', 'priority': 3,
            'depends_on': [training, qa['name'], 'final-' + name + '-noise0'],
            'requires': [str(args.source_receipt)], 'outputs': [str(output / 'evaluation.json'), str(output / 'panel.png')],
            'command': [sys.executable, '-u', '-m', 'experiments.deformnext_retargeting.evaluate_native_checkpoint',
                '--stage', 'ss64', '--config-root', str(cfg.parent), '--arm-config', str(cfg),
                '--checkpoint', str(run / 'ckpts/denoiser_step0000480.pt'), '--conditioner-checkpoint',
                str(run / 'ckpts/deform_conditioner_step0000480.pt'), '--step', '480', '--output', str(output),
                '--panel-file', str(previous / 'panels/panel-32.json'), '--seed-map-file',
                str(previous / 'panels/seed-map.json'), '--generation-batch', '8', '--task-value-control', control]})
    label = 'summarize-ray-value-' + name
    jobs.append({'name': label, 'resource': 'cpu', 'priority': 3, 'depends_on': value_jobs,
        'outputs': [str(values / 'summary.json')], 'command': cpu() + ['-m', MODULE, '--mode', 'value',
            '--root', str(root), '--arm', name, '--previous-work', str(previous)]})
    files = [values / 'summary.json'] + [values / control / f for control in ['normal', 'wrong-address']
                                                        for f in ['evaluation.json', 'panel.png']]
    jobs.append(close_job(root, campaign, work, previous, 'close-ray-value-' + name, [label], files,
        values / 'file4-closure.json'))
    return jobs


def queue(args):
    stage = Path(__file__).resolve().parents[2]
    require_source_contract(args.source_contract, args.source_receipt, args.root.parent)
    current = source_record()
    for template, digest in [(args.epoch_template, args.epoch_sha), (args.final_template, args.final_sha)]:
        if sha(template) != digest:
            raise ValueError('immutable numerical auditor changed')
    jobs = []
    for name in ORDER:
        validate_profile(read(args.root / 'configs' / (name + '.json')), args.root)
        folder = args.root / 'verification' / name / 'cohost-final-native-v4'
        binding, proof = read(folder / 'ray-source-binding-proof.json'), read(folder / 'native-proof.json')
        if (binding['component_SHA256'] != current['component_SHA256'] or not binding['source_stable_through_execution']
                or proof['status'] != 'PASS' or proof['optimizer_updates'] != 0
                or any(value != 0 for value in proof['condition_max_abs_errors'].values())):
            raise ValueError('actual current-source ray hardware native proof required: ' + name)
        jobs += arm_jobs(args, name)
    dependencies = [j['name'] for j in jobs if j['name'].startswith(('audit-final-', 'functional-QA-', 'final-'))]
    summary = 'summarize-five-ray-arms-fixed-final'
    jobs.append({'name': summary, 'resource': 'cpu', 'priority': 2, 'depends_on': dependencies,
        'outputs': [str(args.root / 'fixed-final-report.json'), str(args.root / 'fixed-final-report.md')],
        'command': cpu() + ['-m', MODULE, '--mode', 'summarize', '--root', str(args.root)]})
    jobs.append(close_job(args.root, args.root.parent, args.work_root, args.previous_work,
        'close-five-ray-arms-fixed-final', [summary],
        [args.root / 'fixed-final-report.json', args.root / 'fixed-final-report.md'], args.root / 'fixed-final-file4.json'))
    commands_from_stage(jobs, stage)
    path = args.work_root / 'queue-jobs.json'
    before = path.read_bytes()
    existing = json.loads(before)
    names = [j['name'] for j in existing + jobs]
    if len(names) != len(set(names)) or any(dep not in names for j in jobs for dep in j.get('depends_on', [])):
        raise ValueError('ray queue collision or missing dependency')
    with (args.work_root / 'queue-before-ray-group-five-arm-v1.json').open('xb') as stream:
        stream.write(before)
    declaration = args.root / 'queue-declaration-v1.json'
    write_new(declaration, {'status': 'PREPARED_FOR_QUEUE', 'source_commit': current['commit_at_entry'],
        'source_contract_SHA256': sha(args.source_contract), 'source_receipt_SHA256': sha(args.source_receipt),
        'jobs': jobs, 'training_runs': 5, 'training_seed': 101, 'GPUs': [0,1,2,3],
        'training_budget': 'Each freshSS4210/raw480/global96/DDP2/micro48/acc1;native gate and actual2-updateDDP48 benchmark before each main.',
        'query_unit': 'source-camera ray group,common to all arms', 'retargeter': 'common to all arms',
        'model_adopted': False, 'reserved121_accessed': False})
    if path.read_bytes() != before:
        raise RuntimeError('concurrent queue writer;reconcile before mutation')
    temporary = path.with_suffix('.ray-group.tmp')
    temporary.write_text(json.dumps(existing + jobs, indent=2) + '\n')
    os.replace(temporary, path)
    write_new(args.root / 'queue-installed-v1.json', {'status': 'QUEUED_NOT_TRAINED', 'new_jobs': len(jobs),
        'total_jobs': len(existing + jobs), 'declaration_SHA256': sha(declaration), 'stage': str(stage)})
    print(json.dumps({'status': 'QUEUED_NOT_TRAINED', 'new_jobs': len(jobs), 'training_runs': 5}), flush=True)


def summarize(root):
    campaign = root.parent
    initial = read(campaign / 'seven-arm-final-report.json')['training_audits']['static-c1']['initial_iou']
    records, inputs, audits = {}, [], {}
    for name in ARMS:
        config = root / 'configs' / (name + '.json')
        cfg = read(config)
        validate_profile(cfg, root)
        training = audit_training(root, name, initial)
        folder = root / 'final-weight-audits-v1' / ('train-' + name)
        qa = root / 'functional-QA-v1' / ('s101-' + name) / 'summary.json'
        final, actual_qa = read(folder / 'audit.json'), read(qa)
        expected = dict(status='PASS_ACTUAL_FINAL_TENSORS_AND_OPTIMIZER', frozen_base_tensors_exact=701,
            LoRA_tensors_updated=360, optimizer_states=366 if name in STATIC else 371,
            actual_optimizer_step=480, physical_deformation_query_channels=0, retargeter_input_channels=2048,
            matching_tensors_present={'decoder':0 if name=='ray-no-decoder' else 18,'score':6},
            matching_tensors_changed={'decoder':[],'score':[]})
        if (any(final.get(k) != v for k,v in expected.items())
                or actual_qa['status'] != 'PASS_SAVED_RAY_GROUP_RGB_BF16_CFG_CONTRACT'
                or actual_qa['optimizer_updates'] != 0):
            raise ValueError('actual ray saved-weight/conditioning audit differs: ' + name)
        for one in [folder] + [root / 'epoch-audits-v1' / ('train-' + name) / f'epoch{n:02d}' for n in range(1,6)]:
            receipt = read(one / 'file4-closure.json')
            if receipt['status'] != 'PASS' or not receipt['local_equals_actual_file4_SHA256']:
                raise ValueError('actual file4 epoch/final closure missing')
            audit = one / 'audit.json'
            if not any(row['relative_path'] == str(audit.relative_to(campaign.parent)) and row['sha256']==sha(audit)
                       for row in receipt['files']):
                raise ValueError('closed audit bytes differ')
            inputs += [audit, one / 'file4-closure.json', one / 'once-driver-result-v1.json']
        paths = [root / 'evaluations' / name / f'noise-{n}/evaluation.json' for n in range(3)]
        records[name] = [read(p) for p in paths]
        training['final128_replays_epoch32'] = audit_final_replay(training, records[name][0])
        training['task_branch_gradients'] = task_gradient_audit(root, name)
        audits[name] = training
        inputs += [config, qa, *paths, root / 'verification' / name / 'ray-source-binding-train.json']
    for label, base, name in [('v1', campaign.parent / 'ss-seed-confirmation-v1', 'v1'),
                              ('old-top1-R', campaign, 'static-top1-c1')]:
        paths = [base / 'evaluations' / name / f'noise-{n}/evaluation.json' for n in range(3)]
        records[label] = [read(p) for p in paths]
        inputs += paths
    for values in records.values():
        for record in values:
            validate_evaluation(record)
    zeros, paths = zero_records(campaign)
    records.update(zeros)
    inputs += paths
    pairs = [('ray-full', name) for name in ['ray-top1','ray-weighted','ray-native-spatial','ray-no-decoder']]
    pairs += [('ray-weighted','ray-top1')]
    pairs += [(name, base) for name in ARMS for base in ['v1','old-top1-R',*zeros]]
    contrasts = {candidate+'-vs-'+base:compare([(str(n),records[base][n],records[candidate][n]) for n in range(3)])
                 for candidate,base in pairs}
    report = {'status':'PASS_FIVE_RAY_ARMS_FIXED_FINAL_EXPLORATORY_REPORT',
        'models':{name:aggregate(values) for name,values in records.items()}, 'contrasts':contrasts,
        'training_audits':audits, 'training_seed':101, 'model_adopted':False,
        'scope':'Common ray-group query/R,raw480,fixed128/93source clusters,three generation noises;one training seed,exploratory contrasts without multiplicity correction. No training-seed replication or held-out-test claim.',
        'native_spatial_scope':'Original native spatial3 versus semantic64;natural candidate count differs.',
        'no_decoder_scope':'Physical D1 removal with D1-selected frozen scalar initialization;conditional ablation,not optimally retrainedD0.',
        'historical_scope':'Old Top1-R has a different query unit;context only. Both zero baselines score identical desiredM1:fourM0RGB versus oneM1+threeM0RGB.',
        'inputs':[{'path':str(p),'sha256':sha(p)} for p in dict.fromkeys(inputs)]}
    lines=['# Shared camera-ray query: fixed-final report','','| Model | IoU | F1 |','| --- | ---: | ---: |']
    lines += [f"| {name} | {v['micro_voxel_iou']:.6f} | {v['micro_voxel_f1']:.6f} |" for name,v in report['models'].items()]
    lines += ['','| Contrast | IoU gain, pp | 95% source-cluster CI, pp |','| --- | ---: | --- |']
    for name,value in contrasts.items():
        lo,hi=value['paired_cluster_bootstrap_ci95']
        lines.append(f"| {name} | {100*value['gain']:+.3f} | [{100*lo:+.3f},{100*hi:+.3f}] |")
    lines += ['',report['scope'],'',report['native_spatial_scope'],'',report['no_decoder_scope'],'',report['historical_scope']]
    write_new(root / 'fixed-final-report.json', report)
    with (root / 'fixed-final-report.md').open('x') as stream:
        stream.write('\n'.join(lines)+'\n')


def value(args):
    folder = args.root / 'value-address-v1' / args.arm
    paths = [folder / control / 'evaluation.json' for control in ['normal','wrong-address']]
    reference = args.root / 'evaluations' / args.arm / 'noise-0/evaluation.json'
    records = [read(p) for p in [*paths,reference]]
    ids = read(args.previous_work / 'panels/panel-32.json')['sample_ids']
    seeds = read(args.previous_work / 'panels/seed-map.json')
    if len(ids)!=32 or any([r['sample_id'] for r in e['per_asset']]!=ids for e in records[:2]):
        raise ValueError('actual common32/order differs')
    if any(row['seed']!=seeds[row['sample_id']] for e in records[:2] for row in e['per_asset']):
        raise ValueError('actual sampling noise differs')
    report = value_summary(*records)
    report.update(candidate=args.arm, model_adopted=False,
        input_SHA256={str(p):sha(p) for p in [*paths,reference]},
        scope='Saved raw480/common32/noise0;candidate IDs,keys,logits,weights,support and per-query value multiset unchanged. Value-address sensitivity,not spatial No-corr or training-seed replication.')
    write_new(folder / 'summary.json', report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['queue','summarize','value'],required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--arm',choices=list(ARMS))
    for name in ['work-root','previous-work','source-contract','source-receipt','epoch-template','final-template']:
        p.add_argument('--'+name,type=Path)
    for name in ['epoch-sha','final-sha']:
        p.add_argument('--'+name)
    args=p.parse_args()
    if args.mode=='queue':queue(args)
    elif args.mode=='summarize':summarize(args.root)
    else:value(args)


if __name__=='__main__':main()
