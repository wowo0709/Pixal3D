"""Queue two isolated model changes only after the current fixed-final negative."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from . import utility_softprior_followup as soft
from . import voxel_query_followup as voxel
from .frozen_utility_confirmation import decision_supported, require_closed_decision
from .frozen_utility_strength import NAME as STRENGTH_NAME
from .matching_v2_queue import gpu_ids
from .semantic_topk_confirmation import make_jobs
from .semantic_topk_functional_qa import job as functional_job
from .semantic_topk_search import write_new
from .summarize_frozen_utility_followup import summarize as summarize_saved_profile


NAMESPACE = 'orthogonal-utility-followups-v1'
FACTORY = 'queue-orthogonal-utility-followups-if-needed'
MODULE = 'experiments.deformnext_retargeting.orthogonal_utility_followups'
PROFILES = {voxel.NAME: voxel, soft.NAME: soft}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu():
    return ['taskset', '-c', '84,85', 'env', 'OMP_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=1', sys.executable, '-u']


def close_job(root, campaign, work, previous, name, dependencies, files, receipt):
    manifest = work / (name + '-files.json')
    write_new(manifest, [str(path.relative_to(campaign.parent)) for path in files])
    return {'name': name, 'resource': 'cpu', 'priority': 2, 'depends_on': dependencies,
        'requires': [str(path) for path in files], 'outputs': [str(receipt)],
        'command': cpu() + [str(previous / 'verify_closed_file4_subset.py'), '--manifest', str(manifest),
            '--receipt', str(receipt.relative_to(campaign.parent)),
            '--scope', 'Actual isolated follow-up result,including negatives;not model adoption or reserved121 validation.']}


def study_jobs(profile, campaign, work, previous, devices, *, source_receipt,
               epoch_template, epoch_sha, final_template, final_sha):
    root = campaign / profile.NAMESPACE
    name = profile.NAME
    config = root / 'configs' / (name + '.json')
    run = root / 'runs_ss64' / name
    dependencies = [FACTORY] + (['native-proof-voxel-query-c1'] if profile is voxel else [])
    jobs = make_jobs(root, previous, name, devices, dependencies, 101, prefix='')
    command = jobs[0]['command']
    command[command.index('experiments.deformnext_retargeting.matching_v2_ss')] = profile.__name__
    jobs[0]['requires'].append(str(source_receipt))
    if profile is voxel:
        jobs[0]['requires'].append(str(root / 'verification' / name / 'standalone-native-v1/query-source-binding-proof.json'))
    qa = functional_job(root, root, name, 101, previous / 'panels/panel-128.json')
    jobs.append(qa)
    training = jobs[0]['name']
    epoch_jobs = []
    for number in range(1, 6):
        folder = root / 'epoch-audits-v1' / training / f'epoch{number:02d}'
        label = f'audit-epoch-{name}-{number:02d}'
        epoch_jobs.append(label)
        jobs.append({'name': label, 'resource': 'cpu', 'priority': 0,
            'requires': [str(source_receipt), str(run / 'ckpts' / f'misc_step{96*number:07d}.pt'),
                         str(run / 'validation.jsonl')],
            'outputs': [str(folder / 'file4-closure.json'), str(folder / 'once-driver-result-v1.json')],
            'command': cpu() + ['-m', 'experiments.deformnext_retargeting.run_saved_audit_once',
                '--kind', 'epoch', '--implementation', str(epoch_template), '--expected-sha', epoch_sha,
                '--root', str(root), '--work-root', str(work), '--job-name', training,
                '--config', str(config), '--epoch', str(number)]})
    final_folder = root / 'final-weight-audits-v1' / training
    final_job = 'audit-final-' + name
    jobs.append({'name': final_job, 'resource': 'cpu', 'priority': 0,
        'depends_on': [training, epoch_jobs[-1]], 'requires': [str(source_receipt)],
        'outputs': [str(final_folder / 'file4-closure.json'), str(final_folder / 'once-driver-result-v1.json')],
        'command': cpu() + ['-m', 'experiments.deformnext_retargeting.run_saved_audit_once',
            '--kind', 'final', '--implementation', str(final_template), '--expected-sha', final_sha,
            '--root', str(root), '--work-root', str(work), '--job-name', training, '--config', str(config)]})
    summary_job = 'summarize-fixed-final-' + name
    jobs.append({'name': summary_job, 'resource': 'cpu', 'priority': 2,
        'depends_on': [final_job, qa['name'], *epoch_jobs] + [f'final-{name}-noise{n}' for n in range(3)],
        'outputs': [str(root / 'fixed-final-decision-v1.json'), str(root / 'fixed-final-decision-v1.md')],
        'command': cpu() + ['-m', MODULE, '--mode', 'summarize', '--campaign-root', str(campaign),
                           '--candidate', name]})
    jobs.append(close_job(root, campaign, work, previous, 'close-fixed-final-' + name,
        [summary_job], [root / 'fixed-final-decision-v1.json', root / 'fixed-final-decision-v1.md'],
        root / 'fixed-final-report-file4.json'))
    folder = root / 'task-value-address-v1'
    value_jobs = []
    for control in ['normal', 'wrong-address']:
        label = name + '-value-address-' + control
        value_jobs.append(label)
        jobs.append({'name': label, 'resource': 'evaluation', 'priority': 3,
            'depends_on': [training, qa['name'], f'final-{name}-noise0'],
            'requires': [str(source_receipt)],
            'outputs': [str(folder / control / 'evaluation.json'), str(folder / control / 'panel.png')],
            'command': [sys.executable, '-u', '-m', 'experiments.deformnext_retargeting.evaluate_native_checkpoint',
                '--stage', 'ss64', '--config-root', str(config.parent), '--arm-config', str(config),
                '--checkpoint', str(run / 'ckpts/denoiser_step0000480.pt'),
                '--conditioner-checkpoint', str(run / 'ckpts/deform_conditioner_step0000480.pt'),
                '--step', '480', '--output', str(folder / control),
                '--panel-file', str(previous / 'panels/panel-32.json'),
                '--seed-map-file', str(previous / 'panels/seed-map.json'),
                '--generation-batch', '8', '--task-value-control', control]})
    value_summary = 'summarize-value-address-' + name
    jobs.append({'name': value_summary, 'resource': 'cpu', 'priority': 3, 'depends_on': value_jobs,
        'outputs': [str(folder / 'summary.json')], 'command': cpu() + ['-m',
            'experiments.deformnext_retargeting.summarize_frozen_utility_value_address',
            '--root', str(root), '--candidate', name]})
    files = [folder / 'contract.json', folder / 'summary.json']
    files += [folder / control / filename for control in ['normal', 'wrong-address']
              for filename in ['evaluation.json', 'panel.png']]
    jobs.append(close_job(root, campaign, work, previous, 'close-value-address-' + name,
        [value_summary], files, folder / 'file4-closure.json'))
    return jobs


def require_source_contract(contract_path, receipt_path, campaign):
    receipt = read(receipt_path)
    if receipt.get('status') != 'PASS' or receipt.get('local_equals_actual_file4_SHA256') is not True:
        raise ValueError('actual prelaunch source/config file4 closure required')
    components = read(contract_path)['runtime_components']
    if not components or len({row['runtime_path'] for row in components}) != len(components):
        raise ValueError('nonempty distinct runtime source/config bindings required')
    rows = {row['relative_path']: row for row in receipt['files']}
    for path in [contract_path, *[Path(row['archived_path']) for row in components]]:
        key = str(path.relative_to(campaign.parent))
        if key not in rows or sha(path) != rows[key]['sha256']:
            raise ValueError('closed prelaunch source/contract bytes changed')
    for row in components:
        if sha(Path(row['runtime_path'])) != row['sha256'] or sha(Path(row['archived_path'])) != row['sha256']:
            raise ValueError('executed follow-up source or prepared recipe changed')


def queue(args):
    folder = args.campaign_root / NAMESPACE
    output = folder / 'queue-decision-v1.json'
    if output.exists():
        raise FileExistsError('preserve actual follow-up branch decision')
    strength = args.campaign_root / 'utility-strength-followup-v1'
    development = strength / 'fixed-final-decision-v1.json'
    require_closed_decision(development, strength / 'fixed-final-report-file4.json', args.campaign_root)
    report = read(development)
    for row in report['inputs']:
        if sha(Path(row['path'])) != row['sha256']:
            raise ValueError('closed strength decision input changed')
    positive = decision_supported(report, candidate_name=STRENGTH_NAME)
    seed_branch = read(strength / 'seed202-queue-decision-v1.json')
    expected_branch = 'STRENGTH_CONFIRMATION_QUEUED' if positive else 'NO_STRENGTH_CONFIRMATION_QUEUED'
    if (seed_branch.get('candidate') != STRENGTH_NAME or seed_branch.get('status') != expected_branch
            or seed_branch['eligible'] != positive or seed_branch['development_decision_SHA256'] != sha(development)):
        raise ValueError('strength seed-confirmation branch differs from closed development')
    decision = {'status': 'SKIPPED_FOR_STRENGTH_CONFIRMATION' if positive else 'ORTHOGONAL_FOLLOWUPS_QUEUED',
        'strength_development_SHA256': sha(development), 'reserved121_accessed': False,
        'model_adopted': False, 'full_goal_complete': False}
    if positive:
        write_new(output, decision)
        print(json.dumps(decision), flush=True)
        return
    require_source_contract(args.source_contract, args.source_receipt, args.campaign_root)
    for path, digest in [(args.epoch_template, args.epoch_sha), (args.final_template, args.final_sha)]:
        if sha(path) != digest:
            raise ValueError('immutable owned numerical audit template changed')
    allowed = gpu_ids(args.allowed_gpus)
    devices = [gpu_ids(args.voxel_gpus), gpu_ids(args.soft_gpus)]
    if (any(len(pair) != 2 for pair in devices) or len(set(devices[0] + devices[1])) != 4
            or any(device not in allowed for pair in devices for device in pair)):
        raise ValueError('two disjoint explicitly assigned physical GPU pairs required')
    jobs = []
    for profile, pair in zip([voxel, soft], devices):
        root = args.campaign_root / profile.NAMESPACE
        profile.validate_profile(read(root / 'configs' / (profile.NAME + '.json')), root)
        jobs.extend(study_jobs(profile, args.campaign_root, args.work_root, args.previous_work, pair,
            source_receipt=args.source_receipt, epoch_template=args.epoch_template, epoch_sha=args.epoch_sha,
            final_template=args.final_template, final_sha=args.final_sha))
        contract_folder = root / 'task-value-address-v1'
        panel = args.previous_work / 'panels/panel-32.json'
        seeds = args.previous_work / 'panels/seed-map.json'
        write_new(contract_folder / 'contract.json', {'status': 'QUEUED_NOT_EVALUATED',
            'candidate': profile.NAME, 'candidate_preselected': False,
            'normal': str(contract_folder / 'normal/evaluation.json'),
            'altered': str(contract_folder / 'wrong-address/evaluation.json'),
            'reference': str(root / 'evaluations' / profile.NAME / 'noise-0/evaluation.json'),
            'panel': str(panel), 'seed_map': str(seeds), 'panel_SHA256': sha(panel), 'seed_map_SHA256': sha(seeds),
            'scope': 'Same raw480/common32/noise0 value-address sensitivity,not native-spatial No-corr training or model adoption.'})
    path = args.work_root / 'queue-jobs.json'
    before = path.read_bytes(); existing = json.loads(before)
    names = [job['name'] for job in existing + jobs]
    if len(names) != len(set(names)) or any(dep not in names for job in jobs for dep in job.get('depends_on', [])):
        raise ValueError('follow-up queue name/dependency mismatch')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    for job in jobs:
        job['source_commit'] = commit
    with (args.work_root / 'queue-before-orthogonal-followups-main-v1.json').open('xb') as stream:
        stream.write(before)
    if path.read_bytes() != before:
        raise RuntimeError('concurrent queue change; preserve and reconcile')
    temporary = path.with_suffix('.orthogonal-main.tmp')
    temporary.write_text(json.dumps(existing + jobs, indent=2) + '\n')
    os.replace(temporary, path)
    decision.update(source_commit=commit, queued_jobs=len(jobs), new_training_runs=2,
        actual_budget='Each freshSS4210/seed101/raw480/global96/DDP2/micro48/accumulation1.',
        source_contract_SHA256=sha(args.source_contract), source_receipt_SHA256=sha(args.source_receipt))
    write_new(output, decision)
    print(json.dumps(decision), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['queue', 'summarize'], required=True)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--candidate', choices=list(PROFILES))
    for name in ['work-root', 'previous-work', 'source-contract', 'source-receipt', 'epoch-template', 'final-template']:
        parser.add_argument('--' + name, type=Path)
    for name in ['allowed-gpus', 'voxel-gpus', 'soft-gpus', 'epoch-sha', 'final-sha']:
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if args.mode == 'summarize':
        if args.candidate is None:
            parser.error('summarize requires --candidate')
        profile = PROFILES[args.candidate]
        summarize_saved_profile(args.campaign_root / profile.NAMESPACE, args.campaign_root,
            name=profile.NAME, namespace=profile.NAMESPACE, profile_validator=profile.validate_profile,
            extra_sources={'reference-scale16': (args.campaign_root / 'utility-strength-followup-v1', STRENGTH_NAME),
                           'reference-scale4': (args.campaign_root / 'frozen-utility-followup-v1', 'frozen-utility-c1')})
    else:
        required = ['work_root', 'previous_work', 'source_contract', 'source_receipt', 'epoch_template',
                    'final_template', 'allowed_gpus', 'voxel_gpus', 'soft_gpus', 'epoch_sha', 'final_sha']
        if any(getattr(args, name) is None for name in required):
            parser.error('queue requires explicit source closure,audit templates and assigned GPU pairs')
        queue(args)


if __name__ == '__main__':
    main()
