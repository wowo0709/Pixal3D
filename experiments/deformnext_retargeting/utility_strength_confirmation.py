"""Confirm the strength-study candidate only after its closed fixed-final gain."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from . import frozen_utility_confirmation as common
from .frozen_utility_strength import NAME, NAMESPACE, validate_profile
from .matching_v2_queue import gpu_ids
from .semantic_topk_confirmation import make_jobs, seed_config
from .semantic_topk_functional_qa import job as functional_job
from .semantic_topk_search import write_new


QUEUE_JOB = 'queue-utility-strength-seed202-if-supported'
SUMMARY_JOB = 'summarize-utility-strength-seed202'
AUDIT_JOB = 'audit-final-when-ready-strength-seed202'
STOP_NAME = 'UTILITY_STRENGTH_SEED202_FINAL_WEIGHT_AUDITOR_STOP'
STATUS_NAME = 'utility-strength-seed202-final-weight-auditor-status.json'


def read(path):
    return json.loads(path.read_text())


def confirmation_recipe(source, root):
    validate_profile(source, root)
    folder = root / 'seed-confirmation-v1'
    cfg = seed_config(source, folder, NAME, 202)
    cfg.update(master_port='30445', wandb_id='ss-frozen-utility-scale16-c1-s202',
               wandb_name='SS frozen matching utility scale16 C1 seed202')
    validate_profile(cfg, folder)
    return cfg


def auditor_source(template):
    """Give the existing tensor checks their own lifecycle for this replica."""
    old_stop = 'UTILITY_STRENGTH_FINAL_WEIGHT_AUDITOR_STOP'
    old_status = 'utility-strength-final-weight-auditor-status'
    if template.count(old_stop) != 2 or template.count(old_status) != 2:
        raise ValueError('verified strength auditor lifecycle template required')
    return template.replace(old_stop, STOP_NAME).replace(old_status, STATUS_NAME.removesuffix('.json'))


def confirmation_jobs(root, campaign, previous_work, devices, *, allowed_gpus,
                      work_root, audit_driver, audit_implementation, audit_sha):
    assigned = gpu_ids(','.join(allowed_gpus))
    if len(devices) != 2 or len(set(devices)) != 2 or any(g not in assigned for g in devices):
        raise ValueError('two distinct explicitly assigned physical GPUs required')
    folder = root / 'seed-confirmation-v1'
    jobs = make_jobs(folder, previous_work, NAME, devices, [QUEUE_JOB], 202)
    command = jobs[0]['command']
    command[command.index('experiments.deformnext_retargeting.matching_v2_ss')] = (
        'experiments.deformnext_retargeting.frozen_utility_strength')
    qa = functional_job(root, folder, NAME, 202, previous_work / 'panels/panel-128.json')
    jobs.append(qa)
    training_job = 'confirm-seed202-train-' + NAME
    final_receipt = root / 'final-weight-audits-v1' / training_job / 'file4-closure.json'
    audit_result = final_receipt.parent / 'deferred-driver-result-v1.json'
    jobs.append({'name': AUDIT_JOB, 'resource': 'cpu', 'priority': 0,
        'depends_on': [training_job],
        'requires': [str(root / 'epoch-audits-v1' / training_job / 'epoch05/file4-closure.json'),
                     str(audit_driver), str(audit_implementation)],
        'outputs': [str(final_receipt), str(audit_result)],
        'command': ['taskset', '-c', '84,85', 'env', 'OMP_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=1',
                    sys.executable, '-u', str(audit_driver), '--implementation', str(audit_implementation),
                    '--expected-sha', audit_sha, '--stop-file', str(work_root / STOP_NAME),
                    '--receipt', str(final_receipt), '--status-file', str(work_root / STATUS_NAME),
                    '--job-name', training_job, '--output', str(audit_result)]})
    jobs.append({'name': SUMMARY_JOB, 'resource': 'cpu', 'priority': 1,
        'depends_on': [f'confirm-seed202-final-{arm}-noise{noise}'
                       for arm in [NAME, *common.CONTROLS] for noise in range(3)]
                      + [qa['name'], AUDIT_JOB, *['functional-QA-s202-' + arm for arm in common.CONTROLS]],
        'requires': [str(final_receipt)]
                    + [str(root / 'epoch-audits-v1' / training_job / f'epoch{e:02d}/file4-closure.json')
                       for e in range(1, 6)]
                    + [str(campaign / 'final-weight-audits-v1' / ('confirm-seed202-train-' + arm)
                           / 'file4-closure.json') for arm in common.CONTROLS]
                    + [str(campaign / 'epoch-audits-v1' / ('confirm-seed202-train-' + arm)
                           / f'epoch{e:02d}/file4-closure.json')
                       for arm in common.CONTROLS for e in range(1, 6)],
        'outputs': [str(folder / 'final-confirmation-report.json')],
        'command': ['taskset', '-c', '84,85', 'env', 'OMP_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=1',
                    sys.executable, '-u', '-m', 'experiments.deformnext_retargeting.utility_strength_confirmation',
                    '--mode', 'summarize', '--root', str(root), '--campaign-root', str(campaign)]})
    return jobs


def queue(args):
    path = args.root / 'fixed-final-decision-v1.json'
    common.require_closed_decision(path, args.root / 'fixed-final-report-file4.json', args.campaign_root)
    development = read(path)
    positive = common.decision_supported(development, candidate_name=NAME)
    output = args.root / 'seed202-queue-decision-v1.json'
    if output.exists():
        raise FileExistsError('preserve actual strength confirmation decision')
    for item in development['inputs']:
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('strength development input bytes changed')
    decision = {'status': 'STRENGTH_CONFIRMATION_QUEUED' if positive else 'NO_STRENGTH_CONFIRMATION_QUEUED',
        'candidate': NAME, 'eligible': positive,
        'development_decision_SHA256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'reserved121_evaluated': False, 'model_adopted': False, 'full_goal_complete': False,
        'next': 'Fresh202 against both matched Top1 controls; reserved validation and module effects remain.'
                if positive else 'Preserve failed fixed-final result and continue controlled model search.'}
    if not positive:
        write_new(output, decision)
        print(json.dumps(decision), flush=True)
        return
    if (hashlib.sha256(args.auditor_template.read_bytes()).hexdigest() != args.auditor_template_sha
            or hashlib.sha256(args.audit_driver.read_bytes()).hexdigest() != args.audit_driver_sha):
        raise ValueError('prepared actual audit implementation or lifecycle driver changed')
    cfg = confirmation_recipe(read(args.root / 'configs' / (NAME + '.json')), args.root)
    for arm in common.CONTROLS:
        common.validate_matched_recipe(cfg, read(args.campaign_root / 'seed-confirmation-v1'
                                                / 'configs' / (arm + '.json')), arm)
    allowed = getattr(args, 'allowed_gpus', None) or os.environ.get('MATCHING_V2_ALLOWED_GPUS')
    if allowed is None:
        raise ValueError('explicit allowed-gpus or scheduler assignment required')
    folder = args.root / 'seed-confirmation-v1'
    runtime = args.work_root / NAMESPACE / 'seed-confirmation-v1'
    implementation = runtime / 'audit_final_weights_v1.py'
    template = auditor_source(args.auditor_template.read_text())
    audit_sha = hashlib.sha256(template.encode()).hexdigest()
    jobs = confirmation_jobs(args.root, args.campaign_root, args.previous_work, gpu_ids(args.gpus),
        allowed_gpus=gpu_ids(allowed), work_root=args.work_root, audit_driver=args.audit_driver,
        audit_implementation=implementation, audit_sha=audit_sha)
    queue_path = args.work_root / 'queue-jobs.json'
    old = queue_path.read_bytes()
    existing = json.loads(old)
    names = [j['name'] for j in existing + jobs]
    if len(names) != len(set(names)) or any(dep not in names for j in jobs for dep in j.get('depends_on', [])):
        raise ValueError('strength confirmation dependency/name mismatch')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    cfg['experiment']['source_commit'] = commit
    for job in jobs:
        job['source_commit'] = commit
    write_new(folder / 'configs' / (NAME + '.json'), cfg)
    runtime.mkdir(parents=True, exist_ok=True)
    with implementation.open('x') as stream:
        stream.write(template)
    with (args.work_root / STOP_NAME).open('x') as stream:
        stream.write('DEFER_FINAL_AUDIT_FOR_RAM_V1\n')
    write_new(folder / 'candidate-contract.json', {**decision, 'source_commit': commit,
        'training_seed': 202, 'fresh_original_SS4210': True, 'master_port': 30445,
        'optimizer_and_data_unchanged': True, 'audit_implementation_SHA256': audit_sha,
        'auditor_template_SHA256': args.auditor_template_sha,
        'audit_lifecycle_driver_SHA256': args.audit_driver_sha,
        'final_audit_deferred_until_actual_raw480': True})
    with (args.work_root / 'queue-jobs-before-strength-confirmation-v1.json').open('xb') as stream:
        stream.write(old)
    if queue_path.read_bytes() != old:
        raise RuntimeError('concurrent queue edit; preserve evidence and reconcile')
    tmp = queue_path.with_suffix('.strength-confirmation.tmp')
    tmp.write_text(json.dumps(existing + jobs, indent=2) + '\n')
    os.replace(tmp, queue_path)
    decision.update(queued_jobs=len(jobs), source_commit=commit)
    write_new(output, decision)
    print(json.dumps(decision), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['queue', 'summarize'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--work-root', type=Path)
    parser.add_argument('--previous-work', type=Path)
    parser.add_argument('--gpus')
    parser.add_argument('--allowed-gpus')
    parser.add_argument('--auditor-template', type=Path)
    parser.add_argument('--auditor-template-sha')
    parser.add_argument('--audit-driver', type=Path)
    parser.add_argument('--audit-driver-sha')
    args = parser.parse_args()
    if args.mode == 'summarize':
        common.summarize(args.root, args.campaign_root,
                         candidate_name=NAME, profile_validator=validate_profile,
                         fixed_reference_sources={'v1': (
                             args.campaign_root.parent / 'ss-seed-confirmation-v1', 'v1')})
    else:
        if any(getattr(args, key) is None for key in ['work_root', 'previous_work', 'gpus',
                'auditor_template', 'auditor_template_sha', 'audit_driver', 'audit_driver_sha']):
            parser.error('queue requires explicit work, GPU assignment and verified auditor/driver paths and SHAs')
        queue(args)


if __name__ == '__main__':
    main()
