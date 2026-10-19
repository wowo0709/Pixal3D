"""Conditionally confirm an actually improved frozen-utility SS model."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .compare_ss_noise_panels import compare
from .frozen_matcher_utility import validate_profile
from .semantic_topk_confirmation import make_jobs, seed_config
from .semantic_topk_functional_qa import job as functional_job
from .semantic_topk_search import write_new
from .semantic_topk_selection import supported
from .summarize_frozen_utility_followup import NAME, validate_saved_audits
from .summarize_semantic_topk_search import aggregate, task_gradient_audit, validate_evaluation
from .summarize_ss_module_study import audit_final_replay, audit_training


QUEUE_JOB = 'queue-frozen-utility-seed202-if-supported'
SUMMARY_JOB = 'summarize-frozen-utility-seed202'
CONTROLS = ['static-top1-c1', 'top1-reference']


def read(path):
    return json.loads(path.read_text())


def decision_supported(report):
    if (report['candidate'] != NAME or report['model_adopted'] is not False
            or report['full_goal_complete'] is not False
            or report['first_round_negative_preserved'] is not True):
        raise ValueError('actual frozen-utility development decision required')
    positive = all(supported(report['contrasts'][name])
                   for name in ['v1', *CONTROLS])
    expected = ('PROMISING_FROZEN_UTILITY_REQUIRES_CONFIRMATION' if positive
                else 'NO_FROZEN_UTILITY_IMPROVEMENT_CONFIRMED')
    if (report['status'] != expected
            or report['eligible_for_training_seed_confirmation'] is not positive):
        raise ValueError('development eligibility disagrees with paired quality evidence')
    return positive


def confirmation_recipe(source, root):
    folder = root / 'seed-confirmation-v1'
    cfg = seed_config(source, folder, NAME, 202)
    cfg.update(master_port='30443', wandb_id='ss-frozen-utility-v1-c1-s202',
               wandb_name='SS frozen matching utility C1 seed202')
    validate_profile(cfg, folder)
    return cfg


def confirmation_jobs(root, campaign, previous_work, devices):
    if len(devices) != 2 or len(set(devices)) != 2 or any(g not in {'0', '1', '2', '3'} for g in devices):
        raise ValueError('two distinct assigned physical GPUs from0..3 required')
    folder = root / 'seed-confirmation-v1'
    jobs = make_jobs(folder, previous_work, NAME, devices, [QUEUE_JOB], 202)
    command = jobs[0]['command']
    command[command.index('experiments.deformnext_retargeting.matching_v2_ss')] = (
        'experiments.deformnext_retargeting.frozen_matcher_utility')
    qa = functional_job(root, folder, NAME, 202, previous_work / 'panels/panel-128.json')
    jobs.append(qa)
    jobs.append({
        'name': SUMMARY_JOB, 'resource': 'cpu', 'priority': 1,
        'depends_on': [f'confirm-seed202-final-{arm}-noise{noise}'
                       for arm in [NAME, *CONTROLS] for noise in range(3)]
                      + [qa['name'], *['functional-QA-s202-' + arm for arm in CONTROLS]],
        'requires': [str(root / 'final-weight-audits-v1' / ('confirm-seed202-train-' + NAME)
                         / 'file4-closure.json')]
                    + [str(root / 'epoch-audits-v1' / ('confirm-seed202-train-' + NAME)
                           / f'epoch{number:02d}/file4-closure.json') for number in range(1, 6)]
                    + [str(campaign / 'final-weight-audits-v1'
                           / ('confirm-seed202-train-' + arm) / 'file4-closure.json')
                       for arm in CONTROLS]
                    + [str(campaign / 'epoch-audits-v1' / ('confirm-seed202-train-' + arm)
                           / f'epoch{number:02d}/file4-closure.json')
                       for arm in CONTROLS for number in range(1, 6)],
        'outputs': [str(folder / 'final-confirmation-report.json')],
        'command': ['taskset', '-c', '84,85', 'env', 'OMP_NUM_THREADS=2',
                    'OPENBLAS_NUM_THREADS=1', sys.executable, '-u', '-m',
                    'experiments.deformnext_retargeting.frozen_utility_confirmation',
                    '--mode', 'summarize', '--root', str(root), '--campaign-root', str(campaign)],
    })
    return jobs


def validate_matched_recipe(candidate, control, arm):
    """Permit the required Omega context cache, holding optimization and RGB data fixed."""
    if arm not in CONTROLS:
        raise ValueError('undeclared confirmation control')
    for key in ['models', 'training_data', 'data_dir', 'num_gpus', 'num_nodes', 'training_seed']:
        if candidate[key] != control[key]:
            raise ValueError('confirmation control recipe differs: ' + key)
    left, right = copy.deepcopy(candidate['trainer']), copy.deepcopy(control['trainer'])
    candidate_spec = left['args'].pop('deform_conditioner')
    control_spec = right['args'].pop('deform_conditioner')
    if left != right:
        raise ValueError('confirmation control optimizer or trainer differs')
    for key in ['implementation', 'key_dim', 'value_dim', 'query_dim', 'topk', 'head_checkpoint']:
        if candidate_spec.get(key) != control_spec.get(key):
            raise ValueError('confirmation control matching input differs: ' + key)
    if arm == 'static-top1-c1':
        for spec in [candidate_spec, control_spec]:
            spec.pop('task_attention', None)
            spec.pop('candidate_aggregation', None)
            # Both Top1 consumers select raw-logit argmax before any
            # temperature scaling. This dormant control setting is irrelevant.
            spec.pop('score_temperature', None)
            spec.setdefault('correspondence_mode', 'semantic')
        if candidate_spec != control_spec:
            raise ValueError('same-retargeter control changes additional module settings')
    for key, args in [('dataset', 'args'), ('validation', 'dataset_args')]:
        left, right = copy.deepcopy(candidate[key]), copy.deepcopy(control[key])
        expected_arm = 'c1' if arm == 'static-top1-c1' else 'b2'
        if left[args].pop('transport_arm') != 'c1' or right[args].pop('transport_arm') != expected_arm:
            raise ValueError('confirmation control dataset transport arm differs')
        if left[args].pop('task_matching_features', None) is not True:
            raise ValueError('utility recipe requires contextual Omega features')
        if right[args].pop('task_matching_features', False) is not False or left != right:
            raise ValueError('confirmation control RGB data or validation protocol differs')


def require_closed_decision(path, receipt, campaign):
    closure = read(receipt)
    key = str(path.relative_to(campaign.parent))
    entries = [entry for entry in closure['files'] if entry['relative_path'] == key]
    if (closure['status'] != 'PASS' or closure.get('local_equals_actual_file4_SHA256') is not True
            or len(entries) != 1 or entries[0]['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()
            or entries[0]['bytes'] != path.stat().st_size):
        raise ValueError('actual file4 development decision bytes differ')


def queue(args):
    path = args.root / 'fixed-final-decision-v1.json'
    require_closed_decision(path, args.root / 'fixed-final-report-file4.json', args.campaign_root)
    report = read(path)
    positive = decision_supported(report)
    output = args.root / 'seed202-queue-decision-v1.json'
    if output.exists():
        raise FileExistsError('preserve actual confirmation queue decision')
    for item in report['inputs']:
        source = Path(item['path'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('development decision input bytes changed: ' + str(source))
    decision = {
        'status': 'FROZEN_UTILITY_CONFIRMATION_QUEUED' if positive else 'NO_F1_CONFIRMATION_QUEUED',
        'development_decision_SHA256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'eligible': positive, 'model_adopted': False, 'full_goal_complete': False,
        'next': 'Fresh202 plus same-retargeter Top1 comparison; reserved validation remains unopened.'
                if positive else 'Preserve negative result and continue model search.',
    }
    if not positive:
        write_new(output, decision)
        print(json.dumps(decision), flush=True)
        return
    cfg = confirmation_recipe(read(args.root / 'configs' / (NAME + '.json')), args.root)
    jobs = confirmation_jobs(args.root, args.campaign_root, args.previous_work, args.gpus.split(','))
    queue_path = args.work_root / 'queue-jobs.json'
    old = queue_path.read_bytes()
    existing = json.loads(old)
    names = [job['name'] for job in existing + jobs]
    if len(names) != len(set(names)) or any(
            dep not in set(names) for job in jobs for dep in job.get('depends_on', [])):
        raise ValueError('confirmation job name or dependency mismatch')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    for job in jobs:
        job['source_commit'] = commit
    folder = args.root / 'seed-confirmation-v1'
    write_new(folder / 'configs' / (NAME + '.json'), cfg)
    write_new(folder / 'candidate-contract.json', {
        **decision, 'source_commit': commit, 'training_seed': 202,
        'fresh_original_SS4210': True, 'master_port': 30443,
        'models_and_5epoch_budget_unchanged': True, 'same_retargeter_control_replicated': True,
        'no_reserved121_results_used_for_selection': True,
    })
    with (args.work_root / 'queue-jobs-before-frozen-utility-confirmation.json').open('xb') as stream:
        stream.write(old)
    if queue_path.read_bytes() != old:
        raise RuntimeError('concurrent queue edit; preserve created evidence and reconcile before retry')
    temporary = queue_path.with_suffix('.frozen-confirmation.tmp')
    temporary.write_text(json.dumps(existing + jobs, indent=2) + '\n')
    os.replace(temporary, queue_path)
    decision.update(queued_jobs=len(jobs), source_commit=commit)
    write_new(output, decision)
    print(json.dumps(decision), flush=True)


def verify_closed_audit(folder, campaign):
    closure, audit = read(folder / 'file4-closure.json'), folder / 'audit.json'
    key = str(audit.relative_to(campaign.parent))
    entries = [entry for entry in closure['files'] if entry['relative_path'] == key]
    if (closure['status'] != 'PASS' or closure.get('local_equals_actual_file4_SHA256') is not True
            or len(entries) != 1 or entries[0]['sha256'] != hashlib.sha256(audit.read_bytes()).hexdigest()
            or entries[0]['bytes'] != audit.stat().st_size):
        raise ValueError('actual file4 audit bytes differ')
    return read(audit)


def summarize(root, campaign):
    development = read(root / 'fixed-final-decision-v1.json')
    if not decision_supported(development):
        raise ValueError('no eligible development candidate')
    folder, base = root / 'seed-confirmation-v1', campaign / 'seed-confirmation-v1'
    cfg = read(folder / 'configs' / (NAME + '.json'))
    validate_profile(cfg, folder)
    if cfg['training_seed'] != 202:
        raise ValueError('fresh training seed202 required')
    training, records, models, inputs = {}, {}, {}, []
    for arm, location in [(NAME, folder), *[(name, base) for name in CONTROLS]]:
        training[arm] = audit_training(location, arm, development['training_audit']['initial_iou'])
        paths = [location / 'evaluations' / arm / f'noise-{noise}/evaluation.json' for noise in range(3)]
        records[arm] = [read(path) for path in paths]
        for value in records[arm]:
            validate_evaluation(value)
        training[arm]['final128_replays_epoch32'] = audit_final_replay(training[arm], records[arm][0])
        qa_path = (root if arm == NAME else campaign) / 'functional-QA-v1' / ('s202-' + arm) / 'summary.json'
        qa = read(qa_path)
        if qa['status'] != 'PASS_TRAINED_SS_CACHE_AMP_CFG_CONTRACT' or qa['optimizer_updates'] != 0:
            raise ValueError('actual saved seed202 GPU proof required')
        recipe = location / 'configs' / (arm + '.json')
        if read(recipe)['training_seed'] != 202:
            raise ValueError('control did not use training seed202')
        if arm in CONTROLS:
            control = read(recipe)
            validate_matched_recipe(cfg, control, arm)
            spec = control['trainer']['args']['deform_conditioner']
            if (control['ckpt'] != 'none' or control['load_dir'] != ''
                    or control['trainer']['args'].get('task_attention_optimizer_eps') is not None
                    or spec.get('task_attention') is not None
                    or spec['implementation'] != 'r9-native-v5'
                    or (arm == 'top1-reference' and spec['arm'] != 'b2')
                    or (arm == 'static-top1-c1'
                        and (spec['arm'] != 'c1' or spec['candidate_aggregation'] != 'top1'))):
                raise ValueError('seed202 Top1 control structure differs')
            if (qa.get('saved_frozen_head_exact_to_selected_checkpoint') is not True
                    or len(qa.get('frozen_head_state_keys', [])) != 24):
                raise ValueError('Top1 control lost the fixed original matching head')
            job = 'confirm-seed202-train-' + arm
            audit_folder = campaign / 'final-weight-audits-v1' / job
            final_control = verify_closed_audit(audit_folder, campaign)
            if (final_control['actual_optimizer_step'] != 480
                    or final_control['frozen_base_tensors_exact'] != 701
                    or final_control['LoRA_tensors_updated'] != 360
                    or final_control['matching_tensors_changed'] != {'decoder': [], 'score': []}):
                raise ValueError('Top1 control saved state differs')
            folders = [audit_folder]
            for number in range(1, 6):
                epoch_folder = campaign / 'epoch-audits-v1' / job / f'epoch{number:02d}'
                epoch = verify_closed_audit(epoch_folder, campaign)
                if (epoch['training_seed'] != 202 or epoch['step'] != 96 * number
                        or epoch['global_batch'] != 96 or epoch['microbatch_per_rank'] != 48
                        or epoch['accumulation'] != 1 or epoch['world_size'] != 2
                        or not epoch['actual_remote_four_numeric_fields_verified']
                        or not epoch['actual_remote_PNG_byte_exact']):
                    raise ValueError('Top1 control budget or actual epoch logging differs')
                folders.append(epoch_folder)
            for audit_folder in folders:
                inputs.extend([audit_folder / 'audit.json', audit_folder / 'file4-closure.json'])
        models[arm] = aggregate(records[arm])
        inputs.extend([recipe, qa_path, *paths])
    job = 'confirm-seed202-train-' + NAME
    final_folder = root / 'final-weight-audits-v1' / job
    epoch_folders = [root / 'epoch-audits-v1' / job / f'epoch{n:02d}' for n in range(1, 6)]
    final = verify_closed_audit(final_folder, campaign)
    epochs = [verify_closed_audit(path, campaign) for path in epoch_folders]
    qa_path = root / 'functional-QA-v1' / ('s202-' + NAME) / 'summary.json'
    validate_saved_audits(final, read(qa_path), epochs, training_seed=202)
    training[NAME]['task_branch_gradients'] = task_gradient_audit(folder, NAME)
    for path in [final_folder, *epoch_folders]:
        inputs.extend([path / 'audit.json', path / 'file4-closure.json'])
    first_records = [read(root / 'evaluations' / NAME / f'noise-{n}/evaluation.json') for n in range(3)]
    for noise, value in enumerate(records[NAME]):
        previous = {row['sample_id']: row for row in first_records[noise]['per_asset']}
        for row in value['per_asset']:
            if any(row[key] != previous[row['sample_id']][key] for key in [
                    'source_cluster_id', 'seed', 'target_voxels_sha256', 'target_voxel_shape']):
                raise ValueError('seed confirmation changed development inputs or generation noise')
    contrasts = {arm: compare([(str(n), records[arm][n], records[NAME][n]) for n in range(3)])
                 for arm in CONTROLS}
    replicated = all(supported(value) for value in contrasts.values())
    result = {
        'status': 'PASS_ACTUAL_FROZEN_UTILITY_SEED202_COMPARISON',
        'positive_CI_and_all3noise_direction_at_both_training_seeds': replicated,
        'per_training_seed_contrasts': {'101': {arm: development['contrasts'][arm] for arm in CONTROLS},
                                        '202': contrasts},
        'models': models, 'training_audits': training,
        'inputs': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in dict.fromkeys([root / 'fixed-final-decision-v1.json', *inputs])],
        'scope': 'Two independent optimization seeds on the same development128/93sources/3noises. '
                 'Same-retargeter Top1 effect compared at both seeds; conditional source-cluster CIs, '
                 'no training-seed population interval or independent test claim.',
        'reserved121_evaluated': False, 'model_adopted': False, 'full_goal_complete': False,
        'next': 'Lock candidate, then evaluate reserved source-disjoint121 and module effects.'
                if replicated else 'No adoption; retain failed replication and continue model search.',
    }
    write_new(folder / 'final-confirmation-report.json', result)
    print(json.dumps({'status': result['status'], 'replicated': replicated}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['queue', 'summarize'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--work-root', type=Path)
    parser.add_argument('--previous-work', type=Path)
    parser.add_argument('--gpus')
    args = parser.parse_args()
    if args.mode == 'summarize':
        summarize(args.root, args.campaign_root)
    else:
        if args.work_root is None or args.previous_work is None or args.gpus is None:
            parser.error('queue requires work-root, previous-work and explicit gpus')
        queue(args)


if __name__ == '__main__':
    main()
