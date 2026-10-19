"""Replicate supported orthogonal candidates without changing their first runs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from . import frozen_utility_confirmation as common
from . import orthogonal_utility_followups as study
from .compare_ss_noise_panels import compare
from .loader_runtime_contract import validate_loader_runtime
from .matching_v2_queue import gpu_ids
from .matching_v2_ss import train as train_shared
from .semantic_topk_confirmation import make_jobs, seed_config
from .semantic_topk_functional_qa import job as functional_job
from .semantic_topk_campaign_wandb import validate_saved_qa
from .semantic_topk_search import write_new
from .semantic_topk_selection import supported
from .summarize_frozen_utility_followup import validate_saved_audits
from .summarize_semantic_topk_search import aggregate, task_gradient_audit, validate_evaluation
from .summarize_ss_module_study import audit_final_replay, audit_training


MODULE = 'experiments.deformnext_retargeting.orthogonal_utility_confirmation'
FACTORY = 'queue-orthogonal-utility-confirmation-stage'
DECISION = 'queue-orthogonal-positive-seed202-if-supported'
PORTS = {'voxel-query-c1': '30448', 'utility-softprior-c1': '30449'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replica_recipe(source, root, profile):
    profile.validate_profile(source, root)
    cfg = seed_config(source, root / 'seed-confirmation-v1', profile.NAME, 202)
    cfg.update(master_port=PORTS[profile.NAME], wandb_id=f'ss-{profile.NAME}-s202',
               wandb_name=f'SS {profile.NAME} independent training seed202')
    return cfg


def validate_replica(cfg, folder, profile):
    root = folder.parent
    if folder.name != 'seed-confirmation-v1' or root.name != profile.NAMESPACE:
        raise ValueError('explicit candidate seed-confirmation namespace required')
    source = read(root / 'configs' / (profile.NAME + '.json'))
    expected = replica_recipe(source, root, profile)
    actual = copy.deepcopy(cfg)
    # Entry commit is recorded separately; all numerical/data/runtime settings
    # must equal the immutable seed101 recipe except these declared aliases.
    for value in [actual, expected]:
        value['experiment'].pop('source_commit', None)
    if actual != expected:
        raise ValueError('replica changed more than declared seed,paths,port and logging')


def validate_control(candidate, control, arm, profile):
    normalized = copy.deepcopy(candidate)
    validate_loader_runtime(normalized)
    # The two-worker runtime has an actual source/data/order/RNG parity proof.
    # Its worker count is not an optimization or architecture difference.
    normalized['trainer']['args']['num_workers'] = control['trainer']['args'].get('num_workers', 8)
    if profile is study.voxel:
        if normalized['trainer']['name'] != profile.TRAINER:
            raise ValueError('declared voxel-query trainer required')
        normalized['trainer']['name'] = control['trainer']['name']
        if normalized['trainer']['args']['deform_conditioner'].pop('voxel_chunk_size', None) != 128:
            raise ValueError('declared voxel implementation chunk required')
    common.validate_matched_recipe(normalized, control, arm)


def verify_development(root, campaign, profile):
    path = root / 'fixed-final-decision-v1.json'
    common.require_closed_decision(path, root / 'fixed-final-report-file4.json', campaign)
    report = read(path)
    eligible = common.decision_supported(report, candidate_name=profile.NAME)
    for row in report['inputs']:
        if sha(Path(row['path'])) != row['sha256']:
            raise ValueError('development input bytes changed')
    return report, eligible


def append_jobs(work, jobs, backup):
    path = work / 'queue-jobs.json'
    before = path.read_bytes()
    existing = json.loads(before)
    names = [job['name'] for job in existing + jobs]
    if len(names) != len(set(names)) or any(dep not in names for job in jobs for dep in job.get('depends_on', [])):
        raise ValueError('confirmation queue name/dependency mismatch')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    for job in jobs:
        job['source_commit'] = commit
    with (work / backup).open('xb') as stream:
        stream.write(before)
    if path.read_bytes() != before:
        raise RuntimeError('concurrent queue edit; preserve and reconcile')
    temporary = path.with_suffix('.orthogonal-confirmation.tmp')
    temporary.write_text(json.dumps(existing + jobs, indent=2) + '\n')
    os.replace(temporary, path)


def command(args, mode):
    return study.cpu() + ['-m', MODULE, '--mode', mode, '--campaign-root', str(args.campaign_root),
        '--work-root', str(args.work_root), '--previous-work', str(args.previous_work),
        '--source-contract', str(args.source_contract), '--source-receipt', str(args.source_receipt),
        '--epoch-template', str(args.epoch_template), '--epoch-sha', args.epoch_sha,
        '--final-template', str(args.final_template), '--final-sha', args.final_sha,
        '--allowed-gpus', args.allowed_gpus, '--voxel-gpus', args.voxel_gpus,
        '--soft-gpus', args.soft_gpus]


def stage(args):
    study.require_source_contract(args.source_contract, args.source_receipt, args.campaign_root)
    folder = args.campaign_root / study.NAMESPACE
    branch = read(folder / 'queue-decision-v1.json')
    if branch['status'] == 'SKIPPED_FOR_STRENGTH_CONFIRMATION':
        jobs = []
    elif branch['status'] == 'ORTHOGONAL_FOLLOWUPS_QUEUED':
        jobs = [{'name': DECISION, 'resource': 'cpu', 'priority': 2,
            'depends_on': [FACTORY, *['close-fixed-final-' + name for name in study.PROFILES]],
            'requires': [str(args.source_receipt)],
            'outputs': [str(folder / 'seed202-queue-decision-v1.json')],
            'command': command(args, 'queue')}]
        append_jobs(args.work_root, jobs, 'queue-before-orthogonal-confirmation-decision-v1.json')
    else:
        raise ValueError('actual completed alternative-launch branch required')
    write_new(folder / 'confirmation-stage-v1.json', {'status': 'CONFIRMATION_DECISION_QUEUED' if jobs
        else 'SKIPPED_FOR_EXISTING_STRENGTH_CONFIRMATION', 'queued_jobs': len(jobs),
        'source_branch_SHA256': sha(folder / 'queue-decision-v1.json'),
        'model_adopted': False, 'reserved121_evaluated': False, 'full_goal_complete': False})


def replica_jobs(args, profile, devices):
    campaign = args.campaign_root
    root = campaign / profile.NAMESPACE
    folder = root / 'seed-confirmation-v1'
    name = profile.NAME
    config = folder / 'configs' / (name + '.json')
    run = folder / 'runs_ss64' / name
    training = 'confirm-seed202-train-' + name
    jobs = make_jobs(folder, args.previous_work, name, devices, [DECISION], 202)
    cmd = jobs[0]['command']
    cmd[cmd.index('experiments.deformnext_retargeting.matching_v2_ss')] = MODULE
    cmd.extend(['--candidate', name, '--campaign-root', str(campaign),
                '--source-contract', str(args.source_contract), '--source-receipt', str(args.source_receipt)])
    jobs[0]['requires'].append(str(args.source_receipt))
    qa = functional_job(root, folder, name, 202, args.previous_work / 'panels/panel-128.json')
    jobs.append(qa)
    epochs = []
    for number in range(1, 6):
        label = f'audit-epoch-s202-{name}-{number:02d}'
        epochs.append(label)
        output = folder / 'epoch-audits-v1' / training / f'epoch{number:02d}'
        jobs.append({'name': label, 'resource': 'cpu', 'priority': 0,
            'requires': [str(args.source_receipt), str(run / 'ckpts' / f'misc_step{number*96:07d}.pt'),
                         str(run / 'validation.jsonl')],
            'outputs': [str(output / 'file4-closure.json'), str(output / 'once-driver-result-v1.json')],
            'command': study.cpu() + ['-m', 'experiments.deformnext_retargeting.run_saved_audit_once',
                '--kind', 'epoch', '--implementation', str(args.epoch_template), '--expected-sha', args.epoch_sha,
                '--root', str(folder), '--work-root', str(args.work_root), '--job-name', training,
                '--config', str(config), '--epoch', str(number)]})
    final_label = 'audit-final-s202-' + name
    final = folder / 'final-weight-audits-v1' / training
    jobs.append({'name': final_label, 'resource': 'cpu', 'priority': 0,
        'depends_on': [training, epochs[-1]], 'requires': [str(args.source_receipt)],
        'outputs': [str(final / 'file4-closure.json'), str(final / 'once-driver-result-v1.json')],
        'command': study.cpu() + ['-m', 'experiments.deformnext_retargeting.run_saved_audit_once',
            '--kind', 'final', '--implementation', str(args.final_template), '--expected-sha', args.final_sha,
            '--root', str(folder), '--work-root', str(args.work_root), '--job-name', training,
            '--config', str(config)]})
    summary = 'summarize-s202-' + name
    jobs.append({'name': summary, 'resource': 'cpu', 'priority': 2,
        'depends_on': [final_label, qa['name'], *epochs,
            *[f'confirm-seed202-final-{arm}-noise{noise}' for arm in [name, *common.CONTROLS] for noise in range(3)],
            *['functional-QA-s202-' + arm for arm in common.CONTROLS]],
        'requires': [str(args.source_receipt)],
        'outputs': [str(folder / 'final-confirmation-report.json')],
        'command': study.cpu() + ['-m', MODULE, '--mode', 'summarize', '--campaign-root', str(campaign),
                                '--candidate', name]})
    jobs.append(study.close_job(root, campaign, args.work_root, args.previous_work,
        'close-s202-' + name, [summary], [folder / 'final-confirmation-report.json'],
        folder / 'final-confirmation-report-file4.json'))
    return jobs


def queue(args):
    study.require_source_contract(args.source_contract, args.source_receipt, args.campaign_root)
    if any(sha(path) != digest for path, digest in [(args.epoch_template, args.epoch_sha),
                                                   (args.final_template, args.final_sha)]):
        raise ValueError('immutable owned audit template changed')
    allowed = gpu_ids(args.allowed_gpus)
    pairs = [gpu_ids(args.voxel_gpus), gpu_ids(args.soft_gpus)]
    if (any(len(pair) != 2 for pair in pairs) or len(set(sum(pairs, []))) != 4
            or any(device not in allowed for pair in pairs for device in pair)):
        raise ValueError('two disjoint explicitly assigned physical GPU pairs required')
    jobs, outcomes = [], {}
    for (name, profile), devices in zip(study.PROFILES.items(), pairs):
        root = args.campaign_root / profile.NAMESPACE
        report, positive = verify_development(root, args.campaign_root, profile)
        outcomes[name] = {'eligible': positive, 'development_SHA256': sha(root / 'fixed-final-decision-v1.json')}
        if not positive:
            continue
        cfg = replica_recipe(read(root / 'configs' / (name + '.json')), root, profile)
        for arm in common.CONTROLS:
            validate_control(cfg, read(args.campaign_root / 'seed-confirmation-v1/configs' / (arm + '.json')),
                             arm, profile)
        cfg['experiment']['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        write_new(root / 'seed-confirmation-v1/configs' / (name + '.json'), cfg)
        jobs.extend(replica_jobs(args, profile, devices))
    if jobs:
        append_jobs(args.work_root, jobs, 'queue-before-orthogonal-positive-seed202-v1.json')
    write_new(args.campaign_root / study.NAMESPACE / 'seed202-queue-decision-v1.json', {
        'status': 'SUPPORTED_CANDIDATE_REPLICAS_QUEUED' if jobs else 'NO_ORTHOGONAL_CANDIDATE_SUPPORTED',
        'outcomes': outcomes, 'queued_jobs': len(jobs),
        'policy': 'Independently replicate each fixed-final candidate with positive source-cluster CI '
                  'and all3noise direction against V1 and both Top1 controls; no best-epoch selection.',
        'training_seed': 202, 'fresh_SS4210': True, 'global_batch': 96, 'microbatch_per_rank': 48,
        'accumulation': 1, 'source_receipt_SHA256': sha(args.source_receipt),
        'reserved121_evaluated': False, 'model_adopted': False, 'full_goal_complete': False})


def summarize(campaign, profile):
    root = campaign / profile.NAMESPACE
    development, positive = verify_development(root, campaign, profile)
    if not positive:
        raise ValueError('unsupported candidate cannot be confirmed')
    folder = root / 'seed-confirmation-v1'
    cfg = read(folder / 'configs' / (profile.NAME + '.json'))
    validate_replica(cfg, folder, profile)
    verification = folder / 'verification' / profile.NAME
    entry, exit_record = read(verification / 'replica-entry-source-v1.json'), read(verification / 'replica-exit-source-v1.json')
    if (exit_record['status'] != 'PASS_REPLICA_SOURCE_BINDING'
            or entry['components'] != exit_record['components']
            or entry['module_SHA256'] != exit_record['module_SHA256']
            or exit_record['native_proof_SHA256'] != sha(verification / 'native-proof.json')):
        raise ValueError('replica actual source/native-proof binding differs')
    records, audits, models, inputs = {}, {}, {}, [root / 'fixed-final-decision-v1.json', root / 'fixed-final-report-file4.json',
        verification / 'replica-entry-source-v1.json', verification / 'replica-exit-source-v1.json']
    for arm, location, qa_root in [(profile.NAME, folder, root),
            *[(arm, campaign / 'seed-confirmation-v1', campaign) for arm in common.CONTROLS]]:
        recipe = location / 'configs' / (arm + '.json')
        actual_cfg = read(recipe)
        if actual_cfg['training_seed'] != 202 or actual_cfg['ckpt'] != 'none' or actual_cfg['load_dir'] != '':
            raise ValueError('fresh training seed202 required for candidate and controls')
        if arm in common.CONTROLS:
            validate_control(cfg, actual_cfg, arm, profile)
        audits[arm] = audit_training(location, arm, development['training_audit']['initial_iou'])
        paths = [location / 'evaluations' / arm / f'noise-{noise}/evaluation.json' for noise in range(3)]
        records[arm] = [read(path) for path in paths]
        for value in records[arm]:
            validate_evaluation(value)
        audits[arm]['final128_replays_epoch32'] = audit_final_replay(audits[arm], records[arm][0])
        qa_path = qa_root / 'functional-QA-v1' / ('s202-' + arm) / 'summary.json'
        qa = read(qa_path)
        validate_saved_qa(qa, actual_cfg['trainer']['args']['deform_conditioner'])
        audit_root = folder if arm == profile.NAME else campaign
        job = 'confirm-seed202-train-' + arm
        final_folder = audit_root / 'final-weight-audits-v1' / job
        epochs = [audit_root / 'epoch-audits-v1' / job / f'epoch{number:02d}' for number in range(1, 6)]
        final = common.verify_closed_audit(final_folder, campaign)
        epoch_records = [common.verify_closed_audit(path, campaign) for path in epochs]
        if arm == profile.NAME:
            validate_saved_audits(final, qa, epoch_records, training_seed=202)
            audits[arm]['task_branch_gradients'] = task_gradient_audit(folder, arm)
        else:
            if (final['actual_optimizer_step'] != 480 or final['frozen_base_tensors_exact'] != 701
                    or final['LoRA_tensors_updated'] != 360
                    or final['matching_tensors_changed'] != {'decoder': [], 'score': []}
                    or qa.get('saved_frozen_head_exact_to_selected_checkpoint') is not True
                    or len(qa.get('frozen_head_state_keys', [])) != 24
                    or actual_cfg['trainer']['args']['deform_conditioner'].get('task_attention') is not None):
                raise ValueError('fixed Top1 control weights or architecture differ')
            for number, epoch in enumerate(epoch_records, 1):
                if (epoch['training_seed'] != 202 or epoch['step'] != 96 * number
                        or (epoch['global_batch'], epoch['world_size'], epoch['microbatch_per_rank'], epoch['accumulation']) != (96, 2, 48, 1)
                        or not epoch['actual_remote_four_numeric_fields_verified'] or not epoch['actual_remote_PNG_byte_exact']):
                    raise ValueError('control budget or actual epoch logging differs')
        for path in [final_folder, *epochs]:
            inputs.extend([path / 'audit.json', path / 'file4-closure.json'])
        models[arm] = aggregate(records[arm])
        inputs.extend([recipe, qa_path, *paths])
    # The original V1 checkpoint is a fixed reference, not a retrained seed202 V1.
    v1_paths = [campaign.parent / 'ss-seed-confirmation-v1/evaluations/v1' / f'noise-{n}/evaluation.json' for n in range(3)]
    records['v1'] = [read(path) for path in v1_paths]
    for value in records['v1']:
        validate_evaluation(value)
    models['v1'] = aggregate(records['v1']); inputs.extend(v1_paths)
    for noise, value in enumerate(records[profile.NAME]):
        previous = read(root / 'evaluations' / profile.NAME / f'noise-{noise}/evaluation.json')
        old = {row['sample_id']: row for row in previous['per_asset']}
        for row in value['per_asset']:
            if any(row[key] != old[row['sample_id']][key] for key in [
                    'source_cluster_id', 'seed', 'target_voxels_sha256', 'target_voxel_shape']):
                raise ValueError('confirmation changed development identity or generation noise')
    contrasts = {arm: compare([(str(n), records[arm][n], records[profile.NAME][n]) for n in range(3)])
                 for arm in ['v1', *common.CONTROLS]}
    replicated = all(supported(value) for value in contrasts.values())
    write_new(folder / 'final-confirmation-report.json', {
        'status': 'PASS_ACTUAL_ORTHOGONAL_SEED202_COMPARISON', 'candidate': profile.NAME,
        'positive_CI_and_all3noise_direction_at_both_training_seeds': replicated,
        'per_training_seed_contrasts': {'101': {arm: development['contrasts'][arm] for arm in contrasts}, '202': contrasts},
        'models': models, 'training_audits': audits,
        'inputs': [{'path': str(path), 'sha256': sha(path)} for path in dict.fromkeys(inputs)],
        'scope': 'Two independent optimization seeds; same development128/93sources and3 generation noises. '
                 'Conditional source-cluster CIs, no training-seed population interval. V1 fixed checkpoint reused. '
                 'Reserved121 has prior auxiliary-head validation exposure and is not an independent test.',
        'reserved121_evaluated': False, 'model_adopted': False, 'full_goal_complete': False,
        'next': 'Lock candidate then verify source-disjoint additional SS validation and matched module effects.'
                if replicated else 'Preserve failed replication; do not adopt.'})


def train(args, profile):
    study.require_source_contract(args.source_contract, args.source_receipt, args.campaign_root)
    cfg = read(args.config)
    validate_replica(cfg, args.root, profile)
    before = profile.source_record()['actual_component_SHA256']
    own_sha = sha(Path(__file__))
    folder = args.root / 'verification' / profile.NAME
    write_new(folder / 'replica-entry-source-v1.json', {'module_SHA256': own_sha, 'components': before,
        'training_seed': 202, 'first_run_recipe_SHA256': sha(args.root.parent / 'configs' / (profile.NAME + '.json'))})
    # Shared launch performs a fresh native backward/zero-output identity proof
    # and an actual two-update micro48/DDP2 benchmark before main training.
    train_shared(args)
    if profile.source_record()['actual_component_SHA256'] != before or sha(Path(__file__)) != own_sha:
        raise RuntimeError('replica source changed during execution')
    write_new(folder / 'replica-exit-source-v1.json', {'status': 'PASS_REPLICA_SOURCE_BINDING',
        'module_SHA256': own_sha, 'components': before, 'training_seed': 202,
        'native_proof_SHA256': sha(folder / 'native-proof.json'), 'model_quality_claimed': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['stage', 'queue', 'train', 'summarize'], required=True)
    for name in ['root', 'config', 'campaign-root', 'work-root', 'previous-work', 'source-contract',
                 'source-receipt', 'epoch-template', 'final-template']:
        parser.add_argument('--' + name, type=Path)
    for name in ['allowed-gpus', 'voxel-gpus', 'soft-gpus', 'epoch-sha', 'final-sha']:
        parser.add_argument('--' + name)
    parser.add_argument('--candidate', choices=list(study.PROFILES))
    args = parser.parse_args()
    if args.mode in ['train', 'summarize']:
        if args.candidate is None:
            parser.error('explicit candidate required')
        profile = study.PROFILES[args.candidate]
        if args.mode == 'train':
            if any(getattr(args, key) is None for key in ['root', 'config', 'campaign_root', 'source_contract', 'source_receipt']):
                parser.error('train requires explicit root/config and actual source closure')
            train(args, profile)
        else:
            if args.campaign_root is None:
                parser.error('summarize requires campaign-root')
            summarize(args.campaign_root, profile)
    else:
        if any(getattr(args, name) is None for name in ['campaign_root', 'work_root', 'previous_work',
                'source_contract', 'source_receipt', 'epoch_template', 'final_template', 'allowed_gpus',
                'voxel_gpus', 'soft_gpus', 'epoch_sha', 'final_sha']):
            parser.error('stage/queue require closed source, audit templates and assigned physical GPU pairs')
        (stage if args.mode == 'stage' else queue)(args)


if __name__ == '__main__':
    main()
