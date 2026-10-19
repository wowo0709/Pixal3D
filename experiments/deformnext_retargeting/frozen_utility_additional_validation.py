"""Open reserved121 only after frozen-utility development and seed replication."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .compare_ss_noise_panels import compare
from .frozen_utility_confirmation import CONTROLS, SUMMARY_JOB, decision_supported, require_closed_decision
from .semantic_topk_additional_validation import validate
from .semantic_topk_campaign_wandb import validate_noise_identity
from .semantic_topk_search import write_new
from .semantic_topk_selection import supported
from .summarize_frozen_utility_followup import NAME
from .summarize_semantic_topk_search import MODELS, aggregate


QUEUE_JOB = 'queue-frozen-utility-additional121-if-replicated'
SUMMARY = 'summarize-frozen-utility-additional121'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def factory_job(root, campaign, work, previous_work):
    return {'name': QUEUE_JOB, 'resource': 'cpu', 'priority': 2,
            'depends_on': [SUMMARY_JOB],
            'requires': [str(root / 'seed-confirmation-v1/final-confirmation-report.json')],
            'outputs': [str(root / 'additional-validation-v1/queue-decision.json')],
            'command': ['taskset', '-c', '84,85', sys.executable, '-u', '-m',
                        'experiments.deformnext_retargeting.frozen_utility_additional_validation',
                        '--mode', 'queue', '--root', str(root), '--campaign-root', str(campaign),
                        '--work-root', str(work), '--previous-work', str(previous_work)]}


def replication_supported(report):
    if (report['status'] != 'PASS_ACTUAL_FROZEN_UTILITY_SEED202_COMPARISON'
            or set(report['models']) != {NAME, *CONTROLS}
            or report['model_adopted'] is not False or report['full_goal_complete'] is not False):
        raise ValueError('actual frozen-utility seed202 report required')
    if set(report['per_training_seed_contrasts']) != {'101', '202'}:
        raise ValueError('both optimization seeds required')
    positive = True
    for values in report['per_training_seed_contrasts'].values():
        if set(values) != set(CONTROLS):
            raise ValueError('both Top1 controls required at each optimization seed')
        positive &= all(supported(value) for value in values.values())
    if report['positive_CI_and_all3noise_direction_at_both_training_seeds'] is not positive:
        raise ValueError('seed replication flag disagrees with actual paired evidence')
    return positive


def reservation_inputs(campaign):
    """Read IDs/manifests/seeds only; never inspect reserved RGB, latents or outcomes."""
    folder = campaign / 'additional-validation-v1'
    reservation_path, panel_path = folder / 'panel-contract.json', folder / 'panel-source-disjoint-121.json'
    reservation, panel = read(reservation_path), read(panel_path)
    if (reservation['status'] != 'RESERVED_NOT_EVALUATED' or reservation['additional_assets'] != 121
            or reservation['additional_source_clusters'] != 39 or sha(panel_path) != reservation['panel_SHA256']):
        raise ValueError('original reserved121/39 panel required')
    ids = panel['sample_ids']
    cfg = read(campaign / 'configs/top1-reference.json')
    train, val = Path(cfg['data_dir']), Path(cfg['validation']['roots_file'])
    if any(sha(path) != reservation['manifest_SHA256'][key] for key, path in [('train', train), ('validation', val)]):
        raise ValueError('reserved source manifests changed')
    rows = {row['sample_id']: row for row in (json.loads(line) for line in val.read_text().splitlines() if line)}
    development = read(campaign / 'evaluations/top1-reference/noise-0/evaluation.json')['per_asset']
    clusters = {rows[sid]['source_cluster_id'] for sid in ids}
    dev_clusters = {row['source_cluster_id'] for row in development}
    train_clusters = {json.loads(line)['source_cluster_id'] for line in train.read_text().splitlines() if line}
    if (len(ids) != len(set(ids)) or len(ids) != 121 or len(clusters) != 39
            or len(development) != 128 or len(dev_clusters) != 93
            or clusters & dev_clusters or train_clusters & {row['source_cluster_id'] for row in rows.values()}):
        raise ValueError('reserved source population overlaps training or development')
    maps, files = {}, [reservation_path, panel_path, train, val]
    for noise in range(3):
        path = Path(reservation['seed_maps'][str(noise)]['path'])
        if sha(path) != reservation['seed_maps'][str(noise)]['sha256']:
            raise ValueError('reserved generation seeds changed')
        maps[noise] = read(path)
        if any(not isinstance(maps[noise][sid], int) or isinstance(maps[noise][sid], bool) for sid in ids):
            raise ValueError('explicit integer generation seeds required')
        files.append(path)
    if any(len({maps[n][sid] for n in range(3)}) != 3 for sid in ids):
        raise ValueError('three distinct generation seeds required per reserved case')
    return reservation, panel_path, ids, rows, maps, files


def jobs_and_contract(root, campaign, reservation, panel):
    if (reservation['status'] != 'RESERVED_NOT_EVALUATED' or reservation['additional_assets'] != 121
            or reservation['additional_source_clusters'] != 39):
        raise ValueError('original reserved121/39 panel required')
    models = {}

    def trained(label, base, arm, seed):
        run = base / 'runs_ss64' / arm / 'ckpts'
        models[label] = {'config': str(base / 'configs' / (arm + '.json')),
                         'checkpoint': str(run / 'denoiser_step0000480.pt'),
                         'conditioner_checkpoint': str(run / 'deform_conditioner_step0000480.pt') if arm != 'v1' else None,
                         'step': 480, 'training_seed': seed, 'conditioning_mode': 'edited-rgb'}

    for arm in [*MODELS, 'static-top1-c1']:
        trained('s101-' + arm, campaign, arm, 101)
    for arm in ['v1', 'c1']:
        trained('previous-' + arm, campaign.parent / 'ss-seed-confirmation-v1', arm, 101)
    trained('s101-' + NAME, root, NAME, 101)
    trained('s202-' + NAME, root / 'seed-confirmation-v1', NAME, 202)
    for arm in CONTROLS:
        trained('s202-' + arm, campaign / 'seed-confirmation-v1', arm, 202)
    zero = campaign / 'configs/zero-shot-reference.json'
    checkpoint = read(zero)['trainer']['args']['continuation_ckpt']['denoiser']
    for label, mode in [('zero-source-only', 'source-only'), ('zero-edited-rgb', 'edited-rgb')]:
        models[label] = {'config': str(zero), 'checkpoint': checkpoint, 'conditioner_checkpoint': None,
                         'step': 0, 'training_seed': None, 'conditioning_mode': mode}
    if len(models) != 16:
        raise ValueError('all sixteen reserved evaluation conditions required')
    folder, jobs = root / 'additional-validation-v1', []
    for label, spec in models.items():
        for noise in range(3):
            output, seed_map = folder / 'evaluations' / label / f'noise-{noise}', reservation['seed_maps'][str(noise)]['path']
            command = [sys.executable, '-u', '-m', 'experiments.deformnext_retargeting.evaluate_native_checkpoint',
                       '--stage', 'ss64', '--config-root', str(Path(spec['config']).parent), '--arm-config', spec['config'],
                       '--checkpoint', spec['checkpoint'], '--step', str(spec['step']),
                       '--conditioning-mode', spec['conditioning_mode'], '--output', str(output),
                       '--panel-file', str(panel), '--seed-map-file', seed_map, '--generation-batch', '8']
            if spec['conditioner_checkpoint']:
                command.extend(['--conditioner-checkpoint', spec['conditioner_checkpoint']])
            jobs.append({'name': f'frozen-additional121-{label}-noise{noise}', 'resource': 'evaluation', 'priority': 4,
                         'depends_on': [QUEUE_JOB], 'requires': [str(panel), seed_map, spec['config'], spec['checkpoint'],
                                                               *([spec['conditioner_checkpoint']] if spec['conditioner_checkpoint'] else [])],
                         'outputs': [str(output / 'evaluation.json'), str(output / 'panel.png')], 'command': command})
    jobs.append({'name': SUMMARY, 'resource': 'cpu', 'priority': 3,
                 'depends_on': [job['name'] for job in jobs],
                 'outputs': [str(folder / 'final-report.json'), str(folder / 'final-report.md')],
                 'command': ['taskset', '-c', '84,85', sys.executable, '-u', '-m',
                             'experiments.deformnext_retargeting.frozen_utility_additional_validation',
                             '--mode', 'summarize', '--root', str(root), '--campaign-root', str(campaign)]})
    return jobs, {'status': 'QUEUED_NOT_EVALUATED', 'candidate': NAME, 'models': models,
                  'panel': str(panel), 'panel_SHA256': reservation['panel_SHA256'],
                  'assets': 121, 'source_clusters': 39, 'sampling_noises': 3,
                  'generation_batch': 8, 'sampling_steps': 12, 'guidance': 3.,
                  'optimizer_updates_during_evaluation': 0,
                  'scope': 'Locked frozen-utility candidate after both optimization seeds pass development controls. '
                           'Source-disjoint121/39 within the existing validation split,not an independent test dataset. '
                           'Sixteen saved-model conditions,three generation noises;no training or candidate reselection.'}


def verify_report_inputs(report):
    for item in report['inputs']:
        path = Path(item['path'])
        if sha(path) != item['sha256'] or ('bytes' in item and path.stat().st_size != item['bytes']):
            raise ValueError('selected report input bytes changed')


def close_confirmation(args):
    path = args.root / 'seed-confirmation-v1/final-confirmation-report.json'
    receipt = path.with_name('final-confirmation-report-file4.json')
    if not receipt.exists():
        manifest = args.work_root / 'frozen-utility-seed202-final-confirmation-files.json'
        content = json.dumps([str(path.relative_to(args.campaign_root.parent))], indent=2) + '\n'
        if manifest.exists() and manifest.read_text() != content:
            raise ValueError('preserve previous confirmation manifest')
        if not manifest.exists():
            manifest.write_text(content)
        subprocess.run([sys.executable, str(args.previous_work / 'verify_closed_file4_subset.py'),
                        '--manifest', str(manifest), '--receipt', str(receipt.relative_to(args.campaign_root.parent)),
                        '--scope', 'Actual two-optimization-seed frozen-utility comparison;positive or negative,not adoption.'], check=True)
    require_closed_decision(path, receipt, args.campaign_root)
    return path, receipt


def queue(args):
    output = args.root / 'additional-validation-v1/queue-decision.json'
    if output.exists():
        raise FileExistsError('preserve actual reserved evaluation queue decision')
    development_path = args.root / 'fixed-final-decision-v1.json'
    development = read(development_path)
    require_closed_decision(development_path, args.root / 'fixed-final-report-file4.json', args.campaign_root)
    if not decision_supported(development):
        raise ValueError('no eligible frozen-utility development candidate')
    report_path = args.root / 'seed-confirmation-v1/final-confirmation-report.json'
    report = read(report_path)
    positive = replication_supported(report)
    verify_report_inputs(development)
    verify_report_inputs(report)
    close_confirmation(args)
    decision = {'status': 'RESERVED121_QUEUED_FOR_LOCKED_FROZEN_UTILITY' if positive
                         else 'NO_RESERVED_EVALUATION_REPLICATION_FAILED',
                'candidate': NAME, 'replication_positive': positive,
                'development_decision_SHA256': sha(development_path), 'seed_confirmation_SHA256': sha(report_path),
                'reserved_outcomes_read_during_selection': False, 'model_adopted': False, 'full_goal_complete': False}
    if not positive:
        write_new(output, decision)
        print(json.dumps(decision), flush=True)
        return
    reservation, panel, _, _, _, _ = reservation_inputs(args.campaign_root)
    jobs, contract = jobs_and_contract(args.root, args.campaign_root, reservation, panel)
    contract.update(development_decision_SHA256=sha(development_path), seed_confirmation_SHA256=sha(report_path),
                    reservation_SHA256=sha(args.campaign_root / 'additional-validation-v1/panel-contract.json'))
    queue_path = args.work_root / 'queue-jobs.json'
    old = queue_path.read_bytes()
    existing = json.loads(old)
    names = [job['name'] for job in existing + jobs]
    if len(names) != len(set(names)) or any(dep not in set(names) for job in jobs for dep in job['depends_on']):
        raise ValueError('reserved evaluation job names or dependencies differ')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    for job in jobs:
        job['source_commit'] = commit
    contract['source_commit'] = commit
    contract_path = args.root / 'additional-validation-v1/evaluation-contract.json'
    folder = contract_path.parent
    closed_files = [contract_path, output, folder / 'final-report.json', folder / 'final-report.md']
    closed_files.extend(folder / 'evaluations' / label / f'noise-{noise}' / name
                        for label in contract['models'] for noise in range(3)
                        for name in ['evaluation.json', 'panel.png'])
    manifest = args.work_root / 'frozen-utility-additional121-result-files.json'
    content = json.dumps([str(path.relative_to(args.campaign_root.parent)) for path in closed_files], indent=2) + '\n'
    if manifest.exists() and manifest.read_text() != content:
        raise ValueError('preserve reserved evaluation closure manifest')
    if not manifest.exists():
        manifest.write_text(content)
    receipt = folder / 'file4-closure.json'
    jobs.append({'name': 'close-frozen-utility-additional121-file4', 'resource': 'cpu', 'priority': 3,
                 'depends_on': [SUMMARY], 'requires': [str(path) for path in closed_files],
                 'outputs': [str(receipt)],
                 'command': ['taskset', '-c', '84,85', sys.executable, '-u',
                             str(args.previous_work / 'verify_closed_file4_subset.py'),
                             '--manifest', str(manifest), '--receipt', str(receipt.relative_to(args.campaign_root.parent)),
                             '--scope', contract['scope']]})
    names = [job['name'] for job in existing + jobs]
    if len(names) != len(set(names)):
        raise ValueError('reserved evaluation closure job already exists')
    jobs[-1]['source_commit'] = commit
    write_new(contract_path, contract)
    decision['evaluation_contract_SHA256'] = sha(contract_path)
    with (args.work_root / 'queue-jobs-before-frozen-utility-reserved121.json').open('xb') as stream:
        stream.write(old)
    if queue_path.read_bytes() != old:
        raise RuntimeError('concurrent queue edit; reconcile immutable reserved contract before retry')
    temp = queue_path.with_suffix('.frozen-reserved.tmp')
    temp.write_text(json.dumps(existing + jobs, indent=2) + '\n')
    os.replace(temp, queue_path)
    decision.update(queued_jobs=len(jobs), source_commit=commit)
    write_new(output, decision)
    print(json.dumps(decision), flush=True)


def summarize(root, campaign):
    folder = root / 'additional-validation-v1'
    contract, decision = read(folder / 'evaluation-contract.json'), read(folder / 'queue-decision.json')
    require_closed_decision(root / 'fixed-final-decision-v1.json', root / 'fixed-final-report-file4.json', campaign)
    require_closed_decision(root / 'seed-confirmation-v1/final-confirmation-report.json',
                            root / 'seed-confirmation-v1/final-confirmation-report-file4.json', campaign)
    if (contract['candidate'] != NAME or decision['status'] != 'RESERVED121_QUEUED_FOR_LOCKED_FROZEN_UTILITY'
            or decision['evaluation_contract_SHA256'] != sha(folder / 'evaluation-contract.json')
            or contract['development_decision_SHA256'] != sha(root / 'fixed-final-decision-v1.json')
            or contract['seed_confirmation_SHA256'] != sha(root / 'seed-confirmation-v1/final-confirmation-report.json')
            or not replication_supported(read(root / 'seed-confirmation-v1/final-confirmation-report.json'))):
        raise ValueError('candidate selection or successful seed replication changed')
    reservation, panel, ids, rows, maps, files = reservation_inputs(campaign)
    if (contract['panel'] != str(panel) or contract['panel_SHA256'] != sha(panel)
            or contract['reservation_SHA256'] != sha(campaign / 'additional-validation-v1/panel-contract.json')):
        raise ValueError('original reserved panel changed after selection')
    records = {}
    for label, spec in contract['models'].items():
        paths = [folder / 'evaluations' / label / f'noise-{n}/evaluation.json' for n in range(3)]
        records[label] = [read(path) for path in paths]
        for noise, value in enumerate(records[label]):
            if (value.get('manifest_sha256') != reservation['manifest_SHA256']['validation']
                    or value.get('panel_file') != str(panel)
                    or value.get('seed_map_file') != reservation['seed_maps'][str(noise)]['path']):
                raise ValueError('actual reserved evaluation manifest, panel or seed-map file differs')
            validate(value, spec, ids, rows, maps[noise])
        files.extend(paths)
    if len(records) != 16:
        raise ValueError('all sixteen actual evaluation conditions required')
    validate_noise_identity(records, 121, 39)
    pairs = [(label, 'previous-v1') for label in records if label != 'previous-v1']
    pairs += [(f's{seed}-{NAME}', f's{seed}-{arm}') for seed in [101, 202] for arm in CONTROLS]
    pairs += [(f's101-{NAME}', label) for label in ['previous-c1', 's101-task-full', 's101-task-native-spatial']]
    pairs += [('s101-task-full', 's101-' + arm) for arm in ['task-no-decoder', 'task-no-prior', 'task-native-spatial', 'static-c1']]
    pairs += [('s101-static-c1', 's101-static-top1-c1'), ('s101-static-top1-c1', 's101-top1-reference'),
              ('s101-soft-topk', 's101-top1-reference'), ('zero-edited-rgb', 'zero-source-only')]
    contrasts = {}
    for candidate, baseline in pairs:
        label = candidate + '-vs-' + baseline
        if label not in contrasts:
            contrasts[label] = compare([(str(n), records[baseline][n], records[candidate][n]) for n in range(3)])
            if contrasts[label]['paired_target_hashes_verified'] != 363:
                raise ValueError('all actual paired target hashes required')
    per_seed = {str(seed): {arm: contrasts[f's{seed}-{NAME}-vs-s{seed}-{arm}'] for arm in CONTROLS}
                for seed in [101, 202]}
    positive = all(supported(value) for control in per_seed.values() for value in control.values())
    versus_v1 = {str(seed): contrasts[f's{seed}-{NAME}-vs-previous-v1'] for seed in [101, 202]}
    positive_v1 = all(supported(value) for value in versus_v1.values())
    models = {label: {**aggregate(values), 'secondary_macro_case_iou':
                      sum(row['tp'] / max(1, row['tp'] + row['fp'] + row['fn'])
                          for value in values for row in value['per_asset']) / 363}
              for label, values in records.items()}
    files.extend([folder / 'evaluation-contract.json', folder / 'queue-decision.json',
                  root / 'fixed-final-decision-v1.json', root / 'seed-confirmation-v1/final-confirmation-report.json'])
    result = {'status': 'PASS_FROZEN_UTILITY_ADDITIONAL121_EVALUATION', 'candidate': NAME,
              'models': models, 'contrasts': contrasts, 'per_training_seed_both_Top1_controls': per_seed,
              'positive_CI_and_all_noises_at_both_seeds_vs_both_Top1_controls': positive,
              'per_candidate_training_seed_vs_fixed_V1': versus_v1,
              'positive_vs_fixed_V1_at_both_candidate_training_seeds': positive_v1,
              'passes_locked_candidate_quality_controls': positive and positive_v1,
              'source_disjoint_from_development128': True, 'candidate_reselected': False,
              'inputs': [{'path': str(path), 'sha256': sha(path)} for path in dict.fromkeys(files)],
              'scope': contract['scope'],
              'CI_scope': 'Separate source-cluster CIs conditional on each optimization seed,not a training-seed population CI. '
                          'V1 is the same fixed seed101 checkpoint for both candidate seeds. '
                          'Original joint-Full component contrasts cannot attribute frozen-utility component gains.',
              'model_adopted': False, 'full_goal_complete': False}
    lines = ['# Locked frozen utility: reserved121', '', result['scope'], '',
             '| Model | Micro IoU | Micro F1 | Secondary macro IoU |', '| --- | ---: | ---: | ---: |']
    lines.extend(f"| {label} | {value['micro_voxel_iou']:.6f} | {value['micro_voxel_f1']:.6f} | {value['secondary_macro_case_iou']:.6f} |"
                 for label, value in models.items())
    lines.extend(['', '| Contrast | Gain IoU,pp | 95% source-cluster CI,pp |', '| --- | ---: | --- |'])
    for label, value in contrasts.items():
        lower, upper = value['paired_cluster_bootstrap_ci95']
        lines.append(f"| {label} | {100*value['gain']:+.3f} | [{100*lower:+.3f},{100*upper:+.3f}] |")
    lines.extend(['', result['CI_scope'], '', 'Actual logging/file4 closure,module-effect interpretation and adoption remain required.'])
    write_new(folder / 'final-report.json', result)
    with (folder / 'final-report.md').open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps({'status': result['status'], 'passes_quality_controls': positive and positive_v1,
                      'positive_vs_both_Top1': positive}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['queue', 'summarize'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--work-root', type=Path)
    parser.add_argument('--previous-work', type=Path)
    args = parser.parse_args()
    if args.mode == 'queue':
        if args.work_root is None or args.previous_work is None:
            parser.error('queue requires explicit work-root and previous-work')
        queue(args)
    else:
        summarize(args.root, args.campaign_root)


if __name__ == '__main__':
    main()
