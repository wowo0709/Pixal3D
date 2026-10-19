"""Publish closed utility-model comparisons without changing their selection."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from . import semantic_topk_campaign_wandb as common
from .compare_ss_noise_panels import compare
from .firstwave_wandb import read, sha, validate_visual_sources
from .frozen_utility_confirmation import decision_supported, require_closed_decision
from .semantic_topk_search import write_new
from .semantic_topk_zero_replay import MODES, zero_paths, validate_zero
from .summarize_semantic_topk_search import aggregate, validate_evaluation


MODULE = 'experiments.deformnext_retargeting.utility_followup_wandb'
FOLDER = 'utility-report-wandb-v1'
FUTURE_FACTORY = 'queue-orthogonal-utility-publications-if-launched'
PROFILES = {'frozen-utility-followup-v1': 'frozen-utility-c1',
            'utility-strength-followup-v1': 'utility-scale16-c1',
            'voxel-query-followup-v1': 'voxel-query-c1',
            'utility-softprior-followup-v1': 'utility-softprior-c1'}
LABELS = {'zero-source-only': 'Zero source4 M0', 'zero-edited-rgb': 'Zero edited RGB',
          'v1': 'Vanilla V1', 'top1-reference': 'Top1, no R', 'static-top1-c1': 'Top1+C1 R',
          'previous-c1': 'Old fixed C1/T.005', 'task-full': 'Old joint Full',
          'task-native-spatial': 'Old joint spatial', 'frozen-utility-c1': 'F1 utility4',
          'reference-scale4': 'F1 utility4', 'reference-scale16': 'Utility16 patch',
          'utility-scale16-c1': 'Utility16 patch', 'voxel-query-c1': 'Native voxel query',
          'utility-softprior-c1': 'Soft score T.05'}


def profile(root):
    if root.name not in PROFILES:
        raise ValueError('explicit declared utility development namespace required')
    return PROFILES[root.name]


def sources(root):
    name = profile(root)
    campaign = root.parent
    previous = campaign.parent / 'ss-seed-confirmation-v1'
    result = {arm: (campaign, arm) for arm in ['top1-reference', 'static-top1-c1', 'task-full', 'task-native-spatial']}
    result.update({'v1': (previous, 'v1'), 'previous-c1': (previous, 'c1'), name: (root, name)})
    if name == 'utility-scale16-c1':
        result['frozen-utility-c1'] = (campaign / 'frozen-utility-followup-v1', 'frozen-utility-c1')
    elif name in {'voxel-query-c1', 'utility-softprior-c1'}:
        result.update({'reference-scale4': (campaign / 'frozen-utility-followup-v1', 'frozen-utility-c1'),
                       'reference-scale16': (campaign / 'utility-strength-followup-v1', 'utility-scale16-c1')})
    return result


def closed_file(path, receipt_path, canonical):
    receipt = read(receipt_path)
    entries = [row for row in receipt['files'] if row['relative_path'] == str(path.relative_to(canonical))]
    if (receipt.get('status') != 'PASS' or receipt.get('local_equals_actual_file4_SHA256') is not True
            or len(entries) != 1 or entries[0]['sha256'] != sha(path)
            or entries[0]['bytes'] != path.stat().st_size):
        raise ValueError('actual canonical file receipt does not bind the reported input')


def collect(root):
    name, mapping = profile(root), sources(root)
    campaign, canonical = root.parent, root.parent.parent
    path = root / 'fixed-final-decision-v1.json'
    require_closed_decision(path, root / 'fixed-final-report-file4.json', campaign)
    report = read(path)
    eligible = decision_supported(report, candidate_name=name)
    files = [path, path.with_suffix('.md'), root / 'fixed-final-report-file4.json']
    for row in report['inputs']:
        source = Path(row['path'])
        if sha(source) != row['sha256'] or source.stat().st_size != row['bytes']:
            raise ValueError('actual completed development input changed')
        files.append(source)
    if set(report['models']) != set(mapping) | set(MODES):
        raise ValueError('completed report must retain every declared model and both zeros')
    references = [label for label in mapping if label not in {
        name, 'v1', 'previous-c1', 'top1-reference', 'static-top1-c1', 'task-full', 'task-native-spatial'}]
    order = ['zero-source-only', 'zero-edited-rgb', 'v1', 'top1-reference', 'static-top1-c1',
             'previous-c1', 'task-full', 'task-native-spatial', *references, name]
    records, images = {}, {}
    for label in order:
        if label in MODES:
            paths = zero_paths(campaign, canonical, label)
        else:
            base, arm = mapping[label]
            paths = [base / 'evaluations' / arm / f'noise-{noise}/evaluation.json' for noise in range(3)]
        values = [read(path) for path in paths]
        for value in values:
            validate_zero(value, MODES[label]) if label in MODES else validate_evaluation(value)
        if aggregate(values) != report['models'][label]:
            raise ValueError('actual final voxel counts differ: ' + label)
        records[label] = values
        panel = paths[0].parent / 'panel.png'
        images[label] = np.asarray(Image.open(panel).convert('RGB'))
        files.extend([*paths, panel])
    ids = common.validate_noise_identity(records, 128, 93)
    validate_visual_sources({label: values[0]['per_asset'] for label, values in records.items()}, images)
    for baseline, expected in report['contrasts'].items():
        actual = compare([(str(noise), records[baseline][noise], records[name][noise]) for noise in range(3)])
        if actual != expected:
            raise ValueError('actual final paired comparison differs')
    diagnostic_path = root / 'task-value-address-v1/summary.json'
    diagnostic = read(diagnostic_path)
    if (diagnostic.get('status') != 'PASS_FROZEN_UTILITY_SAME_POOL_VALUE_ADDRESS_DIAGNOSTIC'
            or diagnostic.get('candidate') != name or diagnostic.get('model_adopted') is not False):
        raise ValueError('same actual saved-weight value-address diagnostic required')
    receipt = (campaign / 'provenance/frozen-utility-completed-value-address-and-interaction-v1-file4.json'
               if name == 'frozen-utility-c1' else diagnostic_path.parent / 'file4-closure.json')
    closed_file(diagnostic_path, receipt, canonical)
    for source, digest in diagnostic['actual_evaluation_SHA256'].items():
        source = Path(source)
        if sha(source) != digest:
            raise ValueError('actual same-weight diagnostic input changed')
        files.append(source)
    files.extend([diagnostic_path, receipt])
    expected_ids = report['training_audit']['validation_panel_ids']
    initial = report['training_audit']['validation'][0]['quick_per_asset']
    curves = []
    for label in ['v1', 'previous-c1', 'static-top1-c1', 'task-full', *references, name]:
        base, arm = mapping[label]
        path = base / 'runs_ss64' / arm / 'validation.jsonl'
        values = [json.loads(line) for line in path.read_text().splitlines() if line]
        if ([row['step'] for row in values] != [0, 96, 192, 288, 384, 480]
                or values[0]['quick_per_asset'] != initial
                or any(row['quick_panel_ids'] != expected_ids for row in values)):
            raise ValueError('actual fixed32 curve identity differs')
        curves.extend([label, row['step'] // 96, row['quick_metrics']['voxel_iou'], row['quick_metrics']['voxel_f1']]
                      for row in values)
        files.append(path)
    files.extend([Path(__file__), Path(common.__file__)])
    scope = {'development128_completed': True, 'sampling_noises': 3, 'training_seed': 101,
        'candidate': name, 'candidate_for_seed_confirmation': name if eligible else None,
        'scientific_outcome': report['status'], 'seed_confirmation_included': False,
        'additional121_included': False, 'model_adopted': False, 'full_goal_complete': False,
        'scope': 'Closed raw480/development128/93source/3sampling-noise report,including negatives. '
                 'Common32 curves/value-address sensitivity are separate. Old joint baselines are contextual '
                 'comparisons,not isolated removal effects for this candidate. No seed202 or module-specific '
                 'effect certification. Reserved121 had earlier auxiliary-head validation exposure; '
                 'this isnot an independent test report.'}
    groups = [['zero-source-only', 'zero-edited-rgb', 'v1', 'top1-reference', 'static-top1-c1', name],
              ['previous-c1', 'task-full', 'task-native-spatial', *references, 'static-top1-c1', name]]
    if any(len(group) > 7 for group in groups):
        raise ValueError('fixed gallery supports at most7 methods plusGT')
    return {'report': report, 'diagnostic': diagnostic, 'curves': curves, 'scope': scope,
        'order': order, 'files': list(dict.fromkeys(files)), 'galleries': {'development128': {
            'records': records, 'images': images, 'ids': ids, 'groups': groups,
            'scope': 'Raw480/development128/noise0;3D voxel IoU;occupancy max-projections.'}}}


def payload(root):
    data = collect(root)
    folder = root / FOLDER / 'presentation'
    rendered = read(folder / 'summary.json')
    if (rendered['status'] != 'PASS_COMPLETED_CAMPAIGN_PRESENTATION' or rendered['scope'] != data['scope']
            or rendered['input_SHA256'] != {str(path): sha(path) for path in data['files']}):
        raise ValueError('actual completed presentation/source bytes differ')
    report, metrics = data['report'], {}
    for label, model in report['models'].items():
        for key in ['micro_voxel_iou', 'micro_voxel_f1']:
            metrics[f'development128/{label}/{key}'] = model[key]
        tp, fp, fn = [model['counts'][key] for key in ['tp', 'fp', 'fn']]
        metrics[f'development128/{label}/micro_voxel_precision'] = tp / max(tp + fp, 1)
        metrics[f'development128/{label}/micro_voxel_recall'] = tp / max(tp + fn, 1)
    for label, contrast in report['contrasts'].items():
        for key, number in zip(['gain_pp', 'ci95_lower_pp', 'ci95_upper_pp'],
                               [contrast['gain'], *contrast['paired_cluster_bootstrap_ci95']]):
            metrics[f'development128/candidate-vs-{label}/{key}'] = 100 * number
    sensitivity = data['diagnostic']['normal_minus_wrong_address']
    for key, number in zip(['gain_pp', 'ci95_lower_pp', 'ci95_upper_pp'],
                           [sensitivity['gain'], *sensitivity['paired_cluster_bootstrap_ci95']]):
        metrics['diagnostic_common32/value-address/' + key] = 100 * number
    tables = {'development128/models': {'columns': ['method', 'micro_IoU', 'micro_F1'],
        'data': [[name, report['models'][name]['micro_voxel_iou'], report['models'][name]['micro_voxel_f1']]
                 for name in data['order']]},
        'development128/paired_contrasts': {'columns': ['baseline', 'gain_pp', 'CI95_lower_pp', 'CI95_upper_pp'],
            'data': [[name, 100 * row['gain'], *[100 * x for x in row['paired_cluster_bootstrap_ci95']]]
                     for name, row in report['contrasts'].items()]},
        'common32/epoch_curves': {'columns': ['method', 'epoch', 'micro_IoU', 'micro_F1'], 'data': data['curves']}}
    files = data['files'] + [path for path in folder.iterdir() if path.is_file()]
    if any(path.suffix == '.pt' for path in files):
        raise ValueError('model binaries remain on canonical file4')
    canonical, repo = root.parent.parent, Path(__file__).resolve().parents[2]
    artifacts = {str(path.relative_to(canonical)) if path.is_relative_to(canonical)
                 else 'source/' + str(path.relative_to(repo)): str(path) for path in files}
    return {'metrics': metrics, 'tables': tables, 'images': {path.stem: str(path) for path in folder.glob('*.png')},
        'artifact_files': artifacts, 'scope': data['scope'],
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}


def close_report(args):
    root, folder = args.root, args.root / FOLDER
    audit = read(folder / 'remote-readback/audit.json')
    spec = read(folder / 'publication-spec.json')
    publication = read(folder / 'publication.json')
    if (audit.get('status') != 'PASS_ACTUAL_SS_CAMPAIGN_WANDB_READBACK'
            or audit.get('actual_remote_state') != 'finished' or audit.get('scope') != spec['scope']
            or publication.get('spec_SHA256') != sha(folder / 'publication-spec.json')):
        raise ValueError('completed actual remote report readback required')
    if (folder / 'file4-closure.json').exists():
        raise FileExistsError('preserve completed canonical report closure')
    repo, canonical = Path(__file__).resolve().parents[2], root.parent.parent
    rendered = read(folder / 'presentation/summary.json')
    files = [folder / name for name in ['publication-spec.json', 'publication.json', 'remote-readback/audit.json']]
    files += [path for path in (folder / 'presentation').iterdir() if path.is_file()]
    for value in spec['artifact_files'].values():
        source = Path(value)
        if not source.is_relative_to(repo):
            continue
        if rendered['input_SHA256'].get(str(source)) != sha(source):
            raise ValueError('published source bytes changed before canonical closure')
        target = folder / 'source' / source.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != sha(source):
                raise ValueError('preserve earlier canonical source copy')
        else:
            with target.open('xb') as stream:
                stream.write(source.read_bytes())
        files.append(target)
    manifest = args.work_root / (profile(root) + '-development-report-files-v1.json')
    expected = sorted(str(path.relative_to(canonical)) for path in files)
    if manifest.exists():
        if read(manifest) != expected:
            raise ValueError('preserve earlier canonical report manifest')
    else:
        write_new(manifest, expected)
    subprocess.run([sys.executable, str(args.previous_work / 'verify_closed_file4_subset.py'),
        '--manifest', str(manifest), '--receipt', str((folder / 'file4-closure.json').relative_to(canonical)),
        '--scope', 'Actual finished utility development report,all128case figures,metrics/tables and fresh W&B readback;not model adoption or independent test.'], check=True)


def publication_jobs(root, dependencies, *, work_root, previous_work):
    name = profile(root)
    if name == 'frozen-utility-c1':
        raise ValueError('preserve already-published original F1 report')
    folder = root / FOLDER
    jobs = []
    for mode, output in [('render', folder / 'presentation/summary.json'), ('publish', folder / 'publication.json'),
                         ('readback', folder / 'remote-readback/audit.json'), ('close', folder / 'file4-closure.json')]:
        command = ['taskset', '-c', '84,85', sys.executable, '-u', '-m', MODULE, '--root', str(root), '--mode', mode]
        if mode == 'close':
            command.extend(['--work-root', str(work_root), '--previous-work', str(previous_work)])
        jobs.append({'name': name + '-development-report-' + mode, 'resource': 'cpu', 'priority': 5,
            'depends_on': dependencies if not jobs else [jobs[-1]['name']],
            'requires': [str(root / 'fixed-final-report-file4.json'),
                         str(root / 'task-value-address-v1/file4-closure.json')] if mode == 'render' else [],
            'outputs': [str(output)], 'command': command})
    return jobs


def queue_future(args):
    folder = args.campaign_root / 'orthogonal-utility-followups-v1'
    decision = read(folder / 'queue-decision-v1.json')
    output = folder / 'publication-queue-decision-v1.json'
    if output.exists():
        raise FileExistsError('preserve actual future publication branch')
    if decision['status'] == 'SKIPPED_FOR_STRENGTH_CONFIRMATION':
        write_new(output, {'status': 'NO_ALTERNATIVE_REPORTS_QUEUED', 'model_adopted': False})
        return
    if decision['status'] != 'ORTHOGONAL_FOLLOWUPS_QUEUED' or decision['new_training_runs'] != 2:
        raise ValueError('actual isolated model launch decision required')
    jobs = []
    for namespace in ['voxel-query-followup-v1', 'utility-softprior-followup-v1']:
        root = args.campaign_root / namespace
        name = profile(root)
        jobs.extend(publication_jobs(root, ['close-fixed-final-' + name, 'close-value-address-' + name],
                                     work_root=args.work_root, previous_work=args.previous_work))
    path = args.work_root / 'queue-jobs.json'
    before = path.read_bytes(); existing = json.loads(before)
    names = [job['name'] for job in existing + jobs]
    if len(names) != len(set(names)) or any(dep not in names for job in jobs for dep in job['depends_on']):
        raise ValueError('future publication queue dependency/name mismatch')
    for job in jobs:
        job['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    with (args.work_root / 'queue-before-orthogonal-publications-v1.json').open('xb') as stream:
        stream.write(before)
    if path.read_bytes() != before:
        raise RuntimeError('concurrent queue change; preserve and reconcile')
    temporary = path.with_suffix('.utility-publications.tmp')
    temporary.write_text(json.dumps(existing + jobs, indent=2) + '\n'); os.replace(temporary, path)
    write_new(output, {'status': 'ALTERNATIVE_REPORTS_QUEUED', 'jobs': len(jobs), 'model_adopted': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['render', 'publish', 'readback', 'close', 'queue-future'], required=True)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--campaign-root', type=Path)
    parser.add_argument('--work-root', type=Path)
    parser.add_argument('--previous-work', type=Path)
    args = parser.parse_args()
    if args.mode == 'queue-future':
        if args.campaign_root is None or args.work_root is None or args.previous_work is None:
            parser.error('queue-future requires campaign/work/previous-work roots')
        queue_future(args)
        return
    if args.root is None:
        parser.error('publication mode requires --root')
    name = profile(args.root)
    if name == 'frozen-utility-c1':
        raise ValueError('original F1 is available for read-only collection,already published')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    folder = args.root / FOLDER
    run_id = 'ss-' + name + '-development-v1'
    if args.mode == 'render':
        common.render(args.root, collector=collect, output_folder=folder / 'presentation', labels=LABELS)
    elif args.mode == 'publish':
        common.publish(args.root, payload_factory=payload, output_folder=folder, run_id=run_id,
                       run_name='SS fixed-final development ' + name)
    elif args.mode == 'readback':
        common.readback(args.root, output_folder=folder, run_id=run_id)
    else:
        if args.work_root is None or args.previous_work is None:
            parser.error('close requires explicit work/previous-work roots')
        close_report(args)


if __name__ == '__main__':
    main()
