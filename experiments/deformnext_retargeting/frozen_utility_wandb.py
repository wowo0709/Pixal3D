"""Publish the actual frozen-utility development outcome, including negatives."""
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
from .semantic_topk_zero_replay import MODES, zero_paths, validate_zero
from .summarize_frozen_utility_followup import NAME
from .summarize_semantic_topk_search import aggregate, validate_evaluation


RUN_ID = 'ss-frozen-utility-development-v1'
ORDER = ['zero-source-only', 'zero-edited-rgb', 'v1', 'top1-reference', 'previous-c1',
         'static-top1-c1', 'task-full', 'task-native-spatial', NAME]
LABELS = {**common.LABELS, 'v1': 'Vanilla V1', NAME: 'Frozen utility C1'}


def collect(root):
    campaign, previous = root.parent, root.parent.parent / 'ss-seed-confirmation-v1'
    decision_path = root / 'fixed-final-decision-v1.json'
    require_closed_decision(decision_path, root / 'fixed-final-report-file4.json', campaign)
    report = read(decision_path)
    eligible = decision_supported(report)
    files = [decision_path, decision_path.with_suffix('.md'), root / 'fixed-final-report-file4.json']
    for item in report['inputs']:
        path = Path(item['path'])
        if sha(path) != item['sha256'] or path.stat().st_size != item['bytes']:
            raise ValueError('actual development input changed: ' + str(path))
        files.append(path)
    sources = {name: (campaign, name) for name in ['top1-reference', 'static-top1-c1',
                                                 'task-full', 'task-native-spatial']}
    sources.update({'v1': (previous, 'v1'), 'previous-c1': (previous, 'c1'), NAME: (root, NAME)})
    records, images = {}, {}
    for name in ORDER:
        if name in MODES:
            paths = zero_paths(campaign, campaign.parent, name)
        else:
            base, arm = sources[name]
            paths = [base / 'evaluations' / arm / f'noise-{n}/evaluation.json' for n in range(3)]
        records[name] = [read(path) for path in paths]
        for value in records[name]:
            validate_zero(value, MODES[name]) if name in MODES else validate_evaluation(value)
        if aggregate(records[name]) != report['models'][name]:
            raise ValueError('actual development voxel counts differ: ' + name)
        panel = paths[0].parent / 'panel.png'
        images[name] = np.asarray(Image.open(panel).convert('RGB'))
        files.extend([*paths, panel])
    ids = common.validate_noise_identity(records, 128, 93)
    validate_visual_sources({name: values[0]['per_asset'] for name, values in records.items()}, images)
    for name, expected in report['contrasts'].items():
        actual = compare([(str(n), records[name][n], records[NAME][n]) for n in range(3)])
        if actual != expected:
            raise ValueError('actual paired contrast differs: ' + name)
    diagnostic_path = root / 'task-value-address-v1/summary.json'
    diagnostic = read(diagnostic_path)
    if (diagnostic['status'] != 'PASS_FROZEN_UTILITY_SAME_POOL_VALUE_ADDRESS_DIAGNOSTIC'
            or diagnostic['candidate'] != NAME or diagnostic['model_adopted'] is not False):
        raise ValueError('actual same-weight value-address diagnostic required')
    for path, digest in diagnostic['actual_evaluation_SHA256'].items():
        if sha(Path(path)) != digest:
            raise ValueError('actual value-address diagnostic input changed')
        files.append(Path(path))
    files.append(diagnostic_path)
    curves, expected_ids = [], report['training_audit']['validation_panel_ids']
    initial = report['training_audit']['validation'][0]['quick_per_asset']
    for name in ['v1', 'previous-c1', 'static-top1-c1', 'task-full', NAME]:
        base, arm = sources[name]
        path = base / 'runs_ss64' / arm / 'validation.jsonl'
        values = [json.loads(line) for line in path.read_text().splitlines() if line]
        if ([value['step'] for value in values] != [0, 96, 192, 288, 384, 480]
                or values[0]['quick_per_asset'] != initial
                or any(value['quick_panel_ids'] != expected_ids for value in values)):
            raise ValueError('actual common32 epoch curve identity differs')
        curves.extend([name, value['step'] // 96, value['quick_metrics']['voxel_iou'],
                       value['quick_metrics']['voxel_f1']] for value in values)
        files.append(path)
    files.extend([Path(__file__), Path(common.__file__)])
    scope = {'development128_completed': True, 'sampling_noises': 3, 'training_seed': 101,
             'candidate_for_seed_confirmation': NAME if eligible else None,
             'scientific_outcome': report['status'], 'seed_confirmation_completed': False,
             'additional121_completed': False, 'model_adopted': False,
             'full_campaign_complete': False, 'full_goal_complete': False,
             'scope': 'Frozen-utility development only; raw480,128cases/93sources/3noises. '
                      'Common32 curves and value-address diagnostic are separate intermediate/sensitivity panels. '
                      'This report doesnot certify seed202,reserved121,module-specific gains or model adoption.'}
    return {'report': report, 'diagnostic': diagnostic, 'curves': curves, 'scope': scope,
            'files': list(dict.fromkeys(files)), 'galleries': {'development128': {
                'records': records, 'images': images, 'ids': ids,
                'groups': [['zero-source-only', 'zero-edited-rgb', 'v1', 'top1-reference',
                            'static-top1-c1', NAME],
                           ['previous-c1', 'static-top1-c1', 'task-full', 'task-native-spatial', NAME]],
                'scope': 'Raw480 / development128 / noise0. Numbers:3D voxel IoU;images:occupancy max-projections.'}}}


def payload(root):
    data = collect(root)
    folder = root / 'development-wandb-v1/presentation'
    summary = read(folder / 'summary.json')
    if (summary['status'] != 'PASS_COMPLETED_CAMPAIGN_PRESENTATION'
            or summary['scope'] != data['scope']
            or summary['input_SHA256'] != {str(path): sha(path) for path in data['files']}):
        raise ValueError('actual frozen-utility presentation or inputs differ')
    report, metrics = data['report'], {}
    for name, model in report['models'].items():
        for key in ['micro_voxel_iou', 'micro_voxel_f1']:
            metrics[f'development128/{name}/{key}'] = model[key]
    contrasts = report['contrasts']
    for name, value in contrasts.items():
        for key, number in zip(['gain', 'ci95_lower', 'ci95_upper'],
                               [value['gain'], *value['paired_cluster_bootstrap_ci95']]):
            metrics[f'development128/utility-vs-{name}/{key}'] = number
    sensitivity = data['diagnostic']['normal_minus_wrong_address']
    for key, number in zip(['gain', 'ci95_lower', 'ci95_upper'],
                           [sensitivity['gain'], *sensitivity['paired_cluster_bootstrap_ci95']]):
        metrics['diagnostic_common32/value-address/' + key] = number
    tables = {
        'development128/models': {'columns': ['method', 'micro_IoU', 'micro_F1'],
                                 'data': [[name, report['models'][name]['micro_voxel_iou'],
                                           report['models'][name]['micro_voxel_f1']] for name in ORDER]},
        'development128/utility_contrasts': {'columns': ['baseline', 'gain_IoU', 'ci95_lower', 'ci95_upper'],
                                           'data': [[name, value['gain'], *value['paired_cluster_bootstrap_ci95']]
                                                    for name, value in contrasts.items()]},
        'common32/epoch_curves': {'columns': ['method', 'epoch', 'micro_IoU', 'micro_F1'],
                                'data': data['curves']},
    }
    files = data['files'] + [path for path in folder.iterdir() if path.is_file()]
    if any(path.suffix == '.pt' for path in files):
        raise ValueError('model binaries remain on canonical file4')
    repo, canonical = Path(__file__).resolve().parents[2], root.parent.parent
    artifacts = {}
    for path in files:
        name = str(path.relative_to(canonical)) if path.is_relative_to(canonical) else 'source/' + str(path.relative_to(repo))
        artifacts[name] = str(path)
    return {'metrics': metrics, 'tables': tables,
            'images': {path.stem: str(path) for path in folder.glob('*.png')},
            'artifact_files': artifacts, 'scope': data['scope'],
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}


def publication_jobs(root):
    folder, jobs = root / 'development-wandb-v1', []
    for mode, output in [('render', folder / 'presentation/summary.json'),
                         ('publish', folder / 'publication.json'),
                         ('readback', folder / 'remote-readback/audit.json')]:
        jobs.append({'name': 'frozen-utility-report-' + mode, 'resource': 'cpu', 'priority': 5,
                     'depends_on': ['close-frozen-utility-fixed-final-report', 'summarize-frozen-utility-value-address']
                                   if not jobs else [jobs[-1]['name']],
                     'requires': [str(root / 'fixed-final-report-file4.json')] if mode == 'render' else [],
                     'outputs': [str(output)],
                     'command': ['taskset', '-c', '84,85', sys.executable, '-u', '-m',
                                 'experiments.deformnext_retargeting.frozen_utility_wandb',
                                 '--root', str(root), '--mode', mode]})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=['render', 'publish', 'readback'], required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    folder = args.root / 'development-wandb-v1'
    if args.mode == 'render':
        common.render(args.root, collector=collect, output_folder=folder / 'presentation', labels=LABELS)
    elif args.mode == 'publish':
        common.publish(args.root, payload_factory=payload, output_folder=folder, run_id=RUN_ID,
                       run_name='SS frozen utility fixed-final development comparison')
    else:
        common.readback(args.root, output_folder=folder, run_id=RUN_ID)


if __name__ == '__main__':
    main()
