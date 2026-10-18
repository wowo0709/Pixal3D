"""Fixed-weight 12/50-step diagnostic required when loss and generation diverge."""
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
from .prepare_ss_seed_confirmation import read, write_new

ARMS = ('v1', 'original-c1', 'b3-attention-r', 'c2-query-r')
STEPS = (12, 50)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def evaluation_config(source, work, steps):
    if steps not in STEPS:
        raise ValueError('fixed diagnostic uses only 12 and 50 sampling steps')
    cfg = copy.deepcopy(source)
    cfg['validation']['quick_metrics'].update(panel_size=32,
        panel_file=str(work/'panels/panel-32.json'), seed_map_file=str(work/'panels/seed-map.json'),
        sampling_steps=steps, generation_batch_size=8)
    return cfg


def validate_pair(left, right):
    """The intentional step difference must not bypass any other paired invariant."""
    for value, steps in [(left, 12), (right, 50)]:
        expected = {'status': 'PASS', 'stage': 'ss64', 'step': 480, 'weights': 'raw',
            'sampling_steps': steps, 'guidance': 3., 'generation_batch_size': 8,
            'optimizer_updates': 0, 'optimizer_updates_during_evaluation': 0,
            'conditioning_mode': 'edited-rgb', 'target_modality': 'rgb',
            'transport_control': 'normal', 'r9_path_control': 'normal', 'input_view_count': 4}
        if any(value.get(k) != v for k, v in expected.items()):
            raise ValueError('sampler diagnostic execution protocol changed')
        if value.get('diagnostic_only') or len(value['per_asset']) != 32:
            raise ValueError('sampler diagnostic requires the intact fixed32 input')
        if value.get('input_states') != ['M1','M0','M0','M0'] or value.get('input_camera_ids') != [8,9,10,11]:
            raise ValueError('sampler diagnostic requires edited RGB and three distinct source cameras')
        if value['quick_panel_ids'] != [row['sample_id'] for row in value['per_asset']]:
            raise ValueError('sampler diagnostic image/metric order differs')
        if any(not row.get('target_voxels_sha256') for row in value['per_asset']):
            raise ValueError('actual target occupancy hashes are required')
    for key in ['checkpoint', 'conditioner_checkpoint', 'manifest_sha256', 'panel_file',
                'seed_map_file', 'input_camera_ids', 'input_states', 'quick_panel_ids']:
        if left.get(key) != right.get(key):
            raise ValueError('fixed-weight sampler pair changed: '+key)
    result = compare([('same-common32-noise', {'per_asset': left['per_asset']},
                      {'per_asset': right['per_asset']})])
    if result['assets'] != 32 or result['source_clusters'] != 32 or result['paired_target_hashes_verified'] != 32:
        raise ValueError('sampler diagnostic requires 32 identical source/GT/noise records')
    result.update(status='diagnostic', comparison='same checkpoint, 50 steps minus 12 steps',
        baseline_sampling_steps=12, candidate_sampling_steps=50, selection_use=False)
    return result


def prepare(root, legacy, work):
    stage = root/'sampler-sensitivity-v1'
    if stage.exists():
        raise FileExistsError('preserve the earlier sampler diagnostic')
    primary_path = root/'module-study-native-v4/final-primary-report.json'
    primary = read(primary_path)
    if primary['status'] != 'primary_matrix_complete':
        raise ValueError('complete primary matrix is required')
    rows = primary['training_audit']['b3-attention-r']['validation']
    loss = lambda row: sum(x['K4_loss'] for x in row['sources']) / len(row['sources'])
    if not (loss(rows[-1]) < loss(rows[1]) and
            rows[-1]['quick_metrics']['voxel_iou'] < rows[1]['quick_metrics']['voxel_iou']):
        raise ValueError('planned loss/generation divergence trigger is absent')
    sources = {};configs = {}
    for arm in ARMS:
        if arm in {'v1', 'original-c1'}:
            short = 'v1' if arm == 'v1' else 'c1'
            family = 'runs_world_v4' if arm == 'v1' else 'runs_r9_v5'
            cfg = legacy/family/'ss64'/short/'config_resolved.json'
            ckpts = work/'evaluation-checkpoints'/short
        else:
            cfg = root/'module-study-native-v4/configs'/(arm+'.json')
            ckpts = root/'module-study-native-v4/runs_ss64'/arm/'ckpts'
        checkpoint = ckpts/'denoiser_step0000480.pt'
        conditioner = None if arm == 'v1' else ckpts/'deform_conditioner_step0000480.pt'
        baseline = Path(primary['models'][arm]['evaluations'][0])
        paths = [cfg, checkpoint, baseline] + ([conditioner] if conditioner else [])
        if any(not p.is_file() for p in paths):
            raise FileNotFoundError('explicit saved sampler input is absent: '+arm)
        source = read(cfg)
        if (source['trainer']['args'].get('deform_conditioner') is None) != (conditioner is None):
            raise ValueError('saved conditioner/recipe contract differs')
        sources[arm] = {'config': fingerprint(cfg), 'checkpoint': fingerprint(checkpoint),
            'conditioner': fingerprint(conditioner) if conditioner else None,
            'primary_noise0': fingerprint(baseline)}
        for steps in STEPS:
            recipe = stage/'configs'/f'{arm}-{steps}.json'
            write_new(recipe, evaluation_config(source, work, steps))
            configs[f'{arm}-{steps}'] = fingerprint(recipe)
    write_new(stage/'prepared.json', {'status': 'PASS_CONFIGS_ONLY', 'arms': list(ARMS),
        'sampling_steps': list(STEPS), 'source_inputs': sources, 'evaluation_configs': configs,
        'primary_report': fingerprint(primary_path),
        'panel': fingerprint(work/'panels/panel-32.json'), 'noise': fingerprint(work/'panels/seed-map.json'),
        'trigger': {'arm': 'b3-attention-r', 'step96_loss': loss(rows[1]), 'step480_loss': loss(rows[-1]),
            'step96_iou': rows[1]['quick_metrics']['voxel_iou'], 'step480_iou': rows[-1]['quick_metrics']['voxel_iou']},
        'selection_use': False, 'production_sampler_changed': False,
        'scope': 'Fixed four representative final checkpoints, same32/one noise,12 versus50; not a new primary protocol or a query-module ablation.'})


def queue(root, work):
    stage = root/'sampler-sensitivity-v1';prepared = read(stage/'prepared.json')
    jobs = []; previous = {2: 'summarize-current-regional-original-network-six-v3',
                          3: 'summarize-current-regional-original-network-six-v3'}
    for index, arm in enumerate(ARMS):
        device = 2 + index % 2
        for steps in STEPS:
            name = f'sampler-diagnostic-{arm}-{steps}-v1'
            cfg = stage/'configs'/f'{arm}-{steps}.json';source = prepared['source_inputs'][arm]
            output = stage/'evaluations'/arm/str(steps)
            command = [sys.executable, '-u', '-m', 'experiments.deformnext_retargeting.evaluate_native_checkpoint',
                '--stage', 'ss64', '--config-root', str(cfg.parent), '--arm-config', str(cfg),
                '--checkpoint', source['checkpoint']['path'], '--step', '480', '--output', str(output),
                '--panel-file', prepared['panel']['path'], '--seed-map-file', prepared['noise']['path'],
                '--panel-size', '32', '--generation-batch', '8']
            if source['conditioner']:
                command += ['--conditioner-checkpoint', source['conditioner']['path']]
            jobs.append({'name': name, 'resource': 'evaluation', 'gpu': str(device), 'priority': 2,
                'depends_on': [previous[device]], 'requires': [str(cfg), source['checkpoint']['path']],
                'outputs': [str(output/n) for n in ['evaluation.json', 'panel.png', 'actual_inputs.png']],
                'command': command, 'scope': prepared['scope']})
            previous[device] = name
    summary = 'summarize-sampler-diagnostic-v1'
    jobs.append({'name': summary, 'resource': 'cpu', 'depends_on': list(previous.values()),
        'outputs': [str(stage/'summary.json'), str(stage/'summary.md')],
        'command': [sys.executable, '-u', '-m', 'experiments.deformnext_retargeting.sampler_sensitivity',
                    '--mode', 'summarize', '--root', str(root), '--work-root', str(work)]})
    jobs.append({'name': 'publish-sampler-diagnostic-v1', 'resource': 'cpu', 'depends_on': [summary],
        'outputs': [str(stage/'wandb-publication.json')],
        'command': [sys.executable, '-u', '-m', 'experiments.deformnext_retargeting.sampler_sensitivity',
                    '--mode', 'publish', '--root', str(root), '--work-root', str(work)]})
    path = work/'queue-jobs.json';original = path.read_bytes();existing = json.loads(original)
    names = {j['name'] for j in existing};new = {j['name'] for j in jobs}
    if names & new or len(new) != len(jobs) or any(d not in names | new for j in jobs for d in j['depends_on']):
        raise ValueError('sampler diagnostic queue names/dependencies differ')
    backup = work/'queue-jobs-before-sampler-diagnostic-v1.json'
    with backup.open('xb') as stream: stream.write(original)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    for job in jobs: job['source_commit'] = commit
    tmp = path.with_suffix('.sampler.tmp');tmp.write_text(json.dumps(existing+jobs, indent=2)+'\n');os.replace(tmp, path)
    write_new(root/'provenance/sampler-diagnostic-v1-queue.json', {'status': 'QUEUED_NOT_STARTED',
        'source_commit': commit, 'jobs': jobs, 'training_interrupted': False,
        'physical_GPU_allowlist': [0,1,2,3], 'scope': prepared['scope']})


def summarize(root, work):
    stage = root/'sampler-sensitivity-v1';prepared = read(stage/'prepared.json')
    for ref in [prepared['primary_report'], prepared['panel'], prepared['noise'], *prepared['evaluation_configs'].values()]:
        if fingerprint(Path(ref['path'])) != ref: raise ValueError('frozen sampler provenance changed')
    values = {};within = {};between = {}
    for arm in ARMS:
        refs = prepared['source_inputs'][arm]
        for ref in refs.values():
            if ref and fingerprint(Path(ref['path'])) != ref: raise ValueError('saved sampler input changed: '+arm)
        values[arm] = {steps: read(stage/'evaluations'/arm/str(steps)/'evaluation.json') for steps in STEPS}
        within[arm] = validate_pair(values[arm][12], values[arm][50])
        original = {r['sample_id']: r for r in read(Path(refs['primary_noise0']['path']))['per_asset']}
        for row in values[arm][12]['per_asset']:
            if any(row[k] != original[row['sample_id']][k] for k in ['seed','source_cluster_id','tp','fp','fn']):
                raise ValueError('fresh12 does not replay the primary final checkpoint')
        for steps in STEPS:
            value = values[arm][steps]
            cfg = stage/'configs'/f'{arm}-{steps}.json'
            if value['checkpoint'] != refs['checkpoint']['path'] or value['conditioner_checkpoint'] != (refs['conditioner']['path'] if refs['conditioner'] else None) or value['arm_config'] != str(cfg):
                raise ValueError('evaluation used a different recipe/checkpoint')
    for arm in ARMS[1:]:
        for steps in STEPS:
            between[f'{arm}-vs-v1-{steps}'] = compare([('same-common32-noise', values['v1'][steps], values[arm][steps])])
    result = {'status': 'PASS_SAMPLER_DIAGNOSTIC', 'selection_use': False,
        'production_sampler_changed': False, 'scope': prepared['scope'],
        'prepared_inputs': fingerprint(stage/'prepared.json'), 'within_checkpoint_50_minus_12': within,
        'between_checkpoint_same_sampler': between, 'fresh12_replays_primary_counts': True,
        'evaluations': {arm: {str(n): fingerprint(stage/'evaluations'/arm/str(n)/'evaluation.json') for n in STEPS} for arm in ARMS}}
    write_new(stage/'summary.json', result)
    lines = ['# Fixed-weight SS sampler sensitivity', '', prepared['scope'], '',
        '| Arm | IoU12 | IoU50 | 50−12, pp [95% CI] |', '| --- | ---: | ---: | --- |']
    for arm,row in within.items():
        lo,hi = row['paired_cluster_bootstrap_ci95']
        lines.append(f'| {arm} | {row["baseline_iou"]:.6f} | {row["candidate_iou"]:.6f} | {100*row["gain"]:+.3f} [{100*lo:+.3f}, {100*hi:+.3f}] |')
    lines += ['', 'Same32/one noise only; exploratory source-cluster intervals, conditional on saved weights. All fresh12 counts replay primary noise0. Native inputs, GT, CFG3 and weights retained. Primary final128×3 sampler12 remains unchanged. No module-specific causal or training-seed-population claim.']
    with (stage/'summary.md').open('x') as stream: stream.write('\n'.join(lines)+'\n')


def publish(root):
    import wandb
    stage = root/'sampler-sensitivity-v1';summary = read(stage/'summary.json')
    if summary['status'] != 'PASS_SAMPLER_DIAGNOSTIC': raise ValueError('sampler diagnostic is incomplete')
    run = wandb.init(project='pixal3d-deformnext-retargeting', entity='youngwoojeon-unist',
        id='ss-sampler-sensitivity-v1', name='SS fixed-weight 12 versus50 steps', resume='never',
        dir=str(stage), config={'scope': summary['scope'], 'selection_use': False, 'production_sampler_changed': False})
    artifact = wandb.Artifact('ss-fixed-weight-sampler-sensitivity-v1', type='evaluation-report')
    for file in ['prepared.json', 'summary.json', 'summary.md']: artifact.add_file(str(stage/file))
    for arm in ARMS:
        for steps in STEPS:
            folder = stage/'evaluations'/arm/str(steps);record = read(folder/'evaluation.json')
            run.log({f'{arm}/steps{steps}/'+k: v for k,v in record['quick_metrics'].items()})
            run.log({f'{arm}/steps{steps}/GT_Pred_Diff': wandb.Image(str(folder/'panel.png'))})
            artifact.add_file(str(folder/'evaluation.json'), name=f'{arm}/{steps}/evaluation.json')
    table = wandb.Table(columns=['arm', 'IoU12', 'IoU50', 'gain_pp', 'CI95_lower_pp', 'CI95_upper_pp'],
        data=[[arm,row['baseline_iou'],row['candidate_iou'],100*row['gain'],
               *(100*x for x in row['paired_cluster_bootstrap_ci95'])]
              for arm,row in summary['within_checkpoint_50_minus_12'].items()])
    run.log({'fixed_weight_comparisons': table});run.log_artifact(artifact)
    run.summary.update({'status': summary['status'], 'selection_use': False, 'production_sampler_changed': False})
    url=run.url;run.finish()
    write_new(stage/'wandb-publication.json', {'status': 'PASS_WANDB_FINISHED', 'run_url': url,
        'summary': fingerprint(stage/'summary.json'), 'remote_independent_readback_required': True})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['prepare','queue','summarize','publish'], required=True)
    p.add_argument('--root', type=Path, required=True);p.add_argument('--work-root', type=Path, required=True)
    p.add_argument('--legacy-root', type=Path)
    args = p.parse_args()
    if args.mode == 'prepare':
        if args.legacy_root is None: p.error('prepare requires --legacy-root')
        prepare(args.root, args.legacy_root, args.work_root)
    elif args.mode == 'queue': queue(args.root, args.work_root)
    elif args.mode == 'summarize': summarize(args.root, args.work_root)
    else: publish(args.root)


if __name__ == '__main__': main()
