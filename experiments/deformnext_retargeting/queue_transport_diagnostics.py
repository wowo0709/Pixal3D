"""Queue fixed-checkpoint SS path diagnostics on an explicitly freed GPU pair."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

from .matching_v2_queue import gpu_ids


def build_jobs(root, work, devices, after_job, query_only=False, wrong_object_only=False):
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError('two distinct assigned diagnostic GPUs required')
    if query_only and wrong_object_only:
        raise ValueError('choose one diagnostic append scope')
    study = root / 'module-study-native-v4'
    panel, seeds = work / 'panels/panel-32.json', work / 'panels/seed-map.json'
    configurations = [
        ('p2-matcher-grid', root / 'retarget-native-v3', 'train-ss-p2-matcher-grid', devices[0]),
        ('c1-retarget-r', study, 'train-native-module-c1-retarget-r', devices[1]),
        ('b3-attention-r', study, 'train-native-module-b3-attention-r', devices[0]),
    ]
    if query_only:
        configurations = [('c2-query-r', study, 'train-native-module-c2-query-r', devices[0])]
    jobs, specifications = [], {}
    for arm, stage, train_job, device in configurations:
        run, config = stage / 'runs_ss64' / arm, stage / 'configs' / (arm + '.json')
        checkpoint = run / 'ckpts/denoiser_step0000480.pt'
        conditioner = run / 'ckpts/deform_conditioner_step0000480.pt'
        modes = ['normal', 'shuffled-deform'] if query_only else ['normal', 'source-off', 'wrong-corr']
        if not query_only and arm != 'b3-attention-r':
            modes.insert(2, 'adapter-off')
        outputs, names = {}, {}
        previous = None
        if wrong_object_only:
            modes = ['wrong-object']
            outputs['normal'] = study / 'evaluations/diagnostics' / arm / 'normal/evaluation.json'
            names['normal'] = f'diagnose32-{arm}-normal'
            previous = names['normal']
        for mode in modes:
            out = study / 'evaluations/diagnostics' / arm / mode
            name = f'diagnose32-{arm}-{mode}'
            dependencies = list(dict.fromkeys([after_job, train_job]))
            if previous is not None:
                dependencies.append(previous)
            jobs.append({
                'name': name, 'resource': 'evaluation', 'gpu': str(device), 'priority': 2,
                'depends_on': dependencies,
                'requires': [str(p) for p in [config, checkpoint, conditioner, panel, seeds]],
                'outputs': [str(out / p) for p in ['evaluation.json', 'panel.png', 'actual_inputs.png']],
                'command': [sys.executable, '-u', '-m',
                    'experiments.deformnext_retargeting.evaluate_native_checkpoint',
                    '--stage', 'ss64', '--config-root', str(config.parent), '--arm-config', str(config),
                    '--checkpoint', str(checkpoint), '--conditioner-checkpoint', str(conditioner),
                    '--step', '480', '--output', str(out), '--panel-file', str(panel),
                    '--seed-map-file', str(seeds), '--generation-batch', '8',
                    '--transport-control', mode],
                'scope': 'Diagnostic common32/noise0 only; fixed weights and edited-RGB four-view input. '
                         'Normal and perturbed replicas share a physical device; not a primary matrix arm.',
            })
            outputs[mode], names[mode] = out / 'evaluation.json', name
            previous = name
        for mode in modes if wrong_object_only else modes[1:]:
            name = f'compare-diagnostic-{arm}-{mode}-vs-normal'
            spec = work / 'comparison-inputs' / (name + '.json')
            specifications[spec] = [{'noise': '0', 'baseline': str(outputs['normal']),
                                      'candidate': str(outputs[mode])}]
            out = study / 'comparisons' / f'diagnostic-{arm}-{mode}-vs-normal.json'
            jobs.append({'name': name, 'resource': 'cpu', 'priority': 2,
                'depends_on': [names['normal'], names[mode]],
                'requires': [str(outputs['normal']), str(outputs[mode]), str(spec)],
                'outputs': [str(out)],
                'command': [sys.executable, '-u', '-m',
                    'experiments.deformnext_retargeting.compare_ss_noise_panels',
                    '--input-list', str(spec), '--output', str(out)]})
    return jobs, specifications


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--work-root', type=Path, required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--after-job', required=True)
    scope = p.add_mutually_exclusive_group()
    scope.add_argument('--query-only', action='store_true',
                   help='Append only the saved C2 normal/shuffled-context diagnostic and comparison')
    scope.add_argument('--wrong-object-only', action='store_true',
                   help='Reuse queued normal replicas and append cross-object source-evidence diagnostics')
    a = p.parse_args()
    path = a.work_root / 'queue-jobs.json'
    original = path.read_bytes()
    existing = json.loads(original)
    new, specifications = build_jobs(a.root, a.work_root, gpu_ids(a.gpus), a.after_job,
                                      a.query_only, a.wrong_object_only)
    old_names, new_names = {j['name'] for j in existing}, {j['name'] for j in new}
    if old_names & new_names:
        raise ValueError('transport diagnostics already queued')
    if any(dep not in old_names | new_names for j in new for dep in j.get('depends_on', [])):
        raise ValueError('diagnostic training dependency is not in the campaign')
    if any(f.exists() for f in specifications):
        raise ValueError('diagnostic comparison specification already exists')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    kind = 'query' if a.query_only else 'wrong-object' if a.wrong_object_only else 'transport'
    backup = path.with_name(path.name + '.before-' + kind + '-diagnostics-' + stamp)
    backup.write_bytes(original)
    for f, value in specifications.items():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(value, indent=2) + '\n')
    if path.read_bytes() != original:
        raise ValueError('queue specification changed during diagnostic preparation')
    updated = json.dumps(existing + new, indent=2) + '\n'
    temporary = path.with_suffix('.diagnostics.tmp')
    temporary.write_text(updated)
    os.replace(temporary, path)
    receipt = {'status': 'QUEUED_NOT_EXECUTED', 'at': stamp,
        'after_job': a.after_job, 'assigned_gpus': gpu_ids(a.gpus),
        'original_jobs': len(existing), 'original_jobs_unchanged': True,
        'new_generation_jobs': sum(j['resource'] == 'evaluation' for j in new),
        'new_paired_comparisons': sum(j['resource'] == 'cpu' for j in new),
        'new_training_runs': 0, 'normal_primary_configurations': 14,
        'backup': str(backup), 'queue_before_sha256': hashlib.sha256(original).hexdigest(),
        'queue_after_sha256': hashlib.sha256(updated.encode()).hexdigest(),
        'added_jobs': [j['name'] for j in new],
        'limitations': 'One development-panel32 sampling repeat, no training-seed replication; '
                       'perturbed-minus-normal gain is a path diagnostic. '
                       + ('RAW context cycles across each generation batch; other conditioning stays intact.'
                          if a.query_only else 'Donor foreground values are resampled onto recipient valid queries; '
                          'sensitivity also changes object appearance and does not isolate address correctness.'
                          if a.wrong_object_only else 'Wrong-corr is a weak one-query-slot perturbation.')}
    out = a.root / ('provenance/transport-query-control-queue.json' if a.query_only else
                    'provenance/transport-wrong-object-control-queue.json' if a.wrong_object_only
                    else 'provenance/transport-control-queue.json')
    out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
