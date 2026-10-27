"""Prepare and run a fixed VAL subset for paired 1024 cascade scale checks."""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


MODES = {'image-estimated-v1': 'image-scale', 'legacy-target': 'legacy-target',
         'native-unit': 'native-unit'}


def archive_completed_case(output: Path, args) -> None:
    if args.archive_remote is None:
        return
    prediction = output / 'prediction.npz'
    geometry = output / 'geometry-evaluation.npz'
    if not prediction.exists() and not geometry.exists():
        return  # Previously verified VAL8 outputs are already archived on file4.
    if not prediction.is_file() or not geometry.is_file():
        raise ValueError('partial local cascade output cannot be archived')
    if (not args.archive_remote.startswith('/file4/youngwoo/deformation/experiments/retargeting-v1/')
            or args.archive_host != 'youngwoo@n16.unist.info'):
        raise ValueError('archive must use the declared file4 path through node16')
    remote_dir = (f'{args.archive_remote.rstrip("/")}/'
                  f'{output.parent.parent.name}/{output.parent.name}/{output.name}')
    remote = f'{args.archive_host}:{remote_dir}/'
    ssh = f'ssh -o BatchMode=yes -p {args.archive_port}'
    subprocess.run(['ssh', '-o', 'BatchMode=yes', '-p', str(args.archive_port),
                    args.archive_host, 'mkdir', '-p', remote_dir], check=True)
    subprocess.run(['rsync', '-a', '-e', ssh, str(output) + '/', remote], check=True)
    check = subprocess.run(['rsync', '-a', '--checksum', '--dry-run', '--itemize-changes',
                            '-e', ssh, str(output) + '/', remote],
                           check=True, capture_output=True, text=True)
    if check.stdout.strip():
        raise ValueError('file4 archive checksum differs: ' + check.stdout[:400])
    hashes = {}
    for path in (prediction, geometry):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        hashes[path.name] = digest.hexdigest()
    receipt = output / 'archive-receipt.json'
    receipt.write_text(json.dumps({'status': 'PASS_FILE4_ARCHIVE_CHECKSUM_VERIFIED',
                                   'remote': remote, 'sha256': hashes}, indent=2) + '\n')
    subprocess.run(['rsync', '-a', '-e', ssh, str(receipt), remote], check=True)
    prediction.unlink()
    geometry.unlink()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def prepare(args):
    if args.output.exists() or args.count < 2:
        raise ValueError('fresh campaign output and matched population required')
    panel = json.loads(args.panel_file.read_text())['sample_ids']
    plans = {}
    for path in glob.glob(args.base_plan_glob, recursive=True):
        data = json.loads(Path(path).read_text())
        if data['row']['sample_id'] in plans:
            raise ValueError('ambiguous base cascade plan')
        plans[data['row']['sample_id']] = (path, data)
    selected = []
    seen_clusters = set()
    for sid in panel:
        if sid not in plans:
            continue
        row = plans[sid][1]['row']
        cluster = row['source_cluster_id']
        if cluster in seen_clusters:
            continue
        selected.append(sid)
        seen_clusters.add(cluster)
        if len(selected) == args.count:
            break
    if len(selected) != args.count:
        raise ValueError('too few source-disjoint prepared VAL plans')
    predictions = {row['sample_id']: row for row in read_jsonl(args.scale_predictions)}
    receipt = json.loads(args.scale_predictions.with_suffix('.receipt.json').read_text())
    if (receipt['status'] != 'PASS_IMAGE_ONLY_SCALE_PREDICTIONS'
            or not set(selected) <= set(predictions)):
        raise ValueError('image prediction population or receipt differs')
    freeze = json.loads(args.freeze.read_text())
    if freeze['status'] != 'FROZEN_FROM_DEVELOPMENT_ONLY':
        raise ValueError('development-frozen stage pair required')
    pbr = freeze['models']['V1']['pbr1024']
    import hashlib
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if (sha(args.scale_predictions) != receipt['output_sha256']
            or sha(pbr['checkpoint']['path']) != pbr['checkpoint']['SHA256']):
        raise ValueError('frozen image predictions or PBR checkpoint bytes differ')
    args.output.mkdir(parents=True)
    cases = []
    for sid in selected:
        source_path, original = plans[sid]
        for stage in ('shape512', 'shape1024'):
            frozen = freeze['models']['V1'][stage]
            if (original['stages'][stage]['config'] != frozen['config']['path']
                    or original['stages'][stage]['checkpoint'] != frozen['checkpoint']['path']):
                raise ValueError('VAL cascade plan differs from frozen original Shape stage: ' + stage)
        record = {'sample_id': sid, 'source_cluster_id': original['row']['source_cluster_id'],
                  'plans': {}}
        for mode, folder in MODES.items():
            case = dict(original)
            case.pop('inputs_SHA256', None)
            case.pop('source_SHA256', None)
            scale = (predictions[sid]['predicted_mesh_scale'] if mode == 'image-estimated-v1'
                     else 1.0 if mode == 'native-unit' else original['row']['mesh_scale'])
            case['row'] = dict(original['row'], mesh_scale=scale)
            case['object_scale_policy'] = mode
            case['image_scale_predictions'] = str(args.scale_predictions) if mode == 'image-estimated-v1' else None
            case['SS_evaluation'] = str(args.image_ss_evaluation if mode == 'image-estimated-v1'
                                        else args.native_ss_evaluation if mode == 'native-unit'
                                        else args.legacy_ss_evaluation)
            case['stages'] = dict(original['stages'])
            case['stages']['pbr1024'] = {'config': pbr['native_config']['path'],
                                       'checkpoint': pbr['checkpoint']['path']}
            case['output'] = str(args.output / folder / sid)
            case['status'] = 'PAIRED_VAL1024_SCALE_SENSITIVITY_PLAN'
            case['pilot_rebased_from'] = source_path
            plan_path = args.output / 'plans' / folder / (sid + '.json')
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(json.dumps(case, indent=2) + '\n')
            record['plans'][mode] = str(plan_path)
        cases.append(record)
    campaign = {'status': 'PREPARED_PAIRED_VAL1024_SCALE_SENSITIVITY',
                'samples': len(cases), 'source_clusters': len(seen_clusters),
                'selection': 'first source-disjoint prepared samples in fixed VAL32 panel order',
                'image_scale_receipt': str(args.scale_predictions.with_suffix('.receipt.json')),
                'image_scale_model_sha256': receipt['model_sha256'],
                'frozen_model_set': str(args.freeze),
                'target_M1_scale_is_diagnostic_only': True, 'cases': cases}
    (args.output / 'campaign.json').write_text(json.dumps(campaign, indent=2) + '\n')
    print(json.dumps({'status': campaign['status'], 'samples': len(cases),
                      'source_clusters': len(seen_clusters)}), flush=True)


def worker(args):
    if args.gpu not in {0, 1, 2, 3, 6, 7}:
        raise ValueError('authorized physical GPU 0,1,2,3,6,7 required')
    if importlib.util.find_spec('cumesh') is None:
        raise RuntimeError('Pixal3D CUDA environment with cumesh is required')
    campaign = json.loads((args.output / 'campaign.json').read_text())
    if campaign['status'] != 'PREPARED_PAIRED_VAL1024_SCALE_SENSITIVITY':
        raise ValueError('prepared campaign required')
    folder = MODES[args.mode]
    log_root = args.output / 'logs' / folder
    log_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), MATCHING_V2_ALLOWED_GPUS='0,1,2,3,6,7',
               ATTN_BACKEND='sdpa', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    for i, item in enumerate(campaign['cases']):
        sid = item['sample_id']
        plan_path = Path(item['plans'][args.mode])
        plan = json.loads(plan_path.read_text())
        result_path = Path(plan['output']) / 'generation.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if (result['status'] != 'PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
                    or result['sample_id'] != sid or result['object_scale_policy'] != args.mode
                    or result['actual_resolution'] != 1024):
                raise ValueError('incomplete or inconsistent existing VAL cascade output')
            archive_completed_case(Path(plan['output']), args)
            continue
        if Path(plan['output']).exists():
            raise FileExistsError('partial case requires review: ' + plan['output'])
        Path(plan['output']).parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        command = [sys.executable, '-u', '-m',
                   'experiments.deformnext_retargeting.cascade_from_ss_artifact',
                   '--plan', str(plan_path)]
        log_path = log_root / (sid + '.log')
        with log_path.open('w') as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        if completed.returncode or not result_path.is_file():
            raise RuntimeError('VAL cascade failed; inspect ' + str(log_path))
        result = json.loads(result_path.read_text())
        if (result['status'] != 'PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
                or result['object_scale_policy'] != args.mode
                or result['sample_id'] != sid or result['actual_resolution'] != 1024):
            raise ValueError('invalid completed VAL cascade receipt')
        archive_completed_case(Path(plan['output']), args)
        print(json.dumps({'mode': args.mode, 'index': i + 1, 'samples': len(campaign['cases']),
                          'sample_id': sid, 'seconds': round(time.monotonic()-started, 2),
                          'CD': result['metrics']['chamfer_L1'],
                          'FScore': result['metrics']['surface_FScore']}), flush=True)
    print(json.dumps({'status': 'PASS_PAIRED_VAL1024_SCALE_WORKER',
                      'mode': args.mode, 'samples': len(campaign['cases'])}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('prepare', 'worker'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--panel-file', type=Path)
    p.add_argument('--base-plan-glob')
    p.add_argument('--scale-predictions', type=Path)
    p.add_argument('--image-ss-evaluation', type=Path)
    p.add_argument('--legacy-ss-evaluation', type=Path)
    p.add_argument('--native-ss-evaluation', type=Path)
    p.add_argument('--freeze', type=Path)
    p.add_argument('--count', type=int, default=8)
    p.add_argument('--mode', choices=tuple(MODES))
    p.add_argument('--gpu', type=int)
    p.add_argument('--archive-remote')
    p.add_argument('--archive-host', default='youngwoo@n16.unist.info')
    p.add_argument('--archive-port', type=int, default=55555)
    args = p.parse_args()
    if args.operation == 'prepare':
        if any(value is None for value in (args.panel_file, args.base_plan_glob,
                                            args.scale_predictions, args.image_ss_evaluation,
                                            args.legacy_ss_evaluation,
                                            args.native_ss_evaluation, args.freeze)):
            raise ValueError('all frozen input paths required for preparation')
        prepare(args)
    else:
        if args.mode is None or args.gpu is None:
            raise ValueError('worker mode and GPU assignment required')
        worker(args)


if __name__ == '__main__':
    main()
