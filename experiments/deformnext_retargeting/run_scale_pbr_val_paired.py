"""Paired image-scale VAL8 PBR replay and native render on fixed V1 geometry.

This is a development diagnostic. Paper scores use the common Cycles evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(command: list[str], *, log: Path | None = None, env=None) -> None:
    if log is None:
        subprocess.run(command, check=True, env=env)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, env=env)


def transfer_and_prune(root: Path, remote: str, names: tuple[str, ...]) -> None:
    receipt = root / 'file4-archive.json'
    if receipt.exists():
        if json.loads(receipt.read_text()).get('status') != 'PASS_CHECKSUM_VERIFIED_FILE4_ARCHIVE':
            raise ValueError('invalid existing file4 archive receipt')
        return
    for name in names:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    ssh = 'ssh -o BatchMode=yes -p 55555'
    run(['ssh', '-o', 'BatchMode=yes', '-p', '55555', 'youngwoo@n16.unist.info',
         'mkdir', '-p', remote])
    destination = 'youngwoo@n16.unist.info:' + remote + '/'
    run(['rsync', '-a', '-e', ssh, str(root) + '/', destination])
    check = subprocess.run(['rsync', '-a', '--checksum', '--dry-run', '--itemize-changes',
                            '-e', ssh, str(root) + '/', destination],
                           check=True, capture_output=True, text=True)
    if check.stdout.strip():
        raise ValueError('file4 archive differs: ' + check.stdout[:400])
    receipt.write_text(json.dumps({'status': 'PASS_CHECKSUM_VERIFIED_FILE4_ARCHIVE',
                                   'remote': remote,
                                   'sha256': {name: digest(root / name) for name in names}},
                                  indent=2) + '\n')
    run(['rsync', '-a', '-e', ssh, str(receipt), destination])
    for name in names:
        (root / name).unlink()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--freeze', type=Path, required=True)
    p.add_argument('--vanilla-template', type=Path, required=True)
    p.add_argument('--top1-template', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--file4-archive', required=True)
    p.add_argument('--gpu', type=int, default=1)
    args = p.parse_args()
    if (args.gpu not in (0, 1, 2, 3)
            or args.file4_archive != '/file4/youngwoo/deformation/experiments/retargeting-v1/scale-ungiven-val-20261028/pbr-val8'):
        raise ValueError('authorized GPU and fixed file4 VAL archive required')
    campaign = json.loads((args.campaign / 'campaign.json').read_text())
    freeze = json.loads(args.freeze.read_text())
    if (campaign['samples'] != 8 or campaign['source_clusters'] != 8
            or campaign['frozen_model_set'] != str(args.freeze)
            or freeze['status'] != 'FROZEN_FROM_DEVELOPMENT_ONLY'):
        raise ValueError('paired source-disjoint VAL8 frozen population required')
    ids = [case['sample_id'] for case in campaign['cases']]
    if len(set(ids)) != 8:
        raise ValueError('duplicate VAL case')
    templates = {'vanilla': json.loads(args.vanilla_template.read_text()),
                 'top1-only': json.loads(args.top1_template.read_text())}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), MATCHING_V2_ALLOWED_GPUS='0,1,2,3',
               ATTN_BACKEND='sdpa', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OPENCV_IO_ENABLE_OPENEXR='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    from .generated_r_mediated_pbr import image_only_row
    for index, case in enumerate(campaign['cases'], 1):
        sid = case['sample_id']
        shape_plan = json.loads(Path(case['plans']['image-estimated-v1']).read_text())
        shape_root = Path(shape_plan['output'])
        shape_report = json.loads((shape_root / 'generation.json').read_text())
        if (shape_report['sample_id'] != sid or shape_report['object_scale_policy'] != 'image-estimated-v1'
                or shape_report['GT_geometry_or_coords_used_in_generation']):
            raise ValueError('actual GT-free VAL geometry required')
        shape_prediction = shape_root / 'prediction.npz'
        shape_geometry = shape_root / 'geometry-evaluation.npz'
        if not shape_prediction.exists() or not shape_geometry.exists():
            source = (f'youngwoo@n16.unist.info:{args.file4_archive.rsplit("/pbr-val8", 1)[0]}/'
                      f'{args.campaign.name}/image-scale/{sid}/')
            run(['rsync', '-a', '--include=*.npz', '--exclude=*', '-e',
                 'ssh -o BatchMode=yes -p 55555', source, str(shape_root) + '/'])
        if not shape_prediction.is_file() or not shape_geometry.is_file():
            raise ValueError('frozen VAL geometry was not restored')
        original = Path(shape_plan['pilot_rebased_from'])
        render_template = json.loads(original.with_name(
            original.name.replace('-generation.json', '-render.json')).read_text())
        seed = int.from_bytes(hashlib.sha256(('VAL8-PBR-replay-v1:' + sid).encode()).digest()[:4], 'big')
        results = {}
        for arm, model_alias in (('vanilla', 'V1'), ('top1-only', 'RETARGET')):
            spec = freeze['models'][model_alias]['pbr1024']
            template = templates[arm]
            generation_root = args.output / 'replay' / arm / sid
            plan = dict(template, row=image_only_row(shape_plan['row']),
                        input_states=shape_plan['input_states'], RGB_paths=shape_plan['RGB_paths'],
                        generated_shape_root=str(shape_root), PBR_replay_seed=seed,
                        output=str(generation_root), diagnostic_scale_policy='image-estimated-v1')
            for old_test_binding in ('source_SHA256', 'input_SHA256', 'closed_inputs',
                                     'canonical_local_root', 'cold_reference'):
                plan.pop(old_test_binding, None)
            plan['validation_only_plan_without_old_TEST_closure'] = True
            if (plan['PBR_config'] != spec['native_config']['path']
                    or plan['PBR_checkpoint'] != spec['checkpoint']['path']
                    or plan.get('conditioner_checkpoint') !=
                    (spec['conditioner_checkpoint']['path'] if spec['conditioner_checkpoint'] else None)):
                raise ValueError('PBR replay template differs from frozen arm')
            plan_path = args.output / 'plans' / 'replay' / arm / (sid + '.json')
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            if not plan_path.exists():
                plan_path.write_text(json.dumps(plan, indent=2) + '\n')
            elif json.loads(plan_path.read_text()) != plan:
                raise ValueError('existing PBR plan changed')
            report_path = generation_root / 'generation.json'
            if not report_path.exists():
                if generation_root.exists():
                    raise ValueError('partial PBR generation requires review: ' + str(generation_root))
                command = ("import json,sys;from pathlib import Path;"
                           "from experiments.deformnext_retargeting.replay_r_pbr_on_generated_shape "
                           "import run;run(json.loads(Path(sys.argv[1]).read_text()))")
                run(['/opt/conda/envs/pixal3d/bin/python', '-u', '-c', command, str(plan_path)],
                    log=args.output / 'logs' / 'replay' / arm / (sid + '.log'), env=env)
            report = json.loads(report_path.read_text())
            if (report['status'] != 'PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE'
                    or report['PBR_arm'] != arm or report['PBR_replay_seed'] != seed
                    or not all(report['geometry_array_byte_identity'].values())):
                raise ValueError('paired PBR replay contract differs')
            render_root = args.output / 'render' / arm / sid
            render_plan = dict(render_template, generation_root=str(generation_root),
                               output=str(render_root), row=shape_plan['row'],
                               prediction_render_face_budget=1_000_000,
                               diagnostic_scale_policy='image-estimated-v1', PBR_arm=arm)
            render_path = args.output / 'plans' / 'render' / arm / (sid + '.json')
            render_path.parent.mkdir(parents=True, exist_ok=True)
            if not render_path.exists():
                render_path.write_text(json.dumps(render_plan, indent=2) + '\n')
            elif json.loads(render_path.read_text()) != render_plan:
                raise ValueError('existing native renderer plan changed')
            if not (render_root / 'render-evaluation.json').is_file():
                if render_root.exists():
                    raise ValueError('partial PBR render requires review: ' + str(render_root))
                render_root.parent.mkdir(parents=True, exist_ok=True)
                run(['/opt/conda/envs/pixal3d/bin/python', '-u', '-m',
                     'experiments.deformnext_retargeting.render_cascade_artifact_v2',
                     '--plan', str(render_path)],
                    log=args.output / 'logs' / 'render' / arm / (sid + '.log'), env=env)
            rendered = json.loads((render_root / 'render-evaluation.json').read_text())
            if rendered['status'] != 'PASS_ACTUAL_CASCADE_MULTIVIEW_RGB_ALPHA_AND_CAMERA_AUDIT':
                raise ValueError('native VAL PBR render failed')
            results[arm] = (report, rendered)
            remote = args.file4_archive + '/' + arm + '/' + sid
            if (generation_root / 'prediction.npz').exists():
                transfer_and_prune(generation_root, remote + '/replay',
                                   ('prediction.npz', 'geometry-evaluation.npz'))
            if (render_root / 'camera08.npz').exists():
                transfer_and_prune(render_root, remote + '/render',
                                   ('camera08.npz', 'camera09.npz', 'camera10.npz',
                                    'render-geometry.npz'))
        first, second = results['vanilla'], results['top1-only']
        if (first[0]['actual_initial_texture_noise_SHA256'] != second[0]['actual_initial_texture_noise_SHA256']
                or first[0]['original_prediction_SHA256'] != second[0]['original_prediction_SHA256']):
            raise ValueError('PBR arms do not share actual shape and first sampler noise')
        def metrics(pair):
            views = pair[1]['per_view']
            return {'target8_M_IoU': views[0]['M_IoU'], 'target8_PSNR': views[0]['PSNR'],
                    'target8_SSIM': views[0]['SSIM'],
                    'source9_10_LPIPS': sum(v['LPIPS'] for v in views[1:]) / 2}
        record = {'sample_id': sid, 'source_cluster_id': case['source_cluster_id'],
                  'seed': seed, 'vanilla': metrics(first), 'top1_only': metrics(second),
                  'fixed_geometry_sha256': first[0]['original_prediction_SHA256'],
                  'paired_first_noise_sha256': first[0]['actual_initial_texture_noise_SHA256']}
        (args.output / 'per-case').mkdir(parents=True, exist_ok=True)
        (args.output / 'per-case' / (sid + '.json')).write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({'index': index, 'samples': 8, **record}), flush=True)
        shape_prediction.unlink()
        shape_geometry.unlink()
    print(json.dumps({'status': 'PASS_PAIRED_VAL8_PBR_REPLAY', 'cases': len(ids)}), flush=True)


if __name__ == '__main__':
    main()
