"""Generate and checksum-archive frozen V1/Selected TEST cases on one assigned GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path: Path) -> str:
    state = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            state.update(block)
    return state.hexdigest()


def run(argv: list[str], *, log: Path | None = None, env: dict | None = None,
        timeout: int | None = None) -> None:
    if log is None:
        subprocess.run(argv, check=True, env=env, timeout=timeout)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w') as stream:
            subprocess.run(argv, check=True, env=env, timeout=timeout,
                           stdout=stream, stderr=subprocess.STDOUT)


def archive_case(root: Path, remote_host: str, remote_base: str,
                 arm: str, sid: str, stage: str, ssh_port: int,
                 *, with_export: bool = False) -> None:
    report = root / 'generation.json'
    receipt_path = root / 'archive-receipt.json'
    if receipt_path.is_file():
        record = json.loads(receipt_path.read_text())
        if (record['status'] != 'PASS_FILE4_ARCHIVE_CHECKSUM_VERIFIED'
                or record['remote'] != f'{remote_base}/{arm}/{sid}/{stage}'):
            raise ValueError('existing archive receipt differs: ' + sid + '/' + stage)
        return
    if not report.is_file():
        raise FileNotFoundError(report)
    files = [root / name for name in ('prediction.npz', 'geometry-evaluation.npz', 'generation.json')]
    if with_export:
        files.extend((root / 'benchmark.glb', root / 'benchmark.json'))
    if any(not path.is_file() for path in files):
        raise FileNotFoundError('incomplete predicted case before archive: ' + str(root))
    remote_dir = f'{remote_base}/{arm}/{sid}/{stage}'
    socket = os.environ.get('PIXAL3D_FILE4_CONTROL_SOCKET')
    if socket:
        if remote_host != 'rvi@10.0.12.170' or not Path(socket).is_socket():
            raise ValueError('file4 control socket requires the direct file4 host')
        ssh = ['ssh', '-S', socket, '-o', 'BatchMode=yes', '-p', str(ssh_port)]
    else:
        ssh = ['ssh', '-o', 'BatchMode=yes', '-p', str(ssh_port)]
    rsync_ssh = ' '.join(ssh)
    run([*ssh, remote_host, 'mkdir', '-p', remote_dir])
    destination = f'{remote_host}:{remote_dir}/'
    run(['rsync', '-a', '-e', rsync_ssh, *map(str, files), destination])
    check = subprocess.run(['rsync', '-a', '--checksum', '--dry-run', '--itemize-changes',
                            '-e', rsync_ssh, *map(str, files), destination],
                           check=True, capture_output=True, text=True)
    if check.stdout.strip() or check.stderr.strip():
        raise ValueError('file4 archive checksum differs; preserve local case: ' + str(root))
    hashes = {path.name: sha(path) for path in files}
    record = {'status': 'PASS_FILE4_ARCHIVE_CHECKSUM_VERIFIED',
              'remote': remote_dir, 'sha256': hashes,
              'large_local_NPZ_reclaimed_after_archive': True}
    receipt_path.write_text(json.dumps(record, indent=2) + '\n')
    run(['rsync', '-a', '-e', rsync_ssh, str(receipt_path), destination])
    for path in files[:2]:
        path.unlink()
    if with_export:
        (root / 'benchmark.glb').unlink()
    print(json.dumps({'status': record['status'], 'arm': arm,
                      'sample_id': sid, 'stage': stage, 'remote': remote_dir}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--ss-complete', type=Path, required=True)
    parser.add_argument('--runtime-python', type=Path, required=True)
    parser.add_argument('--remote-host', required=True)
    parser.add_argument('--remote-base', required=True)
    parser.add_argument('--ssh-port', type=int, default=55555)
    parser.add_argument('--physical-gpu', required=True)
    parser.add_argument('--allowed-gpus', required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--skip-sample-id', action='append', default=[],
                        help='Documented recovery exception; the case must be repaired separately')
    args = parser.parse_args()
    if (args.world_size < 1 or not 0 <= args.rank < args.world_size
            or not args.physical_gpu.isdigit()
            or args.physical_gpu not in args.allowed_gpus.split(',')
            or not args.remote_base.startswith('/file4/youngwoo/deformation/experiments/retargeting-v1/')
            or not args.runtime_python.is_file()):
        raise ValueError('frozen campaign, explicit allowed GPU, and file4 archive required')
    ss = json.loads(args.ss_complete.read_text())
    campaign = json.loads(args.campaign.read_text())
    if (ss['status'] != 'PASS_FROZEN_TEST508_V1_SELECTED_SS_IMAGE_SCALE'
            or ss['samples'] != 508
            or campaign['status'] != 'PREPARED_FROZEN_V1_SELECTED_TEST508_IMAGE_SCALE'
            or campaign['samples'] != 508 or len(campaign['plans']) != 1016):
        raise ValueError('frozen SS/cascade campaign closure required')
    entries = {(row['sample_id'], row['arm']): row for row in campaign['plans']}
    ids = [row['sample_id'] for row in campaign['plans'] if row['arm'] == 'V1']
    if (len(ids) != 508 or len(set(ids)) != 508 or len(entries) != 1016
            or any((sid, 'SELECTED') not in entries for sid in ids)):
        raise ValueError('paired V1/Selected TEST508 plan identity differs')
    if (len(args.skip_sample_id) > 1 or len(set(args.skip_sample_id)) != len(args.skip_sample_id)
            or any(sid not in ids[args.rank::args.world_size] for sid in args.skip_sample_id)):
        raise ValueError('at most one own-rank recovery exception may be deferred')
    root = args.campaign.parent
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.physical_gpu,
               MATCHING_V2_ALLOWED_GPUS=args.allowed_gpus,
               BENCHMARK_ASSIGNED_GPU=args.physical_gpu,
               ATTN_BACKEND='sdpa', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OPENCV_IO_ENABLE_OPENEXR='1', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1')
    repo = Path(__file__).resolve().parents[2]
    benchmark_repo = Path('/root/dev/.cache/deformnext-benchmark-runtime')
    if not (benchmark_repo / 'deformnext_benchmark/io.py').is_file():
        raise FileNotFoundError('frozen common benchmark runtime required')

    def ensure_export(pred_root: Path, shape_root: Path, texture_arm: str, sid: str) -> None:
        glb = pred_root / 'benchmark.glb'
        report_path = pred_root / 'benchmark.json'
        if glb.is_file() and report_path.is_file():
            report = json.loads(report_path.read_text())
            if (report['status'] != 'PASS_OURS_FROZEN_GT_SCALE_FREE_1024_EXPORT'
                    or report['sample_id'] != sid or report['texture_arm'] != texture_arm
                    or report['prediction_sha256'] != sha(pred_root / 'prediction.npz')
                    or report['glb_sha256'] != sha(glb)):
                raise ValueError('existing common render export differs: ' + sid)
            return
        if glb.exists() or report_path.exists():
            raise FileExistsError('partial common render export requires inspection: ' + str(glb))
        for budget in (250_000, 500_000, 1_000_000, 2_000_000, 5_000_000):
            command = [str(args.runtime_python), '-u', '-m',
                       'experiments.deformnext_retargeting.export_minimal_test_mesh',
                       '--pixal3d-repo', str(repo), '--benchmark-repo', str(benchmark_repo),
                       '--prediction', str(pred_root / 'prediction.npz'),
                       '--shape-generation', str(shape_root / 'generation.json'),
                       '--anchor-evaluation', str(pred_root / 'geometry-evaluation.npz'),
                       '--texture-arm', texture_arm, '--render-face-budget', str(budget),
                       '--output', str(glb)]
            if texture_arm == 'top1-only':
                command.extend(('--texture-generation', str(pred_root / 'generation.json')))
            log = root / 'logs' / 'benchmark-export' / texture_arm / f'{sid}-{budget}.log'
            old_failure = log.read_text(errors='replace') if log.is_file() else ''
            if (old_failure and not glb.exists() and not report_path.exists()
                    and ('export/untouched metric mesh differ in world space' in old_failure
                         or 'fast simplifier produced' in old_failure
                         or 'fast simplifier did not reach declared render budget' in old_failure)):
                print(json.dumps({'phase': 'REUSE_FAILED_GEOMETRY_BUDGET_CHECK',
                                  'sample_id': sid, 'texture_arm': texture_arm,
                                  'failed_budget': budget}), flush=True)
                continue
            try:
                run(command, log=log, env=env,
                    timeout=3600 if budget >= 2_000_000 else 1200)
                break
            except subprocess.CalledProcessError:
                failure = log.read_text(errors='replace')
                geometry_failure = ('export/untouched metric mesh differ in world space' in failure
                                    or 'fast simplifier produced' in failure
                                    or 'fast simplifier did not reach declared render budget' in failure)
                if (budget == 5_000_000 or glb.exists() or report_path.exists()
                        or not geometry_failure):
                    raise
                print(json.dumps({'phase': 'RETRY_RENDER_BAKE_GEOMETRY_BUDGET',
                                  'sample_id': sid, 'texture_arm': texture_arm,
                                  'failed_budget': budget}), flush=True)
        report = json.loads(report_path.read_text())
        if (report['status'] != 'PASS_OURS_FROZEN_GT_SCALE_FREE_1024_EXPORT'
                or report['sample_id'] != sid
                or report['prediction_sha256'] != sha(pred_root / 'prediction.npz')
                or report['glb_sha256'] != sha(glb)):
            raise ValueError('common render export receipt differs: ' + sid)
    for index in range(args.rank, len(ids), args.world_size):
        sid = ids[index]
        if sid in args.skip_sample_id:
            print(json.dumps({'phase': 'DEFER_DECLARED_RENDER_BAKE_EXCEPTION',
                              'index': index, 'rank': args.rank, 'sample_id': sid}), flush=True)
            continue
        for arm in ('V1', 'SELECTED'):
            entry = entries[sid, arm]
            shape_plan = Path(entry['shape_plan'])
            shape = json.loads(shape_plan.read_text())
            shape_root = Path(shape['output'])
            shape_report = shape_root / 'generation.json'
            shape_archived = (shape_root / 'archive-receipt.json').is_file()
            pbr_archived = (Path(json.loads(Path(entry['top1_pbr_plan']).read_text())['output']) /
                            'archive-receipt.json').is_file() if arm == 'SELECTED' else True
            if shape_archived and pbr_archived:
                continue
            if shape_archived and not pbr_archived:
                raise ValueError('Shape archive cannot precede selected PBR archive: ' + sid)
            if not shape_report.is_file():
                if shape_root.exists():
                    raise FileExistsError('partial frozen Shape case requires inspection: ' + str(shape_root))
                shape_root.parent.mkdir(parents=True, exist_ok=True)
                run([str(args.runtime_python), '-u', '-m',
                     'experiments.deformnext_retargeting.cascade_from_ss_artifact',
                     '--plan', str(shape_plan)],
                    log=root / 'logs' / 'shape' / arm / (sid + '.log'), env=env)
            shape_result = json.loads(shape_report.read_text())
            if (shape_result['status'] != 'PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
                    or shape_result['sample_id'] != sid
                    or shape_result['object_scale_policy'] != 'image-estimated-v1'
                    or shape_result['M1_derived_scale_supplied_to_generation'] is not False):
                raise ValueError('frozen GT-scale-free Shape receipt differs: ' + sid)
            if arm == 'V1':
                ensure_export(shape_root, shape_root, 'vanilla', sid)
            if arm == 'SELECTED':
                pbr_plan = Path(entry['top1_pbr_plan'])
                pbr = json.loads(pbr_plan.read_text())
                pbr_root = Path(pbr['output'])
                pbr_report = pbr_root / 'generation.json'
                if not pbr_report.is_file():
                    if pbr_root.exists():
                        raise FileExistsError('partial frozen Top-1 PBR case requires inspection: ' + str(pbr_root))
                    command = ('import json,sys;from pathlib import Path;'
                               'from experiments.deformnext_retargeting.replay_r_pbr_on_generated_shape '
                               'import run;run(json.loads(Path(sys.argv[1]).read_text()))')
                    run([str(args.runtime_python), '-u', '-c', command, str(pbr_plan)],
                        log=root / 'logs' / 'top1-pbr' / (sid + '.log'), env=env)
                pbr_result = json.loads(pbr_report.read_text())
                if (pbr_result['status'] != 'PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE'
                        or pbr_result['sample_id'] != sid
                        or pbr_result['PBR_arm'] != 'top1-only'
                        or not all(pbr_result['geometry_array_byte_identity'].values())
                        or pbr_result['original_prediction_SHA256'] != sha(shape_root / 'prediction.npz')):
                    raise ValueError('frozen Top-1 PBR receipt/geometry differs: ' + sid)
                ensure_export(pbr_root, shape_root, 'top1-only', sid)
                archive_case(pbr_root, args.remote_host, args.remote_base,
                             arm, sid, 'top1-pbr', args.ssh_port, with_export=True)
            archive_case(shape_root, args.remote_host, args.remote_base,
                         arm, sid, 'shape', args.ssh_port,
                         with_export=arm == 'V1')
        print(json.dumps({'status': 'PASS_TEST_CASE_TWO_ARMS_ARCHIVED',
                          'index': index, 'rank': args.rank, 'sample_id': sid}), flush=True)
    print(json.dumps({'status': 'PASS_TEST_SHARD_TWO_ARMS_ARCHIVED',
                      'rank': args.rank, 'world_size': args.world_size,
                      'cases': len(ids[args.rank::args.world_size])}), flush=True)


if __name__ == '__main__':
    main()
