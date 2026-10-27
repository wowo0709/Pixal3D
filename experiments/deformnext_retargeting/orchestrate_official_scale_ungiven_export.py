"""Export official no-scale textures for the frozen external common-success intersection."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation-pid', type=int, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--common-subset', type=Path, required=True)
    parser.add_argument('--pixal3d-repo', type=Path, required=True)
    parser.add_argument('--benchmark-repo', type=Path, required=True)
    parser.add_argument('--pixal3d-python', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--physical-gpu', type=int, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--render-face-budget', type=int,
                        choices=(250_000, 500_000, 1_000_000, 2_000_000,
                                 5_000_000), default=1_000_000)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError('1–6 workers required')
    if args.physical_gpu != 7:
        raise ValueError('node16 GPU7 is the authorized export device')
    frozen_ids = json.loads(args.common_subset.read_text())['sample_ids']
    if len(frozen_ids) != 502 or len(set(frozen_ids)) != 502:
        raise ValueError('frozen common502 subset required')
    contract = {'status': 'PREPARED_OFFICIAL_SCALE_UNGIVEN_EXPORT',
                'official_root': str(args.official_root), 'manifest': str(args.manifest),
                'common_subset_sha256': sha(args.common_subset),
                'pixal3d_repo': str(args.pixal3d_repo),
                'benchmark_repo': str(args.benchmark_repo),
                'physical_gpu': args.physical_gpu,
                'preferred_render_face_budget': args.render_face_budget,
                'allowed_geometry_fallback_budgets': [budget for budget in
                    (250_000, 500_000, 1_000_000, 2_000_000, 5_000_000)
                    if budget >= args.render_face_budget]}
    contract_path = args.output_root / 'campaign.json'
    if args.output_root.exists():
        if not contract_path.is_file() or json.loads(contract_path.read_text()) != contract:
            raise ValueError('existing export campaign contract differs')
    else:
        args.output_root.mkdir(parents=True)
        contract_path.write_text(json.dumps(contract, indent=2) + '\n')
    while True:
        attempts_path = args.official_root / 'attempts-complete.json'
        completed = len(list((args.official_root / 'cases').glob('*/generation.json')))
        failures = len(list((args.official_root / 'cases').glob('*/failure-receipt.json')))
        print(json.dumps({'phase': 'WAIT_OFFICIAL_ATTEMPT_CLOSURE', 'successes': completed,
                          'model_failures': failures, 'expected_attempts': 508,
                          'pid_live': process_exists(args.generation_pid)}), flush=True)
        if attempts_path.is_file():
            break
        if completed + failures > 508 or not process_exists(args.generation_pid):
            raise RuntimeError('official generation ended before TEST508 attempt closure')
        time.sleep(60)
    attempts = json.loads(attempts_path.read_text())
    successes = set(attempts['success_sample_ids'])
    failed = set(attempts['failure_sample_ids'])
    if (attempts['status'] != 'PASS_OFFICIAL_RGB1_TEST508_ATTEMPTS_CLOSED'
            or attempts['attempted'] != 508 or len(successes) + len(failed) != 508
            or successes & failed or attempts['succeeded'] != len(successes)
            or attempts['model_failed_empty_sparse'] != len(failed)):
        raise ValueError('official TEST508 attempt closure differs')
    ids = [sid for sid in frozen_ids if sid in successes]
    excluded = [sid for sid in frozen_ids if sid in failed]
    if len(ids) + len(excluded) != 502 or not ids:
        raise ValueError('official success × frozen external common502 differs')
    (args.output_root / 'official-success-common.json').write_text(json.dumps({
        'status': 'PASS_OFFICIAL_SUCCESS_INTERSECTION', 'sample_ids': ids,
        'excluded_model_failures': excluded, 'frozen_external_common': 502}, indent=2) + '\n')
    if any(not (args.official_root / 'cases' / sid / 'generation.json').is_file() for sid in ids):
        raise ValueError('common502 not covered by official generation')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.physical_gpu), OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    work = Path(__file__).resolve().parent

    def export(sid: str) -> dict:
        output_dir = args.output_root / 'cases' / sid
        output = output_dir / 'prediction.glb'
        receipt = output_dir / 'prediction.json'
        if output.is_file() and receipt.is_file():
            row = json.loads(receipt.read_text())
            if (row['status'] != 'PASS_OFFICIAL_RGB1_SCALE_UNGIVEN_RENDER_EXPORT'
                    or row['sample_id'] != sid or row['glb_sha256'] != sha(output)
                    or not row.get('pre_simplify')
                    or row.get('render_face_budget') not in contract['allowed_geometry_fallback_budgets']):
                raise ValueError('existing export receipt differs: ' + sid)
            return {'sample_id': sid, 'status': 'REUSED', 'seconds': 0.0}
        if output_dir.exists():
            contents = {path.name for path in output_dir.iterdir()}
            declared_logs = {f'export-{budget}.log' for budget in
                             contract['allowed_geometry_fallback_budgets']}
            if not contents <= declared_logs:
                raise FileExistsError('partial official export needs review: ' + sid)
        else:
            output_dir.mkdir(parents=True)
        source = args.official_root / 'cases' / sid
        started = time.monotonic()
        for budget in contract['allowed_geometry_fallback_budgets']:
            command = [str(args.pixal3d_python), '-u', str(work / 'export_official_pixal3d_scale_ungiven.py'),
                       '--pixal3d-repo', str(args.pixal3d_repo), '--benchmark-repo', str(args.benchmark_repo),
                       '--prediction', str(source / 'prediction.npz'),
                       '--generation', str(source / 'generation.json'),
                       '--manifest', str(args.manifest), '--output', str(output), '--pre-simplify',
                       '--render-face-budget', str(budget)]
            log_path = output_dir / f'export-{budget}.log'
            old_failure = log_path.read_text(errors='replace') if log_path.is_file() else ''
            retryable_old = ('GLB/untouched metric mesh differ in world space' in old_failure
                             or 'fast simplifier produced' in old_failure
                             or ('RuntimeError: [CuMesh] CUDA error' in old_failure
                                 and 'Error code: 9' in old_failure
                                 and 'invalid configuration argument' in old_failure))
            if old_failure and retryable_old:
                print(json.dumps({'phase': 'REUSE_FAILED_OFFICIAL_EXPORT_BUDGET',
                                  'sample_id': sid, 'budget': budget}), flush=True)
                continue
            try:
                with log_path.open('w') as log:
                    subprocess.run(command, check=True, env=env, stdout=log,
                                   stderr=subprocess.STDOUT,
                                   timeout=3600 if budget >= 2_000_000 else 1200)
                break
            except subprocess.CalledProcessError:
                failure = log_path.read_text(errors='replace')
                geometry_failure = ('GLB/untouched metric mesh differ in world space' in failure
                                    or 'fast simplifier produced' in failure
                                    or ('RuntimeError: [CuMesh] CUDA error' in failure
                                        and 'Error code: 9' in failure
                                        and 'invalid configuration argument' in failure))
                if (budget == contract['allowed_geometry_fallback_budgets'][-1]
                        or output.exists() or receipt.exists() or not geometry_failure):
                    raise
        row = json.loads(receipt.read_text())
        if (row['status'] != 'PASS_OFFICIAL_RGB1_SCALE_UNGIVEN_RENDER_EXPORT'
                or row['sample_id'] != sid or row['glb_sha256'] != sha(output)
                or row.get('render_face_budget') not in contract['allowed_geometry_fallback_budgets']):
            raise ValueError('new export receipt failed: ' + sid)
        return {'sample_id': sid, 'status': 'PASS', 'seconds': round(time.monotonic()-started, 2),
                'geometry_q99': row['render_geometry_registration_q99_world']}

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(export, sid): sid for sid in ids}
        for count, future in enumerate(as_completed(pending), 1):
            sid = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {'sample_id': sid, 'status': 'FAIL', 'error': repr(exc)}
                failures.append(result)
            print(json.dumps({'index': count, 'total': len(ids), **result}), flush=True)
    if failures:
        (args.output_root / 'failures.json').write_text(json.dumps(failures, indent=2) + '\n')
        raise RuntimeError(f'{len(failures)} official common502 GLB exports failed')
    (args.output_root / 'failures.json').unlink(missing_ok=True)
    result = {'status': 'PASS_OFFICIAL_RGB1_SCALE_UNGIVEN_COMMON_SUCCESS_EXPORT',
              'samples': len(ids), 'frozen_external_common': 502,
              'excluded_model_failures': excluded,
              'gpu': args.physical_gpu, 'workers': args.workers,
              'preferred_render_face_budget': args.render_face_budget,
              'allowed_geometry_fallback_budgets': contract['allowed_geometry_fallback_budgets'],
              'actual_budget_counts': {str(budget): sum(
                  json.loads(path.read_text())['render_face_budget'] == budget
                  for path in (args.output_root / 'cases').glob('*/prediction.json'))
                  for budget in contract['allowed_geometry_fallback_budgets']},
              'raw_surface_predictions_unchanged': True}
    (args.output_root / 'complete.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
