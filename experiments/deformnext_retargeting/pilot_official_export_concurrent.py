"""Probe one frozen official render bake while the external metric worker runs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--common-subset', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--pixal3d-repo', type=Path, required=True)
    parser.add_argument('--benchmark-repo', type=Path, required=True)
    parser.add_argument('--pixal3d-python', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    attempts = json.loads((args.official_root / 'attempts-complete.json').read_text())
    if attempts['status'] != 'PASS_OFFICIAL_RGB1_TEST508_ATTEMPTS_CLOSED':
        raise ValueError('closed official TEST508 generation required')
    common = json.loads(args.common_subset.read_text())
    successful = set(attempts['success_sample_ids'])
    ids = [sid for sid in common['sample_ids'] if sid in successful]
    if not ids:
        raise ValueError('no official and external common success')
    sid = ids[0]
    source = args.official_root / 'cases' / sid
    if not (source / 'generation.json').is_file() or not (source / 'prediction.npz').is_file():
        raise FileNotFoundError(source)
    args.output_root.mkdir(parents=True)
    output = args.output_root / 'prediction.glb'
    command = [str(args.pixal3d_python), '-u',
               str(Path(__file__).with_name('export_official_pixal3d_scale_ungiven.py')),
               '--pixal3d-repo', str(args.pixal3d_repo),
               '--benchmark-repo', str(args.benchmark_repo),
               '--prediction', str(source / 'prediction.npz'),
               '--generation', str(source / 'generation.json'),
               '--manifest', str(args.manifest), '--output', str(output),
               '--pre-simplify', '--render-face-budget', '250000']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='7', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1')
    print(json.dumps({'phase': 'OFFICIAL_EXPORT_CONCURRENT_PILOT',
                      'sample_id': sid, 'command': command}), flush=True)
    with (args.output_root / 'export.log').open('w') as log:
        subprocess.run(command, check=True, env=env, stdout=log,
                       stderr=subprocess.STDOUT, timeout=1200)
    receipt = json.loads(output.with_suffix('.json').read_text())
    if (receipt['status'] != 'PASS_OFFICIAL_RGB1_SCALE_UNGIVEN_RENDER_EXPORT'
            or receipt['sample_id'] != sid
            or receipt['render_geometry_registration_q99_world'] > 0.01):
        raise ValueError('official pilot render fidelity failed')
    print(json.dumps({'status': 'PASS_OFFICIAL_CONCURRENT_RENDER_BAKE_PILOT',
                      'sample_id': sid, 'geometry_q99':
                      receipt['render_geometry_registration_q99_world'],
                      'render_face_budget': receipt['render_face_budget']}), flush=True)


if __name__ == '__main__':
    main()
