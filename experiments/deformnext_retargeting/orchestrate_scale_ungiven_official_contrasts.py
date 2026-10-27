"""Compute same-population official RGB1 contrasts after the existing paired gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


REMOTE_NINE = ('/home/youngwoo/retargeting-unified-test508/'
               'unified-nine-method-final-evaluation-v4/evaluation')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-complete', type=Path, required=True)
    parser.add_argument('--paired-pid', type=int, required=True)
    parser.add_argument('--paired-root', type=Path, required=True)
    parser.add_argument('--table-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--paper-tex', type=Path, required=True)
    parser.add_argument('--remote-host', default='youngwoo@n16.unist.info')
    parser.add_argument('--ssh-port', type=int, default=55555)
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    if args.output_root.exists() or args.paper_tex.exists() or args.poll_seconds < 10:
        raise ValueError('fresh official contrast output and bounded poll required')
    args.output_root.mkdir(parents=True)
    while not args.paired_complete.is_file():
        live = Path(f'/proc/{args.paired_pid}').exists()
        print(json.dumps({'phase': 'WAIT_OURS_COMMON_PAIRED_GATE',
                          'paired_watcher_live': live}), flush=True)
        if not live:
            raise RuntimeError('Ours paired watcher ended without completion receipt')
        time.sleep(args.poll_seconds)
    paired = json.loads(args.paired_complete.read_text())
    if paired['status'] != 'PASS_COMMON_SUCCESS_OURS_PAIRED_EIGHT_ENDPOINTS':
        raise ValueError('V1 versus Selected common-population paired gate differs')
    official_dir = args.output_root / 'evaluation-nine'
    official_dir.mkdir()
    subprocess.run(['rsync', '-a', '-e',
                    f'ssh -o BatchMode=yes -p {args.ssh_port}',
                    f'{args.remote_host}:{REMOTE_NINE}/*-results.json',
                    str(official_dir) + '/'], check=True)
    if len(list(official_dir.glob('*-results.json'))) != 9:
        raise ValueError('all nine reused common evaluator results required')
    ours_dir = args.paired_root / 'evaluation-ours-four'
    if len(list(ours_dir.glob('*-results.json'))) != 4:
        raise ValueError('all four Ours common evaluator results required')
    output = args.output_root / 'official-contrasts-sixteen-endpoints.json'
    subprocess.run([sys.executable,
                    str(Path(__file__).with_name('score_scale_ungiven_official_contrasts.py')),
                    '--official-evaluation', str(official_dir),
                    '--ours-evaluation', str(ours_dir),
                    '--official-target8', str(args.table_root / 'official-target8.json'),
                    '--ours-target8', str(args.table_root / 'ours-target8.json'),
                    '--common-manifest', str(args.table_root / 'common-manifest.json'),
                    '--output', str(output)], check=True)
    result = json.loads(output.read_text())
    if (result['status']
            != 'PASS_COMMON_TEST_OURS_VERSUS_OFFICIAL_PAIRED_SIXTEEN_ENDPOINTS'
            or result['samples'] != paired['samples']
            or len(result['contrasts']) != 2):
        raise ValueError('official contrasts and frozen paired population differ')
    table = json.loads((args.table_root / 'scale-ungiven-main-table.json').read_text())
    if (table['status'] != 'PASS_SCALE_UNGIVEN_COMMON_SUCCESS_MAIN_TABLE'
            or table['samples'] != result['samples']):
        raise ValueError('official contrasts and ten-metric table population differ')
    for method, effects in result['contrasts'].items():
        for metric, row in effects.items():
            if (abs(row['official_mean']
                    - table['values']['pixal3d_official_rgb1_source_m0'][metric]) > 1e-6
                    or abs(row['ours_mean'] - table['values'][method][metric]) > 1e-6):
                raise ValueError('paired means and published table differ: '
                                 + method + '/' + metric)
    subprocess.run([sys.executable,
                    str(Path(__file__).with_name('compose_scale_ungiven_official_contrasts_tex.py')),
                    '--contrasts', str(output), '--output', str(args.paper_tex)], check=True)
    if not args.paper_tex.is_file() or args.paper_tex.stat().st_size == 0:
        raise ValueError('official paired contrast table missing')
    complete = {'status': 'PASS_COMMON_TEST_OFFICIAL_CONTRASTS_ARCHIVED',
                'samples': result['samples'],
                'source_clusters': result['source_clusters'],
                'output': str(output), 'paper_tex': str(args.paper_tex)}
    (args.output_root / 'complete.json').write_text(json.dumps(complete, indent=2) + '\n')
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()
