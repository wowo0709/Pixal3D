"""Materialize the common-success Table 1 as soon as frozen evaluation closes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


REMOTE = '/home/youngwoo/retargeting-unified-test508'
NINE = REMOTE + '/unified-nine-method-final-evaluation-v4'
OURS = REMOTE + '/ours-four-final-evaluation-v1'
EXPECTED = {
    'nine-summary.json': NINE + '/evaluation/summary.json',
    'ours-summary.json': OURS + '/evaluation-ours-four/summary.json',
    'common-manifest.json': OURS + '/binding-common-thirteen/manifest.json',
    'external-target8.json': NINE + '/target8-external-paired.json',
    'official-target8.json': NINE + '/target8-official-paired.json',
    'ours-target8.json': OURS + '/target8-ours-four.json',
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ours-evaluation-complete', type=Path, required=True)
    parser.add_argument('--ours-evaluation-pid', type=int, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--paper-tex', type=Path, required=True)
    parser.add_argument('--remote-host', default='youngwoo@n16.unist.info')
    parser.add_argument('--ssh-port', type=int, default=55555)
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    if (args.output_root.exists() or args.paper_tex.exists()
            or args.poll_seconds < 10):
        raise ValueError('fresh frozen table output required')
    args.output_root.mkdir(parents=True)
    while not args.ours_evaluation_complete.is_file():
        live = Path(f'/proc/{args.ours_evaluation_pid}').exists()
        print(json.dumps({'phase': 'WAIT_OURS_FOUR_EVALUATION',
                          'watcher_live': live}), flush=True)
        if not live:
            raise RuntimeError('Ours four-condition evaluator ended without completion receipt')
        time.sleep(args.poll_seconds)
    done = json.loads(args.ours_evaluation_complete.read_text())
    if done['status'] != 'PASS_FOUR_NEW_OURS_METHODS_TEN_METRICS_REUSING_NINE':
        raise ValueError('frozen Ours four-condition evaluation did not pass')
    for name, path in EXPECTED.items():
        subprocess.run(['rsync', '-a', '-e',
                        f'ssh -o BatchMode=yes -p {args.ssh_port}',
                        f'{args.remote_host}:{path}', str(args.output_root / name)], check=True)
    script = Path(__file__).with_name('compose_scale_ungiven_main_table.py')
    table_json = args.output_root / 'scale-ungiven-main-table.json'
    subprocess.run([sys.executable, str(script),
                    '--nine-summary', str(args.output_root / 'nine-summary.json'),
                    '--ours-summary', str(args.output_root / 'ours-summary.json'),
                    '--common-manifest', str(args.output_root / 'common-manifest.json'),
                    '--external-target8', str(args.output_root / 'external-target8.json'),
                    '--official-target8', str(args.output_root / 'official-target8.json'),
                    '--ours-target8', str(args.output_root / 'ours-target8.json'),
                    '--output-tex', str(args.paper_tex),
                    '--output-json', str(table_json)], check=True)
    report = json.loads(table_json.read_text())
    if report['samples'] != done['samples']:
        raise ValueError('table and four-condition evaluation sample counts differ')
    registration_tex = Path(report['target8_registration_tex'])
    if not registration_tex.is_file() or registration_tex.stat().st_size == 0:
        raise ValueError('same-N raw/+M0 target8 registration table missing')
    completion = {'status': 'PASS_COMMON_SUCCESS_TABLE_TEN_METRICS',
                  'samples': report['samples'], 'methods': len(report['methods']),
                  'metrics': len(report['metrics']), 'tex': str(args.paper_tex),
                  'target8_registration_tex': str(registration_tex),
                  'json': str(table_json)}
    (args.output_root / 'complete.json').write_text(json.dumps(completion, indent=2) + '\n')
    print(json.dumps(completion), flush=True)


if __name__ == '__main__':
    main()
