"""Close the paired V1/Selected uncertainty analysis after the common table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


REMOTE_EVAL = ('/home/youngwoo/retargeting-unified-test508/'
               'ours-four-final-evaluation-v1/evaluation-ours-four')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table-complete', type=Path, required=True)
    parser.add_argument('--table-pid', type=int, required=True)
    parser.add_argument('--table-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--remote-host', default='youngwoo@n16.unist.info')
    parser.add_argument('--ssh-port', type=int, default=55555)
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    if args.output_root.exists() or args.poll_seconds < 10:
        raise ValueError('fresh paired analysis output required')
    args.output_root.mkdir(parents=True)
    while not args.table_complete.is_file():
        live = Path(f'/proc/{args.table_pid}').exists()
        print(json.dumps({'phase': 'WAIT_COMMON_SUCCESS_TABLE',
                          'table_watcher_live': live}), flush=True)
        if not live:
            raise RuntimeError('common-success table watcher ended without completion')
        time.sleep(args.poll_seconds)
    table = json.loads(args.table_complete.read_text())
    if (table['status'] != 'PASS_COMMON_SUCCESS_TABLE_TEN_METRICS'
            or table['methods'] != 10 or table['metrics'] != 10):
        raise ValueError('completed ten-row ten-metric +M0 table required')
    local_eval = args.output_root / 'evaluation-ours-four'
    local_eval.mkdir()
    subprocess.run(['rsync', '-a', '-e',
                    f'ssh -o BatchMode=yes -p {args.ssh_port}',
                    f'{args.remote_host}:{REMOTE_EVAL}/*-results.json',
                    str(local_eval) + '/'], check=True)
    if len(list(local_eval.glob('*-results.json'))) != 4:
        raise ValueError('all four Ours evaluator results required')
    output = args.output_root / 'paired-eight-endpoints.json'
    subprocess.run([sys.executable,
                    str(Path(__file__).with_name('score_scale_ungiven_ours_paired.py')),
                    '--ours-evaluation', str(local_eval),
                    '--ours-target8', str(args.table_root / 'ours-target8.json'),
                    '--common-manifest', str(args.table_root / 'common-manifest.json'),
                    '--output', str(output)], check=True)
    result = json.loads(output.read_text())
    if (result['status'] != 'PASS_GT_SCALE_FREE_OURS_V1_SELECTED_COMMON_PAIRED_EIGHT_ENDPOINTS'
            or result['samples'] != table['samples']
            or len(result['metrics']) != 8):
        raise ValueError('paired analysis population or endpoints differ')
    completion = {'status': 'PASS_COMMON_SUCCESS_OURS_PAIRED_EIGHT_ENDPOINTS',
                  'samples': result['samples'], 'source_clusters': result['source_clusters'],
                  'output': str(output)}
    (args.output_root / 'complete.json').write_text(json.dumps(completion, indent=2) + '\n')
    print(json.dumps(completion), flush=True)


if __name__ == '__main__':
    main()
