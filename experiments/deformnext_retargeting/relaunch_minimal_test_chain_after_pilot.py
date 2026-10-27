"""Reconnect the frozen TEST cascade after the render-bake pilot recovery."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/root/dev/.cache/deformnext-retargeting/scale-ungiven-test508-20261028')
BENCH = Path('/root/dev/.cache/deformnext-benchmark-runtime')
PILOT_ID = 'shard001_47685d2bec8d49b2a61c18b6d735642a_d2_fast300fpssupplyv1_PA_target08'


def start(name: str, arguments: list[str], log: Path, *, env: dict | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        process = subprocess.Popen([sys.executable, '-u', '-m',
                                    f'experiments.deformnext_retargeting.{name}',
                                    *arguments], cwd=REPO, env=env,
                                   stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'phase': 'LAUNCHED', 'program': name, 'pid': process.pid,
                      'log': str(log)}), flush=True)
    return process.pid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-pid', type=int, required=True)
    parser.add_argument('--pilot-log', type=Path, required=True)
    parser.add_argument('--official-render-pid', type=int, required=True)
    parser.add_argument('--nine-evaluator-pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() or args.poll_seconds < 10:
        raise ValueError('fresh recovery launch receipt required')
    cases = ROOT / 'cascade-two-arm-plans-v1' / 'cases'
    required = [cases / 'V1' / PILOT_ID / 'shape' / 'archive-receipt.json',
                cases / 'SELECTED' / PILOT_ID / 'shape' / 'archive-receipt.json',
                cases / 'SELECTED' / PILOT_ID / 'top1-pbr' / 'archive-receipt.json']
    while True:
        text = args.pilot_log.read_text() if args.pilot_log.is_file() else ''
        done = all(path.is_file() for path in required)
        print(json.dumps({'phase': 'WAIT_PAIRED_PILOT_ARCHIVE',
                          'receipts': [path.is_file() for path in required],
                          'pilot_live': Path(f'/proc/{args.pilot_pid}').exists()}), flush=True)
        if done and 'PASS_TEST_CASE_TWO_ARMS_ARCHIVED' in text:
            break
        if not Path(f'/proc/{args.pilot_pid}').exists():
            raise RuntimeError('paired pilot ended without three archive receipts')
        time.sleep(args.poll_seconds)
    for path in required:
        row = json.loads(path.read_text())
        if row['status'] != 'PASS_FILE4_ARCHIVE_CHECKSUM_VERIFIED':
            raise ValueError('paired pilot archive receipt differs: ' + str(path))
    socket = '/tmp/file4-ctl-v2'
    if not Path(socket).is_socket():
        raise FileNotFoundError('authenticated direct file4 control socket required')
    env = dict(os.environ, PIXAL3D_FILE4_CONTROL_SOCKET=socket)
    cascade = ROOT / 'cascade-two-arm-execution-v2-direct-file4'
    surface = ROOT / 'ours-surface-common502-v2'
    render = BENCH / 'ours-two-render-campaign-v2'
    evaluation = BENCH / 'ours-four-final-evaluation-monitor-v4'
    table = BENCH / 'scale-ungiven-main-table-v4'
    paired = BENCH / 'scale-ungiven-ours-paired-v2'
    for root in (cascade, surface, render, evaluation, table, paired):
        if root.exists():
            raise FileExistsError('recovery output already exists: ' + str(root))
    ids: dict[str, int] = {}
    ids['cascade'] = start('orchestrate_minimal_test_cascade', [
        '--campaign', str(ROOT / 'cascade-two-arm-plans-v1/campaign.json'),
        '--ss-complete', str(ROOT / 'ss-two-arms-v1/complete.json'),
        '--ss-watcher-pid', '539948', '--runtime-python', '/opt/conda/envs/pixal3d/bin/python',
        '--gpus', '0,1,2,3', '--remote-host', 'rvi@10.0.12.170',
        '--remote-base', '/file4/youngwoo/deformation/experiments/retargeting-v1/'
                         'scale-ungiven-test508-20261028/cascade-v1',
        '--output-root', str(cascade)],
        ROOT / 'cascade-two-arm-execution-v2-direct-file4.log', env=env)
    ids['surface'] = start('orchestrate_minimal_test_ours_surface', [
        '--cascade-complete', str(cascade / 'complete.json'),
        '--cascade-watcher-pid', str(ids['cascade']), '--output-root', str(surface)],
        ROOT / 'ours-surface-common502-v2.log')
    ids['render'] = start('orchestrate_minimal_test_ours_render', [
        '--surface-complete', str(surface / 'complete.json'),
        '--surface-watcher-pid', str(ids['surface']),
        '--official-render-complete', str(BENCH / 'official-rgb1-render-campaign-v2/complete.json'),
        '--official-render-watcher-pid', str(args.official_render_pid),
        '--output-root', str(render), '--benchmark-root', str(BENCH),
        '--protocol', str(BENCH / 'protocol.json'),
        '--blender', str(BENCH / 'blender-4.2.3-linux-x64/blender'),
        '--gpus', '0,1,2,3,6,7'],
        BENCH / 'ours-two-render-campaign-v2.log')
    ids['evaluation'] = start('orchestrate_minimal_test_ours_evaluation', [
        '--ours-render-complete', str(render / 'complete.json'),
        '--ours-render-watcher-pid', str(ids['render']),
        '--nine-evaluator-pid', str(args.nine_evaluator_pid),
        '--output-root', str(evaluation)],
        BENCH / 'ours-four-final-evaluation-monitor-v4.log')
    ids['table'] = start('orchestrate_scale_ungiven_main_table', [
        '--ours-evaluation-complete', str(evaluation / 'complete.json'),
        '--ours-evaluation-pid', str(ids['evaluation']),
        '--output-root', str(table),
        '--paper-tex', str(REPO / 'docs/superpowers/paper/final-results/scale-ungiven-main-table.tex')],
        BENCH / 'scale-ungiven-main-table-v4.log')
    ids['paired'] = start('orchestrate_scale_ungiven_ours_paired', [
        '--table-complete', str(table / 'complete.json'),
        '--table-pid', str(ids['table']), '--table-root', str(table),
        '--output-root', str(paired)],
        BENCH / 'scale-ungiven-ours-paired-v2.log')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'status': 'PASS_RECOVERED_FROZEN_TEST_CHAIN_LAUNCHED',
                                       'pilot_id': PILOT_ID, 'pids': ids,
                                       'direct_file4': True}, indent=2) + '\n')


if __name__ == '__main__':
    main()
