"""Wait for frozen CPU M0 fits, audit them, then run the paired render campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


EXPECTED = (126 * 7, 126 * 7, 125 * 7, 125 * 7)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-host', required=True)
    parser.add_argument('--remote-score-root', required=True)
    parser.add_argument('--file4-output-root', required=True)
    parser.add_argument('--local-root', type=Path, required=True)
    parser.add_argument('--target8-jobs', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--frozen-results-root', type=Path, required=True)
    parser.add_argument('--blender', type=Path, required=True)
    parser.add_argument('--benchmark-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--gpus', required=True)
    parser.add_argument('--ssh-port', type=int, default=55555)
    parser.add_argument('--max-wait-hours', type=float, default=8.)
    args = parser.parse_args()
    if not args.file4_output_root.startswith('/file4/youngwoo/deformation/experiments/retargeting-v1/source-sim3-test508-20261028/'):
        raise ValueError('declared file4 source-Sim3 archive required')
    scripts = Path(__file__).parent
    score_dir = args.local_root / 'scores'
    score_dir.mkdir(parents=True, exist_ok=True)
    remote_paths = [args.remote_score_root.rstrip('/') + f'/source-sim3-shard{i}.jsonl' for i in range(4)]
    ssh = ['ssh', '-o', 'BatchMode=yes', '-p', str(args.ssh_port), args.remote_host]
    start = time.monotonic()
    while True:
        code = ("import json,pathlib\n"
                f"paths={remote_paths!r}\n"
                "print(json.dumps([sum(1 for _ in open(p)) if pathlib.Path(p).is_file() else 0 for p in paths]))\n")
        counts = json.loads(subprocess.check_output([*ssh, 'python3', '-'], input=code, text=True))
        if tuple(counts) == EXPECTED:
            break
        if time.monotonic() - start > args.max_wait_hours * 3600:
            raise TimeoutError('source-M0 score shards not complete: ' + str(counts))
        print(json.dumps({'phase': 'WAIT_SOURCE_SCORES', 'counts': counts, 'expected': EXPECTED}), flush=True)
        time.sleep(60)
    local_scores = []
    for index, remote_path in enumerate(remote_paths):
        local = score_dir / f'shard{index}.jsonl'
        run(['scp', '-q', '-P', str(args.ssh_port), args.remote_host + ':' + remote_path, str(local)])
        if sum(1 for _ in local.open()) != EXPECTED[index]:
            raise ValueError('copied source score shard count changed')
        local_scores.append(local)
    combined = score_dir / 'full502.jsonl'
    if not combined.exists():
        combined.write_text(''.join(path.read_text() for path in local_scores))
    if sum(1 for _ in combined.open()) != sum(EXPECTED):
        raise ValueError('full common502 score population incomplete')
    audit = args.local_root / 'raw-parity.json'
    summary = args.local_root / 'paired-surface-summary.json'
    plan = args.local_root / 'render-plan'
    if not audit.exists():
        run([sys.executable, str(scripts / 'audit_source_sim3_raw_parity.py'), '--scores',
             *map(str, local_scores), '--frozen-results-root', str(args.frozen_results_root),
             '--output', str(audit)])
    if not summary.exists():
        run([sys.executable, str(scripts / 'summarize_source_sim3_baselines.py'), '--input',
             str(combined), '--output', str(summary)])
    if not plan.exists():
        run([sys.executable, str(scripts / 'build_source_sim3_render_jobs.py'), '--scores',
             *map(str, local_scores), '--target8-jobs', str(args.target8_jobs),
             '--references', str(args.references), '--output-root', str(plan),
             '--chunk-samples', '16'])
    run([*ssh, 'mkdir', '-p', args.file4_output_root])
    rsync_ssh = 'ssh -p ' + str(args.ssh_port) + ' -o BatchMode=yes'
    run(['rsync', '-a', '-e', rsync_ssh, str(audit), str(summary),
         *map(str, local_scores), args.remote_host + ':' + args.file4_output_root.rstrip('/') + '/'])
    print(json.dumps({'phase': 'RENDER_CAMPAIGN_READY', 'chunks': len(list(plan.glob('chunk*')))}), flush=True)
    run([sys.executable, str(scripts / 'run_source_sim3_render_campaign.py'),
         '--plan-root', str(plan), '--remote-host', args.remote_host,
         '--remote-output-root', args.file4_output_root,
         '--blender', str(args.blender), '--benchmark-root', str(args.benchmark_root),
         '--protocol', str(args.protocol), '--gpus', args.gpus,
         '--ssh-port', str(args.ssh_port)])
    print(json.dumps({'status': 'PASS_M0_SIM3_SOURCE_AND_RENDER_FULL502',
                      'file4_root': args.file4_output_root}), flush=True)


if __name__ == '__main__':
    main()
