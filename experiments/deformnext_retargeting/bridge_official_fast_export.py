"""Replace a paused 1M render-bake campaign with verified 250k bake outputs.

The old watcher remains stopped so already launched render dependents keep a
live prerequisite PID. Its output path is atomically pointed at the completed
replacement only after all common-success cases pass.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def state(pid: int) -> str:
    path = Path(f'/proc/{pid}/stat')
    return path.read_text().split(') ', 1)[1][0] if path.is_file() else 'missing'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-export-pid', type=int, required=True)
    parser.add_argument('--old-gate-pid', type=int, required=True)
    parser.add_argument('--old-root', type=Path, required=True)
    parser.add_argument('--new-root', type=Path, required=True)
    parser.add_argument('--external-completion', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--generation-pid', type=int, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--common-subset', type=Path, required=True)
    parser.add_argument('--pixal3d-repo', type=Path, required=True)
    parser.add_argument('--benchmark-repo', type=Path, required=True)
    parser.add_argument('--pixal3d-python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adopt-existing', action='store_true',
                        help='Adopt a repaired, already completed fast export without rerunning it')
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    if (args.output.exists() or args.new_root.exists() != args.adopt_existing
            or args.old_root.is_symlink()
            or not args.old_root.is_dir() or args.poll_seconds < 10
            or state(args.old_export_pid) != 'T' or state(args.old_gate_pid) != 'T'):
        raise ValueError('fresh recovery and two deliberately stopped old jobs required')
    retired = args.old_root.with_name(args.old_root.name + '-paused-1m')
    if retired.exists() or retired.is_symlink():
        raise FileExistsError('paused 1M campaign archive already exists')
    try:
        if args.adopt_existing:
            if not (args.new_root / 'complete.json').is_file():
                raise FileNotFoundError('repaired official export completion required')
            print(json.dumps({'phase': 'ADOPT_REPAIRED_OFFICIAL_RENDER_BAKE',
                              'root': str(args.new_root)}), flush=True)
        else:
            while not (args.official_root / 'attempts-complete.json').is_file():
                print(json.dumps({'phase': 'WAIT_OFFICIAL_GENERATION_TO_OVERLAP_EXPORT',
                                  'external': args.external_completion.is_file(),
                                  'official': (args.official_root / 'attempts-complete.json').is_file()}),
                      flush=True)
                if state(args.old_export_pid) != 'T' or state(args.old_gate_pid) != 'T':
                    raise RuntimeError('paused GPU7 gate changed state')
                time.sleep(args.poll_seconds)
            command = [sys.executable, '-u', str(Path(__file__).with_name(
                'orchestrate_official_scale_ungiven_export.py')),
                '--generation-pid', str(args.generation_pid),
                '--official-root', str(args.official_root), '--manifest', str(args.manifest),
                '--common-subset', str(args.common_subset),
                '--pixal3d-repo', str(args.pixal3d_repo),
                '--benchmark-repo', str(args.benchmark_repo),
                '--pixal3d-python', str(args.pixal3d_python),
                '--output-root', str(args.new_root), '--physical-gpu', '7',
                '--workers', '4', '--render-face-budget', '250000']
            print(json.dumps({'phase': 'START_250K_OFFICIAL_RENDER_BAKE',
                              'command': command}), flush=True)
            subprocess.run(command, check=True)
        while not args.external_completion.is_file():
            print(json.dumps({'phase': 'WAIT_EXTERNAL_METRICS_AFTER_EXPORT',
                              'official_export_complete': (args.new_root / 'complete.json').is_file()}),
                  flush=True)
            if state(args.old_export_pid) != 'T' or state(args.old_gate_pid) != 'T':
                raise RuntimeError('paused GPU7 gate changed state')
            time.sleep(args.poll_seconds)
        external = json.loads(args.external_completion.read_text())
        if external['status'] != 'PASS_SOURCE_M0_FULL502_SEVEN_METHODS_TEN_METRICS':
            raise ValueError('external ten-metric receipt differs')
        result = json.loads((args.new_root / 'complete.json').read_text())
        rows = list((args.new_root / 'cases').glob('*/prediction.json'))
        budgets = {int(json.loads(row.read_text()).get('render_face_budget', -1))
                   for row in rows}
        if (result['status'] != 'PASS_OFFICIAL_RGB1_SCALE_UNGIVEN_COMMON_SUCCESS_EXPORT'
                or result['preferred_render_face_budget'] != 250_000
                or result['samples'] != len(rows)
                or not budgets <= {250_000, 500_000, 1_000_000,
                                   2_000_000, 5_000_000}):
            raise ValueError('fast official export closure differs')
        args.old_root.rename(retired)
        args.old_root.symlink_to(args.new_root, target_is_directory=True)
        receipt = {'status': 'PASS_OFFICIAL_FAST_EXPORT_BRIDGE',
                   'samples': len(rows), 'preferred_render_face_budget': 250_000,
                   'actual_budget_counts': result['actual_budget_counts'],
                   'old_watcher_pid_still_stopped': args.old_export_pid,
                   'old_gate_pid_still_stopped': args.old_gate_pid,
                   'output_link': str(args.old_root), 'verified_root': str(args.new_root)}
        args.output.write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt), flush=True)
    except Exception:
        # Return to the original, slower 1M path if the replacement fails.
        if not args.old_root.is_symlink() and state(args.old_gate_pid) == 'T':
            os.kill(args.old_gate_pid, signal.SIGCONT)
        raise


if __name__ == '__main__':
    main()
