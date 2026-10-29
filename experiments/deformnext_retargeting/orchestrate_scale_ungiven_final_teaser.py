"""Select, build, and install the scored common-TEST teaser after all gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


OFFICIAL_ARCHIVE = ('/file4/youngwoo/deformation/experiments/retargeting-v1/'
                    'source-sim3-test508-20261028/'
                    'official-rgb1-scale-ungiven-common-success-v2')
OURS_ARCHIVE = ('/file4/youngwoo/deformation/experiments/retargeting-v1/'
                'source-sim3-test508-20261028/'
                'ours-v1-selected-scale-ungiven-common502-v1')
FILES = ('paper-teaser.pdf', 'paper-teaser.png',
         'paper-teaser-provenance.json', 'paper-teaser-caption.tex')


def sha(path: Path) -> str:
    state = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            state.update(block)
    return state.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-complete', type=Path, required=True)
    parser.add_argument('--paired-pid', type=int, required=True)
    parser.add_argument('--paired-root', type=Path, required=True)
    parser.add_argument('--table-complete', type=Path, required=True)
    parser.add_argument('--table-root', type=Path, required=True)
    parser.add_argument('--official-render-complete', type=Path, required=True)
    parser.add_argument('--official-render-root', type=Path, required=True)
    parser.add_argument('--ours-render-complete', type=Path, required=True)
    parser.add_argument('--ours-render-root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--paper-figures', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--file4-host', default='youngwoo@n16.unist.info')
    parser.add_argument('--ssh-port', type=int, default=55555)
    parser.add_argument('--official-archive', default=OFFICIAL_ARCHIVE)
    parser.add_argument('--ours-archive', default=OURS_ARCHIVE)
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    if args.output_root.exists() or args.poll_seconds < 10:
        raise ValueError('fresh final teaser output and bounded poll required')
    args.output_root.mkdir(parents=True)
    while not args.paired_complete.is_file():
        live = Path(f'/proc/{args.paired_pid}').exists()
        print(json.dumps({'phase': 'WAIT_FINAL_COMMON_PAIRED_FOR_TEASER',
                          'paired_watcher_live': live}), flush=True)
        if not live:
            raise RuntimeError('paired evaluator ended without completion receipt')
        time.sleep(args.poll_seconds)
    receipts = [json.loads(path.read_text()) for path in
                (args.paired_complete, args.table_complete,
                 args.official_render_complete, args.ours_render_complete)]
    if (receipts[0]['status'] != 'PASS_COMMON_SUCCESS_OURS_PAIRED_EIGHT_ENDPOINTS'
            or receipts[1]['status'] != 'PASS_COMMON_SUCCESS_TABLE_TEN_METRICS'
            or receipts[2]['status'] != 'PASS_OFFICIAL_RGB1_COMMON502_RENDER_ARCHIVED'
            or receipts[3]['status']
               != 'PASS_OURS_V1_SELECTED_RAW_AND_M0_COMMON502_RENDER_ARCHIVED'
            or not 1 <= receipts[0]['samples'] == receipts[1]['samples']
               == receipts[2]['samples'] <= 502
            or receipts[3]['samples'] != 502):
        raise ValueError('final common table, CI and render closures differ')
    n = receipts[0]['samples']
    official_plan = args.official_render_root / 'render-plan'
    ours_plan = args.ours_render_root / 'render-plan'
    ours_evaluation = args.paired_root / 'evaluation-ours-four'
    common = args.table_root / 'common-manifest.json'
    target = args.table_root / 'ours-target8.json'
    if not (official_plan.is_dir() and ours_plan.is_dir()
            and ours_evaluation.is_dir() and common.is_file()
            and target.is_file() and args.campaign.is_file()):
        raise FileNotFoundError('final teaser scored render/evaluation inputs missing')
    choice = args.output_root / 'selection.json'
    scripts = Path(__file__).parent
    subprocess.run([sys.executable,
                    str(scripts / 'select_scale_ungiven_teaser_case.py'),
                    '--ours-evaluation', str(ours_evaluation),
                    '--ours-target8', str(target),
                    '--common-manifest', str(common),
                    '--output', str(choice)], check=True)
    figure = args.output_root / 'figure'
    subprocess.run([sys.executable,
                    str(scripts / 'build_scale_ungiven_teaser.py'),
                    '--selection', str(choice), '--common-manifest', str(common),
                    '--ours-evaluation', str(ours_evaluation),
                    '--ours-target8', str(target),
                    '--campaign', str(args.campaign),
                    '--official-plan', str(official_plan),
                    '--ours-plan', str(ours_plan),
                    '--official-archive', args.official_archive,
                    '--ours-archive', args.ours_archive,
                    '--file4-host', args.file4_host,
                    '--ssh-port', str(args.ssh_port),
                    '--file4-control-socket', str(args.output_root / 'no-control-socket'),
                    '--staging-dir', str(args.output_root / 'staging'),
                    '--output-dir', str(figure)], check=True)
    selection = json.loads(choice.read_text())
    provenance = json.loads((figure / 'paper-teaser-provenance.json').read_text())
    if (selection['status'] != 'PASS_FAIR_COMMON_TEST_TEASER_CASE_SELECTED'
            or selection['common_samples'] != n
            or provenance['status'] != 'PASS_FAIR_COMMON_TEST_TEASER_RENDER_SHA_VERIFIED'
            or provenance['sample_id'] != selection['sample_id']
            or provenance['selection_sha256'] != sha(choice)
            or provenance['figure_pdf_sha256'] != sha(figure / 'paper-teaser.pdf')
            or len(provenance['render_records']) != 8
            or any(not (figure / name).is_file() for name in FILES)):
        raise ValueError('scored teaser, final population and render SHA differ')
    if not args.paper_figures.is_dir():
        raise FileNotFoundError('manuscript figures directory required')
    retired = args.output_root / 'retired-internal-teaser'
    retired.mkdir()
    for name in FILES:
        old = args.paper_figures / name
        if old.is_file():
            shutil.copy2(old, retired / name)
    for name in FILES:
        staged = args.paper_figures / (name + '.new')
        shutil.copy2(figure / name, staged)
        os.replace(staged, args.paper_figures / name)
        if sha(figure / name) != sha(args.paper_figures / name):
            raise ValueError('installed manuscript teaser differs: ' + name)
    result = {'status': 'PASS_FAIR_COMMON_TEST_TEASER_INSTALLED',
              'samples': n, 'sample_id': selection['sample_id'],
              'selection_rule': selection['selection_rule'],
              'paper_figures': str(args.paper_figures),
              'figure_sha256': {name: sha(args.paper_figures / name) for name in FILES},
              'retired_internal_teaser': str(retired)}
    (args.output_root / 'complete.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
