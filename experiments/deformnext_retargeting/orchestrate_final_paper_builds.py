"""Generate audited findings and compile both papers immediately after closure."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--support-snapshot', type=Path, required=True)
    parser.add_argument('--main-root', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--paired-root', type=Path, required=True)
    parser.add_argument('--raw-paired-root', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--claim-root', type=Path, required=True)
    parser.add_argument('--claim-pid', type=int, required=True)
    parser.add_argument('--teaser-root', type=Path, required=True)
    parser.add_argument('--teaser-pid', type=int, required=True)
    parser.add_argument('--manuscript-build', type=Path, required=True)
    parser.add_argument('--cvpr-build', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    if (args.output_root.exists() or args.manuscript_build.exists()
            or args.cvpr_build.exists() or args.poll_seconds < 10):
        raise ValueError('fresh final build outputs and bounded polling required')
    paper = args.repo_root / 'docs/superpowers/paper'
    findings = paper / 'generate_fair_findings.py'
    if (not findings.is_file() or not args.support_snapshot.is_dir()):
        raise FileNotFoundError('paper source or support snapshot missing')
    args.output_root.mkdir(parents=True)
    claim_receipt = args.claim_root / 'complete.json'
    teaser_receipt = args.teaser_root / 'complete.json'
    while not (claim_receipt.is_file() and teaser_receipt.is_file()):
        state = {'claim': claim_receipt.is_file(),
                 'teaser': teaser_receipt.is_file()}
        if not state['claim'] and not alive(args.claim_pid):
            raise RuntimeError('claim auditor ended without completion receipt')
        if not state['teaser'] and not alive(args.teaser_pid):
            raise RuntimeError('teaser builder ended without completion receipt')
        print(json.dumps({'phase': 'WAIT_CLAIM_AND_TEASER', **state}), flush=True)
        time.sleep(args.poll_seconds)
    claim_done = json.loads(claim_receipt.read_text())
    teaser_done = json.loads(teaser_receipt.read_text())
    if (claim_done['status'] != 'PASS_FAIR_TEST_CLAIM_AUDIT_ARCHIVED'
            or teaser_done['status'] != 'PASS_FAIR_COMMON_TEST_TEASER_INSTALLED'
            or claim_done['samples'] != teaser_done['samples']):
        raise ValueError('claim and teaser closure differ')

    main_json = args.main_root / 'scale-ungiven-main-table.json'
    raw_json = args.raw_root / 'raw-common501-table.json'
    paired_json = args.paired_root / 'paired-eight-endpoints.json'
    raw_paired_json = args.raw_paired_root / 'raw-paired-eight-endpoints.json'
    official_json = args.official_root / 'official-contrasts-sixteen-endpoints.json'
    claim_json = args.claim_root / 'claim-audit.json'
    common_json = args.main_root / 'common-manifest.json'
    shared = [
        '--main-table-json', str(main_json),
        '--raw-table-json', str(raw_json),
        '--paired-json', str(paired_json),
        '--raw-paired-json', str(raw_paired_json),
        '--official-contrasts-json', str(official_json),
        '--claim-audit-json', str(claim_json),
        '--common-manifest', str(common_json),
    ]
    subprocess.run([sys.executable, str(findings), *shared,
                    '--output-dir', str(paper / 'final-results')],
                   cwd=args.repo_root, check=True)
    subprocess.run([
        sys.executable, str(paper / 'build_fair_manuscript.py'),
        '--paper-root', str(paper), '--support-snapshot', str(args.support_snapshot),
        *shared, '--output-dir', str(args.manuscript_build),
    ], cwd=args.repo_root, check=True)
    subprocess.run([
        sys.executable, str(paper / 'build_fair_cvpr2027.py'),
        '--paper-root', str(paper), *shared,
        '--output-dir', str(args.cvpr_build),
    ], cwd=args.repo_root, check=True)
    manuscript_receipt = json.loads(
        (args.manuscript_build / 'build-receipt.json').read_text())
    cvpr_receipt = json.loads((args.cvpr_build / 'build-receipt.json').read_text())
    if (manuscript_receipt['status']
            != 'PASS_FAIR_MANUSCRIPT_COMPILED_AWAITING_VISUAL_REVIEW'
            or cvpr_receipt['status']
            != 'PASS_FAIR_CVPR2027_COMPILED_AWAITING_VISUAL_REVIEW'
            or manuscript_receipt['samples'] != claim_done['samples']
            or cvpr_receipt['samples'] != claim_done['samples']):
        raise ValueError('compiled paper receipts differ from audited population')
    output = {
        'status': 'PASS_BOTH_FINAL_PAPERS_COMPILED_AWAITING_VISUAL_REVIEW',
        'samples': claim_done['samples'],
        'manuscript_pdf': manuscript_receipt['pdf'],
        'manuscript_pdf_sha256': manuscript_receipt['pdf_sha256'],
        'cvpr_pdf': cvpr_receipt['pdf'],
        'cvpr_pdf_sha256': cvpr_receipt['pdf_sha256'],
        'claim_audit_sha256': sha(claim_json),
    }
    (args.output_root / 'complete.json').write_text(
        json.dumps(output, indent=2) + '\n')
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
