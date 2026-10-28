"""Compile the compact CVPR 2027 protocol preview with the official style.

This preview deliberately accepts pending fair TEST slots and must never be
presented as a completed submission. A separate final builder will require
the measured results, common-population receipts, and verified teaser.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import pymupdf


KIT_SHA256 = 'da1ec5db9b73531ea82a87723e4aedb631083331fe655d1972279b48a7bdb2fe'
SOURCE_NAME = 'cvpr2027-submission.tex'


def digest(path: Path) -> str:
    state = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            state.update(block)
    return state.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--kit-zip', type=Path,
                        default=Path('/root/dev/.cache/paper-tools/'
                                     'author-kit-CVPR2027-v1-latex.zip'))
    parser.add_argument('--tectonic', type=Path,
                        default=Path('/root/dev/.cache/paper-tools/'
                                     'tectonic-0.17.0/tectonic'))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('fresh CVPR preview directory required')
    if digest(args.kit_zip) != KIT_SHA256 or not args.tectonic.is_file():
        raise ValueError('official frozen author kit and Tectonic required')
    if not (args.paper_root / SOURCE_NAME).is_file():
        raise FileNotFoundError('CVPR source missing')
    if any((args.paper_root / 'final-results' / name).exists() for name in (
            'cvpr2027-abstract-findings.tex', 'cvpr2027-results.tex',
            'cvpr2027-conclusion.tex')):
        raise ValueError('result-bearing CVPR source must use a final builder')
    source = args.output_dir / 'source'
    source.mkdir(parents=True)
    for path in args.paper_root.rglob('*'):
        if path.is_file() and path.suffix in {'.tex', '.bib', '.pdf', '.png', '.json'}:
            target = source / path.relative_to(args.paper_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    with zipfile.ZipFile(args.kit_zip) as kit:
        for name in ('cvpr.sty', 'ieeenat_fullname.bst'):
            (source / name).write_bytes(kit.read(name))
    command = [str(args.tectonic), '--only-cached', '--untrusted',
               '--keep-logs', SOURCE_NAME]
    with (args.output_dir / 'compile-console.log').open('w') as stream:
        result = subprocess.run(command, cwd=source, stdout=stream,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('CVPR preview compilation failed')
    log = (source / 'cvpr2027-submission.log').read_text()
    if any(token in log for token in ('Overfull \\hbox',
                                      'undefined references',
                                      'undefined citations', '! LaTeX Error')):
        raise RuntimeError('CVPR preview has a layout or citation error')
    pdf = source / 'cvpr2027-submission.pdf'
    document = pymupdf.open(pdf)
    if not 1 <= len(document) <= 9:
        raise ValueError('preview unexpectedly exceeds eight content pages')
    text = '\n'.join(page.get_text() for page in document)
    if ('Frozen common-success ten-method results pending' not in text
            or 'Fair common-success teaser pending' not in text):
        raise ValueError('preview must disclose pending fair results and teaser')
    review = args.output_dir / 'review-pages'
    review.mkdir()
    for number, page in enumerate(document, 1):
        page.get_pixmap(matrix=pymupdf.Matrix(1.25, 1.25), alpha=False).save(
            review / f'page{number:02d}.png')
    receipt = {'status': 'PASS_CVPR2027_PROTOCOL_PREVIEW_NOT_SUBMISSION',
               'pages_including_references': len(document),
               'pdf': str(pdf), 'pdf_sha256': digest(pdf),
               'source_sha256': digest(args.paper_root / SOURCE_NAME),
               'official_kit_sha256': KIT_SHA256,
               'review_pages': str(review)}
    (args.output_dir / 'build-receipt.json').write_text(
        json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
