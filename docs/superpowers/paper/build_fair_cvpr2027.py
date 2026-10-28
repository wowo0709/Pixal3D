"""Build the official CVPR 2027 review PDF only from verified fair TEST results."""
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
RESULT_SLOTS = ('cvpr2027-abstract-findings.tex', 'cvpr2027-results.tex',
                'cvpr2027-conclusion.tex')


def digest(path: Path) -> str:
    state = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            state.update(block)
    return state.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-root', type=Path, required=True)
    parser.add_argument('--main-table-json', type=Path, required=True)
    parser.add_argument('--paired-json', type=Path, required=True)
    parser.add_argument('--official-contrasts-json', type=Path, required=True)
    parser.add_argument('--claim-audit-json', type=Path, required=True)
    parser.add_argument('--common-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--kit-zip', type=Path, default=Path(
        '/root/dev/.cache/paper-tools/author-kit-CVPR2027-v1-latex.zip'))
    parser.add_argument('--tectonic', type=Path, default=Path(
        '/root/dev/.cache/paper-tools/tectonic-0.17.0/tectonic'))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('fresh final CVPR build directory required')
    if digest(args.kit_zip) != KIT_SHA256 or not args.tectonic.is_file():
        raise ValueError('frozen official author kit and Tectonic required')

    table = read_json(args.main_table_json)
    paired = read_json(args.paired_json)
    contrasts = read_json(args.official_contrasts_json)
    claim_audit = read_json(args.claim_audit_json)
    common = read_json(args.common_manifest)
    n = common['samples']
    if (common['status'] != 'PASS_THIRTEEN_METHOD_COMMON_SUCCESS_BINDING'
            or not 1 <= n <= 502
            or len(common['sample_ids']) != n
            or len(set(common['sample_ids'])) != n
            or table['status'] != 'PASS_SCALE_UNGIVEN_COMMON_SUCCESS_MAIN_TABLE'
            or table['samples'] != n
            or len(table['methods']) != 10
            or len(table['metrics']) != 10
            or paired['status']
               != 'PASS_GT_SCALE_FREE_OURS_V1_SELECTED_COMMON_PAIRED_EIGHT_ENDPOINTS'
            or paired['samples'] != n
            or len(paired['metrics']) != 8
            or contrasts['status']
               != 'PASS_COMMON_TEST_OURS_VERSUS_OFFICIAL_PAIRED_SIXTEEN_ENDPOINTS'
            or contrasts['samples'] != n
            or set(contrasts['contrasts'])
               != {'ours_v1_source_m0', 'ours_selected_source_m0'}
            or any(len(rows) != 8 for rows in contrasts['contrasts'].values())):
        raise ValueError('final fair population/table/paired result gates incomplete')
    if (claim_audit['status'] != 'PASS_FAIR_TEST_CLAIM_AUDIT'
            or claim_audit['samples'] != n
            or len(claim_audit['comparisons']['selected_vs_v1']) != 8
            or set(claim_audit['comparisons']['ours_vs_official'])
               != {'ours_v1_source_m0', 'ours_selected_source_m0'}):
        raise ValueError('fair TEST claim audit incomplete')
    for method, endpoints in contrasts['contrasts'].items():
        for metric, row in endpoints.items():
            if (abs(row['official_mean'] - table['values']
                    ['pixal3d_official_rgb1_source_m0'][metric]) > 1e-6
                    or abs(row['ours_mean'] - table['values'][method][metric]) > 1e-6):
                raise ValueError('official paired contrast and table differ')

    figures = args.paper_root / 'final-results/figures'
    provenance_path = figures / 'paper-teaser-provenance.json'
    provenance = read_json(provenance_path)
    caption_path = figures / 'paper-teaser-caption.tex'
    caption = caption_path.read_text()
    if (provenance['status'] != 'PASS_FAIR_COMMON_TEST_TEASER_RENDER_SHA_VERIFIED'
            or provenance['sample_id'] not in common['sample_ids']
            or provenance['selection']['common_manifest_sha256']
               != digest(args.common_manifest)
            or provenance['figure_pdf_sha256'] != digest(figures / 'paper-teaser.pdf')
            or 'Common-success TEST qualitative comparison' not in caption
            or 'source-M0-only Sim(3)' not in caption
            or str(n) + ' edits' not in caption):
        raise ValueError('final fair teaser/caption/population gate incomplete')
    required = [args.paper_root / SOURCE_NAME,
                args.paper_root / 'final-results/scale-ungiven-main-table.tex',
                *[args.paper_root / 'final-results' / name for name in RESULT_SLOTS]]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError('final CVPR source, table, or results missing')
    for path in required[2:]:
        content = path.read_text()
        if not content.strip() or any(token in content.lower() for token in (
                'layout test', 'synthetic', 'results pending', 'evaluation in progress')):
            raise ValueError('CVPR result slot is empty or provisional: ' + str(path))

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
    with (args.output_dir / 'compile-console.log').open('w') as stream:
        result = subprocess.run([str(args.tectonic), '--only-cached', '--untrusted',
                                 '--keep-logs', SOURCE_NAME], cwd=source,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('official CVPR final compilation failed')
    log = (source / 'cvpr2027-submission.log').read_text()
    if any(token in log for token in ('Overfull \\hbox', 'undefined references',
                                      'undefined citations', '! LaTeX Error')):
        raise RuntimeError('CVPR layout, reference, or LaTeX error')
    pdf = source / 'cvpr2027-submission.pdf'
    document = pymupdf.open(pdf)
    pages = [page.get_text() for page in document]
    refs = [index + 1 for index, content in enumerate(pages)
            if 'References\n' in content]
    if not refs or min(refs) > 9:
        raise ValueError('CVPR main content exceeds eight pages before references')
    content = '\n'.join(pages)
    if any(token in content for token in ('results pending', 'teaser pending',
                                          'LAYOUT TEST', 'Synthetic results')):
        raise ValueError('final CVPR PDF contains provisional content')
    if not all(token in content for token in ('LLM Usage Statement',
                                              'Common-success TEST comparison',
                                              'Matched DEV128 geometry',
                                              'Matched DEV32 appearance')):
        raise ValueError('final CVPR paper misses required sections or tables')
    review = args.output_dir / 'review-pages'
    review.mkdir()
    for number, page in enumerate(document, 1):
        page.get_pixmap(matrix=pymupdf.Matrix(1.25, 1.25), alpha=False).save(
            review / f'page{number:02d}.png')
    receipt = {
        'status': 'PASS_FAIR_CVPR2027_COMPILED_AWAITING_VISUAL_REVIEW',
        'samples': n, 'methods': 10, 'metrics': 10,
        'main_pages_before_references': min(refs) - 1,
        'pages_including_references': len(document),
        'pdf': str(pdf), 'pdf_sha256': digest(pdf),
        'review_pages': str(review), 'official_kit_sha256': KIT_SHA256,
        'main_table_json_sha256': digest(args.main_table_json),
        'paired_json_sha256': digest(args.paired_json),
        'official_contrasts_json_sha256': digest(args.official_contrasts_json),
        'claim_audit_json_sha256': digest(args.claim_audit_json),
        'common_manifest_sha256': digest(args.common_manifest),
        'teaser_provenance_sha256': digest(provenance_path),
    }
    (args.output_dir / 'build-receipt.json').write_text(
        json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
