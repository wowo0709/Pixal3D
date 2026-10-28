"""Assemble and compile the final scale-unprovided manuscript after all gates.

This creates a reviewable build in a fresh output directory. Copying its PDF to
the reports release path and publishing remain separate review steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pymupdf


def sha(path:Path)->str:
    state=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            state.update(block)
    return state.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-root',type=Path,required=True)
    parser.add_argument('--support-snapshot',type=Path,required=True)
    parser.add_argument('--main-table-json',type=Path,required=True)
    parser.add_argument('--paired-json',type=Path,required=True)
    parser.add_argument('--official-contrasts-json',type=Path,required=True)
    parser.add_argument('--claim-audit-json',type=Path,required=True)
    parser.add_argument('--common-manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--tectonic',type=Path,default=Path('/root/dev/.cache/paper-tools/tectonic-0.17.0/tectonic'))
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError('fresh manuscript build directory required')
    table=json.loads(args.main_table_json.read_text())
    paired=json.loads(args.paired_json.read_text())
    official_contrasts=json.loads(args.official_contrasts_json.read_text())
    claim_audit=json.loads(args.claim_audit_json.read_text())
    common=json.loads(args.common_manifest.read_text())
    teaser_json=args.paper_root/'final-results/figures/paper-teaser-provenance.json'
    teaser=json.loads(teaser_json.read_text())
    teaser_caption=args.paper_root/'final-results/figures/paper-teaser-caption.tex'
    n=common['samples']
    if (table['status']!='PASS_SCALE_UNGIVEN_COMMON_SUCCESS_MAIN_TABLE'
            or table['samples']!=n or len(table['methods'])!=10 or len(table['metrics'])!=10
            or paired['status']!='PASS_GT_SCALE_FREE_OURS_V1_SELECTED_COMMON_PAIRED_EIGHT_ENDPOINTS'
            or paired['samples']!=n or len(paired['metrics'])!=8
            or official_contrasts['status']
               !='PASS_COMMON_TEST_OURS_VERSUS_OFFICIAL_PAIRED_SIXTEEN_ENDPOINTS'
            or official_contrasts['samples']!=n
            or set(official_contrasts['contrasts'])
               !={'ours_v1_source_m0','ours_selected_source_m0'}
            or any(len(metrics)!=8 for metrics in official_contrasts['contrasts'].values())
            or common['status']!='PASS_THIRTEEN_METHOD_COMMON_SUCCESS_BINDING'
            or claim_audit['status']!='PASS_FAIR_TEST_CLAIM_AUDIT'
            or claim_audit['samples']!=n
            or len(claim_audit['comparisons']['selected_vs_v1'])!=8
            or set(claim_audit['comparisons']['ours_vs_official'])
               !={'ours_v1_source_m0','ours_selected_source_m0'}
            or teaser['status']!='PASS_FAIR_COMMON_TEST_TEASER_RENDER_SHA_VERIFIED'
            or teaser['sample_id'] not in common['sample_ids']
            or teaser['selection']['common_manifest_sha256']!=sha(args.common_manifest)
            or teaser['figure_pdf_sha256']!=sha(args.paper_root/'final-results/figures/paper-teaser.pdf')):
        raise ValueError('fair table, paired CI, population and teaser must agree')
    if (not teaser_caption.is_file()
            or 'Common-success TEST qualitative comparison' not in teaser_caption.read_text()
            or 'source-M0-only Sim(3)' not in teaser_caption.read_text()
            or str(n)+' edits' not in teaser_caption.read_text()):
        raise ValueError('fair teaser caption must describe the final common population')
    for method, effects in official_contrasts['contrasts'].items():
        for metric, row in effects.items():
            if (abs(row['official_mean']
                    - table['values']['pixal3d_official_rgb1_source_m0'][metric])>1e-6
                    or abs(row['ours_mean']-table['values'][method][metric])>1e-6):
                raise ValueError('official paired contrast and paper table differ: '
                                 +method+'/'+metric)
    required=('final-results/scale-ungiven-main-table.tex',
              'final-results/scale-ungiven-target8-registration.tex',
              'final-results/scale-ungiven-official-contrasts.tex',
              'final-results/unified-abstract-findings.tex',
              'final-results/unified-results.tex',
              'final-results/unified-discussion.tex',
              'final-results/unified-conclusion.tex',
              'sections/unified-evaluation-protocol-draft.tex')
    if any(not (args.paper_root/p).is_file() for p in required):
        raise FileNotFoundError('new protocol/table/abstract/conclusion files incomplete')
    if not (args.support_snapshot/'typeset').is_dir() or not args.tectonic.is_file():
        raise FileNotFoundError('typeset support snapshot and cached Tectonic required')
    source=args.output_dir/'source'
    shutil.copytree(args.support_snapshot,source)
    for p in args.paper_root.rglob('*'):
        if p.is_file() and p.suffix in {'.tex','.bib','.pdf','.png','.json'}:
            q=source/p.relative_to(args.paper_root)
            q.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,q)
    command=[str(args.tectonic),'--only-cached','--untrusted','--keep-logs',
             'retargeting-final-manuscript.tex']
    log=args.output_dir/'compile-console.log'
    with log.open('w') as stream:
        result=subprocess.run(command,cwd=source,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'LaTeX compilation failed: {log}')
    latex_log=source/'retargeting-final-manuscript.log'
    if any(token in latex_log.read_text() for token in ('Overfull \\hbox','undefined references',
                                                        'undefined citations','! LaTeX Error')):
        raise RuntimeError('manuscript log has unresolved/overfull/fatal issue')
    pdf=source/'retargeting-final-manuscript.pdf'
    doc=pymupdf.open(pdf)
    texts='\n'.join(page.get_text() for page in doc)
    for token in ('Unified benchmark protocol','Unified scale-unprovided comparison',
                  'Internal oracle-scale held-out comparison'):
        if token not in texts:raise ValueError('missing fair/internal manuscript section: '+token)
    review=args.output_dir/'review-pages';review.mkdir()
    for i,page in enumerate(doc):
        page.get_pixmap(matrix=pymupdf.Matrix(1.25,1.25),alpha=False).save(review/f'page{i+1:02d}.png')
    result={'status':'PASS_FAIR_MANUSCRIPT_COMPILED_AWAITING_VISUAL_REVIEW',
            'samples':n,'methods':10,'metrics':10,'paired_endpoints':8,
            'official_contrast_endpoints':16,
            'pages':len(doc),'teaser_sample_id':teaser['sample_id'],
            'pdf':str(pdf),'pdf_sha256':sha(pdf),'review_pages':str(review),
            'main_table_json_sha256':sha(args.main_table_json),
            'paired_json_sha256':sha(args.paired_json),
            'official_contrasts_json_sha256':sha(args.official_contrasts_json),
            'claim_audit_json_sha256':sha(args.claim_audit_json),
            'teaser_provenance_sha256':sha(teaser_json)}
    (args.output_dir/'build-receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
