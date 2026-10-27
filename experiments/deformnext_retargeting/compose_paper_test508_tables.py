"""Derive an input-audited diagnostic and matched ablation from common scores.

The combined thirteen-row table remains the audit artifact. Both paper views
reuse its predictions, success population, evaluator, and metric definitions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from compose_unified_test508_table import (EXTERNAL, INTERNAL, INPUT, LABEL,
                                           METRIC_DIRECTIONS, marked, ranks)


DIAGNOSTIC = (*EXTERNAL, 'SELECTED')
ABLATION = INTERNAL
METRICS = ('LPIPS', 'DreamSim', 'CLIP', 'FID', 'KID', 'PSNR', 'SSIM',
           'M_IoU', 'CD', 'FScore')


def load_scores(source_path: Path, target_path: Path) -> tuple[dict, dict, dict]:
    source = json.loads(source_path.read_text())
    target = json.loads(target_path.read_text())
    methods = set(EXTERNAL) | set(INTERNAL)
    if (source['schema'] != 'deformnext-common-source502-v1'
            or source['common_deformations'] != 502
            or source['distribution_images'] != 1004
            or set(source['methods']) != methods
            or target['schema'] != 'deformnext-common-target8-v1'
            or target['render_protocol_sha256'] != source['protocol_sha256']
            or target['common_subset_sha256'] != source['subset_sha256']
            or target['camera_id'] != 8
            or target['common_deformations'] != 502
            or set(target['methods']) != methods):
        raise ValueError('paper views require the complete, paired common-502 scores')
    values = {}
    for method in methods:
        report = source['methods'][method]
        target_report = target['methods'][method]
        if (report['common']['successful_deformations'] != 502
                or target_report['common']['successful_deformations'] != 502
                or report['full_successes'] != target_report['successful_deformations']
                or report['distribution']['image_count'] != 1004):
            raise ValueError('population mismatch: ' + method)
        source_metrics = report['common']['deformation_weighted']
        target_metrics = target_report['common']['deformation_weighted']
        distribution = report['distribution']
        values[method] = {metric: (distribution[metric] if metric in ('FID', 'KID')
                                   else target_metrics[metric] if metric in ('PSNR', 'SSIM', 'M_IoU')
                                   else source_metrics[metric]) for metric in METRICS}
    return source, target, values


def compose(methods: tuple[str, ...], source: dict, values: dict,
            caption: str, label: str) -> str:
    ranking = ranks({method: values[method] for method in methods})
    lines = [r'\begin{table*}[t]', r'\centering\scriptsize',
             r'\resizebox{\textwidth}{!}{%',
             r'\begin{tabular}{llrrrrrrrrrrr}', r'\toprule',
             r'& & &\multicolumn{3}{c}{Source views (9/10)}'
             r'&\multicolumn{2}{c}{Distribution (9/10)}'
             r'&\multicolumn{3}{c}{Target edit (8)}'
             r'&\multicolumn{2}{c}{Geometry}\\',
             r'Method&Input&Success&LPIPS$\downarrow$&DreamSim$\downarrow$'
             r'&CLIP$\uparrow$&FID$\downarrow$&KID$\downarrow$'
             r'&PSNR$\uparrow$&SSIM$\uparrow$&M-IoU$\uparrow$'
             r'&CD$\downarrow$&F@0.02$\uparrow$\\', r'\midrule']
    for method in methods:
        m, rank = values[method], ranking[method]
        row = [LABEL[method], INPUT[method],
               f"{source['methods'][method]['full_successes']}/508",
               marked(m['LPIPS'], 4, rank.get('LPIPS')),
               marked(m['DreamSim'], 4, rank.get('DreamSim')),
               marked(m['CLIP'], 4, rank.get('CLIP')),
               marked(m['FID'], 2, rank.get('FID')),
               marked(m['KID'], 3, rank.get('KID'), 1000.),
               marked(m['PSNR'], 2, rank.get('PSNR')),
               marked(m['SSIM'], 4, rank.get('SSIM')),
               marked(m['M_IoU'], 4, rank.get('M_IoU')),
               marked(m['CD'], 5, rank.get('CD')),
               marked(m['FScore'], 4, rank.get('FScore'))]
        if method == 'SELECTED' and methods == DIAGNOSTIC:
            lines.append(r'\midrule')
        lines.append('&'.join(row) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}}',
              r'\caption{' + caption + '}',
              r'\label{' + label + '}', r'\end{table*}']
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--common-source-score', type=Path, required=True)
    parser.add_argument('--target8-scores', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    main_path = args.output_dir / 'heterogeneous-input-diagnostic-test508-table.tex'
    ablation_path = args.output_dir / 'ablation-test508-table.tex'
    if main_path.exists() and ablation_path.exists():
        print(json.dumps({'status': 'ALREADY_COMPLETE', 'main': str(main_path),
                          'ablation': str(ablation_path)}))
        return
    if main_path.exists() or ablation_path.exists():
        raise FileExistsError('only one paper table exists; inspect before retry')
    source, _, values = load_scores(args.common_source_score, args.target8_scores)
    shared = (r' All metrics use the same 502 of 508 edits (133 source clusters)'
              r' and the same node16 Blender/Cycles and surface evaluator.'
              r' Source metrics use cameras 9/10; target metrics use camera 8;'
              r' KID is shown in $10^{-3}$ raw units.'
              r' C denotes supplied camera-to-world, FOV, distance, and target-M1-derived mesh scale in this diagnostic run; it is not a GT-free camera-given result.'
              r' Bold and underline mark first and second within this table.')
    main_caption = (r'Input-heterogeneous diagnostic comparison with external methods.'
                    r' This is not a matched-input primary baseline table. Ours is the identical'
                    r' selected checkpoint and prediction set used in the matched ablation table.'
                    r' Inputs differ by method: T1 is edited target RGB; S3 is three original'
                    r' source RGB views; A1 is the original anchor; our method additionally uses C.'
                    r' Official Pixal3D estimates its camera with MoGe and runs a native 1536 cascade;'
                    r' Ours uses a 1024 projective cascade and supplied camera metadata plus target-derived scale.'
                    r' Other external methods do not receive that dataset camera metadata at generation.'
                    r' Cross-method differences cannot establish a camera-matched advantage.' + shared)
    ablation_caption = (r'Matched Ours-stack ablation and input controls.'
                       r' V1, Top-1 R, and Selected use the same T1+S3+C inputs and 1024 projective'
                       r' cascade; Selected is exactly the Ours row in the diagnostic table.'
                       r' Projective RGB1 is a single-view control using the original Pixal3D weights,'
                       r' not a reproduction of the official Pixal3D runner.'
                       r' T1 is the edited target RGB view, S3 is three original source RGB views,'
                       r' and S4 denotes four original source RGB views.' + shared)
    main_text = compose(DIAGNOSTIC, source, values, main_caption, 'tab:test508-heterogeneous-diagnostic')
    ablation_text = compose(ABLATION, source, values, ablation_caption, 'tab:test508-ablation')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_path.write_text(main_text)
    ablation_path.write_text(ablation_text)
    print(json.dumps({'status': 'PASS_DIAGNOSTIC_AND_ABLATION_TABLES', 'diagnostic': str(main_path),
                      'ablation': str(ablation_path), 'shared_ours_method': 'SELECTED',
                      'common_edits': 502}))


if __name__ == '__main__':
    main()
