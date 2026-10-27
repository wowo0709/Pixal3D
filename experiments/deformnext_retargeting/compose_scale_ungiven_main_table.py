"""Compose the common-success camera-given, scale-unprovided +M0 Table 1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXTERNAL = ('trellis2', 'pixal3d', 'reconviagen', 'mvadapter',
            'corradapter', 'shapeup', 'spacecontrol')
METHODS = ('ours_selected_source_m0', 'ours_v1_source_m0',
           'pixal3d_official_rgb1_source_m0', *EXTERNAL)
LABEL = {'trellis2': 'TRELLIS.2', 'pixal3d': 'Pixal3D official (estimated C)',
         'reconviagen': 'ReconViaGen', 'mvadapter': 'MVAdapter+ReconViaGen',
         'corradapter': 'CorrAdapter+ReconViaGen', 'shapeup': 'ShapeUp',
         'spacecontrol': 'TRELLIS.2+SpaceControl',
         'pixal3d_official_rgb1_source_m0': 'Pixal3D official 1024 (given C)',
         'ours_v1_source_m0': 'Ours vanilla V1',
         'ours_selected_source_m0': 'Ours selected'}
INPUT = {'trellis2': 'T1', 'pixal3d': 'T1', 'reconviagen': 'T1+S3',
         'mvadapter': 'T1', 'corradapter': 'T1', 'shapeup': 'T1+M0',
         'spacecontrol': 'T1+P+A1',
         'pixal3d_official_rgb1_source_m0': 'T1+C',
         'ours_v1_source_m0': 'T1+S3+C',
         'ours_selected_source_m0': 'T1+S3+C'}
METRICS = ('LPIPS', 'DreamSim', 'CLIP', 'FID', 'KID',
           'PSNR', 'SSIM', 'M_IoU', 'CD', 'FScore')
DIRECTION = {'LPIPS': -1, 'DreamSim': -1, 'CLIP': 1,
             'FID': -1, 'KID': -1, 'PSNR': 1, 'SSIM': 1,
             'M_IoU': 1, 'CD': -1, 'FScore': 1}
DIGITS = {'LPIPS': 4, 'DreamSim': 4, 'CLIP': 4, 'FID': 2,
          'KID': 3, 'PSNR': 2, 'SSIM': 4, 'M_IoU': 4,
          'CD': 5, 'FScore': 4}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nine-summary', type=Path, required=True)
    parser.add_argument('--ours-summary', type=Path, required=True)
    parser.add_argument('--common-manifest', type=Path, required=True)
    parser.add_argument('--external-target8', type=Path, required=True)
    parser.add_argument('--official-target8', type=Path, required=True)
    parser.add_argument('--ours-target8', type=Path, required=True)
    parser.add_argument('--output-tex', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args()
    if args.output_tex.exists() or args.output_json.exists():
        raise FileExistsError('immutable fair table outputs already exist')
    registration_tex = args.output_tex.with_name('scale-ungiven-target8-registration.tex')
    if registration_tex.exists():
        raise FileExistsError('immutable paired registration table already exists')
    nine = json.loads(args.nine_summary.read_text())
    ours = json.loads(args.ours_summary.read_text())
    common = json.loads(args.common_manifest.read_text())
    ext_t = json.loads(args.external_target8.read_text())
    off_t = json.loads(args.official_target8.read_text())
    ours_t = json.loads(args.ours_target8.read_text())
    n = common['samples']
    if (common['status'] != 'PASS_THIRTEEN_METHOD_COMMON_SUCCESS_BINDING'
            or not 1 <= n <= 502
            or len(common['sample_ids']) != n
            or len(set(common['sample_ids'])) != n
            or set(nine['methods']) != (set(EXTERNAL)
                | {'pixal3d_official_rgb1_raw', 'pixal3d_official_rgb1_source_m0'})
            or set(ours['methods']) != {'ours_v1_raw', 'ours_v1_source_m0',
                                       'ours_selected_raw', 'ours_selected_source_m0'}
            or nine['protocol_sha256'] != ours['protocol_sha256']
            or nine['reference_manifest_sha256'] != ours['reference_manifest_sha256']
            or nine['split'] != ours['split'] or nine['split'] != 'test'
            or ext_t['samples'] != n or off_t['samples'] != n or ours_t['samples'] != n):
        raise ValueError('same TEST/common-success/metric protocol required')
    source = {**nine['methods'], **ours['methods']}
    values = {}
    for method in METHODS:
        report = source[method]
        if (report['successful_predictions'] != n
                or not report['full_metric_computation']
                or report['distribution']['image_count'] != n * 2):
            raise ValueError('complete same-N source/distribution score required: ' + method)
        m = {metric: float(report['deformation_weighted'][metric])
             for metric in ('LPIPS', 'DreamSim', 'CLIP', 'CD', 'FScore')}
        m.update({metric: float(report['distribution'][metric])
                  for metric in ('FID', 'KID')})
        for metric in ('PSNR', 'SSIM', 'M_IoU'):
            if method in EXTERNAL:
                m[metric] = float(ext_t['methods'][method][metric]['source_aligned_mean'])
            elif method == 'pixal3d_official_rgb1_source_m0':
                m[metric] = float(off_t['metrics'][metric]['source_m0_mean'])
            else:
                comparison = ours_t['metrics'][metric]['selected_minus_v1_source_m0']
                m[metric] = float(comparison['left_mean'] if method == 'ours_v1_source_m0'
                                  else comparison['right_mean'])
        values[method] = m
    ranks = {}
    for metric in METRICS:
        ordered = sorted(METHODS, key=lambda method: (-DIRECTION[metric] * values[method][metric],
                                                      method))
        ranks[metric] = {ordered[0]: 'best', ordered[1]: 'second'}
    lines = [r'\begin{table*}[t]', r'\centering\scriptsize',
             r'\resizebox{\textwidth}{!}{%',
             r'\begin{tabular}{llrrrrrrrrrr}', r'\toprule',
             r'& & \multicolumn{3}{c}{Source preservation (9/10)}'
             r' & \multicolumn{2}{c}{Distribution (9/10)}'
             r' & \multicolumn{5}{c}{Target alignment (8/M1)}\\',
             r'Method & Input & LPIPS$\downarrow$ & DreamSim$\downarrow$'
             r' & CLIP$\uparrow$ & FID$\downarrow$ & KID$\downarrow$'
             r' & PSNR$\uparrow$ & SSIM$\uparrow$ & M-IoU$\uparrow$'
             r' & CD$\downarrow$ & F@0.02$\uparrow$\\', r'\midrule']
    for method in METHODS:
        if method == 'pixal3d_official_rgb1_source_m0' or method == EXTERNAL[0]:
            lines.append(r'\midrule')
        row = [LABEL[method], INPUT[method]]
        for metric in METRICS:
            value = values[method][metric] * (1000 if metric == 'KID' else 1)
            cell = f'{value:.{DIGITS[metric]}f}'
            if ranks[metric].get(method) == 'best':
                cell = r'\textbf{' + cell + '}'
            elif ranks[metric].get(method) == 'second':
                cell = r'\underline{' + cell + '}'
            row.append(cell)
        lines.append(' & '.join(row) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}}',
              (r'\caption{Common-success TEST comparison on ' + str(n) +
               r'/508 edits. All rows are evaluated with the same cameras, Cycles renderer,'
               r' metric code, and post-generation source-M0-only Sim(3) alignment; no'
               r' target-derived object scale is supplied at generation. T1 is the edited'
               r' target RGB, S3 three original source RGBs, M0 the textured source mesh,'
               r' A1 an original anchor RGB, P a TRELLIS.2 target proxy predicted from T1,'
               r' and C provided camera metadata. Native model input access differs by row;'
               r' the older Pixal3D baseline estimates C and uses a 1536 cascade, while'
               r' the new RGB1 and Ours rows use given C and a 1024 cascade. Source metrics'
               r' use cameras 9/10; target image metrics use camera 8; geometry is measured'
               r' against edited M1. KID is displayed in $10^{-3}$ raw units. Bold and'
               r' underline mark the first and second values within this table.}'),
              r'\label{tab:scale-ungiven-common-main}', r'\end{table*}']
    result = {'status': 'PASS_SCALE_UNGIVEN_COMMON_SUCCESS_MAIN_TABLE',
              'samples': n, 'methods': list(METHODS), 'metrics': list(METRICS),
              'values': values, 'ranks': ranks,
              'protocol_sha256': nine['protocol_sha256'],
              'common_manifest': str(args.common_manifest),
              'target8_registration_tex': str(registration_tex)}
    registration = [r'\begin{table*}[t]', r'\centering\scriptsize',
                    r'\begin{tabular}{lrrrrrrrrr}', r'\toprule',
                    r'& \multicolumn{3}{c}{PSNR (dB)$\uparrow$}'
                    r' & \multicolumn{3}{c}{SSIM$\uparrow$}'
                    r' & \multicolumn{3}{c}{Mask IoU$\uparrow$}\\',
                    r'Method & Raw & +M0 & $\Delta$ & Raw & +M0 & $\Delta$'
                    r' & Raw & +M0 & $\Delta$\\', r'\midrule']
    for method in METHODS:
        if method in EXTERNAL:
            rows = ext_t['methods'][method]
            pair = {metric: (float(rows[metric]['raw_mean']),
                             float(rows[metric]['source_aligned_mean']))
                    for metric in ('PSNR', 'SSIM', 'M_IoU')}
        elif method == 'pixal3d_official_rgb1_source_m0':
            pair = {metric: (float(off_t['metrics'][metric]['raw_mean']),
                             float(off_t['metrics'][metric]['source_m0_mean']))
                    for metric in ('PSNR', 'SSIM', 'M_IoU')}
        else:
            key = ('v1_source_m0_minus_raw' if method == 'ours_v1_source_m0'
                   else 'selected_source_m0_minus_raw')
            pair = {metric: (float(ours_t['metrics'][metric][key]['left_mean']),
                             float(ours_t['metrics'][metric][key]['right_mean']))
                    for metric in ('PSNR', 'SSIM', 'M_IoU')}
        if any(abs(aligned - values[method][metric]) > 1e-8
               for metric, (_, aligned) in pair.items()):
            raise ValueError('target8 paired and main table means differ: ' + method)
        row = [LABEL[method]]
        for metric in ('PSNR', 'SSIM', 'M_IoU'):
            raw, aligned = pair[metric]
            digits = 2 if metric == 'PSNR' else 4
            row.extend(f'{value:+.{digits}f}' if offset == 2 else f'{value:.{digits}f}'
                       for offset, value in enumerate((raw, aligned, aligned - raw)))
        registration.append(' & '.join(row) + r'\\')
    registration += [r'\bottomrule', r'\end{tabular}',
                     (r'\caption{Paired effect of post-generation Sim(3) alignment'
                      r' to the input source M0 mesh on the exact ' + str(n) +
                      r'/508 common-success edits. Target-camera metrics are'
                      r' computed from raw and rerendered +M0 outputs with the'
                      r' same references; $\Delta$ is +M0 minus raw. The edited'
                      r' target mesh is never used for alignment.}'),
                     r'\label{tab:target8-source-m0-effect}', r'\end{table*}']
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text('\n'.join(lines) + '\n')
    registration_tex.write_text('\n'.join(registration) + '\n')
    args.output_json.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'samples': n,
                      'methods': len(METHODS), 'metrics': len(METRICS)}))


if __name__ == '__main__':
    main()
