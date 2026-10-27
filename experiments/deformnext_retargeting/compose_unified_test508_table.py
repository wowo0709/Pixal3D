"""Compose the common-502 TEST table after all score files close.

The first five image metrics come from the frozen node16 cameras 9/10
protocol. PSNR/SSIM/M-IoU come from the additional common-renderer camera 8
evaluation. Surface metrics use the original node16 source-defined world.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXTERNAL = ('trellis2', 'pixal3d', 'reconviagen', 'mvadapter', 'corradapter',
            'shapeup', 'spacecontrol')
INTERNAL = ('RGB1_CALIBRATED', 'zero-source4', 'zero-edited4', 'V1',
            'RETARGET', 'SELECTED')
LABEL = {
    'trellis2': 'TRELLIS.2', 'pixal3d': 'Pixal3D official',
    'reconviagen': 'ReconViaGen', 'mvadapter': 'MVAdapter+ReconViaGen',
    'corradapter': 'CorrAdapter+ReconViaGen', 'shapeup': 'ShapeUp',
    'spacecontrol': 'TRELLIS.2+SpaceControl',
    'RGB1_CALIBRATED': 'Pixal3D RGB1 projective zero-shot',
    'zero-source4': 'Ours backbone, source RGB4',
    'zero-edited4': 'Ours backbone, edited RGB4',
    'V1': 'Ours vanilla V1', 'RETARGET': 'Ours Top-1 R',
    'SELECTED': 'Ours selected',
}
INPUT = {
    'trellis2': 'T1', 'pixal3d': 'T1', 'reconviagen': 'T1+S3',
    'mvadapter': 'T1', 'corradapter': 'T1', 'shapeup': 'T1',
    'spacecontrol': 'T1+A1', 'RGB1_CALIBRATED': 'T1+C',
    'zero-source4': 'S4+C', 'zero-edited4': 'T1+S3+C',
    'V1': 'T1+S3+C', 'RETARGET': 'T1+S3+C', 'SELECTED': 'T1+S3+C',
}
METRIC_DIRECTIONS = {'LPIPS': 'min', 'DreamSim': 'min', 'CLIP': 'max',
                     'FID': 'min', 'KID': 'min', 'PSNR': 'max', 'SSIM': 'max',
                     'M_IoU': 'max', 'CD': 'min', 'FScore': 'max'}


def latex_number(value: float | None, digits: int, multiplier: float = 1.) -> str:
    return '--' if value is None else f'{value * multiplier:.{digits}f}'


def ranks(values: dict[str, dict[str, float]]) -> dict[str, dict[str, int]]:
    """Rank exact, unrounded values across all thirteen methods per metric."""
    result = {method: {} for method in values}
    for metric, direction in METRIC_DIRECTIONS.items():
        unique = {values[method][metric] for method in values}
        if any(not math.isfinite(value) for value in unique):
            raise ValueError('non-finite metric value: ' + metric)
        ordered = sorted(unique, reverse=direction == 'max')
        if len(ordered) < 2:
            raise ValueError('at least two distinct metric values required: ' + metric)
        for method in values:
            if values[method][metric] == ordered[0]:
                result[method][metric] = 1
            elif values[method][metric] == ordered[1]:
                result[method][metric] = 2
    return result


def marked(value: float, digits: int, rank: int | None, multiplier: float = 1.) -> str:
    formatted = latex_number(value, digits, multiplier)
    if rank == 1:
        return r'\textbf{' + formatted + '}'
    if rank == 2:
        return r'\underline{' + formatted + '}'
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--common-source-score', type=Path, required=True)
    parser.add_argument('--target8-scores', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('table output is immutable')
    source = json.loads(args.common_source_score.read_text())
    reports = source['methods']
    if (source['schema'] != 'deformnext-common-source502-v1'
            or source['common_deformations'] != 502
            or source['distribution_images'] != 1004
            or set(reports) != set(EXTERNAL) | set(INTERNAL)):
        raise ValueError('source score must contain all 13 methods on common 502')
    protocol_hash = source['protocol_sha256']
    target = json.loads(args.target8_scores.read_text())
    if (target['schema'] != 'deformnext-common-target8-v1'
            or target['render_protocol_sha256'] != protocol_hash
            or target['camera_id'] != 8
            or target['common_deformations'] != 502
            or target['common_subset_sha256'] != source['subset_sha256']
            or set(target['methods']) != set(reports)):
        raise ValueError('target-camera scores do not match frozen source protocol or rows')
    values = {}
    for method, report in reports.items():
        source_metrics = report['common']['deformation_weighted']
        target_metrics = target['methods'][method]['common']['deformation_weighted']
        dist = report['distribution']
        values[method] = {metric: (dist[metric] if metric in ('FID', 'KID')
                                   else target_metrics[metric] if metric in ('PSNR', 'SSIM', 'M_IoU')
                                   else source_metrics[metric])
                          for metric in METRIC_DIRECTIONS}
    ranking = ranks(values)
    lines = [r'\begin{table*}[t]', r'\centering\scriptsize',
             r'\resizebox{\textwidth}{!}{%',
             r'\begin{tabular}{llrrrrrrrrrrr}', r'\toprule',
             r'& & &\multicolumn{3}{c}{Source-view fidelity (9/10)}'
             r'&\multicolumn{2}{c}{Distribution (9/10)}'
             r'&\multicolumn{3}{c}{Target edit (8)}'
             r'&\multicolumn{2}{c}{Geometry (world)}\\',
             r'Method&Input&Success&LPIPS$\downarrow$&DreamSim$\downarrow$'
             r'&CLIP$\uparrow$&FID$\downarrow$&KID$\downarrow$'
             r'&PSNR$\uparrow$&SSIM$\uparrow$&M-IoU$\uparrow$'
             r'&CD$\downarrow$&F@0.02$\uparrow$\\', r'\midrule']
    for group in (EXTERNAL, INTERNAL):
        for method in group:
            report = reports[method]
            n = report['full_successes']
            if not 0 < n <= 508 or target['methods'][method]['successful_deformations'] != n:
                raise ValueError('success population mismatch for ' + method)
            if (report['common']['successful_deformations'] != 502
                    or target['methods'][method]['common']['successful_deformations'] != 502):
                raise ValueError('common population mismatch for ' + method)
            metrics = report['common']['deformation_weighted']
            t = target['methods'][method]['common']['deformation_weighted']
            dist = report['distribution']
            if dist['image_count'] != 1004:
                raise ValueError('distribution image population mismatch for ' + method)
            cells = [LABEL[method], INPUT[method], f'{n}/508',
                     marked(metrics['LPIPS'], 4, ranking[method].get('LPIPS')),
                     marked(metrics['DreamSim'], 4, ranking[method].get('DreamSim')),
                     marked(metrics['CLIP'], 4, ranking[method].get('CLIP')),
                     marked(dist['FID'], 2, ranking[method].get('FID')),
                     marked(dist['KID'], 3, ranking[method].get('KID'), 1000.),
                     marked(t['PSNR'], 2, ranking[method].get('PSNR')),
                     marked(t['SSIM'], 4, ranking[method].get('SSIM')),
                     marked(t['M_IoU'], 4, ranking[method].get('M_IoU')),
                     marked(metrics['CD'], 5, ranking[method].get('CD')),
                     marked(metrics['FScore'], 4, ranking[method].get('FScore'))]
            lines.append('&'.join(cells) + r'\\')
        if group is EXTERNAL:
            lines.append(r'\midrule')
    lines.extend([r'\bottomrule', r'\end{tabular}}',
                  r'\caption{Common-renderer DeformNeXt TEST508. Success reports $n/508$;'
                  r' all ten metrics use the same 502 edits (133 source clusters) on which every method succeeded.'
                  r' T1 is edited M1 RGB at camera 8; S3 is original M0 RGB at 9/10/11;'
                  r' S4 includes original M0 RGB at 8/9/10/11; A1 is the original anchor at 8;'
                  r' C is supplied camera-to-world, FOV, distance, and per-case mesh scale computed from target M1 geometry in this diagnostic run; this is not a GT-free camera-given result.'
                  r' Official Pixal3D instead estimates camera parameters with MoGe and runs its native 1536 cascade;'
                  r' the calibrated RGB1 row uses original single-view weights in a projective 1024 cascade.'
                  r' All outputs use the same source-defined world, Blender/Cycles lighting,'
                  r' original-material M1 references, and no per-output alignment.'
                  r' Source-view metrics use cameras 9/10, which are input views for S3/S4 methods'
                  r' but held out for target-only methods. Target edit metrics use camera 8.'
                  r' FID/KID use the same 1,004-image 9/10 pool; KID is shown in $10^{-3}$ raw units.'
                  r' Best values are bold and second-best values are underlined per metric'
                  r' across all 13 rows, ranked before rounding.'
                  r' The six cases excluded from metrics remain in the success-rate denominator.'
                  r' GeoStyle full TEST was cancelled and is not a completed row.}',
                  r'\label{tab:unified-test508}', r'\end{table*}'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': 'PASS_UNIFIED_TEST508_TABLE', 'rows': len(reports),
                      'source_protocol_sha256': protocol_hash, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
