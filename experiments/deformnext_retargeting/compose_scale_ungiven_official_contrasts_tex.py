"""Typeset the predeclared common-TEST official RGB1 paired comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = (('ours_v1_source_m0', 'Vanilla V1'),
           ('ours_selected_source_m0', 'Ours selected'))
METRICS = ('LPIPS', 'DreamSim', 'CLIP', 'PSNR', 'SSIM', 'M_IoU', 'CD', 'FScore')
LABEL = {'M_IoU': 'M-IoU', 'FScore': 'F@0.02'}
DIGITS = {'LPIPS': 5, 'DreamSim': 5, 'CLIP': 5, 'PSNR': 3,
          'SSIM': 5, 'M_IoU': 5, 'CD': 6, 'FScore': 5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contrasts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('immutable paired contrast table already exists')
    data = json.loads(args.contrasts.read_text())
    if (data['status']
            != 'PASS_COMMON_TEST_OURS_VERSUS_OFFICIAL_PAIRED_SIXTEEN_ENDPOINTS'
            or data['left'] != 'pixal3d_official_rgb1_source_m0'
            or set(data['contrasts']) != {method for method, _ in METHODS}
            or any(set(data['contrasts'][method]) != set(METRICS)
                   for method, _ in METHODS)):
        raise ValueError('final two-by-eight official comparison required')
    lines = [r'\begin{table*}[t]', r'\centering\scriptsize',
             r'\begin{tabular}{llrrl}', r'\toprule',
             r'Ours method & Metric & $\Delta$ to official & Family-adjusted 95\% CI'
             r' & Favorable interval?\\', r'\midrule']
    for method, label in METHODS:
        for metric in METRICS:
            row = data['contrasts'][method][metric]
            delta = row['ours_minus_official']
            lo, hi = row['family16_bonferroni_ci95']
            digits = DIGITS[metric]
            cell = f'{delta:+.{digits}f}'
            interval = f'$[{lo:+.{digits}f},{hi:+.{digits}f}]$'
            verdict = 'yes' if row['family16_favorable_interval'] else 'uncertain'
            lines.append(' & '.join((label, LABEL.get(metric, metric), cell,
                                     interval, verdict)) + r'\\')
        if method != METHODS[-1][0]:
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}',
              (r'\caption{Paired source-cluster comparisons of frozen Vanilla V1'
               r' and Ours selected against the official given-camera Pixal3D RGB1'
               r' on the same ' + str(data['samples'])
               + r'/508 successful edits after source-M0-only post-generation'
               r' registration. $\Delta$ is Ours minus official; favorable'
               r' directions are lower LPIPS, DreamSim and CD, and higher'
               r' CLIP, PSNR, SSIM, M-IoU and F@0.02. The family contains'
               r' two comparisons times eight endpoints, with Bonferroni'
               r' adjusted source-cluster intervals. ``Uncertain'' does not'
               r' establish equality. The official control uses T1+C and Ours'
               r' uses T1+S3+C; these rows compare complete systems, not an'
               r' isolated module. FID/KID are distribution point estimates'
               r' in Table~\ref{tab:scale-ungiven-common-main}.}'),
              r'\label{tab:scale-ungiven-official-contrasts}', r'\end{table*}']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': 'PASS_OFFICIAL_CONTRASTS_TEX',
                      'samples': data['samples'], 'rows': 16,
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
