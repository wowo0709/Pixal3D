"""Compose the raw counterpart of the frozen common-success main table."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from experiments.deformnext_retargeting.compose_scale_ungiven_main_table import (
    EXTERNAL, METHODS, LABEL, INPUT, METRICS, DIRECTION, DIGITS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('raw-external', 'nine-summary', 'ours-summary', 'common-manifest',
                 'external-target8', 'official-target8', 'ours-target8',
                 'native-success', 'output-tex', 'output-json'):
        parser.add_argument('--' + flag, type=Path, required=True)
    a = parser.parse_args()
    if a.output_tex.exists() or a.output_json.exists():
        raise FileExistsError('fresh raw table outputs required')
    raw, nine, ours, common, ext, off, ot, native = (
        json.loads(p.read_text()) for p in (a.raw_external, a.nine_summary, a.ours_summary,
        a.common_manifest, a.external_target8, a.official_target8, a.ours_target8, a.native_success))
    ids, n = common['sample_ids'], common['samples']
    if (raw['status'] != 'PASS_RAW_EXTERNAL_COMMON501_FEATURE_REPLAY'
            or ids != raw['sample_ids'] or n != 501 or len(set(ids)) != n
            or set(raw['methods']) != set(EXTERNAL)
            or common['status'] != 'PASS_THIRTEEN_METHOD_COMMON_SUCCESS_BINDING'
            or nine['protocol_sha256'] != ours['protocol_sha256']
            or nine['reference_manifest_sha256'] != ours['reference_manifest_sha256']
            or nine['split'] != 'test' or ours['split'] != 'test'
            or native['status'] != 'PASS_FROZEN_TEST508_NATIVE_SUCCESS_COUNTS'
            or native['methods']['pixal3d_official_rgb1_source_m0'] != 507):
        raise ValueError('frozen raw common population or provenance differs')
    if (ext['schema'] != 'deformnext-input-M0-Sim3-target8-paired-common-success-v2'
            or off['status'] != 'PASS_OFFICIAL_RGB1_RAW_AND_M0_TARGET8_PAIRED'
            or ot['status'] != 'PASS_OURS_V1_SELECTED_RAW_AND_M0_TARGET8_PAIRED'
            or any(x['samples'] != n or set(x['sample_ids']) != set(ids) for x in (ext, off, ot))):
        raise ValueError('raw target scores do not bind the common population')
    values = {}
    for method in METHODS:
        key = (method.replace('_source_m0', '_raw') if method not in EXTERNAL else method)
        report = raw['methods'][method] if method in EXTERNAL else (nine['methods'][key]
                  if method == 'pixal3d_official_rgb1_source_m0' else ours['methods'][key])
        if method not in EXTERNAL and (report['successful_predictions'] != n
                or not report['full_metric_computation']
                or report['distribution']['image_count'] != n * 2):
            raise ValueError('incomplete raw full-metric result: ' + method)
        row = {k: float(report['deformation_weighted'][k])
               for k in ('LPIPS', 'DreamSim', 'CLIP', 'CD', 'FScore')}
        row.update({k: float(report['distribution'][k]) for k in ('FID', 'KID')})
        for metric in ('PSNR', 'SSIM', 'M_IoU'):
            if method in EXTERNAL:
                row[metric] = float(ext['methods'][method][metric]['raw_mean'])
            elif method == 'pixal3d_official_rgb1_source_m0':
                row[metric] = float(off['metrics'][metric]['raw_mean'])
            else:
                pair = ot['metrics'][metric]['selected_minus_v1_raw']
                row[metric] = float(pair['left_mean'] if method == 'ours_v1_source_m0' else pair['right_mean'])
        if any(not math.isfinite(x) for x in row.values()):
            raise ValueError('nonfinite raw score: ' + method)
        values[method] = row
    ranks = {}
    for metric in METRICS:
        order = sorted(METHODS, key=lambda m: (-DIRECTION[metric] * values[m][metric], m))
        ranks[metric] = {order[0]: 'best', order[1]: 'second'}
    lines = [r'\begin{table*}[t]', r'\centering\normalsize', r'\setlength{\tabcolsep}{2.5pt}',
             r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{llrrrrrrrrrrr}', r'\toprule',
             r'& & & \multicolumn{3}{c}{Source preservation} & \multicolumn{2}{c}{Distribution} & \multicolumn{3}{c}{Target view} & \multicolumn{2}{c}{Geometry}\\',
             r'Method & Input & Success & LPIPS$\downarrow$ & DreamSim$\downarrow$ & CLIP$\uparrow$ & FID$\downarrow$ & KID$\downarrow$ & PSNR$\uparrow$ & SSIM$\uparrow$ & IoU$\uparrow$ & CD$\downarrow$ & F$\uparrow$\\', r'\midrule']
    for method in METHODS:
        if method in (EXTERNAL[0], 'pixal3d_official_rgb1_source_m0'):
            lines.append(r'\midrule')
        row = [LABEL[method], INPUT[method], f'{native["methods"][method]}/508']
        for metric in METRICS:
            cell = f'{values[method][metric] * (1000 if metric == "KID" else 1):.{DIGITS[metric]}f}'
            if ranks[metric].get(method) == 'best': cell = r'\textbf{' + cell + '}'
            elif ranks[metric].get(method) == 'second': cell = r'\underline{' + cell + '}'
            row.append(cell)
        lines.append(' & '.join(row) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}}',
              (r'\caption{Raw outputs before source-M0 registration, evaluated on the same '
               + str(n) + r'/508 common-success edits and cameras as the registered table. '
               r'Native success counts and heterogeneous image, camera, and mesh inputs are '
               r'defined in the main table. KID is multiplied by 1,000. Bold and underline '
               r'mark the first and second values within this table.}'),
              r'\label{tab:scale-ungiven-common-raw}', r'\end{table*}']
    result = {'status': 'PASS_RAW_COMMON501_FULL_TABLE', 'samples': n, 'methods': list(METHODS),
              'metrics': list(METRICS), 'values': values, 'ranks': ranks,
              'raw_external_sha256': hashlib.sha256(a.raw_external.read_bytes()).hexdigest(),
              'common_manifest_sha256': hashlib.sha256(a.common_manifest.read_bytes()).hexdigest()}
    a.output_tex.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_tex.write_text('\n'.join(lines) + '\n')
    a.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'samples': n, 'methods': len(METHODS)}))


if __name__ == '__main__':
    main()
