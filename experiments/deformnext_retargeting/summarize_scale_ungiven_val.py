"""Paired DEV summary for image-scale SS cascade and fixed-shape PBR arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARM_DIRS = {'V1': 'v1', 'Retarget': 'retarget', 'Selected': 'selected'}
GEOMETRY_METRICS = ('chamfer_L1', 'surface_FScore')
PBR_METRICS = ('target8_M_IoU', 'target8_PSNR', 'target8_SSIM', 'source9_10_LPIPS')


def bootstrap(differences: np.ndarray, *, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    replicates = differences[rng.integers(0, len(differences), size=(20000, len(differences)))].mean(1)
    return [float(value) for value in np.quantile(replicates, [.025, .975])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pbr', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    records = {}
    ss = {}
    campaigns = {}
    for name, directory in ARM_DIRS.items():
        campaign = args.root / f'cascade-val31-{directory}-v1'
        campaigns[name] = json.loads((campaign / 'campaign.json').read_text())
        expected = {case['sample_id'] for case in campaigns[name]['cases']}
        if len(expected) != 31 or campaigns[name]['source_clusters'] != 31:
            raise ValueError('expected frozen 31 source-disjoint cases')
        existing = {}
        for path in (campaign / 'image-scale').glob('*/generation.json'):
            result = json.loads(path.read_text())
            sid = result['sample_id']
            if sid not in expected or sid in existing or result['actual_resolution'] != 1024:
                raise ValueError('unexpected or duplicate cascade case')
            if (result['object_scale_policy'] != 'image-estimated-v1'
                    or result['M1_derived_scale_supplied_to_generation']
                    or result['source_M0_mesh_used_in_generation']
                    or result['GT_geometry_or_coords_used_in_generation']):
                raise ValueError('GT-free generation provenance failed')
            existing[sid] = result
        records[name] = existing
        if not args.allow_partial and set(existing) != expected:
            raise ValueError(f'{name}: only {len(existing)}/31 cascade cases completed')
        evaluation = json.loads((args.root / f'{name.upper()}-image-scale-v1/evaluation.json').read_text())
        ss[name] = {row['sample_id']: row for row in evaluation['per_asset']}
        if not expected <= set(ss[name]):
            raise ValueError('SS evaluation lacks cascade cases')
    if not all({case['sample_id'] for case in campaigns['V1']['cases']} ==
               {case['sample_id'] for case in campaigns[name]['cases']} for name in ARM_DIRS):
        raise ValueError('geometry arms have different frozen panels')
    ids = sorted(set.intersection(*(set(rows) for rows in records.values())))
    clusters = {ss['V1'][sid]['source_cluster_id'] for sid in ids}
    if len(clusters) != len(ids):
        raise ValueError('paired panel is not source-disjoint')
    for name in ARM_DIRS:
        if any(ss[name][sid]['source_cluster_id'] != ss['V1'][sid]['source_cluster_id'] for sid in ids):
            raise ValueError('source cluster identity changed between arms')
    result = {'status': 'PASS_PAIRED_VAL_SUMMARY' if len(ids) == 31 else 'PARTIAL_PAIRED_VAL_SUMMARY',
              'protocol': 'camera given, object-specific GT scale absent, image-only frozen scale head',
              'source_disjoint_cases': len(ids), 'expected_cases': 31,
              'geometry': {}, 'pbr': None}
    for name in ARM_DIRS:
        row = {}
        for metric in GEOMETRY_METRICS:
            values = np.array([records[name][sid]['metrics'][metric] for sid in ids])
            row[metric] = float(values.mean())
            if name != 'V1':
                difference = values - np.array([records['V1'][sid]['metrics'][metric] for sid in ids])
                row[f'delta_{metric}_vs_V1'] = float(difference.mean())
                row[f'delta_{metric}_CI95'] = bootstrap(difference, seed=20261028)
                row[f'{metric}_improved_cases'] = int((difference < 0).sum() if metric == 'chamfer_L1'
                                                        else (difference > 0).sum())
        voxel = np.array([ss[name][sid]['voxel_iou'] for sid in ids])
        row['SS_macro_voxel_IoU'] = float(voxel.mean())
        if name != 'V1':
            voxel_delta = voxel - np.array([ss['V1'][sid]['voxel_iou'] for sid in ids])
            geometry_delta = np.array([records[name][sid]['metrics']['surface_FScore']
                                       - records['V1'][sid]['metrics']['surface_FScore'] for sid in ids])
            row['delta_SS_macro_voxel_IoU_vs_V1'] = float(voxel_delta.mean())
            row['delta_SS_macro_voxel_IoU_CI95'] = bootstrap(voxel_delta, seed=20261029)
            row['SS_IoU_to_mesh_FScore_delta_correlation'] = float(np.corrcoef(voxel_delta, geometry_delta)[0, 1])
        result['geometry'][name] = row
    if args.pbr:
        rows = [json.loads(path.read_text()) for path in sorted((args.pbr / 'per-case').glob('*.json'))]
        if not args.allow_partial and len(rows) != 8:
            raise ValueError(f'PBR: only {len(rows)}/8 paired cases completed')
        if len({row['sample_id'] for row in rows}) != len(rows):
            raise ValueError('duplicate paired PBR case')
        if len({row['source_cluster_id'] for row in rows}) != len(rows):
            raise ValueError('PBR panel is not source-disjoint')
        if any(not row['fixed_geometry_sha256'] or not row['paired_first_noise_sha256'] for row in rows):
            raise ValueError('PBR geometry/noise identity missing')
        summary = {'source_disjoint_cases': len(rows), 'expected_cases': 8,
                   'renderer': 'native DEV diagnostic, not common Cycles final'}
        for metric in PBR_METRICS:
            vanilla = np.array([row['vanilla'][metric] for row in rows])
            top1 = np.array([row['top1_only'][metric] for row in rows])
            difference = top1 - vanilla
            summary[metric] = {'vanilla': float(vanilla.mean()), 'top1_only': float(top1.mean()),
                               'delta_top1_vs_vanilla': float(difference.mean()),
                               'delta_CI95': bootstrap(difference, seed=20261030),
                               'top1_improved_cases': int((difference < 0).sum() if metric == 'source9_10_LPIPS'
                                                          else (difference > 0).sum())}
        result['pbr'] = summary
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'geometry_cases': len(ids),
                      'pbr_cases': result['pbr']['source_disjoint_cases'] if result['pbr'] else 0,
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
