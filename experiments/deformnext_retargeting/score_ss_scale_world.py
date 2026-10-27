"""Compare archived SS occupancies in one world-frame lattice.

The target latent was encoded with its original dataset mesh_scale. A sample
generated with the native unit-scale fallback must be transformed into the
same physical frame before its overlap with that target is interpreted.
This remains an SS diagnostic; final quality requires decoded surfaces.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def occupancy(packed: np.ndarray) -> torch.Tensor:
    value = np.unpackbits(packed, count=64**3).reshape(64, 64, 64)
    return torch.from_numpy(value.copy()).float()[None, None]


def world_grid(resolution: int, half_extent: float) -> torch.Tensor:
    axis = (torch.arange(resolution, dtype=torch.float32) + .5) * (
        2 * half_extent / resolution) - half_extent
    gx, gy, gz = torch.meshgrid(axis, axis, axis, indexing='ij')
    # grid_sample sees input [D,H,W] as [x,y,z] in this archive.
    return torch.stack((gz, gy, gx), dim=-1)[None]


def reframe(value: torch.Tensor, scale: float, grid: torch.Tensor) -> torch.Tensor:
    if not 0 < scale < 2:
        raise ValueError('invalid positive canonical-to-world scale')
    # Native voxel center (i+.5)/64-.5 maps to world / mesh_scale.
    return F.grid_sample(value, grid * (2 * scale), mode='nearest',
                         padding_mode='zeros', align_corners=False)[0, 0] > .5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--world-resolution', type=int, default=128)
    parser.add_argument('--world-half-extent', type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists() or args.world_resolution < 64 or args.world_half_extent < .8:
        raise ValueError('fresh output and adequate world lattice required')
    report = json.loads(args.evaluation.read_text())
    if report.get('status') != 'PASS' or len(report['quick_panel_ids']) != len(report['native_SS_artifacts']['artifacts']):
        raise ValueError('complete frozen SS evaluation required')
    manifest = {row['sample_id']: row for row in map(json.loads, args.manifest.read_text().splitlines())}
    panel = report['quick_panel_ids']
    if len(panel) != len(set(panel)) or not set(panel) <= set(manifest):
        raise ValueError('manifest does not contain the complete panel')
    contract = json.loads((args.evaluation.parent / 'scale-contract.json').read_text())
    if contract['panel'] != panel or contract['status'] != 'PASS_VAL_SS_SCALE_SENSITIVITY':
        raise ValueError('scale-mode receipt differs from archived panel')
    pred_scales = contract.get('image_estimated_scales', {})
    grid = world_grid(args.world_resolution, args.world_half_extent)
    per_asset = []
    for item in report['native_SS_artifacts']['artifacts']:
        sid = item['sample_id']
        target_scale = float(manifest[sid]['mesh_scale'])
        if contract['mode'] == 'scale-ungiven':
            prediction_scale = 1.0
        elif contract['mode'] == 'image-estimated':
            prediction_scale = float(pred_scales[sid])
        else:
            prediction_scale = target_scale
        with np.load(args.evaluation.parent / item['artifact']) as saved:
            if str(saved['sample_id']) != sid or tuple(saved['occupancy_shape']) != (64, 64, 64):
                raise ValueError('archived SS identity or shape mismatch')
            pred = occupancy(saved['prediction_occupancy_packed'])
            target = occupancy(saved['target_occupancy_packed'])
        p = reframe(pred, prediction_scale, grid)
        t = reframe(target, target_scale, grid)
        tp = int((p & t).sum())
        fp = int((p & ~t).sum())
        fn = int((~p & t).sum())
        per_asset.append({'sample_id': sid, 'prediction_scale': prediction_scale,
                          'target_scale': target_scale, 'tp': tp, 'fp': fp, 'fn': fn,
                          'world_ss_iou': tp / (tp + fp + fn) if tp + fp + fn else 1.0})
    if [x['sample_id'] for x in per_asset] != panel:
        raise ValueError('archive order differs from paired panel')
    sums = {key: sum(row[key] for row in per_asset) for key in ('tp', 'fp', 'fn')}
    output = {'status': 'PASS_WORLD_FRAME_SS_DIAGNOSTIC', 'arm': contract['arm'],
              'mode': contract['mode'], 'samples': len(panel),
              'world_resolution': args.world_resolution,
              'world_half_extent': args.world_half_extent,
              'world_ss_micro_iou': sums['tp'] / sum(sums.values()),
              'world_ss_macro_iou': float(np.mean([x['world_ss_iou'] for x in per_asset])),
              'old_lattice_iou_for_reference_only': report['quick_metrics']['voxel_iou'],
              'GT_scale_used_for_scoring_only': contract['mode'] == 'scale-ungiven',
              'M1_derived_scale_supplied_to_generation': contract['mode'] == 'legacy-target',
              'source_M0_mesh_used_for_generation': False,
              'surface_or_render_metric': False, 'per_asset': per_asset}
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({key: output[key] for key in ('status', 'arm', 'mode', 'samples',
                'world_ss_micro_iou', 'world_ss_macro_iou', 'old_lattice_iou_for_reference_only')}))


if __name__ == '__main__':
    main()
