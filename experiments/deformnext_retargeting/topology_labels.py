"""Offline M1 barycentric -> M0 topology and z-buffer visibility teacher.

These files are supervision only. Dataset conditioning never reads them.
Spatially projecting X1 into M0 cameras is not a topology label.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

SCHEMA = 'deformnext-topology-visible-v1'


class RasterAgreementError(ValueError):
    """An asset's independent raster cannot certify its topology teacher."""


def mask_agreement(actual, expected, *, alpha_cutouts=False):
    iou = float((actual & expected).sum() / (actual | expected).sum().clamp_min(1))
    # Cycles can render transparent lenses/cutout materials, whereas the
    # geometry raster is opaque. Require the observed silhouette to lie in
    # the geometric one (one pixel edge tolerance), not equal opaque areas.
    dilated = F.max_pool2d(actual[None, None].float(), 3, 1, 1)[0, 0].bool()
    containment = float((dilated & expected).sum() / expected.sum().clamp_min(1))
    if (alpha_cutouts and containment < .99) or (not alpha_cutouts and iou < .95):
        raise RasterAgreementError(f'Topology raster disagrees with input image: IoU={iou}, containment={containment}')
    return iou, containment


def fingerprint(row, resolve):
    record = {key: row[key] for key in ('sample_id', 'split', 'transform_matrix', 'camera_angle_x')}
    for key in ('source_geometry', 'target_vertices', 'target_anchor', 'source_images'):
        paths = row[key] if isinstance(row[key], list) else [row[key]]
        record[key] = []
        for path in paths:
            with resolve(path).open('rb') as stream:
                record[key].append(hashlib.file_digest(stream, 'sha256').hexdigest())
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


class TopologyTeacher:
    def __init__(self, depth_tolerance=.005):
        from pixal3d.renderers import MeshRenderer
        self.renderer = MeshRenderer({'resolution': 512, 'near': .01, 'far': 100, 'ssaa': 1, 'antialias': False})
        self.depth_tolerance = float(depth_tolerance)
        if self.depth_tolerance <= 0:
            raise ValueError('depth tolerance must be positive')

    @torch.no_grad()
    def compute(self, row, resolve):
        from pixal3d.representations import Mesh
        from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import project_points_to_image_batch
        import utils3d
        with np.load(resolve(row['source_geometry'])) as cache:
            v0 = torch.from_numpy(cache['V'].copy()).float().cuda()
            faces = torch.from_numpy(cache['F'].copy()).int().cuda()
        path = resolve(row['target_vertices'])
        if path.suffix == '.npy':
            v1 = torch.from_numpy(np.load(path)).float().cuda()
        else:
            with np.load(path) as cache:
                v1 = torch.from_numpy(cache['V'].copy()).float().cuda()
        if v0.shape != v1.shape or v0.ndim != 2 or v0.shape[-1] != 3:
            raise ValueError('M0/M1 do not share vertex topology')
        transforms = torch.tensor(row['transform_matrix'], dtype=torch.float32, device='cuda')
        fovs = torch.tensor(row['camera_angle_x'], dtype=torch.float32, device='cuda')
        conversion = torch.diag(torch.tensor([1., -1., -1., 1.], device='cuda'))
        def render(vertices, view, attrs=None):
            ext = conversion @ torch.linalg.inv(transforms[view])
            intr = utils3d.torch.intrinsics_from_fov(fov_x=fovs[view], fov_y=fovs[view])
            return self.renderer.render(Mesh(vertices, faces, vertex_attrs=attrs), ext, intr,
                return_types=['mask', 'depth', 'coord'] + (['attr'] if attrs is not None else []))
        def mask(path):
            with Image.open(resolve(path)) as image:
                if image.size != (512, 512): raise ValueError('teacher render must be 512px')
                return torch.from_numpy((np.asarray(image.convert('RGBA'))[..., 3] > 127).copy()).cuda()
        target = render(v1, 0, v0)
        actual, expected = target['mask'] > .5, mask(row['target_anchor'])
        agreement = [mask_agreement(actual, expected)]
        centers = torch.arange(8, 512, 16, device='cuda')
        yy, xx = torch.meshgrid(centers, centers, indexing='ij')
        foreground = (actual & expected)[yy, xx].flatten()
        # Rasterized attributes interpolate V0 with the barycentrics of M1.
        x0 = target['attr'][:, yy, xx].permute(1, 2, 0).reshape(-1, 3)
        indices, visible = [], []
        for view in range(1, 4):
            source = render(v0, view)
            source_input_mask = mask(row['source_images'][view-1])
            agreement.append(mask_agreement(source['mask'] > .5, source_input_mask, alpha_cutouts=True))
            xy, depth, inside = project_points_to_image_batch(x0[None], transforms[view:view+1], fovs[view:view+1], 512)
            xy, depth, inside = xy[0], depth[0], inside[0]
            pixel = xy.floor().long().clamp(0, 511)
            zbuffer = source['depth'][pixel[:, 1], pixel[:, 0]]
            vis = (foreground & inside & source_input_mask[pixel[:, 1], pixel[:, 0]]
                   & (source['mask'][pixel[:, 1], pixel[:, 0]] > .5)
                   & ((zbuffer - depth).abs() < self.depth_tolerance))
            patch = (xy / 16).floor().long().clamp(0, 31)
            indices.append(patch[:, 1] * 32 + patch[:, 0] + (view-1)*1024)
            visible.append(vis)
        return {'positive_indices': torch.stack(indices, -1).cpu().numpy().astype(np.int32),
                'positive_visible': torch.stack(visible, -1).cpu().numpy(),
                'target_foreground': foreground.cpu().numpy(),
                'input_mask_iou': np.asarray([a[0] for a in agreement], dtype=np.float32),
                'input_mask_containment': np.asarray([a[1] for a in agreement], dtype=np.float32)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--path-map', action='append', default=[])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--continue-on-validation-error', action='store_true',
                        help='Record uncertified teacher rows; never publish labels for them')
    args = parser.parse_args()
    from .precompute_matches import remap
    mappings = [tuple(x.split('=', 1)) for x in args.path_map]
    resolve = lambda path: remap(path, mappings)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    rows = rows[args.rank::args.world_size][:args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    teacher = TopologyTeacher()
    failures=[]
    for n, row in enumerate(rows):
        output = args.output_root / f"{row['sample_id']}.npz"
        digest = fingerprint(row, resolve)
        if output.exists():
            with np.load(output) as old:
                if (str(old['schema']) == SCHEMA and str(old['input_sha256']) == digest
                        and 'input_mask_containment' in old):
                    continue
                if str(old['input_sha256']) != digest:
                    raise ValueError(f'Existing teacher provenance differs: {output}')
        try:
            arrays = teacher.compute(row, resolve)
        except RasterAgreementError as error:
            if not args.continue_on_validation_error:
                raise
            failures.append({'sample_id': row['sample_id'], 'split': row['split'],
                             'input_sha256': digest, 'reason': str(error)})
            failure_path=args.output_root.parent/f'teacher-rejections-rank{args.rank:03d}.json'
            failure_path.write_text(json.dumps(failures,indent=2)+'\n')
            print(json.dumps({'rejected_teacher': failures[-1]}),flush=True)
            continue
        tmp = output.with_name(f'.{output.name}.{os.getpid()}.tmp.npz')
        np.savez(tmp, schema=np.asarray(SCHEMA), sample_id=np.asarray(row['sample_id']),
                 split=np.asarray(row['split']), input_sha256=np.asarray(digest),
                 visibility_policy=np.asarray('opaque-mesh-zbuffer-and-input-alpha-conservative'),
                 visibility_depth_tolerance=np.asarray(teacher.depth_tolerance), **arrays)
        os.replace(tmp, output)
        print(json.dumps({'completed': n+1, 'assigned': len(rows), 'sample_id': row['sample_id'],
                          'visible_queries': int(arrays['positive_visible'].any(-1).sum()),
                          'minimum_mask_iou': float(arrays['input_mask_iou'].min())}), flush=True)
    print(json.dumps({'status':'COMPLETED_WITH_REJECTIONS' if failures else 'PASS',
                      'rank':args.rank,'assigned':len(rows),'rejected_teachers':len(failures)}),flush=True)


if __name__ == '__main__': main()
