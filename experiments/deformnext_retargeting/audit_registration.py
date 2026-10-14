"""Independent GT registration audit; does not load a trained denoiser."""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ss-config', type=Path, required=True)
    parser.add_argument('--pbr-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--skip-integrity', action='store_true', help='Rerender only; report does not certify whole-dataset integrity.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.ss_config.read_text())
    pcfg = json.loads(args.pbr_config.read_text())
    mappings = cfg['validation']['dataset_args']['path_map']
    def resolve(value):
        for source, destination in mappings.items():
            if value.startswith(source + '/'):
                return Path(destination + value[len(source):])
        return Path(value)
    splits = {split: [json.loads(line) for line in Path(path).read_text().splitlines() if line]
              for split, path in [('train', cfg['data_dir']), ('validation', cfg['validation']['roots_file'])]}
    def check(row):
        with np.load(resolve(row['shape_latent'])) as shape, np.load(resolve(row['pbr_latent'])) as pbr, np.load(resolve(row['ss_latent'])) as ss:
            coords = shape['coords']
            assert np.array_equal(coords, pbr['coords']), 'shape/PBR support'
            assert str(ss['schema'].item()) == row['ss_latent_schema'], 'SS schema'
            assert str(ss['source_coords_sha256'].item()) == hashlib.sha256(coords.tobytes()).hexdigest(), 'SS source hash'
            for feats in (shape['feats'], pbr['feats'], ss['mean'] if 'mean' in ss else ss['z']):
                assert np.isfinite(feats).all(), 'nonfinite latent'
            assert coords.min() >= 0 and coords.max() < 64, 'coordinate bounds'
            assert row['mesh_scale'] > 0, 'invalid scale'
        return row['sample_id']
    errors = []
    def checked(row):
        try:
            check(row)
        except Exception as error:
            errors.append({'sample_id': row['sample_id'], 'error': str(error)})
    if not args.skip_integrity:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(checked, splits['train'] + splits['validation']))
    leakage = {}
    for field in ('sample_id', 'pair_id', 'source_cluster_id', 'source_uid'):
        leakage[field] = sorted({row[field] for row in splits['train']} & {row[field] for row in splits['validation']})
    report = {'assets_checked': 0 if args.skip_integrity else sum(map(len, splits.values())), 'integrity_errors': errors, 'split_overlap': leakage, 'renders': []}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'renders'}), flush=True)
    if errors or any(leakage.values()):
        raise RuntimeError('Dataset integrity or split isolation failed')

    from pixal3d import datasets, models
    from pixal3d.representations import Mesh
    from pixal3d.renderers import MeshRenderer
    from pixal3d.utils.data_utils import recursive_to_device
    from experiments.deformnext_retargeting.quick_metrics import stratified_indices, binary_metrics
    from experiments.deformnext_retargeting.diagnose_ss import render_mask
    from data_toolkit.utils import sphere_normalize_torch, transform_mesh
    import utils3d
    torch.set_num_threads(4)
    ssds = getattr(datasets, cfg['validation']['dataset_name'])(cfg['validation']['roots_file'], **cfg['validation']['dataset_args'])
    pds = getattr(datasets, pcfg['validation']['dataset_name'])(pcfg['validation']['roots_file'], **pcfg['validation']['dataset_args'])
    pi = {row['sample_id']: i for i, row in enumerate(pds.rows)}
    decoder = models.from_pretrained(pds.pretrained_shape_slat_dec).cuda().eval()
    pbr_decoder = models.from_pretrained(pds.pretrained_pbr_slat_dec).cuda().eval()
    decoder.set_resolution(pds.resolution)
    ssds._loading_ss_dec()
    renderer = MeshRenderer({'resolution': 512, 'near': .01, 'far': 100, 'ssaa': 1})
    with torch.inference_mode():
        for i in stratified_indices(ssds, args.samples):
            row = ssds.rows[i]
            pack = recursive_to_device(ssds.collate_fn([ssds[i]]), 'cuda')
            ppack = recursive_to_device(pds.collate_fn([pds[pi[row['sample_id']]]]), 'cuda')
            z = pack['x_0']
            if ssds.normalization:
                z = z * ssds.std.cuda() + ssds.mean.cuda()
            occupancy = ssds.ss_dec(z)[0, 0] > 0
            with np.load(resolve(row['shape_latent'])) as archive:
                support = torch.from_numpy(archive['coords'].astype(np.int64)).cuda()
            encoded_occupancy = torch.zeros_like(occupancy)
            encoded_occupancy[support[:, 0], support[:, 1], support[:, 2]] = True
            ssmask = render_mask(occupancy, pack)
            shape = ppack['concat_cond']
            if pds.shape_slat_normalization is not None:
                shape = shape * pds.shape_slat_std.cuda() + pds.shape_slat_mean.cuda()
            meshes, subs = decoder(shape, return_subs=True)
            mesh = meshes[0]
            pbr_roundtrip = None
            if len(report['renders']) < 4:
                pbr = ppack['x_0']
                if pds.pbr_slat_normalization is not None:
                    pbr = pbr * pds.pbr_slat_std.cuda() + pds.pbr_slat_mean.cuda()
                decoded_pbr = pbr_decoder(pbr, guide_subs=subs)
                if not torch.isfinite(decoded_pbr.feats).all():
                    raise ValueError('nonfinite decoded PBR attributes')
                pbr_roundtrip = {'finite': True, 'decoded_tokens': len(decoded_pbr.feats)}
            scale = float(pack['mesh_scale'][0])
            radius = float(pack['camera_distance'][0, 0])
            fov = float(pack['camera_angle_x'][0, 0])
            ext = utils3d.torch.extrinsics_look_at(torch.tensor([0., 0., radius], device='cuda'), torch.zeros(3, device='cuda'), torch.tensor([0., 1., 0.], device='cuda'))
            intr = utils3d.torch.intrinsics_from_fov(fov_x=torch.tensor(fov, device='cuda'), fov_y=torch.tensor(fov, device='cuda'))
            mask = renderer.render(Mesh(mesh.vertices / scale, mesh.faces), ext, intr, return_types=['mask'])['mask'] > .5
            expected = torch.from_numpy(np.array(Image.open(resolve(row['target_anchor'])).convert('RGBA'))[:, :, 3].copy()).cuda() > 127
            dump_path = resolve(str(Path(row['shape_latent']).parent.parent / 'pbr_dumps' / (row['sample_id'] + '.pickle')))
            with dump_path.open('rb') as stream:
                dump = pickle.load(stream)
            vertices = torch.from_numpy(np.concatenate([obj['vertices'] for obj in dump['objects']])).float()
            normalized, _, sphere_radius = sphere_normalize_torch(vertices)
            transformed = transform_mesh(normalized, {'transform_matrix': row['transform_matrix'][0]})
            rebuilt_scale = .49999 / transformed.abs().max().item() / sphere_radius.item()
            fixed_mask = renderer.render(Mesh(mesh.vertices / rebuilt_scale, mesh.faces), ext, intr, return_types=['mask'])['mask'] > .5
            fixed_pack = dict(pack, mesh_scale=torch.tensor([rebuilt_scale], device='cuda'))
            fixed_ssmask = render_mask(occupancy, fixed_pack)
            faces = []
            offset = 0
            for obj in dump['objects']:
                faces.append(obj['faces'] + offset)
                offset += len(obj['vertices'])
            canonical_vertices = transformed * (.49999 / transformed.abs().max().item())
            raw_mask = renderer.render(Mesh(canonical_vertices.cuda() / rebuilt_scale,
                                          torch.from_numpy(np.concatenate(faces)).long().cuda()),
                                       ext, intr, return_types=['mask'])['mask'] > .5
            original_path = resolve(row['target_vertices'])
            if original_path.suffix == '.npy':
                original_vertices = np.load(original_path)
            else:
                with np.load(original_path) as original:
                    original_vertices = original['V'].copy()
            center = (original_vertices.min(0) + original_vertices.max(0)) / 2
            extent = float(np.ptp(original_vertices, axis=0).max())
            restored_vertices = vertices * extent + torch.from_numpy(center).float()
            restored_sphere, _, restored_radius = sphere_normalize_torch(restored_vertices)
            restored = transform_mesh(restored_sphere, {'transform_matrix': row['transform_matrix'][0]})
            restored_scale = .49999 / restored.abs().max().item() / restored_radius.item()
            restored = restored * (.49999 / restored.abs().max().item())
            restored_mask = renderer.render(Mesh(restored.cuda() / restored_scale,
                                               torch.from_numpy(np.concatenate(faces)).long().cuda()),
                                            ext, intr, return_types=['mask'])['mask'] > .5
            result = {'sample_id': row['sample_id'], 'category': row['category'],
                      'ss_vae_roundtrip_voxel_iou': binary_metrics(occupancy, encoded_occupancy)['voxel_iou'],
                      'ss_gt_mask_iou': binary_metrics(ssmask, expected)['voxel_iou'],
                      'shape_gt_mask_iou': binary_metrics(mask, expected)['voxel_iou'],
                      'ss_shape_mask_iou': binary_metrics(ssmask, mask)['voxel_iou'],
                      'corrected_scale_shape_mask_iou': binary_metrics(fixed_mask, expected)['voxel_iou'],
                      'corrected_scale_ss_mask_iou': binary_metrics(fixed_ssmask, expected)['voxel_iou'],
                      'raw_mesh_mask_iou': binary_metrics(raw_mask, expected)['voxel_iou'],
                      'restored_dump_world_mask_iou': binary_metrics(restored_mask, expected)['voxel_iou'],
                      'restored_dump_world_scale': restored_scale,
                      'pbr_attribute_roundtrip': pbr_roundtrip,
                      'manifest_scale': scale, 'rebuilt_scale': rebuilt_scale}
            report['renders'].append(result)
            panels = torch.cat((expected, ssmask, mask, fixed_ssmask, fixed_mask, raw_mask, restored_mask), 1).cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(panels).save(args.output / f'{len(report["renders"]):02d}-{row["category"]}.png')
            (args.output / 'report.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
