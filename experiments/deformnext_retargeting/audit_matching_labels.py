"""Evaluator-only topology correspondence audit; never writes training caches."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=32)
    args = parser.parse_args()
    from pixal3d.representations import Mesh
    from pixal3d.renderers import MeshRenderer
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import project_points_to_image_batch
    import utils3d
    torch.set_num_threads(4)
    def resolve(value):
        return Path('/root/dev/.cache/deformnext-file4' + value) if value.startswith('/file4/') else Path(value)
    rows = [json.loads(line) for line in (args.data_root / 'manifests_world_v4/validation.jsonl').read_text().splitlines()]
    groups = {}
    for row in rows:
        groups.setdefault(row['category'], []).append(row)
    selected = [groups[name][0] for name in sorted(groups)][:args.samples]
    renderer = MeshRenderer({'resolution': 512, 'near': .01, 'far': 100, 'ssaa': 1, 'antialias': False})
    counters = dict(foreground_queries=0, visible_queries=0, visible_pairs=0,
                    spatial_label_correct_pairs=0, raw_top1_correct=0,
                    q2c_top1_correct=0, proposal_recall=0, q2c_changes=0,
                    frame_top1_correct=0, frame_proposal_recall=0, frame_top1_changes=0,
                    background_queries=0, background_queries_with_source=0)
    results = []
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for row in selected:
            with np.load(resolve(row['source_geometry'])) as archive:
                v0 = torch.from_numpy(archive['V'].copy()).float().cuda()
                faces = torch.from_numpy(archive['F'].copy()).int().cuda()
            target_path = resolve(row['target_vertices'])
            if target_path.suffix == '.npy':
                v1 = torch.from_numpy(np.load(target_path)).float().cuda()
            else:
                with np.load(target_path) as archive:
                    v1 = torch.from_numpy(archive['V'].copy()).float().cuda()
            assert v0.shape == v1.shape
            transforms = torch.tensor(row['transform_matrix'], device='cuda', dtype=torch.float32)
            fovs = torch.tensor(row['camera_angle_x'], device='cuda', dtype=torch.float32)
            conversion = torch.diag(torch.tensor([1., -1., -1., 1.], device='cuda'))
            def render(vertices, view, attrs=None):
                ext = conversion @ torch.linalg.inv(transforms[view])
                intr = utils3d.torch.intrinsics_from_fov(fov_x=fovs[view], fov_y=fovs[view])
                return renderer.render(Mesh(vertices, faces, vertex_attrs=attrs), ext, intr,
                                       return_types=['mask', 'depth', 'coord'] + (['attr'] if attrs is not None else []))
            target = render(v1, 0, v0)
            expected = torch.from_numpy(np.array(Image.open(resolve(row['target_anchor'])).convert('RGBA'))[:, :, 3].copy()).cuda() > 127
            actual = target['mask'] > .5
            mask_iou = float((actual & expected).sum() / (actual | expected).sum())
            if mask_iou < .95:
                raise ValueError(f'Independent topology raster disagrees with M1 anchor: {mask_iou}')
            centers = torch.arange(8, 512, 16, device='cuda')
            yy, xx = torch.meshgrid(centers, centers, indexing='ij')
            foreground = (actual & expected)[yy, xx].flatten()
            source_points = target['attr'][:, yy, xx].permute(1, 2, 0).reshape(-1, 3)
            target_points = target['coord'][:, yy, xx].permute(1, 2, 0).reshape(-1, 3)
            gt_indices, visible = [], []
            for view in range(1, 4):
                source = render(v0, view)
                xy, depth, valid = project_points_to_image_batch(source_points[None], transforms[view:view+1], fovs[view:view+1], 512)
                xy, depth, valid = xy[0], depth[0], valid[0]
                pixel = xy.floor().long().clamp(0, 511)
                zbuffer = source['depth'][pixel[:, 1], pixel[:, 0]]
                visibility = foreground & valid & (source['mask'][pixel[:, 1], pixel[:, 0]] > .5) & ((zbuffer - depth).abs() < .005)
                patch = (xy / 16).floor().long().clamp(0, 31)
                gt_indices.append(patch[:, 1] * 32 + patch[:, 0] + (view - 1) * 1024)
                visible.append(visibility)
            gt = torch.stack(gt_indices, -1)
            visibility = torch.stack(visible, -1)
            eligible = visibility.any(-1)
            with np.load(resolve(row['transport_cache'])) as archive:
                old = torch.from_numpy(archive['spatial_indices'].copy()).cuda().long().T + torch.arange(3, device='cuda')[None] * 1024
                source_valid = torch.from_numpy(archive['source_valid'].copy()).cuda().bool().flatten()
                source_keys = torch.from_numpy(archive['source_keys'].copy()).cuda().float().reshape(-1, 2048)
                target_keys = torch.from_numpy(archive['target_keys'].copy()).cuda().float()
            with np.load(args.data_root / 'matches' / (row['sample_id'] + '.npz')) as archive:
                candidates = torch.from_numpy(archive['semantic_indices'].astype(np.int64)).cuda()
                logits = torch.from_numpy(archive['matching_logits'].copy()).cuda()
            valid_candidates = source_valid[candidates]
            logits = logits.masked_fill(~valid_candidates, -torch.inf)
            chosen = candidates.gather(1, logits.argmax(-1, keepdim=True)).squeeze(1)
            correct = (candidates[..., None] == gt[:, None]) & visibility[:, None]
            raw_correct = correct[:, 0].any(-1)
            q2c_correct = ((chosen[:, None] == gt) & visibility).any(-1)
            frame_scores = F.normalize(target_keys[:, :1024], dim=-1) @ F.normalize(source_keys[:, :1024], dim=-1).T
            frame_indices = frame_scores.masked_fill(~source_valid[None], -torch.inf).topk(64, dim=-1).indices
            frame_correct = (frame_indices[..., None] == gt[:, None]) & visibility[:, None]
            old_patch = old % 1024
            gt_patch = gt % 1024
            patch_distance = torch.stack((old_patch % 32 - gt_patch % 32,
                                          old_patch // 32 - gt_patch // 32), -1).float().norm(dim=-1)
            record = {'sample_id': row['sample_id'], 'category': row['category'], 'target_mask_iou': mask_iou,
                      'foreground_queries': int(foreground.sum()), 'visible_queries': int(eligible.sum()),
                      'visible_pairs': int(visibility.sum()),
                      'spatial_label_correct_pairs': int(((old == gt) & visibility).sum()),
                      'raw_top1_correct': int((raw_correct & eligible).sum()),
                      'q2c_top1_correct': int((q2c_correct & eligible).sum()),
                      'proposal_recall': int((correct.any(-1).any(-1) & eligible).sum()),
                      'q2c_changes': int(((chosen != candidates[:, 0]) & eligible).sum()),
                      'frame_top1_correct': int((frame_correct[:, 0].any(-1) & eligible).sum()),
                      'frame_proposal_recall': int((frame_correct.any(-1).any(-1) & eligible).sum()),
                      'frame_top1_changes': int(((frame_indices[:, 0] != candidates[:, 0]) & eligible).sum()),
                      'background_queries': int((~foreground).sum()),
                      'background_queries_with_source': int((~foreground & valid_candidates.any(-1)).sum()),
                      'mean_spatial_label_error_patches': float(patch_distance[visibility].mean()) if visibility.any() else None,
                      'spatial_label_error_over_one_patch_pairs': int(((patch_distance > 1) & visibility).sum()),
                      'mean_surface_displacement': float((source_points[foreground] - target_points[foreground]).norm(dim=-1).mean())}
            for key in counters:
                counters[key] += record[key]
            results.append(record)
            report = {'diagnostic_only': True, 'samples': len(results), 'visibility_depth_tolerance': .005,
                      'counts': counters, 'results': results}
            denominator = max(counters['visible_queries'], 1)
            report['metrics'] = {key: counters[key] / denominator for key in ['raw_top1_correct', 'q2c_top1_correct', 'proposal_recall', 'q2c_changes', 'frame_top1_correct', 'frame_proposal_recall', 'frame_top1_changes']}
            report['metrics']['spatial_labels_same_exact_patch_fraction'] = counters['spatial_label_correct_pairs'] / max(counters['visible_pairs'], 1)
            report['metrics']['background_assigned_source_fraction'] = counters['background_queries_with_source'] / max(counters['background_queries'], 1)
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(record), flush=True)
    print(json.dumps(report['metrics']), flush=True)


if __name__ == '__main__':
    main()
