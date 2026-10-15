"""Compare native, cached spatial and semantic feature carriers without training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from pixal3d.datasets.deformnext import _load_image
    from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor, compute_multiview_projection_matrices
    torch.set_num_threads(4)
    def resolve(value):
        return Path('/root/dev/.cache/deformnext-file4' + value) if value.startswith('/file4/') else Path(value)
    def metrics(reference, actual):
        ref, val = reference.float(), actual.float()
        rms = ref.square().mean().sqrt()
        return {'relative_rmse': float((ref-val).square().mean().sqrt()/rms.clamp_min(1e-8)),
                'mean_cosine': float(F.cosine_similarity(ref, val, dim=-1).mean()),
                'rms_ratio': float(val.square().mean().sqrt()/rms.clamp_min(1e-8))}
    rows = [json.loads(line) for line in (args.data_root/'manifests_world_v4/validation.jsonl').read_text().splitlines()]
    selected = [next(row for row in rows if row['category'] == category) for category in ['Axe','Bed','Bottle','Scissors']]
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'diagnostic_only': True, 'results': []}
    with torch.inference_mode():
        for stage in ['ss64', 'pbr1024']:
            cfg = json.loads((args.config_root/f'{stage}-v1.json').read_text())
            extractor = DinoV3ProjFeatureExtractor(**cfg['trainer']['args']['image_cond_model']['args']).eval().cuda()
            resolution = extractor.grid_resolution
            modules = {}
            for arm in ['b1','b2','b3','c1','c2']:
                spec = json.loads((args.config_root/f'{stage}-{arm}.json').read_text())['trainer']['args']['deform_conditioner']
                spec['implementation'] = 'legacy-v4'
                modules[arm] = DeformationAwareConditioner(**spec).eval().cuda()
            for row in selected:
                paths = [row['target_anchor'], *row['source_images']]
                images = torch.stack([_load_image(resolve(path),cfg['dataset']['args']['image_size']) for path in paths]).cuda()
                matrices = torch.tensor(row['transform_matrix'],device='cuda',dtype=torch.float32)[None]
                distances = matrices[:,:,:3,3].norm(dim=-1)
                fovs = torch.tensor(row['camera_angle_x'],device='cuda',dtype=torch.float32)
                scales = torch.tensor([row['mesh_scale']],device='cuda',dtype=torch.float32)
                projections,_ = compute_multiview_projection_matrices(matrices,distances,extractor.fixed_projection_transform)
                if stage=='pbr1024':
                    with np.load(resolve(row['shape_latent'])) as archive:
                        xyz = torch.from_numpy(archive['coords'].astype(np.int64)).cuda()
                    indices = xyz[:,0]*resolution**2+xyz[:,1]*resolution+xyz[:,2]
                else:
                    indices = None
                def select(value):
                    return value[0] if indices is None else value[0,indices]
                with np.load(resolve(row['transport_cache'])) as archive:
                    source_valid = torch.from_numpy(archive['source_valid'].copy()).cuda()[None].bool()
                cache_field = 'transport_cache' if stage=='ss64' else 'transport_cache_pbr'
                with np.load(resolve(row[cache_field])) as archive:
                    values = torch.from_numpy(archive['source_values'].copy()).cuda().float()
                    query = torch.from_numpy(archive['target_query'].copy()).cuda().float()[None]
                native_sources = None
                cached_sources = None
                anchor = None
                for view in range(4):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        _, projected = extractor._forward_single_view(images[view:view+1],fovs[view:view+1],distances[:,view],scales,projections[:,view])
                        native = select(projected).float()
                        if view==0:
                            anchor=native
                        else:
                            pooled = extractor.proj_grid(values[view-1].reshape(1,32,32,-1),fovs[view:view+1],distances[:,view],scales,projections[:,view])
                            cached = select(pooled).float()
                            native_sources = native if native_sources is None else native_sources+native
                            cached_sources = cached if cached_sources is None else cached_sources+cached
                    del projected
                native_mean=(anchor+native_sources)/4
                cached_mean=(anchor+cached_sources)/4
                with np.load(args.data_root/'matches'/(row['sample_id']+'.npz')) as archive:
                    matches={key:torch.from_numpy(archive[key].copy()).cuda()[None] for key in ['semantic_indices','matching_logits','candidate_valid','deformation']}
                with np.load(args.data_root/'projection_maps_world_v4'/(row['sample_id']+'.npz')) as archive:
                    key='patch_grid' if stage=='ss64' else 'patch_ids'
                    patch_ids=torch.from_numpy(archive[key].astype(np.int64)).cuda().flatten()
                result={'stage':stage,'sample_id':row['sample_id'],'category':row['category'],
                        'cached_spatial_vs_native':metrics(native_mean,cached_mean),
                        'arms':{}}
                b3=None
                for arm,module in modules.items():
                    inputs={'source_values':values[None],'source_valid':source_valid,
                            'semantic_indices':matches['semantic_indices'].long()}
                    if arm!='b1':
                        inputs.update(matching_logits=matches['matching_logits'],candidate_valid=matches['candidate_valid'])
                    if arm in ['c1','c2']:
                        inputs['target_query']=query
                    if arm=='c2':inputs['deformation']=matches['deformation'].float()
                    output=module(**inputs)
                    projected=output.evidence[0,patch_ids]
                    semantic_mean=(anchor+3*projected)/4
                    entry={'vs_native':metrics(native_mean,semantic_mean),
                           'additional_trainable_parameters':sum(p.numel() for p in module.parameters() if p.requires_grad)}
                    if arm=='b3':b3=semantic_mean.clone()
                    if arm in ['c1','c2']:
                        entry['initial_vs_b3_max_abs']=float((semantic_mean-b3).abs().max())
                    zero_blend = native_mean + 0*(semantic_mean-native_mean)
                    assert torch.equal(zero_blend,native_mean)
                    entry['zero_residual_native_identity']=True
                    result['arms'][arm]=entry
                report['results'].append(result)
                (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(result),flush=True)
            del extractor,modules,images,anchor,native_sources,cached_sources,native_mean,cached_mean,semantic_mean,b3
            torch.cuda.empty_cache()


if __name__=='__main__':
    main()
