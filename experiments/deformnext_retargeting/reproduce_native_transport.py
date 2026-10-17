"""Run the historical per-source-view transport through current native SS.

All cases are GT diagnostic-only, including the native replay reference. They
cannot enter the14-arm primary collector as production inference results.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .historical_reproduction import route_tensors
from .reproduction_transport_context import MODES, native_slot_context
from .quick_metrics import binary_metrics, panel_indices, panel_seeds, _panel_image
from .cache_reproduction_routes import SCHEMA, frame_contract


def load_routes(root, rows, device):
    grids, masks, regions = [], [], []
    for row in rows:
        with np.load(root / 'data' / (row['sample_id'] + '.npz')) as data:
            if (str(data['schema']) != SCHEMA
                    or str(data['sample_id']) != row['sample_id']
                    or str(data['conditioning_input_sha256']) != row['conditioning_input_sha256']
                    or str(data['frame_contract']) != frame_contract(row)
                    or not bool(data['diagnostic_only'])):
                raise ValueError('GT route/cache input identity mismatch')
            grid, mask = route_tensors(data, device=device)
            grids.append(grid)
            masks.append(mask)
            regions.append({r: data[f'route_hidden_coords{r}'].copy() for r in [64, 32]})
    return torch.cat(grids), torch.cat(masks), regions


def region_metrics(prediction, target, coords):
    """Sampled routed hidden surface, explicitly distinct from full SS regions."""
    result = {}
    for resolution in [64, 32]:
        pred = prediction[None, None].float()
        gt = target[None, None].float()
        if resolution == 32:
            pred = torch.nn.functional.max_pool3d(pred, 2, 2)
            gt = torch.nn.functional.max_pool3d(gt, 2, 2)
        index = torch.from_numpy(coords[resolution]).to(prediction.device).long()
        if index.ndim != 2 or index.shape[1] != 3 or torch.any((index < 0) | (index >= resolution)):
            raise ValueError('sampled hidden surface coordinates differ from native SS axes')
        observed = gt[0, 0, index[:, 0], index[:, 1], index[:, 2]].bool()
        hit = pred[0, 0, index[:, 0], index[:, 1], index[:, 2]].bool() & observed
        count = int(observed.sum())
        result[f'route_sampled_hidden_gt_count{resolution}'] = count
        result[f'route_sampled_hidden_surface_cells{resolution}'] = len(index)
        result[f'route_sampled_hidden_gt_tp{resolution}'] = int(hit.sum())
        result[f'route_sampled_hidden_gt_recall{resolution}'] = int(hit.sum()) / count if count else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--routes', type=Path, required=True)
    parser.add_argument('--panel-file', type=Path, required=True)
    parser.add_argument('--seed-map-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', default=','.join(MODES))
    parser.add_argument('--generation-batch', type=int, default=4)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    cases = args.cases.split(',')
    if len(set(cases)) != len(cases) or not set(cases) <= set(MODES):
        raise ValueError('invalid or duplicate reproduction cases')
    if args.generation_batch < 1:
        raise ValueError('positive generation batch required')
    route_summary = json.loads((args.routes / 'summary.json').read_text())
    if route_summary['status'] != 'PASS':
        raise ValueError('three-view GT route construction has not passed')
    from pixal3d import datasets, models, trainers
    from pixal3d.utils.data_utils import recursive_to_device
    from PIL import Image
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    cfg = json.loads(args.config.read_text())
    ds_args = copy.deepcopy(cfg['validation']['dataset_args'])
    ds_args['transport_arm'] = 'v1'
    for key in ['match_root', 'projection_root', 'deformation_root', 'expected_q2c_sha256']:
        ds_args.pop(key, None)
    dataset = getattr(datasets, cfg['dataset']['name'])(cfg['validation']['roots_file'], **ds_args)
    quick = cfg['validation']['quick_metrics']
    quick.update(panel_file=str(args.panel_file), seed_map_file=str(args.seed_map_file))
    indices = panel_indices(dataset, quick)
    seeds = panel_seeds(dataset, indices, quick)
    if set(route_summary['records'][i]['sample_id'] for i in range(route_summary['samples'])) != {
            dataset.rows[i]['sample_id'] for i in indices}:
        raise ValueError('evaluation and materialized route populations differ')
    with torch.device('meta'):
        model_dict = {name: getattr(models, spec['name'])(**spec['args'])
                      for name, spec in cfg['models'].items()}
    kw = copy.deepcopy(cfg['trainer']['args'])
    kw.update(continuation_ckpt={'denoiser': str(args.checkpoint)}, deform_conditioner=None,
              batch_size_per_gpu=1, batch_split=1, num_workers=0, prefetch_data=False, max_steps=1)
    kw.get('model_learning_rates', {}).pop('deform_conditioner', None)
    args.output.mkdir(parents=True, exist_ok=True)
    trainer = getattr(trainers, cfg['trainer']['name'])(model_dict, dataset, **kw,
                output_dir=str(args.output), load_dir=None, step=None, wandb_run=None)
    trainer.models['denoiser'].eval().requires_grad_(False)
    trainer.p_uncond = 0
    summaries = {}
    for case in cases:
        out = args.output / case
        if (out / 'evaluation.json').exists():
            raise FileExistsError('immutable reproduction evaluation already exists: ' + str(out))
        out.mkdir(parents=True, exist_ok=True)
        predictions, targets, all_regions, events = [], [], [], []
        current_indices = indices[:1] if args.preflight_only else indices
        for start in range(0, len(current_indices), args.generation_batch):
            selected = current_indices[start:start + args.generation_batch]
            rows = [dataset.rows[i] for i in selected]
            pack = dataset.collate_fn([dataset[i] for i in selected])
            if isinstance(pack, list):
                if len(pack) != 1:
                    raise ValueError('evaluation collator unexpectedly split the fixed generation batch')
                pack = pack[0]
            pack = recursive_to_device(pack, trainer.device)
            target = pack.pop('x_0')
            pack.pop('_dataset_name', None)
            pack.pop('_sha256', None)
            grid, valid, regions = load_routes(args.routes, rows, trainer.device)
            noises = []
            for offset in range(len(selected)):
                seed = seeds[start + offset]
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                noises.append(torch.randn_like(target[offset:offset+1]))
            with torch.no_grad(), torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
                with native_slot_context(trainer, case, grid, valid,
                                         [row['sample_id'] for row in rows], events):
                    condition = trainer.get_inference_cond(**pack)
                if not args.preflight_only:
                    predicted = trainer.get_sampler().sample(trainer.models['denoiser'],
                        noise=torch.cat(noises), **condition, steps=12,
                        guidance_strength=3., verbose=False).samples
                    predictions.append(predicted)
                    targets.append(target)
                    all_regions.extend(regions)
            print(f'REPRODUCTION {case} completed {min(start + len(selected), len(current_indices))}/{len(current_indices)}', flush=True)
        if args.preflight_only:
            record = {'status': 'PASS_PREFLIGHT_ONLY', 'diagnostic_only': True,
                      'case': case, 'sampling_performed': False, 'events': events}
        else:
            count = len(indices)
            with torch.no_grad(), torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
                decoded = dataset.decode_latent(torch.cat(predictions + targets), batch_size=2)
            pred_occ, target_occ = decoded[:count, 0] > 0, decoded[count:, 0] > 0
            per_asset = []
            for index, seed, pred, gt, region in zip(indices, seeds, pred_occ, target_occ, all_regions, strict=True):
                row = dataset.rows[index]
                per_asset.append({'sample_id': row['sample_id'], 'source_cluster_id': row['source_cluster_id'],
                    'seed': seed, 'tp': int((pred & gt).sum()), 'fp': int((pred & ~gt).sum()),
                    'fn': int((~pred & gt).sum()), **binary_metrics(pred, gt), **region_metrics(pred, gt, region),
                    'target_voxels_sha256': hashlib.sha256(np.packbits(gt.cpu().numpy().reshape(-1)).tobytes()).hexdigest(),
                    'prediction_voxels_sha256': hashlib.sha256(np.packbits(pred.cpu().numpy().reshape(-1)).tobytes()).hexdigest(),
                    'target_voxel_shape': list(gt.shape)})
            pred_panel = pred_occ.float().amax(1, keepdim=True).repeat(1, 3, 1, 1)
            gt_panel = target_occ.float().amax(1, keepdim=True).repeat(1, 3, 1, 1)
            Image.fromarray(_panel_image(pred_panel, gt_panel)).save(out / 'panel.png')
            record = {'status': 'PASS', 'diagnostic_only': True, 'GT_addresses': case in ['gt-corr', 'wrong-corr'],
                      'case': case, 'checkpoint': str(args.checkpoint), 'sampling_steps': 12, 'guidance': 3.,
                      'optimizer_updates_during_evaluation': 0, 'per_asset': per_asset,
                      'quick_metrics': binary_metrics(pred_occ, target_occ), 'events': events,
                      'input_view_count': 4, 'input_states': ['M1', 'M0', 'M0', 'M0'],
                      'geometry_protocol': 'per-source-camera GT slots before unchanged native ProjGrid',
                      'regional_metric_scope': 'sampled routed hidden surface cells intersected with decoded target SS; not full historical SS region authority',
                      'route_summary_sha256': hashlib.sha256((args.routes / 'summary.json').read_bytes()).hexdigest()}
        (out / 'evaluation.json').write_text(json.dumps(record, indent=2) + '\n')
        summaries[case] = {k: record[k] for k in ['status', 'quick_metrics'] if k in record}
        print(json.dumps({'case': case, **summaries[case]}), flush=True)
    (args.output / 'summary.json').write_text(json.dumps({'execution_status': 'PASS',
        'diagnostic_only': True, 'performance_promotion': False, 'cases': summaries}, indent=2) + '\n')


if __name__ == '__main__':
    main()
