"""Matched current TRAIN32 frozen-controller transfer of the original G2 loss.

This isolated128-update experiment does not change production LoRA/global96
training. Calibration GT opens only after the fixed final checkpoint freeze.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .cache_current_regions import SCHEMA
from .current_geometry_runtime import load_current_trainer, move_tensors
from .geometry_content_adapter import GeometryContentRetargetAdapter
from .historical_geometry_runtime import _seed, _seeded_noise, geometry_derangement
from .historical_reproduction import make_loss_regions, locked_endpoint_loss, differentiable_flow_euler_samples
from .quick_metrics import binary_metrics, _panel_image
from .reproduce_q2c import checked, read, reference, write_new

ARMS = ('adapter-only', 'adapter-query')


def training_order(rows, seed, update):
    if len(rows) != 32:
        raise ValueError('regional controller pilot requires all32 fixed TRAIN pairs')
    epoch, position = divmod(update, len(rows))
    ordered = sorted(rows, key=lambda row: hashlib.sha256(
        f"{seed}|{epoch}|{row['sample_id']}".encode()).hexdigest())
    row = ordered[position]
    return row, epoch, _seed('ss-geometry-factorial', seed, row['sample_id'], epoch)


def load_pair(row, labels, *, role, with_query, consumer, device):
    # Reject a role before opening any of its tensor or GT files.
    if row['role'] != role or labels['role'] != role or row['sample_id'] != labels['sample_id']:
        raise ValueError('controller train/evaluator role or identity mismatch')
    saved = torch.load(checked(row['artifact']), map_location='cpu', weights_only=True)
    if (saved['consumer'] != consumer or saved['role'] != role or saved['sample_id'] != row['sample_id']
            or saved['source_cluster_id'] != labels['source_cluster_id']):
        raise ValueError('current cached native condition/label consumer differs')
    with np.load(checked(labels['artifact']), allow_pickle=False) as data:
        if (str(data['schema']) != SCHEMA or str(data['role']) != role
                or str(data['sample_id']) != row['sample_id'] or not bool(data['supervision_only'])
                or str(data['frame_contract']) != labels['frame_contract']):
            raise ValueError('current native1024 GT authority differs')
        arrays = {key: data[key].copy() for key in ['source64', 'target64', 'free64', 'surface64',
            'matched_source_exclusive64', 'matched_source_exclusive32', 'stale_visible64', 'stale_visible32']}
    target = torch.zeros((1, 1, 64, 64, 64), dtype=torch.bool, device=device)
    coords = torch.as_tensor(arrays['target64'], device=device).long()
    target[0, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = True
    geometry = None
    if with_query:
        with np.load(checked(row['geometry_query']), allow_pickle=False) as data:
            if str(data['sample_id']) != row['sample_id'] or data['geometry9'].shape != (4096, 9):
                raise ValueError('predicted RGB-only O/D query differs')
            geometry = torch.as_tensor(data['geometry9'].copy(), device=device)[None]
    return {'sample_id': row['sample_id'], 'source_cluster_id': labels['source_cluster_id'],
        'native': move_tensors(saved['native_condition'], device), 'target_proj': saved['target_proj'].to(device),
        'geometry': geometry, 'regions': move_tensors(make_loss_regions(arrays, arrays), device),
        'targets': {64: target, 32: F.max_pool3d(target.float(), 2, 2).bool()}}


def condition(pair, adapter, *, source_off=False, shuffled=False):
    if shuffled and pair['geometry'] is None:
        raise ValueError('query shuffle requires physical query channels')
    geometry = pair['geometry']
    if shuffled:
        geometry = geometry[:, geometry_derangement(pair['sample_id']).to(geometry.device)]
    original = pair['native']
    source = pair['target_proj'] if source_off else original['cond']['proj']
    projected, trace = adapter(pair['target_proj'], source, geometry)
    return {'cond': {**original['cond'], 'proj': projected}, 'neg_cond': original['neg_cond']}, trace


def score(logits, pair):
    support = logits > 0
    result = binary_metrics(support, pair['targets'][64])
    gt = pair['targets'][64]
    result.update(tp=int((support & gt).sum()), fp=int((support & ~gt).sum()), fn=int((~support & gt).sum()),
        target_voxels_sha256=hashlib.sha256(np.packbits(gt.cpu().numpy().reshape(-1)).tobytes()).hexdigest(),
        predicted_voxels_sha256=hashlib.sha256(np.packbits(support.cpu().numpy().reshape(-1)).tobytes()).hexdigest())
    loss, _ = locked_endpoint_loss(logits, pair['regions'])
    result['regional_bce'] = float(loss)
    for resolution in [64, 32]:
        predicted = support if resolution == 64 else F.max_pool3d(logits, 2, 2) > 0
        masks = pair['regions'][resolution]
        for label, mask_name in [('source_recall', 'source_positive'), ('stale_leak', 'stale')]:
            mask = masks[mask_name]
            result[label+str(resolution)] = float(predicted[mask].float().mean()) if mask.any() else None
        positive, negative = masks['visible_positive'], masks['visible_negative']
        result['visible_state_acc'+str(resolution)] = (float((predicted[positive].float().mean() +
            (~predicted[negative]).float().mean())/2) if positive.any() and negative.any() else None)
    return result


def evaluate(pairs, adapter, flow, decoder, sampler, dtype, output, run, generation_batch):
    cases = ['native', 'normal', 'projected-source-off'] + (['shuffled-query'] if adapter.with_query else [])
    result = {}
    with torch.inference_mode(), torch.autocast('cuda', dtype=dtype):
        for case in cases:
            rows = []
            for noise_index in [0, 1, 2]:
                for start in range(0, len(pairs), generation_batch):
                    batch = pairs[start:start+generation_batch]
                    conditions = [pair['native'] if case == 'native' else condition(pair, adapter,
                        source_off=case == 'projected-source-off', shuffled=case == 'shuffled-query')[0] for pair in batch]
                    merged = {branch: {key: torch.cat([one[branch][key] for one in conditions])
                        for key in conditions[0][branch]} for branch in ['cond', 'neg_cond']}
                    noise = torch.cat([_seeded_noise(pair['sample_id'], noise_index,
                        shape=(1, 8, 16, 16, 16), device='cuda:0') for pair in batch])
                    generated = sampler.sample(flow, noise=noise, **merged, steps=12, rescale_t=1.,
                        guidance_strength=3., verbose=False).samples
                    logits = decoder(generated).float()
                    if noise_index == 0 and start == 0:
                        from PIL import Image
                        count = min(4, len(batch))
                        pred = (logits[:count, 0] > 0).float().amax(1, keepdim=True).repeat(1, 3, 1, 1)
                        gt = torch.cat([p['targets'][64] for p in batch[:count]])[:, 0].float().amax(
                            1, keepdim=True).repeat(1, 3, 1, 1)
                        image_path = output/(case+'-GT-Pred-Diff.png')
                        Image.fromarray(_panel_image(pred, gt)).save(image_path)
                        if run:
                            import wandb
                            run.log({'calibration/'+case+'/GT_Pred_Diff': wandb.Image(str(image_path),
                                caption='Y voxel max-projection: actual encoder GT / generated / abs diff; first four fixed cal pairs')})
                    for index, pair in enumerate(batch):
                        rows.append({'sample_id': pair['sample_id'], 'source_cluster_id': pair['source_cluster_id'],
                            'noise_index': noise_index, **score(logits[index:index+1], pair)})
                    print(json.dumps({'evaluation_case': case, 'noise': noise_index,
                        'completed': len(rows), 'of': 96}), flush=True)
            write_new(output/'evaluation'/(case+'.json'), {'status': 'PASS', 'case': case, 'per_asset': rows,
                'target_representation': 'actual SS encoder input native1024 pooled64, not decoded-latent GT of primary14',
                'voxel_iou_aggregation': 'macro over fixed asset/noise rows, not primary14 micro IoU',
                'sampling_steps': 12, 'guidance': 3., 'rescale_t': 1., 'generation_batch': generation_batch})
            result[case] = {metric: float(np.mean([r[metric] for r in rows if r[metric] is not None]))
                for metric in ['voxel_iou', 'voxel_f1', 'source_recall32', 'stale_leak32', 'visible_state_acc32', 'regional_bce']
                if any(r[metric] is not None for r in rows)}
            if run:
                run.log({f'calibration/{case}/{key}': value for key, value in result[case].items()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'checkpoint', 'manifest', 'conditions', 'regions', 'gradient-proof', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, choices=[101, 202, 303], required=True)
    parser.add_argument('--generation-batch', type=int, default=32)
    parser.add_argument('--wandb-id')
    parser.add_argument('--cpu-affinity', help='Explicit comma-separated permitted CPU cores; defaults to the last two inherited cores')
    args = parser.parse_args()
    if hasattr(os, 'sched_getaffinity'):
        available = os.sched_getaffinity(0)
        assigned = [int(value) for value in args.cpu_affinity.split(',')] if args.cpu_affinity else sorted(available)[-2:]
        if len(assigned) != 2 or len(set(assigned)) != 2 or not set(assigned) <= available:
            raise ValueError('controller requires two distinct permitted CPU cores')
        os.sched_setaffinity(0, assigned)
    torch.set_num_threads(2); torch.cuda.set_device(0)
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output/name).exists() for name in ['summary.json', 'checkpoint-final.pt', 'train-log.jsonl']):
        raise FileExistsError('preserve the earlier controller attempt')
    proof, cache, labels = read(args.gradient_proof), read(args.conditions/'summary.json'), read(args.regions/'summary.json')
    consumer = reference(args.checkpoint)
    if (proof['status'] != 'PASS' or proof['consumer'] != consumer or cache['status'] != 'PASS'
            or labels['status'] != 'PASS' or labels['schema'] != SCHEMA
            or cache['consumer'] != consumer or cache['population'] != labels['population']):
        raise ValueError('current native1024 region/condition/actual gradient gates have not passed')
    if read(checked(proof['region_preflight']))['schema'] != SCHEMA:
        raise ValueError('gradient proof used another GT authority')
    label_rows = {row['sample_id']: row for row in labels['records']}
    training = [row for row in cache['records'] if row['role'] == 'train']
    calibration = [row for row in cache['records'] if row['role'] == 'calibration']
    if len(training) != 32 or len(calibration) != 32 or len(label_rows) != 64:
        raise ValueError('current fixed TRAIN32/cal32 population differs')
    if {row['source_cluster_id'] for row in training} & {row['source_cluster_id'] for row in calibration}:
        raise ValueError('source cluster split overlaps')
    torch.manual_seed(args.seed)
    trainer, dataset = load_current_trainer(read(args.config), args.checkpoint, args.manifest, args.output/'runtime')
    dataset._loading_ss_dec(); decoder = dataset.ss_dec.eval().requires_grad_(False)
    flow, sampler = trainer.models['denoiser'], trainer.get_sampler()
    adapter = GeometryContentRetargetAdapter(with_query=args.arm == 'adapter-query').cuda().train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=.0001, weight_decay=.01)
    pairs = {row['sample_id']: load_pair(row, label_rows[row['sample_id']], role='train',
        with_query=adapter.with_query, consumer=consumer, device='cuda:0') for row in training}
    run, started = None, time.monotonic()
    if args.wandb_id:
        import wandb
        run = wandb.init(project='pixal3d-deformnext-retargeting', entity='youngwoojeon-unist',
            id=args.wandb_id, resume='never', dir=str(args.output), config={
                'scope': 'current TRAIN32 frozen-controller original regional loss transfer pilot',
                'arm': args.arm, 'seed': args.seed, 'updates': 128, 'controller_asset_batch': 1,
                'production_global96_recipe_changed': False, 'V1_retrained': False,
                'consumer': consumer, 'sampler': 'native12 CFG3 rescale1', 'LoRA_consumer_frozen': True,
                'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                'calibration_open_after_fixed_final_freeze': True})
    with (args.output/'train-log.jsonl').open('x') as stream:
        for update in range(128):
            row, epoch, noise_seed = training_order(training, args.seed, update)
            pair = pairs[row['sample_id']]
            generator = torch.Generator(device='cpu').manual_seed(noise_seed)
            noise = torch.randn((1, 8, 16, 16, 16), generator=generator).cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
                current, trace = condition(pair, adapter)
                if update == 0 and not torch.equal(current['cond']['proj'], pair['native']['cond']['proj']):
                    raise ValueError('controller regressed initial native projection')
                latent = differentiable_flow_euler_samples(sampler, flow, noise, current,
                    steps=12, rescale_t=1., checkpoint_model=True, guidance_strength=3.)
                endpoint, _ = locked_endpoint_loss(decoder(latent).float(), pair['regions'])
                correction = trace['correction'].float().square().mean()
                loss = endpoint + .001 * correction
            if not torch.isfinite(loss):
                raise ValueError('nonfinite original regional objective')
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.))
            if not np.isfinite(gradient) or gradient == 0:
                raise ValueError('missing/nonfinite controller gradient')
            if any(p.grad is not None for model in [flow, decoder] for p in model.parameters()):
                raise ValueError('frozen current SS consumer acquired gradients')
            optimizer.step()
            record = {'update': update+1, 'epoch': epoch, 'sample_id': row['sample_id'], 'noise_seed': noise_seed,
                'loss': float(loss.detach()), 'endpoint_loss': float(endpoint.detach()),
                'correction_loss': float(correction.detach()), 'gradient_norm': gradient,
                'seconds': time.monotonic()-started, 'peak_GPU_allocated_gib': torch.cuda.max_memory_allocated()/1024**3}
            stream.write(json.dumps(record)+'\n'); stream.flush()
            if update == 0 or (update+1) % 8 == 0:
                print(json.dumps(record), flush=True)
                if run: run.log({'train/'+key: value for key, value in record.items() if isinstance(value, (int, float))})
    checkpoint = args.output/'checkpoint-final.pt'
    torch.save({'adapter': adapter.state_dict(), 'optimizer': optimizer.state_dict(), 'arm': args.arm,
        'seed': args.seed, 'updates': 128, 'train_pairs': sorted(pairs), 'consumer': consumer,
        'conditions': reference(args.conditions/'summary.json'), 'regions': reference(args.regions/'summary.json'),
        'scope': 'unpromoted current frozen regional controller, not a production LoRA checkpoint'}, checkpoint)
    write_new(args.output/'checkpoint-freeze.json', {'checkpoint': reference(checkpoint),
        'calibration_GT_values_opened': False, 'completed_updates': 128})
    del pairs
    adapter.eval().requires_grad_(False)
    heldout = [load_pair(row, label_rows[row['sample_id']], role='calibration', with_query=adapter.with_query,
        consumer=consumer, device='cuda:0') for row in calibration]
    metrics = evaluate(heldout, adapter, flow, decoder, sampler, trainer.mix_precision_dtype,
        args.output, run, args.generation_batch)
    write_new(args.output/'summary.json', {'status': 'PASS', 'scope': 'current frozen-controller regional pilot',
        'arm': args.arm, 'seed': args.seed, 'updates': 128, 'checkpoint': reference(checkpoint),
        'train_clusters': 32, 'calibration_clusters': 32, 'calibration_noise_seeds': 3,
        'metrics': metrics, 'seconds': time.monotonic()-started, 'production_promotion': False,
        'production_global96_recipe_changed': False, 'main14_comparison_replaced': False})
    if run:
        run.summary.update({'status': 'PASS', 'production_promotion': False}); run.finish()


if __name__ == '__main__':
    main()
