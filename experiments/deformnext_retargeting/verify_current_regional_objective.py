"""Actual frozen native rollout/regional-BCE proof on one current TRAIN pair."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .current_geometry_runtime import load_current_trainer, move_tensors
from .geometry_content_adapter import GeometryContentRetargetAdapter
from .historical_geometry_runtime import _seed
from .historical_reproduction import make_loss_regions, locked_endpoint_loss, differentiable_flow_euler_samples
from .cache_current_regions import SCHEMA
from .reproduce_q2c import checked, read, reference, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'checkpoint', 'manifest', 'conditions', 'region-preflight', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--wandb-id')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    if (args.output/'summary.json').exists():
        raise FileExistsError('immutable regional-objective proof exists')
    condition_summary, region_summary = read(args.conditions/'summary.json'), read(args.region_preflight)
    if (condition_summary['status'] != 'PASS' or region_summary['status'] != 'PASS_PREFLIGHT_ONLY'
            or region_summary['schema'] != SCHEMA or len(region_summary['records']) != 1):
        raise ValueError('actual current condition/region preflight has not passed')
    label = region_summary['records'][0]
    if label['role'] != 'train':
        raise ValueError('regional gradient proof must use a TRAIN pair')
    row = next(r for r in condition_summary['records'] if r['sample_id'] == label['sample_id'])
    if row['role'] != 'train' or row['source_cluster_id'] != label['source_cluster_id']:
        raise ValueError('native condition and regional label identities differ')
    saved = torch.load(checked(row['artifact']), map_location='cpu', weights_only=True)
    if saved['consumer'] != reference(args.checkpoint) or saved['role'] != 'train':
        raise ValueError('current native frozen consumer differs')
    with np.load(checked(label['artifact']), allow_pickle=False) as data:
        if str(data['schema']) != SCHEMA or not bool(data['supervision_only']):
            raise ValueError('regional GT authority differs')
        arrays = {key: data[key].copy() for key in ['source64', 'target64', 'free64', 'surface64',
            'matched_source_exclusive64', 'matched_source_exclusive32', 'stale_visible64', 'stale_visible32']}
    regions = move_tensors(make_loss_regions(arrays, arrays), 'cuda:0')
    with np.load(checked(row['geometry_query']), allow_pickle=False) as query:
        if str(query['sample_id']) != row['sample_id']:
            raise ValueError('predicted query identity differs')
        geometry = torch.from_numpy(query['geometry9'].copy())[None].cuda()
    native = move_tensors(saved['native_condition'], 'cuda:0')
    target_proj = saved['target_proj'].cuda()
    trainer, dataset = load_current_trainer(read(args.config), args.checkpoint, args.manifest, args.output/'runtime')
    dataset._loading_ss_dec()
    decoder = dataset.ss_dec.eval().requires_grad_(False)
    flow, sampler = trainer.models['denoiser'], trainer.get_sampler()
    generator = torch.Generator(device='cpu').manual_seed(_seed('ss-geometry-factorial', 101, row['sample_id'], 0))
    initial_noise = torch.randn((1, 8, 16, 16, 16), generator=generator).cuda()
    with torch.no_grad(), torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
        latent = sampler.sample(flow, noise=initial_noise.clone(), **native, steps=12,
            guidance_strength=3., rescale_t=1., verbose=False).samples
        baseline, _ = locked_endpoint_loss(decoder(latent).float(), regions)
    cases, run_log = [], None
    if args.wandb_id:
        import wandb
        args.output.mkdir(parents=True, exist_ok=True)
        run_log = wandb.init(project='pixal3d-deformnext-retargeting', entity='youngwoojeon-unist',
            id=args.wandb_id, resume='never', dir=str(args.output), config={
                'scope': 'one TRAIN pair original regional objective actual GPU proof',
                'sampler': 'current native12steps CFG3 rescale1', 'SS_consumer': reference(args.checkpoint),
                'sample_id': row['sample_id'], 'query_GT_geometry': False,
                'production_global96_recipe_changed': False})
    for with_query in [False, True]:
        torch.manual_seed(101)
        adapter = GeometryContentRetargetAdapter(with_query=with_query).cuda().train()
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=.0001, weight_decay=.01)
        history, started = [], time.monotonic()
        for update in range(2):
            generator = torch.Generator(device='cpu').manual_seed(_seed('ss-geometry-factorial', 101, row['sample_id'], update))
            noise = torch.randn((1, 8, 16, 16, 16), generator=generator).cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
                projected, trace = adapter(target_proj, native['cond']['proj'], geometry if with_query else None)
                if update == 0 and not torch.equal(projected, native['cond']['proj']):
                    raise ValueError('initial adapter has regressed native projected input')
                condition = {'cond': {**native['cond'], 'proj': projected}, 'neg_cond': native['neg_cond']}
                latent = differentiable_flow_euler_samples(sampler, flow, noise, condition,
                    steps=12, rescale_t=1., checkpoint_model=True, guidance_strength=3.)
                endpoint, _ = locked_endpoint_loss(decoder(latent).float(), regions)
                correction = trace['correction'].float().square().mean()
                loss = endpoint + .001 * correction
            if update == 0 and not torch.equal(endpoint, baseline):
                raise ValueError(f'initial differentiable regional loss differs from native sampler: {float(endpoint)} vs {float(baseline)}')
            if not torch.isfinite(loss):
                raise ValueError('nonfinite actual regional rollout loss')
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.))
            if not np.isfinite(gradient) or gradient == 0:
                raise ValueError('regional native loss has no finite/nonzero adapter gradient')
            if any(p.grad is not None for model in [flow, decoder] for p in model.parameters()):
                raise ValueError('frozen SS/decoder acquired parameter gradients')
            query_gradient = (sum(float(p.grad.abs().sum()) for p in adapter.content_encoder.parameters()
                if p.grad is not None) if with_query else None)
            if with_query and update == 1 and not query_gradient > 0:
                raise ValueError('predicted query disconnected from full native regional loss')
            record = {'update': update+1, 'regional_endpoint_loss': float(endpoint.detach()),
                'correction_penalty_unweighted': float(correction.detach()), 'loss': float(loss.detach()),
                'gradient_norm_before_clip': gradient, 'query_encoder_gradient_L1_after_clip': query_gradient,
                'initial_native_regional_loss_exact': torch.equal(endpoint, baseline) if update == 0 else None,
                'seconds': time.monotonic()-started, 'peak_GPU_allocated_gib': torch.cuda.max_memory_allocated()/1024**3}
            history.append(record); optimizer.step()
            print(json.dumps({'with_query': with_query, **record}), flush=True)
            if run_log:
                run_log.log({('query' if with_query else 'adapter')+'/'+k: v for k,v in record.items() if v is not None})
        cases.append({'with_query': with_query, 'raw_query_channels': 9 if with_query else 0,
            'residual_input_channels': adapter.residual[0].normalized_shape[0], 'history': history})
    write_new(args.output/'summary.json', {'status': 'PASS', 'scope': 'actual original regional objective transfer gradient proof',
        'sample_id': row['sample_id'], 'cases': cases, 'native_baseline_regional_loss': float(baseline),
        'consumer': reference(args.checkpoint), 'condition': reference(args.conditions/'summary.json'),
        'region_preflight': reference(args.region_preflight), 'frozen_SS_and_decoder_gradients_absent': True,
        'quality_evaluation_performed': False, 'production_global96_recipe_changed': False,
        'new_five_epoch_training_started': False, 'SS_generation_gain_claim': False})
    if run_log:
        run_log.summary.update({'status': 'PASS', 'SS_gain_claim': False}); run_log.finish()


if __name__ == '__main__':
    main()
