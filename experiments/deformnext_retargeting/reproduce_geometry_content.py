"""Replay G2 content math, real frozen-rollout gradients and archived endpoints.

This isolated reference uses its recorded historical consumer and masks. The
production mixed385 LoRA initializer and DeformNeXt evaluation stay separate.
Only the two feature/content reference arms are exposed, not a 2x2 experiment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from .historical_reproduction import (
    GeometryFactorialVoxelAdapter, geometry_from_logits, make_loss_regions,
    coords_to_mask, locked_endpoint_loss, differentiable_flow_euler_samples,
)
from .historical_geometry_runtime import adapted_condition, _score, _seed, _seeded_noise, geometry_derangement
from .reproduce_q2c import checked, read, reference, write_new, sha

ARMS = ['r2_feat', 'r4_content']


def copy_verified(source, target, expected=None):
    if expected is not None and sha(source) != expected:
        raise ValueError('historical runtime/source checksum differs: ' + str(source))
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha(source) != sha(target):
        raise ValueError('staged reference file differs')
    return reference(target)


def prepare(root, checkpoint, output):
    if (output / 'summary.json').exists():
        raise FileExistsError('immutable G2 input export exists')
    archived = torch.load(checkpoint, map_location='cpu', weights_only=False)
    runtime = archived['runtime']
    flow = runtime['flow_spec']
    flow_config = Path(flow['config_path'])
    if sha(flow_config) != flow['config_sha256']:
        raise ValueError('recorded full-FT reference model architecture changed')
    if (flow['step'], flow['weight_kind']) != (20000, 'ema'):
        raise ValueError('unexpected archived G2 consumer; never substitute the current LoRA model')
    model_spec = read(flow_config)['models']['denoiser']
    pipeline_file = Path(runtime['pipeline_snapshot']) / 'pipeline.json'
    pipeline = read(pipeline_file)['args']
    decoder = Path(runtime['pipeline_snapshot']) / pipeline['models']['sparse_structure_decoder']
    runtime_out = {'scope': 'archived-full-FT-reference-only; not current mixed385-LoRA initialization',
                   'model_spec': model_spec, 'original_flow': flow,
                   'original_pipeline': reference(pipeline_file),
                   'sampler': pipeline['sparse_structure_sampler'],
                   'flow_checkpoint': copy_verified(Path(flow['path']), output / 'runtime/ss64-reference.pt', flow['sha256']),
                   'decoder_prefix': str(output / 'runtime/ss-decoder'),
                   'decoder_files': [copy_verified(Path(str(decoder) + ext), output / ('runtime/ss-decoder' + ext))
                                     for ext in ['.json', '.safetensors']]}
    write_new(output / 'runtime.json', runtime_out)
    manifest = read(root / 'ss_tto_adapter0/split_manifest.json')
    rows = sorted(manifest['records'], key=lambda r: r['pair_id'])
    if len([r for r in rows if r['split'] == 'train']) != 32 or len([r for r in rows if r['split'] == 'dev']) != 8:
        raise ValueError('historical TRAIN32/DEV8 split differs')
    records = []
    for row in rows:
        pair = row['pair_id']
        pair_path = root / 'ss_static_ray0/CACHE_HIDDEN_ONLY/pairs' / (pair + '.pt')
        payload = torch.load(pair_path, map_location='cpu', weights_only=True)
        pair_hash = sha(pair_path)
        if payload['pair_id'] != pair or payload['split'] != row['split']:
            raise ValueError('historical condition identity/split differs')
        u0 = root / 'ss_union_surface_q2c0/U0_SCAFFOLD'
        u0_row = read(u0 / 'rows' / (pair + '-seed0.json'))
        if (u0_row['status'], u0_row['split'], u0_row['seed'], u0_row['cache_sha256']) != (
                'completed', row['split'], 0, pair_hash):
            raise ValueError('predicted U0 geometry lineage differs')
        if ([r['state'] for r in u0_row['original_input_views']],
                [r['state'] for r in u0_row['desired_input_views']]) != (['source'] * 4, ['target']):
            raise ValueError('predicted O/D geometry image state differs')
        u0_file = checked({'path': u0_row['artifact'], 'sha256': u0_row['artifact_sha256']})
        with np.load(u0_file, allow_pickle=False) as data:
            geometry = geometry_from_logits(data['original_logit64'], data['desired_logit64'])
        authority_path = root / 'ss_tto_adapter0/S4_CACHE/pairs' / (pair + '.pt')
        authority = torch.load(authority_path, map_location='cpu', weights_only=True)
        if authority['pair_id'] != pair:
            raise ValueError('historical full SS authority differs')
        region_path = root / 'ss_premean_asym0/SS_SOURCE_UTILITY_AUDIT1/MATCHED_REGIONS/data' / (pair + '.npz')
        with np.load(region_path, allow_pickle=False) as data:
            regions = make_loss_regions(authority, {key: data[key].copy() for key in data.files})
        target_row_path = root / 'ss_ray64_anchor_v3/rows' / (pair + '.json')
        target_row = read(target_row_path)
        if target_row['status'] != 'completed':
            raise ValueError('physical SS64 authority is incomplete')
        with np.load(target_row['artifact'], allow_pickle=False) as data:
            target = coords_to_mask(data['target64'], 64)
        targets = {64: target, 32: F.max_pool3d(target.float(), 2, 2).bool()}
        path = output / 'pairs' / (pair + '.pt')
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        torch.save({'pair_id': pair, 'split': row['split'], 'geometry': geometry, 'regions': regions,
                    'targets': targets, 'clean_target_condition': payload['clean_target_condition'],
                    'native_mismatch_condition': payload['native_mismatch_condition']}, path)
        records.append({'pair_id': pair, 'split': row['split'], 'staged': reference(path),
                        'source_pair': reference(pair_path), 'source_geometry': reference(u0_file),
                        'source_authority': reference(authority_path), 'source_regions': reference(region_path),
                        'source_target_row': reference(target_row_path)})
        print('prepared G2 reference', len(records), '/40', flush=True)
    models = {}
    factorial = root / 'ss_union_surface_q2c0/GEOM_FACTORIAL_G2'
    for arm in ARMS:
        models[arm] = {}
        for seed in [101, 202, 303]:
            old_summary = read(factorial / 'TRAIN32' / arm / f'seed{seed}/summary.json')
            source = Path(old_summary['checkpoint'])
            staged = output / 'checkpoints' / arm / f'seed{seed}.pt'
            models[arm][str(seed)] = copy_verified(source, staged, old_summary['checkpoint_sha256'])
    smoke_history = {arm: read(factorial / 'SMOKE_ONEPAIR' / arm / 'seed101/summary.json')['history']
                     for arm in ARMS}
    expected = []
    for row in records:
        if row['split'] != 'dev':
            continue
        for arm in ARMS:
            for seed in [101, 202, 303]:
                for noise in [0, 1, 2]:
                    path = factorial / 'ENDPOINT/dev' / arm / f"{row['pair_id']}-model{seed}-noise{noise}.json"
                    old = read(path)
                    if old['pair_cache_sha256'] != row['source_pair']['sha256']:
                        raise ValueError('archived endpoint condition changed')
                    expected.append({'pair_id': row['pair_id'], 'arm': arm, 'seed': seed, 'noise': noise,
                                     'metrics': old['metrics'], 'source': reference(path)})
    write_new(output / 'summary.json', {'status': 'PASS', 'records': records, 'models': models,
              'historical_smoke_history': smoke_history, 'historical_endpoints': expected,
              'runtime': reference(output / 'runtime.json'), 'source_archived_checkpoint': reference(checkpoint),
              'source_split': reference(root / 'ss_tto_adapter0/split_manifest.json'),
              'claim': 'historical reference geometry/regions only; not current DeformNeXt preprocessing'})


def load_runtime(spec):
    from pixal3d import models
    from pixal3d.pipelines import samplers
    cfg = spec['model_spec']
    model = getattr(models, cfg['name'])(**cfg['args'])
    state = torch.load(checked(spec['flow_checkpoint']), map_location='cpu', weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    if set(incompatible.missing_keys) - {'rope_phases'} or incompatible.unexpected_keys:
        raise ValueError('archived full-FT checkpoint/model mismatch: ' + str(incompatible))
    for entry in spec['decoder_files']:
        checked(entry)
    decoder = models.from_pretrained(spec['decoder_prefix'])
    sampler = getattr(samplers, spec['sampler']['name'])(**spec['sampler']['args'])
    return SimpleNamespace(models={'sparse_structure_flow_model': model.cuda().eval().requires_grad_(False),
                   'sparse_structure_decoder': decoder.cuda().eval().requires_grad_(False)},
                   sparse_structure_sampler=sampler, sparse_structure_sampler_params=spec['sampler']['params'])


def to_device(tree):
    if isinstance(tree, torch.Tensor):
        return tree.cuda()
    if isinstance(tree, dict):
        return {key: to_device(value) for key, value in tree.items()}
    return tree


def smoke(inputs, output, arm):
    """Four real frozen-flow gradient updates, paired to archived one-pair smoke."""
    summary = read(inputs / 'summary.json')
    runtime_spec = read(checked(summary['runtime']))
    pipeline = load_runtime(runtime_spec)
    row = next(r for r in summary['records'] if r['split'] == 'train')
    pair = to_device(torch.load(checked(row['staged']), map_location='cpu', weights_only=True))
    torch.manual_seed(101)
    adapter = GeometryFactorialVoxelAdapter().cuda().train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=.01)
    history = []
    for update in range(4):
        generator = torch.Generator(device='cpu').manual_seed(_seed('ss-geometry-factorial', 101, row['pair_id'], update))
        noise = torch.randn((1, 8, 16, 16, 16), generator=generator).cuda()
        optimizer.zero_grad(set_to_none=True)
        condition, _, mismatch, trace = adapted_condition(pair, adapter, pair['geometry'], arm, noise.device)
        parity = torch.equal(condition['cond']['proj'], mismatch['cond']['proj'])
        if update == 0 and not parity:
            raise ValueError('zero adapter regressed before learning')
        params = dict(pipeline.sparse_structure_sampler_params)
        steps, rescale = params.pop('steps'), params.pop('rescale_t')
        latent = differentiable_flow_euler_samples(pipeline.sparse_structure_sampler,
                    pipeline.models['sparse_structure_flow_model'], noise, condition,
                    steps=steps, rescale_t=rescale, checkpoint_model=True, **params)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = pipeline.models['sparse_structure_decoder'](latent).float()
        endpoint, _ = locked_endpoint_loss(logits, pair['regions'])
        correction = trace['correction'].float().square().mean()
        loss = endpoint + .001 * correction
        if not torch.isfinite(loss):
            raise ValueError('nonfinite actual native endpoint loss')
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.))
        if not np.isfinite(gradient) or gradient == 0:
            raise ValueError('missing actual native endpoint gradient')
        if any(p.grad is not None for model in pipeline.models.values() for p in model.parameters()):
            raise ValueError('frozen historical SS acquired parameter gradients')
        optimizer.step()
        previous = summary['historical_smoke_history'][arm][update]
        record = {'update': update+1, 'total_loss': float(loss.detach()), 'gradient_norm': gradient,
                  'initial_native_projection_exact': parity if update == 0 else None,
                  'historical_total_loss': previous['total_loss'],
                  'historical_loss_abs_difference': abs(float(loss.detach()) - previous['total_loss'])}
        history.append(record)
        print(json.dumps(record), flush=True)
        if os.environ.get('FAITHFUL_GEOMETRY_WANDB') == '1':
            import wandb
            wandb.log(record, step=update+1)
    output.mkdir(parents=True, exist_ok=True)
    write_new(output / 'summary.json', {'status': 'PASS', 'scope': 'archived G2 reference smoke only',
              'arm': arm, 'updates': 4, 'history': history, 'frozen_SS_gradients_absent': True,
              'current_DeformNeXt_SS_gain_claim': False})


def evaluate(inputs, output, arm, seed):
    summary = read(inputs / 'summary.json')
    pipeline = load_runtime(read(checked(summary['runtime'])))
    adapter = GeometryFactorialVoxelAdapter().cuda().eval().requires_grad_(False)
    checkpoint = checked(summary['models'][arm][str(seed)])
    adapter.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['adapter'], strict=True)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for row in summary['records']:
            if row['split'] != 'dev':
                continue
            pair = to_device(torch.load(checked(row['staged']), map_location='cpu', weights_only=True))
            normal, _, _, _ = adapted_condition(pair, adapter, pair['geometry'], arm, torch.device('cuda:0'))
            off, _, _, _ = adapted_condition(pair, adapter, pair['geometry'], arm, torch.device('cuda:0'), source_off=True)
            conditions = {'FACTORIAL': normal, 'PROJ_SOURCE_OFF': off}
            if arm == 'r4_content':
                shuffled = pair['geometry'][:, geometry_derangement(row['pair_id']).cuda()]
                conditions['GEOMETRY_SHUFFLED'] = adapted_condition(pair, adapter, shuffled, arm, torch.device('cuda:0'))[0]
            for noise_seed in [0, 1, 2]:
                noise = _seeded_noise(row['pair_id'], noise_seed, shape=(1, 8, 16, 16, 16), device='cuda:0')
                metrics = {case: _score(pipeline, condition, noise, pair['regions'],
                    pipeline.sparse_structure_sampler_params, pair['targets']) for case, condition in conditions.items()}
                old = next(e for e in summary['historical_endpoints'] if
                    (e['pair_id'], e['arm'], e['seed'], e['noise']) == (row['pair_id'], arm, seed, noise_seed))
                differences = {case: {key: abs(value - old['metrics'][case][key])
                    for key, value in values.items() if isinstance(value, (float, int))}
                    for case, values in metrics.items()}
                record = {'pair_id': row['pair_id'], 'arm': arm, 'model_seed': seed, 'noise_seed': noise_seed,
                          'metrics': metrics, 'historical_abs_differences': differences}
                write_new(output / 'rows' / f"{row['pair_id']}-noise{noise_seed}.json", record)
                rows.append(record)
                if os.environ.get('FAITHFUL_GEOMETRY_WANDB') == '1':
                    import wandb
                    wandb.log({case + '/' + key: value for case, values in metrics.items()
                               for key, value in values.items() if isinstance(value, (int, float))})
                print(json.dumps({'completed': len(rows), 'of': 24, 'arm': arm, 'seed': seed,
                                  'source_recall32': metrics['FACTORIAL']['source_recall32']}), flush=True)
    write_new(output / 'summary.json', {'status': 'PASS', 'scope': 'archived G2 frozen checkpoint replay',
              'arm': arm, 'seed': seed, 'rows': rows, 'checkpoint': reference(checkpoint),
              'current_DeformNeXt_SS_gain_claim': False, 'performance_promotion': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'smoke', 'evaluate'])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-checkpoint', type=Path)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int, choices=[101, 202, 303], default=101)
    parser.add_argument('--wandb-id')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
    if args.mode == 'prepare':
        if args.reference_checkpoint is None:
            parser.error('prepare requires the recorded G2 checkpoint/runtime authority')
        prepare(args.input, args.reference_checkpoint, args.output)
    else:
        torch.cuda.set_device(0)
        if args.arm is None:
            parser.error('smoke/evaluate requires one of the two reference arms')
        run = None
        if args.wandb_id:
            import wandb
            args.output.mkdir(parents=True, exist_ok=True)
            run = wandb.init(project='pixal3d-deformnext-retargeting', entity='youngwoojeon-unist',
                id=args.wandb_id, resume='never', dir=str(args.output), config={
                    'scope': 'archived-G2-reference-only', 'consumer': 'full-FT-step20000-EMA',
                    'sampler': '12steps-CFG7.5-rescale5', 'arm': args.arm, 'seed': args.seed,
                    'mode': args.mode, 'production_initializer_changed': False})
            os.environ['FAITHFUL_GEOMETRY_WANDB'] = '1'
        if args.mode == 'smoke':
            smoke(args.input, args.output, args.arm)
        else:
            evaluate(args.input, args.output, args.arm, args.seed)
        if run:
            run.summary.update(status='PASS_EXECUTION', current_DeformNeXt_SS_gain_claim=False)
            run.finish()


if __name__ == '__main__':
    main()
