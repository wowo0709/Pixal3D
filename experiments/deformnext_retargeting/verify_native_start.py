"""Real-checkpoint all-arm native identity, sampling, and gradient acceptance.

This is a diagnostic, not training: denoiser weights are never updated. A fresh
R9 head is used only to exercise routing; no legacy matcher/cache is accepted.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch
import numpy as np


def feats(value):
    return value.feats if hasattr(value, 'feats') else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['ss64', 'pbr1024'], required=True)
    parser.add_argument('--config-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, required=True)
    parser.add_argument('--q2c-checkpoint', type=Path)
    parser.add_argument('--raw-match-root', type=Path)
    args = parser.parse_args()
    from pixal3d import datasets, models, trainers
    from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
    from pixal3d.utils.data_utils import recursive_to_device
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import compute_multiview_projection_matrices
    from experiments.deformnext_retargeting.quick_metrics import stratified_indices, binary_metrics, _render_pbr_anchor
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((args.config_root / f'{args.stage}-v1.json').read_text())
    ds_args = dict(cfg['validation']['dataset_args'])
    ds_args.update(transport_arm='c2', transport_implementation='r9-native-v5', match_root=None,
                   projection_root=str(Path(cfg['validation']['roots_file']).parent.parent / 'projection_maps_world_v4'))
    ds = getattr(datasets, cfg['dataset']['name'])(cfg['validation']['roots_file'], **ds_args)
    with torch.device('meta'):
        md = {key: getattr(models, spec['name'])(**spec['args']) for key, spec in cfg['models'].items()}
    kw = dict(cfg['trainer']['args'])
    kw.update(num_workers=0, prefetch_data=False, batch_size_per_gpu=1, batch_split=1, max_steps=1)
    trainer = getattr(trainers, cfg['trainer']['name'])(md, ds, **kw, output_dir=str(args.output), load_dir=None, step=None, wandb_run=None)
    trainer.p_uncond = 0
    trainer.models['denoiser'].eval()
    width = 1024 if args.stage == 'ss64' else 2048
    modules = {}
    torch.manual_seed(42017)
    shared = DeformationAwareConditioner(arm='b3', key_dim=1024, value_dim=width, query_dim=1024,
        **({'q2c_checkpoint': str(args.q2c_checkpoint), 'freeze_q2c': True} if args.q2c_checkpoint else {})).cuda().eval()
    for arm in ['b1', 'b2', 'b3', 'c1', 'c2']:
        module = DeformationAwareConditioner(arm=arm, key_dim=1024, value_dim=width, query_dim=1024,
                                            **({'deformation_dim': 2048} if arm == 'c2' else {})).cuda().eval()
        if module.q2c is not None:
            module.q2c.load_state_dict(shared.q2c.state_dict())
            module.q2c.requires_grad_(False)
        modules[arm] = module
    del shared
    indices = stratified_indices(ds, args.samples)
    report = {'status': 'RUNNING', 'stage': args.stage, 'optimizer_updates': 0,
              'conditioning_schema': 'r9-native-residual-v5',
              'matcher_state': 'fresh promoted R9 D1; identity diagnostic only',
              'native_condition_max_abs_error': 0., 'native_denoiser_max_abs_error': 0.,
              'native_sampling_max_abs_error': 0., 'gradient_checks': {}, 'samples': [], 'metrics': {}}
    if args.q2c_checkpoint:
        from .precompute_matches import sha256
        report['q2c_sha256'] = sha256(args.q2c_checkpoint)
        report['matcher_state'] = 'strict frozen R9 checkpoint: ' + str(args.q2c_checkpoint)
    if args.raw_match_root:
        from pixal3d.modules.r9_matching import RAW_MATCH_CONTRACT
        report['raw_matching_contract']=RAW_MATCH_CONTRACT
    started = time.monotonic()
    for offset, index in enumerate(indices):
        pack = recursive_to_device(ds.collate_fn([ds[index]]), 'cuda')
        target = pack.pop('x_0'); pack.pop('_dataset_name', None); pack.pop('_sha256', None)
        if args.stage == 'pbr1024': pack['coords'] = target.coords
        transport = {key.removeprefix('transport_'): pack.pop(key) for key in list(pack) if key.startswith('transport_')}
        with torch.no_grad(), torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
            trainer.deform_conditioner_config = None
            native = trainer.get_inference_cond(**pack)
            image_model = getattr(trainer, '_image_cond_module', trainer.image_cond_model)
            projection, _ = compute_multiview_projection_matrices(pack['transform_matrix'], pack['camera_distance'], image_model.fixed_projection_transform)
            anchor, _ = trainer.encode_image_proj(pack['cond'][:, 0], camera_angle_x=pack['camera_angle_x'][:, 0],
                distance=pack['camera_distance'][:, 0], mesh_scale=pack['mesh_scale'], transform_matrix=projection[:, 0], coords=pack.get('coords'))
            # Compute the fixed proposal/decoder once. Cache consumers and the
            # public online path are compared on the first sample below.
            match_call = {k: v for k, v in transport.items() if k not in {'patch_grid', 'patch_ids', 'deformation', 'target_context', 'source_context'}}
            match_call.pop('target_query', None)
            routed = modules['b3'](**match_call)
            raw_indices = routed.candidate_indices
            cached = {key: value for key, value in transport.items() if key not in {'target_keys', 'source_keys', 'target_context', 'source_context'}}
            cached.update(semantic_indices=raw_indices, matching_logits=routed.matching_logits,
                          candidate_valid=torch.isfinite(routed.matching_logits))
            # Context uses the same RAW FRAME tokens and categorical weights.
            flat_keys = transport['source_context'].flatten(1, 2)
            keys = modules['b3']._gather(flat_keys, raw_indices)
            weights = routed.candidate_weights
            cached['deformation'] = (transport['target_context'].float() - (keys.float() * weights[..., None]).sum(2)).half()
            raw_cached=None
            if args.raw_match_root:
                from .precompute_raw_matches import valid_cache
                sample_id=ds.rows[index]['sample_id']
                path=args.raw_match_root/f'{sample_id}.npz'
                if not valid_cache(path,sample_id):raise ValueError('unverified B1 RAW cache')
                raw_cached={k:v for k,v in cached.items() if k not in {'matching_logits','candidate_valid','deformation','target_query'}}
                with np.load(path) as archive:
                    raw_cached['semantic_indices']=torch.from_numpy(archive['semantic_indices'].astype(np.int64)).cuda()[None]
                online=modules['b1'](**match_call)
                offline=modules['b1'](**{k:v for k,v in raw_cached.items() if k not in {'patch_grid','patch_ids'}})
                torch.testing.assert_close(online.evidence,offline.evidence,rtol=0,atol=0)
                assert torch.equal(online.candidate_indices,offline.candidate_indices)
            torch.manual_seed(42 + offset); torch.cuda.manual_seed(42 + offset)
            noise = target.replace(torch.randn_like(target.feats)) if hasattr(target, 'feats') else torch.randn_like(target)
            t = torch.tensor([500.], device='cuda')
            extra = {'concat_cond': pack['concat_cond']} if 'concat_cond' in pack else {}
            reference = trainer.models['denoiser'](noise, t, native['cond'], **extra)
            conditions = {'v1': native['cond']}
            for arm, module in modules.items():
                trainer.deform_conditioner = module
                call = dict(raw_cached if arm=='b1' and raw_cached is not None else cached)
                if arm != 'c2': call.pop('deformation', None)
                if arm not in {'c1', 'c2'}: call.pop('target_query', None)
                condition = trainer._inject_deform_transport(dict(native['cond']), call, pack.get('coords'), anchor_condition=anchor)
                for key in native['cond']:
                    torch.testing.assert_close(condition[key], native['cond'][key], rtol=0, atol=0)
                prediction = trainer.models['denoiser'](noise, t, condition, **extra)
                torch.testing.assert_close(feats(prediction), feats(reference), rtol=0, atol=0)
                conditions[arm] = condition
            if offset == 0:
                predictions = []
                for arm, condition in conditions.items():
                    case_noise = noise.replace(noise.feats.clone()) if hasattr(noise, 'feats') else noise.clone()
                    pred = trainer.get_sampler().sample(trainer.models['denoiser'], noise=case_noise,
                        **dict(native, cond=condition), steps=12, guidance_strength=3., verbose=False).samples
                    if predictions:
                        torch.testing.assert_close(feats(pred), feats(predictions[0]), rtol=0, atol=0)
                    predictions.append(pred)
                if args.stage == 'ss64':
                    decoded = ds.decode_latent(torch.cat(predictions + [target]), batch_size=2)[:, 0] > 0
                    for j, arm in enumerate(conditions): report['metrics'][arm] = binary_metrics(decoded[j:j+1], decoded[-1:])
                else:
                    from pixal3d.modules import sparse as sp
                    with torch.autocast('cuda', enabled=False):
                        z = sp.sparse_cat(predictions + [target])
                        shape = sp.sparse_cat([pack['concat_cond']] * len(z.layout))
                        reps = ds.decode_latent(z.float(), shape.float(), batch_size=1)
                        camera = (float(pack['camera_angle_x'][0, 0]), float(pack['camera_distance'][0, 0]), float(pack['mesh_scale'][0]))
                        renders = _render_pbr_anchor(ds, reps, [camera] * len(reps), 256)
                        render_error = max(float((renders[key][:len(conditions)] - renders[key][0:1]).abs().max())
                                           for key in ('shaded', 'base_color', 'alpha'))
                        if render_error != 0:
                            raise AssertionError(f'identical sampled PBR latents render differently: {render_error}')
                        report['native_render_max_abs_error'] = render_error
                        # Compare identical PBR attributes in a single renderer
                        # invocation to exclude cross-run lighting jitter.
                        import math
                        for j, arm in enumerate(conditions):
                            mask = ((renders['alpha'][j] > .5) | (renders['alpha'][-1] > .5)).reshape(1, 256, 256).expand_as(renders['shaded'][j])
                            mse = float((renders['shaded'][j] - renders['shaded'][-1]).square()[mask].mean())
                            report['metrics'][arm] = {'foreground_psnr': -10 * math.log10(mse)}
                        del z, shape, reps, renders
                del predictions, pred
        if offset == 0:
            # Exercise actual public train/inference encoding, and actual
            # denoising backward into LoRA, gate, and zero-output retargeter.
            for arm, module in modules.items():
                trainer.deform_conditioner = module
                trainer.deform_conditioner_config = {'arm': arm, 'implementation': 'r9-native-v5'}
                train_pack = dict(pack)
                for key, value in (raw_cached if arm=='b1' and raw_cached is not None else cached).items():
                    if key == 'deformation' and arm != 'c2': continue
                    if key == 'target_query' and arm not in {'c1', 'c2'}: continue
                    train_pack['transport_' + key] = value
                trainer.models['denoiser'].zero_grad(set_to_none=True); module.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=trainer.mix_precision_dtype, cache_enabled=False):
                    inf = trainer.get_inference_cond(**train_pack)['cond']
                    trained = trainer.get_cond(**train_pack)
                    for key in native['cond']:
                        torch.testing.assert_close(inf[key], native['cond'][key], rtol=0, atol=0)
                        torch.testing.assert_close(trained[key], native['cond'][key], rtol=0, atol=0)
                    y = trainer.models['denoiser'](noise, t, trained, **extra)
                    loss = feats(y).square().mean()
                loss.backward()
                gate_grad = float(module.fusion_gate.grad.abs())
                lora_grad = sum(float(p.grad.float().abs().sum()) for n, p in trainer.models['denoiser'].named_parameters() if 'lora_' in n and p.grad is not None)
                adapter_grad = float(module.retargeter.net[-1].weight.grad.abs().sum()) if module.retargeter is not None else None
                assert gate_grad > 0 and lora_grad > 0
                assert adapter_grad is None or adapter_grad > 0
                report['gradient_checks'][arm] = {'gate': gate_grad, 'lora': lora_grad, 'adapter_output': adapter_grad}
                module.zero_grad(set_to_none=True); trainer.models['denoiser'].zero_grad(set_to_none=True)
                del inf, trained, y, loss
        report['samples'].append(ds.rows[index]['sample_id'])
        report['elapsed_seconds'] = time.monotonic() - started
        (args.output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'completed': offset + 1, 'total': len(indices), 'elapsed_seconds': report['elapsed_seconds']}), flush=True)
        del pack, target, transport, native, anchor, conditions, condition, reference, prediction, cached, match_call, keys, routed
        torch.cuda.empty_cache()
    report['status'] = 'PASS'
    report['arms'] = [{'arm': arm} for arm in ('v1', 'b1', 'b2', 'b3', 'c1', 'c2')]
    report['max_memory_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    (args.output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
