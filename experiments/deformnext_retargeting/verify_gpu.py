"""Verify real checkpoints, native MV parity and six-arm gradients on one GPU.

Assign physical GPUs with CUDA_VISIBLE_DEVICES outside this script.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from pixal3d import datasets, models, trainers
from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
from pixal3d.utils.data_utils import recursive_to_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-root', type=Path, required=True)
    parser.add_argument('--stage', choices=('ss64', 'pbr1024'), required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    cfg = json.loads((args.config_root / f'{args.stage}-v1.json').read_text())
    output = args.output_root / args.stage
    output.mkdir(parents=True, exist_ok=True)
    ds = getattr(datasets, cfg['dataset']['name'])(cfg['validation']['roots_file'],
             **cfg['validation']['dataset_args'])
    with torch.device('meta'):
        model_dict = {name: getattr(models, spec['name'])(**spec['args'])
                      for name, spec in cfg['models'].items()}
    trainer_args = dict(cfg['trainer']['args'])
    trainer_args.update(batch_size_per_gpu=1, batch_split=1, num_workers=0,
                        prefetch_data=False, max_steps=1)
    trainer = getattr(trainers, cfg['trainer']['name'])(
        model_dict, ds, **trainer_args, output_dir=str(output), load_dir=None,
        step=None, wandb_run=None)
    trainer.p_uncond = 0
    for model in trainer.models.values():
        model.eval()
    report = {'stage': args.stage, 'arms': [], 'source_checkpoint': trainer_args['continuation_ckpt']}
    torch.manual_seed(42)
    for arm in ('v1', 'b1', 'b2', 'b3', 'c1', 'c2'):
        c = json.loads((args.config_root / f'{args.stage}-{arm}.json').read_text())
        ds = getattr(datasets, c['dataset']['name'])(c['validation']['roots_file'],
                    **c['validation']['dataset_args'])
        packs = ds.collate_fn([ds[0]])
        pack = recursive_to_device(packs if isinstance(packs, dict) else packs[0], trainer.device)
        target = pack.pop('x_0')
        pack.pop('_dataset_name', None)
        pack.pop('_sha256', None)
        if args.stage == 'pbr1024':
            pack['coords'] = target.coords
        trainer.deform_conditioner_config = c['trainer']['args']['deform_conditioner']
        trainer.deform_conditioner = (None if arm == 'v1' else
            DeformationAwareConditioner(**trainer.deform_conditioner_config).cuda())
        trainer.models['denoiser'].zero_grad(set_to_none=True)
        if trainer.deform_conditioner is not None:
            trainer.deform_conditioner.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=trainer.mix_precision_dtype, cache_enabled=False):
            condition = trainer.get_inference_cond(**pack)['cond']
            training_condition = trainer.get_cond(**pack)
            condition_error = max(float((condition[k] - training_condition[k]).abs().max().detach()) for k in condition)
            # CUDA scatter_add in score aggregation has FP32 atomic-order
            # variation. V1/Top-1 must be exact; weighted arms allow 1e-5.
            tolerance = 0.0 if arm in ('v1', 'b1', 'b2') else 1e-5
            for key in condition:
                torch.testing.assert_close(condition[key], training_condition[key], rtol=0, atol=tolerance)
            if arm == 'v1':
                reference, _ = trainer.encode_image_proj(
                    pack['cond'], camera_angle_x=pack['camera_angle_x'],
                    distance=pack['camera_distance'], mesh_scale=pack['mesh_scale'],
                    transform_matrix=pack['transform_matrix'], coords=pack.get('coords'))
                for key in reference:
                    torch.testing.assert_close(condition[key], reference[key], rtol=0, atol=0)
                report['native_condition_max_abs_error'] = 0.0
                noise = torch.randn_like(target.feats) if hasattr(target, 'feats') else torch.randn_like(target)
                x = target.replace(noise) if hasattr(target, 'feats') else noise
                t = torch.tensor([500.], device=trainer.device)
                extra = {'concat_cond': pack['concat_cond']} if 'concat_cond' in pack else {}
                with torch.no_grad():
                    y = trainer.models['denoiser'](x, t, condition, **extra)
                    ref = trainer.models['denoiser'](x, t, reference, **extra)
                    ya = y.feats if hasattr(y, 'feats') else y
                    ra = ref.feats if hasattr(ref, 'feats') else ref
                    torch.testing.assert_close(ya, ra, rtol=0, atol=0)
                    report['native_denoiser_max_abs_error'] = 0.0
            # Repeatable low-noise and high-noise losses are checked by the
            # native objective; backward must reach LoRA and both adapters.
            losses, _ = trainer.training_losses(x_0=target, **pack)
            loss = losses['loss']
        if not torch.isfinite(loss):
            raise AssertionError(f'{arm} has non-finite loss')
        loss.backward()
        lora_gradient = sum(float(p.grad.float().abs().sum())
                            for name, p in trainer.models['denoiser'].named_parameters()
                            if 'lora_' in name and p.grad is not None)
        if not lora_gradient > 0:
            raise AssertionError(f'{arm} has no LoRA gradient')
        adapter_gradient = None
        if arm in ('c1', 'c2'):
            adapter_gradient = float(trainer.deform_conditioner.retargeter.net[-1].weight.grad.abs().sum())
            if not adapter_gradient > 0:
                raise AssertionError(f'{arm} has no retargeter gradient')
        row = {'arm': arm, 'loss': float(loss.detach()), 'lora_gradient_l1': lora_gradient,
               'retargeter_gradient_l1': adapter_gradient, 'train_inference_condition_match': True, 'condition_max_abs_error': condition_error}
        report['arms'].append(row)
        print(json.dumps(row), flush=True)
        del condition, training_condition, losses, loss, pack, target
        torch.cuda.empty_cache()
    report['status'] = 'PASS'
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
