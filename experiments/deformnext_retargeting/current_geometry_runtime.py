"""The verified current LoRA consumer for isolated native-voxel experiments."""
from __future__ import annotations

import copy

import torch


def load_current_trainer(cfg, checkpoint, manifest, output):
    from pixal3d import datasets, models, trainers
    ds_args = copy.deepcopy(cfg['dataset']['args'])
    ds_args.update(transport_arm='v1', expected_split='train')
    for key in ['match_root', 'projection_root', 'deformation_root', 'expected_q2c_sha256']:
        ds_args.pop(key, None)
    dataset = getattr(datasets, cfg['dataset']['name'])(str(manifest), **ds_args)
    if dataset.normalization is not None:
        raise ValueError('current native raw-latent pilot requires its recorded unnormalized SS config')
    with torch.device('meta'):
        model_dict = {name: getattr(models, spec['name'])(**spec['args']) for name, spec in cfg['models'].items()}
    kwargs = copy.deepcopy(cfg['trainer']['args'])
    kwargs.update(continuation_ckpt={'denoiser': str(checkpoint)}, deform_conditioner=None,
                  batch_size_per_gpu=1, batch_split=1, num_workers=0, prefetch_data=False, max_steps=1)
    kwargs.get('model_learning_rates', {}).pop('deform_conditioner', None)
    trainer = getattr(trainers, cfg['trainer']['name'])(model_dict, dataset, **kwargs,
        output_dir=str(output), load_dir=None, step=None, wandb_run=None)
    trainer.models['denoiser'].eval().requires_grad_(False)
    trainer.p_uncond = 0
    return trainer, dataset


def capture_native(trainer, pack):
    if trainer.image_cond_model is None:
        trainer._init_image_cond_model()
    captures = []
    hook = trainer._image_cond_module.proj_grid.register_forward_hook(
        lambda module, arguments, output: captures.append(output.detach()))
    try:
        native = trainer.get_inference_cond(**pack)
    finally:
        hook.remove()
    if len(captures) != 4 or set(native) != {'cond', 'neg_cond'} or set(native['cond']) != {'global', 'proj'}:
        raise ValueError('expected exactly four current native per-view projections')
    total = captures[0].float()
    for value in captures[1:]:
        total = total + value.float()
    if not torch.equal((total/4).to(native['cond']['proj'].dtype), native['cond']['proj']):
        raise ValueError('cached per-view projections do not reproduce native multiview mean')
    return native, captures[0]


def move_tensors(tree, device):
    return {key: move_tensors(value, device) if isinstance(value, dict) else value.to(device)
            for key, value in tree.items()}
