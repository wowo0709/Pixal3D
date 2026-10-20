"""Actual native SS proof and initial32 carrier comparison; no optimizer updates.

Zero correction means the target-only carrier, not native four-view V1.
The separate four-view initial panel makes that architectural change visible.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from types import MethodType

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    allowed = os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(',')
    if len(visible) != 1 or visible[0] not in allowed or visible[0] not in {'0', '1', '2', '3'}:
        raise ValueError('one explicitly assigned physical GPU0–3 required')
    from pixal3d import datasets, models
    from pixal3d.trainers.flow_matching.target_first_ray import TargetFirstRayImageConditionedProjFlowMatchingCFGTrainer as Trainer
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin
    from pixal3d.utils.data_utils import recursive_to_device
    from .quick_metrics import _ss_metrics, panel_indices
    from experiments.multiview_finetune.runtime import install_hooks
    from train import setup_rng
    from PIL import Image

    install_hooks()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    cfg = json.loads(args.config.read_text())
    if (cfg['trainer']['name'] != Trainer.__name__ or cfg['training_seed'] != 101
            or cfg['dataset']['args']['match_root'] is not None
            or cfg['validation']['dataset_args']['match_root'] is not None):
        raise ValueError('explicit target-first/seed101/fresh ray proposal recipe required')
    setup_rng(0, 101)
    ds = getattr(datasets, cfg['dataset']['name'])(cfg['validation']['roots_file'], **cfg['validation']['dataset_args'])
    indices = panel_indices(ds, cfg['validation']['quick_metrics'])
    if len(indices) != 32:
        raise ValueError('same fixed32 development assets required')
    with torch.device('meta'):
        md = {name:getattr(models, spec['name'])(**spec['args']) for name,spec in cfg['models'].items()}
    kw = copy.deepcopy(cfg['trainer']['args'])
    kw.update(batch_size_per_gpu=1, batch_split=1, num_workers=0, max_steps=1, prefetch_data=False)
    args.output.mkdir(parents=True, exist_ok=False)
    trainer = Trainer(md, ds, **kw, output_dir=str(args.output/'native-trainer'),
        load_dir=None, step=None, wandb_run=None)
    trainer.models['denoiser'].eval()
    trainer.deform_conditioner.eval()
    trainer.p_uncond = 0
    module = trainer.deform_conditioner
    trainable = {name:p for name,p in module.named_parameters() if p.requires_grad}
    originals = {name:p.detach().clone() for name,p in trainable.items()}

    def frozen_digest():
        h = hashlib.sha256()
        for name,p in module.q2c.named_parameters():
            if p.requires_grad:
                raise ValueError('the original TRAIN matching head must be frozen')
            h.update(name.encode());h.update(p.detach().contiguous().cpu().numpy().tobytes())
        return h.hexdigest()

    frozen = frozen_digest()
    pack = ds.collate_fn([ds[indices[0]]])
    pack = pack if isinstance(pack, dict) else pack[0]
    pack = recursive_to_device(pack, trainer.device)
    pack.pop('_dataset_name', None);pack.pop('_sha256', None)
    target = pack.pop('x_0')
    camera = trainer._extract_camera_info(dict(pack))
    transport = trainer._pop_deform_transport(dict(pack))
    with torch.no_grad(), torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
        actual, negative = trainer._encode_deform_views(pack['cond'], camera, transport)
        anchor, anchor_negative = trainer._encode_deform_views(pack['cond'], camera, None)
        native, _ = ImageConditionedProjMixin._encode_deform_views(trainer, pack['cond'], camera, None)
    equality = {key:bool(torch.equal(actual[key], anchor[key])) for key in anchor}
    equality.update({'negative_'+key:bool(torch.equal(negative[key], anchor_negative[key])) for key in negative})
    if not all(equality.values()):
        raise ValueError('zero source correction must exactly retain the native target-only carrier')
    native_difference = {key:float((actual[key]-native[key]).abs().max()) for key in actual}
    with torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
        loss, _ = trainer.training_losses(x_0=target, **pack)
    loss['loss'].backward()
    initial = dict(fusion_gate=float(module.fusion_gate.grad.abs()),
        retargeter_output=float(module.retargeter.net[-1].weight.grad.float().abs().sum()),
        NN_LoRA=sum(float(p.grad.float().abs().sum()) for name,p in trainer.models['denoiser'].named_parameters()
                    if 'lora_' in name and p.grad is not None))
    if any(not np.isfinite(x) or x <= 0 for x in initial.values()):
        raise ValueError('source gate, R output and NN LoRA require actual initial SS-loss gradients')
    direction = module.retargeter.net[-1].weight.grad.detach().sign()
    try:
        with torch.no_grad():
            module.retargeter.net[-1].weight.add_(1e-3*direction)
            module.fusion_gate.fill_(.2)
            module.evidence_use.network[-1].weight.fill_(1e-3)
            if module.task_attention is not None:
                module.task_attention.content_output.weight.fill_(1e-3)
        for model in trainer.models.values():
            model.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
            probe, _ = trainer.training_losses(x_0=target, **pack)
        probe['loss'].backward()
        gradient = {name:(float(p.grad.float().abs().sum()) if p.grad is not None else None)
                    for name,p in trainable.items()}
        if any(value is None or not np.isfinite(value) or value <= 0 for value in gradient.values()):
            raise ValueError('every source/use/attention tensor must connect to the actual SS loss after a restored probe')
        with torch.no_grad(), torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
            empty = dict(transport, source_valid=torch.zeros_like(transport['source_valid']))
            absent, _ = trainer._encode_deform_views(pack['cond'], camera, empty)
            if not all(torch.equal(absent[key], anchor[key]) for key in anchor):
                raise ValueError('missing correspondences must retain target even with nonzero source weights')
    finally:
        with torch.no_grad():
            for name,p in trainable.items():
                p.copy_(originals[name])
        for model in trainer.models.values():
            model.zero_grad(set_to_none=True)
    if frozen_digest() != frozen or any(not torch.equal(p, originals[name]) for name,p in trainable.items()):
        raise ValueError('mechanical proof must restore every temporary weight and preserve matcher')
    proof = dict(status='PASS_ACTUAL_TARGET_FIRST_RAY_NATIVE_SS_MECHANICS',
        sample_id=ds.rows[indices[0]]['sample_id'], exact_target_carrier=equality,
        native_four_view_V1_condition_max_difference=native_difference,
        initial_SS_gradient_L1=initial, temporary_probe_gradient_L1=gradient,
        all_temporary_parameters_restored=True, optimizer_updates=0,
        missing_correspondence_returns_target_exact=True, physical_voxel_visibility_gate=False,
        query='native target projected DINO/Omega pooled per source-camera ray;not predictedSS proxy',
        initial_V1_function_identity=False, initial_quality_comparison_still_required=True,
        trained_quality_gain_established=False)
    (args.output/'mechanics.json').write_text(json.dumps(proof,indent=2)+'\n')
    quick = dict(cfg['validation']['quick_metrics'], generation_batch_size=1, log_progress=True)
    panels = {}
    original_method = trainer._encode_deform_views
    original_config = trainer.deform_conditioner_config
    try:
        # Mechanics above binds the zero model to this target-only carrier.
        # Compare carriers directly, without running unused zero source heads.
        trainer.deform_conditioner_config = None
        for name in ['target-carrier', 'native-four-view-V1']:
            if name == 'native-four-view-V1':
                trainer._encode_deform_views = MethodType(ImageConditionedProjMixin._encode_deform_views, trainer)
            with torch.no_grad(), torch.autocast('cuda', dtype=trainer.mix_precision_dtype):
                metrics, image = _ss_metrics(trainer, ds, indices, quick)
            panels[name] = dict(metrics=metrics, per_asset=copy.deepcopy(trainer._quick_per_asset))
            Image.fromarray(image).save(args.output/(name+'.png'))
    finally:
        trainer._encode_deform_views = original_method
        trainer.deform_conditioner_config = original_config
    if any(a['sample_id']!=b['sample_id'] or a['seed']!=b['seed'] or a['target_voxels_sha256']!=b['target_voxels_sha256']
           for a,b in zip(panels['target-carrier']['per_asset'],panels['native-four-view-V1']['per_asset'])):
        raise ValueError('paired initial carriers require identical sample/noise/target occupancy')
    difference = panels['target-carrier']['metrics']['voxel_iou'] - panels['native-four-view-V1']['metrics']['voxel_iou']
    report = dict(status='PASS_ACTUAL_TARGET_FIRST_INITIAL32_CARRIER_COMPARISON', panels=panels,
        target_minus_native_V1_micro_IoU=difference,
        observed_initial_micro_IoU_regression=difference < 0,
        initial_V1_identity=False, source_module_optimizer_updates=0,
        mechanical_PASS_does_not_prove_capacity_or_quality=True,
        full_training_not_queued=True, heldout_test_measured=False,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30, full_goal_complete=False)
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='panels'}),flush=True)


if __name__ == '__main__':
    main()
