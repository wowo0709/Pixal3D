"""Actual native SS/four-head gradients and paired32 V1 identity; no updates."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from types import MethodType

import torch
from .member_feature_control_recipe import control_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    allowed = os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(',')
    if (len(visible) != 1 or visible[0] not in allowed or visible[0] not in {'0','1','2','3'}
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'):
        raise ValueError('one explicitly assigned GPU0–3 and pinned CUBLAS workspace required')
    from pixal3d import datasets, models
    from pixal3d.trainers.flow_matching.member_feature_controls import MemberFeatureControlImageConditionedProjFlowMatchingCFGTrainer as Trainer
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin
    from pixal3d.utils.data_utils import recursive_to_device
    from experiments.multiview_finetune.runtime import install_hooks
    from .quick_metrics import _ss_metrics, panel_indices
    from .probe_target_first_carriers_v2 import CarrierOnlyDataset
    from .member_feature_control_value import feature_value_control
    from .native_ffn_chunk_parity import audit_native_ffn_chunk_parity
    from train import setup_rng
    from PIL import Image

    cfg = json.loads(args.config.read_text()); control = control_spec(cfg)
    if (cfg['trainer']['name'] != Trainer.__name__ or cfg['training_seed'] != 101
            or cfg['dataset']['name'] != 'DeformNeXtSS64'
            or cfg['trainer']['args']['deform_conditioner']['freeze_q2c'] is not True
            or cfg['dataset']['args']['match_root'] is not None
            or cfg['validation']['dataset_args']['match_root'] is not None):
        raise ValueError('fresh original-LoRA native-prior RGB SS recipe required')
    install_hooks()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    setup_rng(0, 101)
    ds = getattr(datasets, cfg['dataset']['name'])(cfg['validation']['roots_file'], **cfg['validation']['dataset_args'])
    indices = panel_indices(ds, cfg['validation']['quick_metrics'])
    if len(indices) != 32:
        raise ValueError('fixed development32 required')
    with torch.device('meta'):
        md = {name: getattr(models, spec['name'])(**spec['args']) for name, spec in cfg['models'].items()}
    kw = copy.deepcopy(cfg['trainer']['args'])
    kw.update(batch_size_per_gpu=1,batch_split=1,num_workers=0,max_steps=1,prefetch_data=False)
    args.output.mkdir(parents=True, exist_ok=False)
    trainer = Trainer(md, ds, **kw, output_dir=str(args.output/'native-trainer'),load_dir=None,step=None,wandb_run=None)
    for model in trainer.models.values():
        model.eval()
    trainer.p_uncond = 0
    ffn_parity = audit_native_ffn_chunk_parity(trainer.models['denoiser'])
    (args.output/'native-ffn-chunk-parity.json').write_text(json.dumps(ffn_parity,indent=2)+'\n')
    module = trainer.deform_conditioner
    parameters = {name: value for name,value in module.named_parameters() if value.requires_grad}
    originals = {name: value.detach().clone() for name,value in parameters.items()}
    head = {name: value.detach().cpu().clone() for name,value in module.q2c.state_dict().items()}
    pack = ds.collate_fn([ds[indices[0]]])
    pack = recursive_to_device(pack if isinstance(pack,dict) else pack[0],trainer.device)
    pack.pop('_dataset_name',None);pack.pop('_sha256',None)
    target = pack.pop('x_0')
    camera = trainer._extract_camera_info(dict(pack))
    transport = trainer._pop_deform_transport(dict(pack))
    with torch.no_grad(),torch.autocast('cuda',dtype=trainer.mix_precision_dtype):
        actual,negative = trainer._encode_deform_views(pack['cond'],camera,transport)
        native,native_negative = ImageConditionedProjMixin._encode_deform_views(trainer,pack['cond'],camera,None)
    equality = {key:bool(torch.equal(value,native[key])) for key,value in actual.items()}
    equality.update({'negative_'+key:bool(torch.equal(value,native_negative[key])) for key,value in negative.items()})
    if not all(equality.values()):
        raise ValueError('zero correction must retain native MV4 global/projection/negative exactly')
    with torch.autocast('cuda',dtype=trainer.mix_precision_dtype):
        loss,_ = trainer.training_losses(x_0=target,**pack)
    loss['loss'].backward()
    if len(parameters) != control['source_tensors']:
        raise ValueError('all declared physical control source/use/feature tensors required')
    initial = dict(feature_output=float(module.ray_readout.output.weight.grad.float().abs().sum()),
        fusion_gate=float(module.fusion_gate.grad.abs()),
        retargeter_output=float(module.retargeter.net[-1].weight.grad.float().abs().sum()),
        NN_LoRA=sum(float(value.grad.float().abs().sum()) for name,value in trainer.models['denoiser'].named_parameters() if 'lora_' in name and value.grad is not None))
    if any(not torch.isfinite(torch.tensor(value)) or value<=0 for value in initial.values()):
        raise ValueError('native SS loss must train both source paths and original NN LoRA initially')
    try:
        with torch.no_grad():
            module.fusion_gate.fill_(.2)
            module.retargeter.net[-1].weight.add_(1e-3*module.retargeter.net[-1].weight.grad.sign())
            module.evidence_use.network[-1].weight.fill_(1e-3)
            module.ray_readout.output.weight.add_(1e-3*module.ray_readout.output.weight.grad.sign())
            if module.task_attention is not None:
                module.task_attention.content_output.weight.fill_(1e-3)
        for model in trainer.models.values():
            model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=trainer.mix_precision_dtype):
            enabled,_ = trainer.training_losses(x_0=target,**pack)
        enabled['loss'].backward()
        gradients = {name:float(value.grad.float().abs().sum()) if value.grad is not None else None for name,value in parameters.items()}
        if any(value is None or not torch.isfinite(torch.tensor(value)) or value<=0 for value in gradients.values()):
            raise ValueError('all source/use/readout tensors must reach actual SS loss')
        with torch.no_grad(),torch.autocast('cuda',dtype=trainer.mix_precision_dtype):
            normal,_ = trainer._encode_deform_views(pack['cond'],camera,transport)
            for field in ['source_valid','target_valid']:
                absent,_ = trainer._encode_deform_views(pack['cond'],camera,dict(transport,**{field:torch.zeros_like(transport[field])}))
                if not all(torch.equal(absent[key],native[key]) for key in native):
                    raise ValueError('no-query/no-source must preserve native MV4 prior, not observed zeros')
            events=[]
            with feature_value_control(module,events):
                wrong,_ = trainer._encode_deform_views(pack['cond'],camera,transport)
            if torch.equal(wrong['proj'],normal['proj']) or not events:
                raise ValueError('actual fixed-use wrong-value intervention must affect native source correction')
            paired_losses={}
            for role in ['normal','wrong-value-fixed-use']:
                with torch.random.fork_rng(devices=[trainer.device]):
                    torch.manual_seed(80217);torch.cuda.manual_seed(80217)
                    if role=='normal':
                        measured,_=trainer.training_losses(x_0=target,**pack)
                    else:
                        with feature_value_control(module,events):
                            measured,_=trainer.training_losses(x_0=target,**pack)
                paired_losses[role]=float(measured['loss'])
            if paired_losses['normal']==paired_losses['wrong-value-fixed-use']:
                raise ValueError('same-noise native SS loss must respond to source value intervention')
            observed = transport['target_valid'].gather(1,transport['patch_grid'].flatten(1).long())
            delivered = int((((normal['proj']-native['proj']).abs().amax(-1)>0)&~observed).sum())
            if delivered<=0:
                raise ValueError('native source rays must deliver corrections beyond observed target-image voxels')
    finally:
        with torch.no_grad():
            for name,value in parameters.items():
                value.copy_(originals[name])
        for model in trainer.models.values():
            model.zero_grad(set_to_none=True)
    if (len(trainer.optimizer.state)!=0
            or any(not torch.equal(value,originals[name]) for name,value in parameters.items())
            or any(not torch.equal(value.detach().cpu(),head[name]) for name,value in module.q2c.state_dict().items())):
        raise ValueError('mechanics must restore source weights, freeze matcher and perform no optimizer update')
    mechanics = dict(status='PASS_ACTUAL_MEMBER_FEATURE_CONTROL_SOURCE_RAY_SS_MECHANICS',
        initial_native_MV4_global_projection_negative_exact=equality,
        initial_SS_gradient_L1=initial,activated_SS_source_gradient_L1=gradients,
        no_query_or_source_preserves_native_MV4=True,target_only_fallback=False,
        actual_fixed_use_wrong_value_events=events,actual_same_noise_SS_value_losses=paired_losses,
        native_correction_reaches_unobserved_target_voxels=delivered,
        physical_voxel_visibility_write_gate=False,all_temporary_source_parameters_restored=True,
        matching_head_exact=True,optimizer_updates=0,physical_deformation_query_channels=0,
        ffn_chunks=trainer._native_ffn_chunk_report,actual_native_ffn_parity=ffn_parity,
        actual_source_tensors=control['source_tensors'],actual_feature_head_tensors=control['head_tensors'],
        ray_feature_utility=trainer._ray_feature_utility_initialization,
        direct_feature_delivery_bypasses_zero_semantic_gate=True)
    (args.output/'mechanics.json').write_text(json.dumps(mechanics,indent=2)+'\n')
    quick = dict(cfg['validation']['quick_metrics'],generation_batch_size=1,log_progress=True)
    panels={};saved_method=trainer._encode_deform_views;saved_config=trainer.deform_conditioner_config
    prior_determinism=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        for role in ['native-prior-source-zero','native-four-view-V1']:
            dataset=ds
            if role=='native-four-view-V1':
                trainer.deform_conditioner_config=None
                trainer._encode_deform_views=MethodType(ImageConditionedProjMixin._encode_deform_views,trainer)
                dataset=CarrierOnlyDataset(ds,ImageConditionedProjMixin._DEFORM_TRANSPORT_KEYS)
            with torch.no_grad(),torch.autocast('cuda',dtype=trainer.mix_precision_dtype):
                metrics,panel=_ss_metrics(trainer,dataset,indices,quick)
            panels[role]=dict(metrics=metrics,per_asset=copy.deepcopy(trainer._quick_per_asset))
            Image.fromarray(panel).save(args.output/(role+'.png'))
    finally:
        trainer._encode_deform_views=saved_method;trainer.deform_conditioner_config=saved_config
        torch.use_deterministic_algorithms(prior_determinism)
    left,right=[panels[role] for role in ['native-prior-source-zero','native-four-view-V1']]
    fields=['sample_id','source_cluster_id','seed','tp','fp','fn','target_voxels_sha256','target_voxel_shape']
    if (left['metrics']!=right['metrics'] or any(a[key]!=b[key] for a,b in zip(left['per_asset'],right['per_asset']) for key in fields)):
        raise ValueError('actual SS32 native V1 zero-source predictions and GT/noise identities must be exact')
    report=dict(status='PASS_ACTUAL_MEMBER_FEATURE_CONTROL_NATIVE_V1_INITIAL32_IDENTITY',panels=panels,
        all32_voxel_counts_and_noise_GT_identities_exact=True,initial_V1_identity=True,
        initial_micro_IoU_regression=False,source_module_optimizer_updates=0,
        source_module_weights_restored=True,capacity_or_trained_quality_not_proven=True,
        heldout_TEST_measured=False,full_goal_complete=False)
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='panels'}),flush=True)


if __name__=='__main__':
    main()
