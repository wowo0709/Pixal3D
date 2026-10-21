"""Actual native SS warm-source parity and flow-gradient proof on one input.

This performs no optimizer update and exports no scientific checkpoint.
Micro96 capacity and joint training require their own subsequent executions.
"""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from .cascade_view_roles import file_sha256
from .source_input_joint_runtime import science_recipe_binding
from .source_teacher_initialization import CLASSES, source_input_binding


@contextmanager
def temporary_source_module(trainer, reference):
    """Swap only the source path for a read-only reference forward, then restore.

    Native world1 can alias ``models`` and ``training_models``. Avoid replacing
    an optimizer parameter or restoring an aliased dictionary twice.
    """
    original = trainer.deform_conditioner
    if (trainer.world_size != 1
            or trainer.models.get('deform_conditioner') is not original
            or trainer.training_models.get('deform_conditioner') is not original):
        raise ValueError('one unwrapped native source model on world1 required')
    mappings = {id(values): values for values in (trainer.models, trainer.training_models)}
    try:
        trainer.deform_conditioner = reference
        for values in mappings.values(): values['deform_conditioner'] = reference
        yield
    finally:
        trainer.deform_conditioner = original
        for values in mappings.values(): values['deform_conditioner'] = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--initialization-spec', type=Path, required=True)
    parser.add_argument('--sample-id', required=True)
    parser.add_argument('--expected-conditioning-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (gpu not in {'0','1','2','3'} or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(',')
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'):
        raise ValueError('one assigned GPU0-3 and pinned CUBLAS workspace required')
    cfg = json.loads(args.config.read_text())
    spec = json.loads(args.initialization_spec.read_text()); variant = spec['variant']
    from .member_feature_recipe import validate_feature_recipe
    holder = SimpleNamespace(validate_feature_recipe=validate_feature_recipe)
    with science_recipe_binding(holder, variant=variant, teacher_enabled=True):
        holder.validate_feature_recipe(cfg)
    if cfg['dataset']['name'] != 'DeformNeXtSS64' or cfg['dataset']['args']['expected_split'] != 'train':
        raise ValueError('native TRAIN flow-loss input, no validation fitting required')
    from pixal3d import datasets, models
    from pixal3d.trainers.flow_matching import member_feature_source_ray as trainer_module
    from pixal3d.trainers.flow_matching.member_feature_source_ray import MemberFeatureSourceRayImageConditionedProjFlowMatchingCFGTrainer as Trainer
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin
    from pixal3d.utils.data_utils import recursive_to_device
    from experiments.multiview_finetune.runtime import install_hooks
    from train import setup_rng
    torch.set_num_threads(2);torch.cuda.set_device(0);setup_rng(0,cfg['training_seed']);install_hooks()
    original_determinism=torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    dataset = getattr(datasets,cfg['dataset']['name'])(cfg['data_dir'],**cfg['dataset']['args'])
    indices = [i for i,r in enumerate(dataset.rows) if r['sample_id']==args.sample_id]
    if (len(indices)!=1 or dataset.rows[indices[0]]['split']!='train'
            or dataset.rows[indices[0]]['conditioning_input_sha256']!=args.expected_conditioning_sha256
            or dataset.rows[indices[0]]['conditioning_camera_ids']!=[8,9,10,11]):
        raise ValueError('one exact predeclared TRAIN sample required')
    with torch.device('meta'):
        md={name:getattr(models,s['name'])(**s['args']) for name,s in cfg['models'].items()}
    kwargs=copy.deepcopy(cfg['trainer']['args'])
    kwargs.update(batch_size_per_gpu=1,batch_split=1,num_workers=0,max_steps=1,prefetch_data=False)
    args.output.mkdir(parents=True,exist_ok=False)
    with source_input_binding(trainer_module,variant=variant,teacher=spec) as instances:
        trainer=Trainer(md,dataset,**kwargs,output_dir=str(args.output/'native-trainer'),
                        load_dir=None,step=None,wandb_run=None)
    module=trainer.deform_conditioner
    if (len(instances)!=1 or instances[0]is not module or len(trainer.optimizer.state)!=0
            or trainer.world_size!=1 or trainer.step!=0):
        raise ValueError('fresh native optimizer and actual integrated source teacher required')
    for model in trainer.models.values():model.eval()
    trainer.p_uncond=0
    with torch.random.fork_rng(devices=[trainer.device]):
        reference=CLASSES[variant](**cfg['trainer']['args']['deform_conditioner']).cuda().eval()
        reference.load_state_dict(torch.load(spec['checkpoint'],map_location='cpu',weights_only=True),strict=True)
    saved=torch.load(spec['checkpoint'],map_location='cpu',weights_only=True)
    if any(not torch.equal(value.detach().cpu(),saved[name]) for name,value in module.state_dict().items()):
        raise ValueError('integrated post-utility teacher must equal the standalone checkpoint')
    trainable={name:p for name,p in module.named_parameters() if p.requires_grad}
    lora={name:p for name,p in trainer.models['denoiser'].named_parameters() if p.requires_grad}
    source_group=next(g for g in trainer.optimizer.param_groups if g.get('model_name')=='deform_conditioner')
    if (len(trainable)!=19 or len(lora)!=360 or any('lora_'not in n for n in lora)
            or {id(p)for p in source_group['params']}!={id(p)for p in trainable.values()}):
        raise ValueError('original360 LoRA and all19 integrated source parameters must be optimized')
    lora_before={n:p.detach().cpu().clone()for n,p in lora.items()}
    pack=dataset.collate_fn([dataset[indices[0]]])
    pack=recursive_to_device(pack if isinstance(pack,dict)else pack[0],trainer.device)
    pack.pop('_dataset_name',None);pack.pop('_sha256',None);target=pack.pop('x_0')
    camera=trainer._extract_camera_info(dict(pack));transport=trainer._pop_deform_transport(dict(pack))
    cpu_rng,cuda_rng=torch.random.get_rng_state(),torch.cuda.get_rng_state()
    with torch.no_grad(),torch.autocast('cuda',dtype=trainer.mix_precision_dtype):
        integrated,negative=trainer._encode_deform_views(pack['cond'],camera,transport)
        with temporary_source_module(trainer,reference):
            standalone,standalone_negative=trainer._encode_deform_views(pack['cond'],camera,transport)
        native,native_negative=ImageConditionedProjMixin._encode_deform_views(trainer,pack['cond'],camera,None)
    equal={k:bool(torch.equal(v,standalone[k]))for k,v in integrated.items()}
    equal.update({'negative_'+k:bool(torch.equal(v,standalone_negative[k]))for k,v in negative.items()})
    if (not all(equal.values()) or not torch.equal(integrated['global'],native['global'])
            or any(not torch.equal(v,native_negative[k])for k,v in negative.items())
            or not torch.equal(cpu_rng,torch.random.get_rng_state())
            or not torch.equal(cuda_rng,torch.cuda.get_rng_state())):
        raise ValueError('warm source native fields/negative/RNG must equal standalone teacher; global must remain native')
    correction_RMS=float((integrated['proj']-native['proj']).float().square().mean().sqrt())
    if not correction_RMS>0:
        raise ValueError('predeclared supported input must actually execute the learned source correction')
    outputs=[]
    hook=trainer.models['denoiser'].register_forward_hook(lambda _,inp,out:outputs.append(out.detach().cpu().clone()))
    paired_losses=[]
    try:
        for use_reference in [False,True]:
            with torch.random.fork_rng(devices=[trainer.device]):
                torch.manual_seed(80217);torch.cuda.manual_seed(80217)
                with torch.no_grad(),torch.autocast('cuda',dtype=trainer.mix_precision_dtype):
                    if use_reference:
                        with temporary_source_module(trainer,reference):loss,_=trainer.training_losses(x_0=target,**pack)
                    else:loss,_=trainer.training_losses(x_0=target,**pack)
                paired_losses.append(float(loss['loss']))
    finally:hook.remove()
    if len(outputs)!=2 or not torch.equal(outputs[0],outputs[1])or paired_losses[0]!=paired_losses[1]:
        raise ValueError('actual native same-noise velocity and flow loss must equal standalone warm source')
    for model in trainer.models.values():model.zero_grad(set_to_none=True)
    # The production source projection uses CUDA bilinear grid-sample
    # backward. Keep its native training algorithm; deterministic mode above
    # belongs only to the paired read-only forwards, not a new train contract.
    torch.use_deterministic_algorithms(False)
    try:
        with torch.random.fork_rng(devices=[trainer.device]):
            torch.manual_seed(80217);torch.cuda.manual_seed(80217)
            with torch.autocast('cuda',dtype=trainer.mix_precision_dtype):loss,_=trainer.training_losses(x_0=target,**pack)
            loss['loss'].backward()
    finally:torch.use_deterministic_algorithms(original_determinism)
    gradients={name:float(p.grad.detach().float().abs().sum())if p.grad is not None else None for name,p in trainable.items()}
    lora_gradient=sum(float(p.grad.detach().float().abs().sum())for p in lora.values()if p.grad is not None)
    if (not math.isfinite(float(loss['loss']))or not lora_gradient>0
            or any(v is None or not math.isfinite(v)or not v>0 for v in gradients.values())
            or any(p.grad is not None for p in module.q2c.parameters())):
        raise ValueError('actual native SS flow loss must reach all19 source tensors and original LoRA; matcher frozen')
    if (len(trainer.optimizer.state)!=0 or any(not torch.equal(v.detach().cpu(),saved[k])for k,v in module.state_dict().items())
            or any(not torch.equal(p.detach().cpu(),lora_before[n])for n,p in lora.items())
            or trainer.deform_conditioner is not module
            or trainer.models['deform_conditioner']is not module
            or trainer.training_models['deform_conditioner']is not module):
        raise ValueError('no optimization and complete native source restoration required')
    report=dict(status='PASS_ACTUAL_NATIVE_SS_WARM_TEACHER_FIELDS_VELOCITY_AND_GRADIENT',variant=variant,
        sample_id=args.sample_id,split='train',actual_native_global_projection_negative_exact_to_standalone=equal,
        conditioning_input_sha256=args.expected_conditioning_sha256,
        actual_RGB_tensor_SHA256=hashlib.sha256(pack['cond'].detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
        actual_GT_SS_latent_SHA256=hashlib.sha256(target.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
        actual_RGB_shape=list(pack['cond'].shape),actual_GT_SS_latent_shape=list(target.shape),
        actual_same_noise_velocity_exact=True,actual_same_noise_flow_losses=paired_losses,
        native_global_and_negative_unchanged=True,warm_source_projection_correction_RMS=correction_RMS,
        actual_source_gradient_L1=gradients,actual_NN_LoRA_gradient_L1=lora_gradient,
        source_tensors=19,NN_LoRA_tensors=360,frozen_matcher_unchanged=True,
        source_and_NN_LoRA_weights_exact_after_backward=True,optimizer_states=0,optimizer_updates=0,
        native_SS_probe_batch=1,p_uncond_during_probe=0,teacher_initialization_updates=112,
        deterministic_readonly_forwards=True,native_nondeterministic_CUDA_backward_preserved=True,
        teacher_labels_not_loaded_for_native_SS_loss=True,cold_native_zero_identity_claim=False,
        micro96_capacity_or_joint_training_or_quality_not_proven=True,
        initialization=module._source_input_initialization,
        input_SHA256={str(p):file_sha256(p)for p in [args.config,args.initialization_spec,Path(spec['checkpoint'])]},
        physical_GPU=gpu,peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
        model_adopted=False,full_goal_complete=False)
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    for model in trainer.models.values():model.zero_grad(set_to_none=True)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
