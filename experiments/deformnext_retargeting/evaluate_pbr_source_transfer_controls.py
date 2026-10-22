"""Read-only fixed4 source-delivery interventions on own trained PBR1900.

This is an explicitly deterministic diagnostic protocol. Independent normal
loads must reproduce each other; differences from the original epoch run are
reported, never renamed as an exact original-epoch replay.
"""
import argparse
import copy
import importlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .cascade_view_roles import ClosedInputs, file_sha256 as sha
from .evaluate_pbr_streaming import aggregate_metrics, tensor_sha
from .evaluate_r_mediated_pbr_initial import audited_sampling, IDENTITY_FIELDS
from .native_pbr_stage_summary import validate_completion
from .pbr_source_transfer_controls import CONTROLS, SavedPBRSourceTransferControl
from .probe_r_mediated_pbr_native_v2 import preload_frozen_naf
from .r_mediated_native_runtime import state_sha
from .r_mediated_pbr_science_metrics import optimizer_token


def fingerprint(value):
    if value is None:
        return {'none':True}
    if isinstance(value,dict):
        return {k:fingerprint(v) for k,v in value.items()}
    if isinstance(value,torch.Tensor) or (hasattr(value,'feats') and hasattr(value,'coords')):
        return tensor_sha(value)
    raise ValueError('explicit native tensor or sparse conditioning required')


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text())
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if gpu not in {'0','1','2','3'} or gpu not in os.environ['MATCHING_V2_ALLOWED_GPUS'].split(','):
        raise ValueError('one exclusive assigned GPU0..3 required')
    if plan['control'] not in CONTROLS or plan['model_mode']!='top1-only':
        raise ValueError('explicit physical Top1 PBR saved-weight control required')
    closed=ClosedInputs(plan['canonical_local_root'])
    def guard():
        closed.check(a.plan,a.plan.parent/'file4-closure.json')
        for path,digest in {**plan['source_SHA256'],**plan['input_SHA256']}.items():
            if sha(path)!=digest:raise ValueError('bound diagnostic source/input changed: '+path)
        for path,receipt in plan['closed_inputs'].items():closed.check(path,receipt)
    guard()
    snapshot=Path(plan['completed_snapshot']);closed.check(snapshot,snapshot.parent/'file4-closure.json')
    trained=json.loads(snapshot.read_text())
    if trained['status']!='PASS_ACTUAL_NATIVE_PBR1900_FIXED4_STAGE_SNAPSHOT' or trained['mode']!='top1-only':
        raise ValueError('own actual Top1 PBR1900 completion required')
    validate_completion('top1-only',**trained['completion_proof'])
    native=Path(trained['native_output'])
    if str(native)!=plan['science_native_output']:raise ValueError('actual model namespace differs')
    cfg=json.loads(Path(plan['config']).read_text())
    expected=copy.deepcopy(cfg);expected['config']=plan['config']
    closed.check(native/'config.json',native/'file4-closure.json')
    if json.loads((native/'config.json').read_text())!=expected:
        raise ValueError('actual complete training configuration differs')
    weights={n:native/'ckpts'/f'{n}_step0001900.pt' for n in ['denoiser','deform_conditioner']}
    for f in weights.values():closed.check(f,native/'checkpoint-archives/step0001900/file4-closure.json')
    validation=cfg['validation'];options=validation['quick_metrics']
    if (options['panel_size'],options['sampling_steps'],options['render_resolution'],options['generation_batch_size'])!=(4,12,256,1):
        raise ValueError('same actual fixed4/native12/CFG3/render256 protocol required')
    from PIL import Image
    from pixal3d import models
    from pixal3d.datasets.deformnext_feature import DeformNeXtFeaturePBR1024
    from pixal3d.trainers.flow_matching.r_mediated_pbr_source import RMediatedPBRNativeTrainer
    from train import setup_rng
    from . import quick_metrics as quick
    from .frozen_projection_streaming import install_frozen_projection_streaming

    torch.cuda.set_device(0);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    setup_rng(0,cfg['training_seed'])
    output=Path(plan['output']);output.mkdir(parents=True,exist_ok=False)
    install_frozen_projection_streaming(max_assets=1,audit_directory=output/'projection-streaming-audit')
    train=DeformNeXtFeaturePBR1024(cfg['data_dir'],**cfg['dataset']['args'])
    dataset=DeformNeXtFeaturePBR1024(validation['roots_file'],**validation['dataset_args'])
    if len(train)!=9139:raise ValueError('original eligible TRAIN constructor9139 required')
    indices=quick.panel_indices(dataset,options);seeds=quick.panel_seeds(dataset,indices,options)
    reference=trained['validation_rows'][-1]
    if ([dataset.rows[i]['sample_id'] for i in indices]!=reference['quick_panel_ids'] or len(indices)!=4
            or any(dataset.rows[i]['split']!='validation' for i in indices)):
        raise ValueError('actual original fixed4 panel required')
    with torch.device('meta'):
        md={n:getattr(models,spec['name'])(**spec['args']) for n,spec in cfg['models'].items()}
    args=copy.deepcopy(cfg['trainer']['args']);args['num_workers']=0
    trainer=RMediatedPBRNativeTrainer(md,train,**args,output_dir=str(output/'constructor'),
        load_dir=None,step=None,wandb_run=None)
    preload=preload_frozen_naf(trainer)
    frozen_base=state_sha((n,v) for n,v in trainer.models['denoiser'].state_dict().items() if 'lora_' not in n)
    matcher=state_sha(trainer.deform_conditioner.q2c.state_dict().items())
    for name,path in weights.items():
        trainer.models[name].load_state_dict(torch.load(path,map_location='cpu',weights_only=True,mmap=True),strict=True)
    if (frozen_base!=state_sha((n,v) for n,v in trainer.models['denoiser'].state_dict().items() if 'lora_' not in n)
            or matcher!=state_sha(trainer.deform_conditioner.q2c.state_dict().items())
            or trainer.optimizer.state or trainer.step!=0):
        raise ValueError('strict own1900 load must retain frozen original base/matcher and fresh unused eval optimizer')
    if (sum(p.requires_grad for p in trainer.models['denoiser'].parameters())!=360
            or sum(p.requires_grad for p in trainer.deform_conditioner.parameters())!=10):
        raise ValueError('own native360 LoRA and physical Top1 source10 required')
    trainer.step=1900;trainer.p_uncond=0.
    for model in trainer.models.values():model.eval()
    # Finish lazy frozen image/position caches before taking immutable hashes.
    # This warmup samples no texture and restores Torch RNG; GT Shape is the
    # declared stage condition, while PBR GT values never enter conditioning.
    from pixal3d.utils.data_utils import recursive_to_device
    with torch.random.fork_rng(devices=[0]),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        pack=dataset.collate_fn([dataset[indices[0]]]);pack=recursive_to_device(pack,trainer.device)
        pack.pop('_dataset_name',None);pack.pop('_sha256',None)
        target=pack.pop('x_0');pack['coords']=target.coords
        trainer.get_inference_cond(**pack)
        del pack,target
    before={n:state_sha(m.state_dict().items()) for n,m in trainer.models.items()}
    encoder_before=state_sha(trainer.image_cond_model.state_dict().items())
    token=optimizer_token(trainer)
    hook=SavedPBRSourceTransferControl(trainer.deform_conditioner,plan['control'])
    pbr_module=importlib.import_module('pixal3d.trainers.flow_matching.r_mediated_pbr_source')
    original_add=pbr_module.add_source_ray_correction;original_condition=trainer.get_inference_cond
    consumer,conditions=[],[]
    def added(native_condition,delta):
        result=original_add(native_condition,delta)
        if result['global'] is not native_condition['global']:
            raise ValueError('source intervention changed native global object')
        if plan['control']=='both-source-contributions-off' and (
                torch.count_nonzero(delta) or not torch.equal(result['proj'],native_condition['proj'])):
            raise ValueError('both-off must restore exact native projected conditioning')
        consumer.append(dict(native_global_SHA256=tensor_sha(native_condition['global']),
            native_proj_SHA256=tensor_sha(native_condition['proj']),
            actual_delta_SHA256=tensor_sha(delta),actual_delta_RMS=float(delta.float().square().mean().sqrt()),
            actual_consumer_proj_SHA256=tensor_sha(result['proj']),
            native_global_object_preserved=True))
        return result
    def conditioned(**kwargs):
        result=original_condition(**kwargs)
        conditions.append(dict(native_global=fingerprint(result['cond']['global']),
            negative=fingerprint(result['neg_cond']),shape_condition=fingerprint(result['concat_cond'])))
        return result
    pbr_module.add_source_ray_correction=added;trainer.get_inference_cond=conditioned
    events,numeric,panels,pixels=[],[],[],[];began=time.monotonic();torch.cuda.reset_peak_memory_stats()
    try:
        with audited_sampling(quick,trainer,events),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for number,(index,seed) in enumerate(zip(indices,seeds)):
                metrics,panel=quick._pbr_metrics(trainer,dataset,[index],dict(options,seed=seed))
                if (len(events)!=number+1 or len(hook.events)!=number+1 or len(consumer)!=number+1
                        or len(conditions)!=number+1 or not all(math.isfinite(v) for v in metrics.values())
                        or panel.shape!=(256,768,3) or panel.dtype!=np.uint8):
                    raise ValueError('one actual complete fixed4 control event per generated asset required')
                row=events[-1];row['metrics']=metrics
                original=reference['quick_per_asset'][number]
                if any(row[k]!=original[k] for k in IDENTITY_FIELDS if k!='actual_prediction_PBR_latent_SHA256'):
                    raise ValueError('actual diagnostic RGB/noise/GT/Shape/camera changed from own native1900')
                numeric.append(metrics);panels.append(panel)
                pixels.append(dict(GT=sha_pixels(panel[:,:256]),prediction=sha_pixels(panel[:,256:512])))
                Image.fromarray(panel).save(output/f'case{number:03d}.png')
                (output/'progress.json').write_text(json.dumps(dict(completed=number+1,expected=4))+'\n')
    finally:
        hook.close();pbr_module.add_source_ray_correction=original_add;trainer.get_inference_cond=original_condition
    after={n:state_sha(m.state_dict().items()) for n,m in trainer.models.items()}
    if before!=after or encoder_before!=state_sha(trainer.image_cond_model.state_dict().items()) or token!=optimizer_token(trainer):
        raise ValueError('diagnostic modified weights/encoder/optimizer/sampler/step')
    report=dict(status='PASS_ACTUAL_NATIVE_PBR1900_SOURCE_TRANSFER_READOUT_FIXED4',
        diagnostic_key=plan['diagnostic_key'],control=plan['control'],model_mode='top1-only',
        actual_trained_model_updates=1900,source_multiview_step=16850,physical_GPU=gpu,
        optimizer_updates_in_diagnostic=0,unused_eval_optimizer_states=0,
        actual_model_files={k:dict(path=str(v),SHA256=sha(v)) for k,v in weights.items()},
        actual_model_state_SHA256=before,actual_frozen_encoder_SHA256=encoder_before,
        frozen_original_base_and_matcher_exact=True,all_weights_unchanged=True,
        original_science_config=plan['config'],evaluation_constructor_only_override=dict(num_workers=0),
        NAF_preload=preload,frozen_condition_warmup_before_hash_and_no_sampling=True,actual_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        protocol='new deterministic saved-weight fixed4 source-transfer readout; independent normal repeat required',
        actual_original_epoch_prediction_exact=[a['actual_prediction_PBR_latent_SHA256']==b['actual_prediction_PBR_latent_SHA256'] for a,b in zip(events,reference['quick_per_asset'])],
        actual_original_epoch_GT_pixels_exact=[v['GT']==trained['GT_PNG_pixel_SHA256']['1900'][row['sample_id']] for v,row in zip(pixels,events)],
        original_epoch_metrics=reference['quick_metrics'],per_asset=events,quick_metrics=aggregate_metrics(numeric),
        module_events=hook.events,consumer_events=consumer,condition_events=conditions,panel_pixel_SHA256=pixels,
        fixed_GT_target_shape=True,full_cascade=False,source_preservation=False,
        sampling_steps=12,guidance=3.,render_resolution=256,generation_batch=1,
        inference_interventions_not_retrained_ablation=True,no_uniform_NoScore_or_spatial_NoCorr_claim=True,
        quality_gain_proven=False,model_adopted=False,full_goal_complete=False,
        verified_inputs=closed.inputs,seconds=time.monotonic()-began,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
    guard();(output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    Image.fromarray(np.concatenate(panels,axis=0)).save(output/'panel.png')
    print(json.dumps(dict(status=report['status'],diagnostic_key=plan['diagnostic_key'],assets=4)),flush=True)


def sha_pixels(array):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


if __name__=='__main__':main()
