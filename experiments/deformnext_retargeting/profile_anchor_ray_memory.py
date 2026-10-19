"""Bounded cohost capacity test; discard all temporary composer updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import psutil
import torch
from PIL import Image
import numpy as np

from pixal3d import datasets,models
from pixal3d.utils.data_utils import recursive_to_device
from .anchor_ray_runtime import AnchorRayCompositionTrainer
from .verify_anchor_ray_ss_gradient import tensor_digest


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--identity-proof',type=Path,required=True)
    p.add_argument('--SS-gradient-proof',type=Path,required=True);p.add_argument('--training-cache-index',type=Path,required=True)
    p.add_argument('--microbatch',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--physical-gpu',required=True);p.add_argument('--protected-train-pid',type=int,required=True)
    args=p.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==args.physical_gpu
    assert args.physical_gpu in os.environ['MATCHING_V2_ALLOWED_GPUS'].split(',')
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.25,0)
    torch.use_deterministic_algorithms(True)
    free,total=torch.cuda.mem_get_info();assert free > .25*total+6*2**30,'cohost margin unavailable'
    protected=psutil.Process(args.protected_train_pid);created=protected.create_time();command=protected.cmdline()
    gradient=json.loads(args.SS_gradient_proof.read_text());identity=json.loads(args.identity_proof.read_text())
    assert gradient['status']=='PASS_ACTUAL_SS_LOSS_COMPOSER_GRADIENT_FROZEN_BACKBONE_AND_R'
    sources=dict(gradient['source_SHA256']);sources[str(Path(__file__).resolve())]=sha(__file__)
    assert all(sha(path)==digest for path,digest in sources.items())
    args.output.mkdir(exist_ok=False,parents=True)
    cfg=json.loads(args.config.read_text());ds=getattr(datasets,cfg['dataset']['name'])(cfg['data_dir'],**cfg['dataset']['args'])
    rows=[json.loads(line) for line in args.training_cache_index.read_text().splitlines()]
    eligible=[r['sample_id'] for r in rows if r['role']=='train' and r['usable_queries']>0][:32]
    index={r['sample_id']:i for i,r in enumerate(ds.rows)};areas=[]
    for sid in eligible:
        row=ds.rows[index[sid]]
        with Image.open(ds._resolve(row['target_anchor'])) as image:
            alpha=np.asarray(image.convert('RGBA').resize((512,512),Image.Resampling.NEAREST))[...,3]
        areas.append(int((alpha[8::16,8::16]>127).sum()))
    selected=eligible[int(np.argmax(areas))];item=ds[index[selected]]
    with torch.device('meta'):
        modules={name:getattr(models,spec['name'])(**spec['args']) for name,spec in cfg['models'].items()}
    kw=dict(cfg['trainer']['args']);kw.update(batch_size_per_gpu=args.microbatch,batch_split=1,num_workers=0,
        prefetch_data=False,max_steps=2,p_uncond=0.,lr_scheduler=None)
    trainer=AnchorRayCompositionTrainer(modules,ds,**kw,output_dir=str(args.output),load_dir=None,step=None,wandb_run=None)
    module=trainer.deform_conditioner;denoiser=trainer.models['denoiser'];module.train();denoiser.train()
    module.ray_mix.data.fill_(.2)
    before=tensor_digest(denoiser)
    reference={key:value.detach().cpu().clone() for key,value in module.reference_state().items()}
    pack=recursive_to_device(ds.collate_fn([item]*args.microbatch),torch.device('cuda:0'))
    torch.manual_seed(101);torch.cuda.manual_seed_all(101);steps=[];started=time.monotonic()
    try:
        for update in range(2):
            torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
            result=trainer.run_step([pack]);torch.cuda.synchronize()
            assert np.isfinite(result['loss']['loss']) and len(trainer.optimizer.state)==6
            assert 0<=float(module.ray_mix.detach())<=1
            steps.append(dict(update=update+1,loss=result['loss']['loss'],elapsed_seconds=time.monotonic()-begin,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,gate=float(module.ray_mix.detach())))
            print(json.dumps(steps[-1]),flush=True)
    except torch.OutOfMemoryError as error:
        result=dict(status='BOUNDED_CAPACITY_OOM_NOT_SCIENTIFIC_TRAINING',microbatch=args.microbatch,
            allocator_fraction=.25,completed_temporary_updates=len(steps),error=str(error),
            protected_PID=args.protected_train_pid,protected_create_time=created,SS_and_R_updates=0,
            scientific_training_started=False,source_SHA256=sources)
        with (args.output/'capacity.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
        print(json.dumps(result),flush=True);return
    assert tensor_digest(denoiser)==before
    assert all(torch.equal(value,module.reference_state()[key].detach().cpu()) for key,value in reference.items())
    assert all(sha(path)==digest for path,digest in sources.items())
    protected=psutil.Process(args.protected_train_pid)
    assert protected.create_time()==created and protected.cmdline()==command and protected.status()!=psutil.STATUS_ZOMBIE
    result=dict(status='PASS_ACTUAL_ANCHOR_RAY_SS_FORWARD_BACKWARD_ADAM_COHOST_CAPACITY',microbatch=args.microbatch,
        accumulation=1,physical_GPU=args.physical_gpu,allocator_fraction=.25,
        train_stress_asset=selected,stress_target_foreground_patches=max(areas),input_RGB_stress_selection_assets=32,
        input_selection_uses_GT=False,steps=steps,peak_allocated_GiB=max(x['peak_allocated_GiB'] for x in steps),
        elapsed_seconds=time.monotonic()-started,optimizer_states=6,disposable_composer_updates=2,
        SS_and_R_updates=0,all_original_SS_R_matching_fusion_tensors_exact=True,
        protected_PID=args.protected_train_pid,protected_create_time=created,protected_main_training_live_unchanged=True,
        denoiser_state_SHA256=before,source_SHA256=sources,config_SHA256=sha(args.config),
        scientific_training_started=False,probe_model_discarded=True,
        scope='ActualfrozenSSpluscomposer6 trainloss,twoAdamupdates,declaredpoint2gate;25percentallocatorwhilecanonicalmainlive. RepeatedinputRGBstressasset;notwhole-datasetmaxVRAMorqualityproof.')
    with (args.output/'capacity.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=result['status'],microbatch=args.microbatch,peak_allocated_GiB=result['peak_allocated_GiB'],elapsed_seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
