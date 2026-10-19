"""Actual SS-loss gradient proof on a disposable, memory-bounded training model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from pixal3d import datasets,models
from pixal3d.utils.data_utils import recursive_to_device
from .anchor_ray_runtime import AnchorRayCompositionTrainer


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(module):
    digest=hashlib.sha256()
    for name,value in module.state_dict().items():
        digest.update((name+str(value.dtype)+str(tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--identity-proof',type=Path,required=True)
    parser.add_argument('--training-cache-index',type=Path,required=True)
    parser.add_argument('--frozen-input-contract',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--physical-gpu',required=True)
    args=parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']==args.physical_gpu
    assert args.physical_gpu in os.environ['MATCHING_V2_ALLOWED_GPUS'].split(',')
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.2,0)
    torch.use_deterministic_algorithms(True)
    cfg=json.loads(args.config.read_text());identity=json.loads(args.identity_proof.read_text())
    assert identity['status']=='PASS_ACTUAL_OLD_R_NEAREST_ZERO_GATE_CONDITION_IDENTITY_AND_PROJECTION_GRADIENT'
    frozen=json.loads(args.frozen_input_contract.read_text())['source_SHA256']
    sources={**frozen,**identity['source_SHA256'],str(Path(__file__).resolve()):sha(__file__)}
    assert all(sha(path)==digest for path,digest in sources.items())
    args.output.mkdir(exist_ok=False,parents=True)
    ds=getattr(datasets,cfg['dataset']['name'])(cfg['data_dir'],**cfg['dataset']['args'])
    rows=[json.loads(line) for line in args.training_cache_index.read_text().splitlines()]
    ids=[row['sample_id'] for row in rows if row['role']=='train' and row['usable_queries']>0][:2]
    index={row['sample_id']:i for i,row in enumerate(ds.rows)}
    batch=ds.collate_fn([ds[index[sid]] for sid in ids])
    with torch.device('meta'):
        modules={name:getattr(models,spec['name'])(**spec['args']) for name,spec in cfg['models'].items()}
    kwargs=dict(cfg['trainer']['args']);kwargs.update(batch_size_per_gpu=2,batch_split=1,num_workers=0,
        prefetch_data=False,max_steps=1,p_uncond=0.)
    trainer=AnchorRayCompositionTrainer(modules,ds,**kwargs,output_dir=str(args.output),load_dir=None,step=None,wandb_run=None)
    module=trainer.deform_conditioner;denoiser=trainer.models['denoiser']
    module.train();denoiser.train()
    assert len(trainer.model_params)==6 and all(not parameter.requires_grad for parameter in denoiser.parameters())
    before=tensor_digest(denoiser);reference={key:value.detach().cpu().clone() for key,value in module.reference_state().items()}
    initial={key:value.detach().cpu().clone() for key,value in module.state_dict().items()}
    pack=recursive_to_device(batch,torch.device('cuda:0'));results=[]
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    # Stage0 measures the exact initial model. Stage1 activates the soft branch
    # as a declared diagnostic, then takes ONE disposable optimizer update.
    for stage,gate,update in [('initial-zero',0.,False),('active-soft',.2,True),('after-soft-update',None,False)]:
        if gate is not None:module.ray_mix.data.fill_(gate)
        trainer.optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(101);torch.cuda.manual_seed_all(101)
        with torch.autocast('cuda',dtype=torch.bfloat16):losses,_=trainer.training_losses(**pack)
        loss=losses['loss'];assert torch.isfinite(loss);loss.backward()
        gradients={name:float(parameter.grad.detach().float().abs().sum()) if parameter.grad is not None else None
            for name,parameter in module.named_parameters() if parameter.requires_grad}
        assert all(value is not None and torch.isfinite(torch.tensor(value)) for value in gradients.values())
        assert gradients['ray_mix']>0
        assert all(parameter.grad is None for parameter in denoiser.parameters())
        assert all(parameter.grad is None for name,parameter in module.named_parameters()
            if name!='ray_mix' and not name.startswith('ray_composer.'))
        if stage=='active-soft':assert gradients['ray_composer.content_output.weight']>0
        if stage=='after-soft-update':
            assert all(gradients['ray_composer.'+key+'.weight']>0 for key in ['query','context','key'])
        if update:
            torch.nn.utils.clip_grad_norm_(trainer.model_params,1.);trainer.optimizer.step()
            assert len(trainer.optimizer.state)==6
        row=dict(stage=stage,loss=float(loss.detach()),gate=float(module.ray_mix.detach()),gradient_L1=gradients,
            disposable_optimizer_update=update,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
        results.append(row);print(json.dumps(row),flush=True)
        del losses,loss
    torch.cuda.synchronize();after=tensor_digest(denoiser)
    assert before==after
    assert all(torch.equal(value,module.reference_state()[key].detach().cpu()) for key,value in reference.items())
    assert all(sha(path)==digest for path,digest in sources.items())
    module.load_state_dict(initial,strict=True);trainer.optimizer.zero_grad(set_to_none=True)
    assert float(module.ray_mix.detach())==0
    result=dict(status='PASS_ACTUAL_SS_LOSS_COMPOSER_GRADIENT_FROZEN_BACKBONE_AND_R',
        training_assets=ids,microbatch=2,gradient_stages=results,denoiser_state_SHA256_before=before,
        denoiser_state_SHA256_after=after,all_original_R_matching_fusion_tensors_exact=True,
        trainable_tensors=6,optimizer_states_after_temporary_update=6,disposable_composer_optimizer_updates=1,
        SS_LoRA_optimizer_updates=0,R_optimizer_updates=0,scientific_training_started=False,
        model_restored_to_exact_zero_gate_initializer=True,probe_model_discarded=True,GT_in_forward=False,
        GT_SS_latent_used_only_in_flow_loss=True,p_uncond_for_capacity_proof=0.,
        GPU=args.physical_gpu,allocator_fraction=.2,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        elapsed_seconds=time.monotonic()-started,source_SHA256=sources,
        scope='ThreeactualSSforward/backwards on2trainassets;gatezero and activebranches,one disposable composer update;not scientifictraining,largemicrocapacity orqualityproof.')
    with (args.output/'audit.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=result['status'],peak_allocated_GiB=result['peak_allocated_GiB'],elapsed_seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
