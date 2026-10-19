"""Actual SS initial-loss identity and zero-output feature-readout learnability."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from pixal3d import datasets, models
from pixal3d.modules.ray_feature_attention import RayFeatureAttentionConditioner
from pixal3d.utils.data_utils import recursive_to_device
from .anchor_ray_runtime import AnchorRayCompositionTrainer
from .verify_anchor_ray_ss_gradient import sha, tensor_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cache-index', type=Path, required=True)
    parser.add_argument('--utility-head', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible not in {'0','1','2','3'} or visible not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','):
        raise ValueError('one assigned physical GPU0-3 required')
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.2, 0)
    torch.use_deterministic_algorithms(True)
    cfg = json.loads(args.config.read_text())
    sources = {str(Path(__file__).resolve()): sha(__file__),
        str(Path(__file__).parents[2]/'pixal3d/modules/ray_feature_attention.py'):
            sha(Path(__file__).parents[2]/'pixal3d/modules/ray_feature_attention.py'),
        str(args.config): sha(args.config), str(args.utility_head): sha(args.utility_head)}
    args.output.mkdir(parents=True, exist_ok=False)
    ds = getattr(datasets, cfg['dataset']['name'])(cfg['data_dir'], **cfg['dataset']['args'])
    rows = [json.loads(line) for line in args.cache_index.read_text().splitlines()]
    selected = [row['sample_id'] for row in rows if row['role']=='train' and row['usable_queries']>0][:2]
    index = {row['sample_id']: i for i,row in enumerate(ds.rows)}
    batch = ds.collate_fn([ds[index[sid]] for sid in selected])
    with torch.device('meta'):
        md = {name: getattr(models,spec['name'])(**spec['args']) for name,spec in cfg['models'].items()}
    kwargs = dict(cfg['trainer']['args'])
    kwargs.update(batch_size_per_gpu=2,batch_split=1,num_workers=0,prefetch_data=False,max_steps=1,p_uncond=0.)
    trainer = AnchorRayCompositionTrainer(md,ds,**kwargs,output_dir=str(args.output),load_dir=None,step=None,wandb_run=None)
    denoiser = trainer.models['denoiser'];denoiser.train()
    old = trainer.deform_conditioner
    reference = {name: value.detach().cpu().clone() for name,value in old.reference_state().items()}
    pack = recursive_to_device(batch, torch.device('cuda:0'))

    def loss():
        torch.manual_seed(101);torch.cuda.manual_seed_all(101)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            value,_ = trainer.training_losses(**pack)
        return value['loss']

    with torch.no_grad(): original_loss = float(loss())
    original_evidence = trainer._last_deform_condition.evidence.detach().clone()
    feature = RayFeatureAttentionConditioner(**cfg['trainer']['args']['deform_conditioner']).cuda()
    feature.load_reference(reference)
    feature.ray_readout.score_selector.load_state_dict(torch.load(args.utility_head,map_location='cpu',weights_only=True),strict=True)
    # This is an explicit single-process diagnostic constructor intervention;
    # it never changes any saved training model or frozen registry file.
    trainer.deform_conditioner = feature
    trainer.models['deform_conditioner'] = feature
    trainer.training_models['deform_conditioner'] = feature
    parameters = list(feature.ray_readout.parameters())
    if len(parameters)!=9 or any(p.requires_grad for p in denoiser.parameters()):
        raise ValueError('exactly nine feature tensors and frozen NN required')
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.01)
    before = tensor_digest(denoiser)
    results=[];began=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True);value=loss()
        if not torch.isfinite(value):raise ValueError('nonfinite SS loss')
        if step==0:
            if float(value.detach())!=original_loss or not torch.equal(trainer._last_deform_condition.evidence,original_evidence):
                raise ValueError('new zero-output feature branch changed actual reference SS computation')
        value.backward()
        gradients={name:float(p.grad.detach().float().abs().sum()) if p.grad is not None else None
                   for name,p in feature.ray_readout.named_parameters()}
        if any(v is None or not torch.isfinite(torch.tensor(v)) for v in gradients.values()):
            raise ValueError('nonfinite or absent feature-readout gradient')
        if gradients['output.weight']<=0:
            raise ValueError('zero-output projection must learn immediately without a gate')
        if step==2 and any(v<=0 for v in gradients.values()):
            raise ValueError('all nine feature tensors must learn after zero-output warm-up')
        if any(p.grad is not None for p in denoiser.parameters()):
            raise ValueError('frozen denoiser received a gradient')
        torch.nn.utils.clip_grad_norm_(parameters,1.)
        optimizer.step()
        row=dict(step=step+1,loss=float(value.detach()),gradient_L1=gradients,
                 feature_stats=feature.last_feature_attention_stats)
        results.append(row);print(json.dumps(row),flush=True)
    if tensor_digest(denoiser)!=before or any(not torch.equal(value,feature.reference_state()[name].detach().cpu()) for name,value in reference.items()):
        raise ValueError('NN or original R/matcher/fusion changed in the disposable head-only proof')
    if any(sha(path)!=digest for path,digest in sources.items()):raise ValueError('probe source changed')
    result=dict(status='PASS_ACTUAL_SS_ZERO_FEATURE_OUTPUT_IDENTITY_AND_NINE_TENSOR_GRADIENT',
        reference_initial_loss=original_loss,new_initial_loss=results[0]['loss'],
        actual_initial_evidence_exact=True,training_assets=selected,temporary_optimizer_updates=3,
        scientific_training_updates=0,SS_LoRA_updates=0,R_updates=0,
        all_original_NN_R_matching_fusion_exact=True,scalar_zero_gate=False,
        nine_feature_tensors_receive_gradient_by_update3=True,optimizer_states=len(optimizer.state),
        gradients=results,physical_GPU=visible,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        seconds=time.monotonic()-began,source_SHA256=sources,probe_model_discarded=True,
        scope='Actual twoTRAINasset SS loss/conditioning proof,not generatedSS32 parity or performance improvement.')
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
